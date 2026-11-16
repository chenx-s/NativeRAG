"""OpenAI 兼容接口（/v1）测试：用假的 RAG 对象，不依赖模型、向量库和网络。"""

import json
import os
import unittest
from unittest.mock import patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.openai_compat import (
    REGENERATE_NOTICE,
    ChatMessage,
    build_router,
    clean_assistant_text,
    format_sources,
    parse_messages,
)

SOURCES = [
    {"source_type": "local", "filename": "a.pdf", "page": 3, "source": "data/a.pdf"},
    {"source_type": "web", "filename": "示例网页", "source": "https://example.com"},
]


class FakeRAG:
    def __init__(self):
        self.calls = []

    async def aanswer(self, question, **kwargs):
        self.calls.append((question, kwargs))
        return {"answer": f"答：{question}", "sources": SOURCES}

    async def astream_events(self, question, **kwargs):
        self.calls.append((question, kwargs))
        yield {"event": "step", "node": "retrieve", "label": "检索知识库"}
        yield {"event": "token", "text": "草"}
        yield {"event": "reset", "reason": "regenerate"}
        yield {"event": "token", "text": "终"}
        yield {"event": "token", "text": "稿"}
        yield {"event": "done", "result": {"answer": "终稿", "sources": SOURCES}}


def _client(rag):
    app = FastAPI()
    app.include_router(build_router(lambda: rag))
    return TestClient(app)


class MessageParsingTests(unittest.TestCase):
    def test_question_and_history(self):
        messages = [
            ChatMessage(role="system", content="你是助手"),
            ChatMessage(role="user", content="RRF 是什么"),
            ChatMessage(role="assistant", content="排名融合" + format_sources(SOURCES)),
            ChatMessage(role="user", content=[{"type": "text", "text": "它的公式呢"}]),
        ]
        question, history = parse_messages(messages)

        self.assertEqual(question, "它的公式呢")
        # 参考来源列表不应进入历史
        self.assertEqual(history, [("RRF 是什么", "排名融合")])

    def test_clean_assistant_text_keeps_final_version_only(self):
        text = "<think>检索中</think>草稿内容" + REGENERATE_NOTICE + "最终内容" + format_sources(SOURCES)
        self.assertEqual(clean_assistant_text(text), "最终内容")

    def test_format_sources(self):
        rendered = format_sources(SOURCES)
        self.assertIn("a.pdf，第 3 页", rendered)
        self.assertIn("[示例网页](https://example.com)", rendered)


class ChatCompletionsTests(unittest.TestCase):
    def test_models(self):
        response = _client(FakeRAG()).get("/v1/models")
        ids = [m["id"] for m in response.json()["data"]]
        self.assertEqual(ids, ["rag-agent", "rag-agent-local"])

    def test_non_stream(self):
        rag = FakeRAG()
        response = _client(rag).post("/v1/chat/completions", json={
            "model": "rag-agent-local",
            "messages": [{"role": "user", "content": "你好"}],
        })
        self.assertEqual(response.status_code, 200)
        content = response.json()["choices"][0]["message"]["content"]
        self.assertTrue(content.startswith("答：你好"))
        self.assertIn("参考来源", content)
        # -local 模型不联网
        self.assertFalse(rag.calls[0][1]["allow_web"])

    def test_stream(self):
        rag = FakeRAG()
        with _client(rag).stream("POST", "/v1/chat/completions", json={
            "model": "rag-agent",
            "stream": True,
            "messages": [{"role": "user", "content": "你好"}],
        }) as response:
            lines = [line for line in response.iter_lines() if line.startswith("data: ")]

        self.assertEqual(lines[-1], "data: [DONE]")
        deltas = [json.loads(line[6:])["choices"][0]["delta"] for line in lines[:-1]]
        content = "".join(d.get("content", "") for d in deltas)
        reasoning = "".join(d.get("reasoning_content", "") for d in deltas)

        self.assertIn("检索知识库", reasoning)
        self.assertIn(REGENERATE_NOTICE.strip(), content)
        self.assertIn("终稿", content)
        self.assertTrue(rag.calls[0][1]["allow_web"])

    def test_unknown_model(self):
        response = _client(FakeRAG()).post("/v1/chat/completions", json={
            "model": "gpt-4", "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(response.status_code, 404)

    def test_session_header_uses_server_memory(self):
        rag = FakeRAG()
        _client(rag).post(
            "/v1/chat/completions",
            headers={"X-Session-Id": "user-1"},
            json={"model": "rag-agent", "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(rag.calls[0][1]["session_id"], "user-1")
        self.assertNotIn("history", rag.calls[0][1])

    def test_api_token(self):
        client = _client(FakeRAG())
        with patch("app.config.settings.API_TOKEN", "secret"):
            self.assertEqual(client.get("/v1/models").status_code, 401)
            ok = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
            self.assertEqual(ok.status_code, 200)


if __name__ == "__main__":
    unittest.main()
