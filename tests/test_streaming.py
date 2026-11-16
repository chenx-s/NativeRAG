"""流式事件（RAGChain.astream_events）的契约测试：不访问网络、不打开 Milvus、不调用真实 LLM。"""

import os
import unittest
from unittest.mock import AsyncMock, patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.rag import RAGChain
from app.rag.prompts import FALLBACK_ANSWER

DOC = Document(page_content="RRF 用排名融合多路检索结果。", metadata={"source": "a.txt", "filename": "a.txt"})


def _final_text(events):
    """按约定还原最终答案：最后一次 reset 之后的 token 拼接。"""
    text = ""
    for event in events:
        if event["event"] == "reset":
            text = ""
        elif event["event"] == "token":
            text += event["text"]
    return text


class StreamEventTests(unittest.IsolatedAsyncioTestCase):
    async def _collect(self, *, docs, llm_responses, groundedness_retries=1):
        llm = FakeListChatModel(responses=llm_responses)
        patches = [
            patch("app.config.settings.QUERY_TRANSFORM", "none"),
            patch("app.config.settings.MAX_RETRIES", 0),
            patch("app.config.settings.WEB_SEARCH_ENABLED", False),
            patch("app.config.settings.RERANK_ENABLED", False),
            patch("app.config.settings.GROUNDEDNESS_ACTION", "retry"),
            patch("app.config.settings.MAX_GROUNDEDNESS_RETRIES", groundedness_retries),
            patch("app.vector_store.amulti_query_search", new=AsyncMock(return_value=docs)),
            patch("app.rag.llm.get_llm", new=lambda **_: llm),
        ]
        for p in patches:
            p.start()
        try:
            return [event async for event in RAGChain(warm=False).astream_events("什么是 RRF")]
        finally:
            for p in reversed(patches):
                p.stop()

    async def test_tokens_steps_and_done(self):
        # 调用顺序：相关性评分 yes -> 生成 -> 忠实性检查 yes
        events = await self._collect(docs=[DOC], llm_responses=["yes", "RRF 是排名融合", "yes"])

        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[-1], "done")
        self.assertIn("step", kinds)
        self.assertNotIn("reset", kinds)
        self.assertEqual(_final_text(events), "RRF 是排名融合")
        self.assertEqual(events[-1]["result"]["answer"], "RRF 是排名融合")
        labels = [e["label"] for e in events if e["event"] == "step"]
        self.assertIn("生成回答", labels)

    async def test_regeneration_emits_reset(self):
        # 草稿未通过 -> 严格模式重生成 -> 通过
        events = await self._collect(
            docs=[DOC], llm_responses=["yes", "草稿", "no", "终稿", "yes"]
        )

        resets = [e for e in events if e["event"] == "reset"]
        self.assertEqual([r["reason"] for r in resets], ["regenerate"])
        self.assertEqual(_final_text(events), "终稿")
        result = events[-1]["result"]
        self.assertEqual(result["answer"], "终稿")
        self.assertEqual(result["groundedness_retries"], 1)

    async def test_rejected_answer_is_replaced(self):
        # 没有重生成额度：草稿未通过 -> 替换为拒答
        events = await self._collect(
            docs=[DOC], llm_responses=["yes", "草稿", "no"], groundedness_retries=0
        )

        resets = [e for e in events if e["event"] == "reset"]
        self.assertEqual([r["reason"] for r in resets], ["replaced"])
        self.assertEqual(_final_text(events), FALLBACK_ANSWER)

    async def test_fallback_answer_is_sent_as_one_token(self):
        events = await self._collect(docs=[], llm_responses=["unused"])

        tokens = [e for e in events if e["event"] == "token"]
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0]["text"], FALLBACK_ANSWER)
        self.assertEqual(events[-1]["result"]["groundedness"], "skipped")


if __name__ == "__main__":
    unittest.main()
