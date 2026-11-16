import os
import unittest
from unittest.mock import AsyncMock, patch

# Tests must not emit external LangSmith traces from the developer's .env.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.rag import RAGChain
from app.rag.helpers import parse_route
from app.rag.prompts import WEB_FALLBACK_ANSWER

# 依赖统一在"被调用的模块"上替换：节点通过 vector_store.xxx / web_search.xxx / llm.get_llm 调用
SETTINGS = "app.config.settings"


def _settings(**values):
    return [patch(f"{SETTINGS}.{key}", value) for key, value in values.items()]


class AsyncRAGTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_retrieval_runs_through_async_graph(self):
        rag = RAGChain(warm=False)

        patches = _settings(QUERY_TRANSFORM="none", MAX_RETRIES=0, WEB_SEARCH_ENABLED=False) + [
            patch("app.vector_store.amulti_query_search", new=AsyncMock(return_value=[])),
        ]
        for p in patches:
            p.start()
        try:
            result = await rag.aanswer("没有命中的问题")
        finally:
            for p in reversed(patches):
                p.stop()

        self.assertEqual(result["num_documents"], 0)
        self.assertEqual(result["groundedness"], "skipped")
        self.assertEqual(result["queries"], ["没有命中的问题"])
        self.assertFalse(result["web_searched"])
        self.assertEqual(result["steps"][0], "route_question")
        self.assertNotIn("web_search", result["steps"])


class WebFallbackGraphTests(unittest.IsolatedAsyncioTestCase):
    """联网搜索作为图节点的路由行为（不访问网络、不打开 Milvus）。"""

    def _patches(self, search_mock, *, kb_count=10, routing="off", max_retries=2):
        return _settings(
            QUERY_TRANSFORM="none",
            MAX_RETRIES=max_retries,
            WEB_FALLBACK_AFTER_RETRIES=0,
            WEB_SEARCH_ENABLED=True,
            TAVILY_API_KEY="tvly-test",
            WEB_ROUTING=routing,
            GROUNDEDNESS_ACTION="off",
        ) + [
            patch("app.vector_store.acount", new=AsyncMock(return_value=kb_count)),
            patch("app.vector_store.amulti_query_search", new=AsyncMock(return_value=[])),
            patch("app.web_search.asearch_web", new=search_mock),
        ]

    async def _run(self, patches, question="2026 年最新的新闻", **kwargs):
        rag = RAGChain(warm=False)
        for p in patches:
            p.start()
        try:
            return await rag.aanswer(question, **kwargs)
        finally:
            for p in reversed(patches):
                p.stop()

    async def test_local_miss_falls_back_to_web_without_rewrite(self):
        search = AsyncMock(return_value=[])
        result = await self._run(self._patches(search))

        search.assert_awaited_once()
        self.assertTrue(result["web_searched"])
        self.assertEqual(result["source_type"], "web")
        # 联网可用且 WEB_FALLBACK_AFTER_RETRIES=0 时，不再做无谓的改写重检索
        self.assertEqual(
            result["steps"],
            ["route_question", "transform_query", "retrieve",
             "rerank_documents", "grade_document", "web_search",
             "generate", "check_groundedness"],
        )
        self.assertEqual(result["answer"], WEB_FALLBACK_ANSWER)

    async def test_allow_web_false_never_searches(self):
        search = AsyncMock(return_value=[])
        # 不联网时本地失败会走"改写重检索"，这里关掉改写，避免触发 LLM 调用
        result = await self._run(self._patches(search, max_retries=0), allow_web=False)

        search.assert_not_awaited()
        self.assertFalse(result["web_searched"])

    async def test_empty_knowledge_base_goes_straight_to_web(self):
        search = AsyncMock(return_value=[])
        result = await self._run(self._patches(search, kb_count=0, routing="llm"))

        self.assertEqual(result["route"], "web")
        self.assertEqual(result["steps"][:2], ["route_question", "web_search"])

    async def test_web_documents_are_returned_as_sources(self):
        docs = [
            Document(
                page_content="网页正文",
                metadata={"source": "https://example.com/a", "filename": "示例网页",
                          "source_type": "web", "relevance_score": 0.9},
            )
        ]
        search = AsyncMock(return_value=docs)
        answer = "答案 [示例网页](https://example.com/a)"
        patches = self._patches(search) + [
            patch("app.rag.llm.get_llm", new=lambda **_: FakeListChatModel(responses=[answer])),
        ]
        result = await self._run(patches)

        self.assertEqual(result["source_type"], "web")
        self.assertEqual(result["sources"][0]["source_type"], "web")
        self.assertEqual(result["sources"][0]["source"], "https://example.com/a")
        self.assertEqual(result["answer"], answer)

    async def test_web_failure_is_reported_not_raised(self):
        from app.web_search import WebSearchError

        search = AsyncMock(side_effect=WebSearchError("Tavily 额度已用完"))
        result = await self._run(self._patches(search))

        self.assertTrue(result["web_searched"])
        self.assertIn("额度", result["web_error"])
        self.assertIn("额度", result["answer"])


class ParseRouteTests(unittest.TestCase):
    def test_parse_route(self):
        self.assertEqual(parse_route("路由：web\n问题：今天北京天气"), ("web", "今天北京天气"))
        self.assertEqual(parse_route("路由: LOCAL\n问题: 什么是 RRF"), ("local", "什么是 RRF"))
        self.assertEqual(parse_route("随便输出"), (None, None))


if __name__ == "__main__":
    unittest.main()
