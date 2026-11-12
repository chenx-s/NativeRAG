import os
import unittest
from unittest.mock import AsyncMock, patch

# Tests must not emit external LangSmith traces from the developer's .env.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.rag_chain import RAGChain, WEB_FALLBACK_ANSWER


class AsyncRAGTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_retrieval_runs_through_async_graph(self):
        rag = RAGChain(warm=False)

        with (
            patch("app.rag_chain.settings.QUERY_TRANSFORM", "none"),
            patch("app.rag_chain.settings.MAX_RETRIES", 0),
            patch("app.rag_chain.settings.WEB_SEARCH_ENABLED", False),
            patch(
                "app.rag_chain.amulti_query_search",
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = await rag.aanswer("没有命中的问题")

        self.assertEqual(result["num_documents"], 0)
        self.assertEqual(result["groundedness"], "skipped")
        self.assertEqual(result["queries"], ["没有命中的问题"])
        self.assertFalse(result["web_searched"])
        self.assertEqual(result["steps"][0], "route_question")
        self.assertNotIn("web_search", result["steps"])


class WebFallbackGraphTests(unittest.IsolatedAsyncioTestCase):
    """联网搜索作为图节点的路由行为（不访问网络、不打开 Milvus）。"""

    def _patches(self, search_mock, *, kb_count=10, routing="off"):
        return (
            patch("app.rag_chain.settings.QUERY_TRANSFORM", "none"),
            patch("app.rag_chain.settings.MAX_RETRIES", 2),
            patch("app.rag_chain.settings.WEB_FALLBACK_AFTER_RETRIES", 0),
            patch("app.rag_chain.settings.WEB_SEARCH_ENABLED", True),
            patch("app.rag_chain.settings.TAVILY_API_KEY", "tvly-test"),
            patch("app.rag_chain.settings.WEB_ROUTING", routing),
            patch("app.rag_chain.settings.GROUNDEDNESS_ACTION", "off"),
            patch("app.rag_chain.acount", new=AsyncMock(return_value=kb_count)),
            patch("app.rag_chain.amulti_query_search", new=AsyncMock(return_value=[])),
            patch("app.rag_chain.asearch_web", new=search_mock),
        )

    async def _run(self, patches, question="2026 年最新的新闻"):
        rag = RAGChain(warm=False)
        for p in patches:
            p.start()
        try:
            return await rag.aanswer(question)
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
        self.assertNotIn("rewrite_query", result["steps"])
        self.assertEqual(
            result["steps"],
            ["route_question", "transform_query", "retrieve",
             "rerank_documents", "grade_document", "web_search",
             "generate", "check_groundedness"],
        )
        self.assertEqual(result["answer"], WEB_FALLBACK_ANSWER)

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
        with patch(
            "app.rag_chain.get_llm",
            new=lambda **_: FakeListChatModel(responses=[answer]),
        ):
            result = await self._run(self._patches(search))

        self.assertEqual(result["source_type"], "web")
        self.assertEqual(result["sources"][0]["source_type"], "web")
        self.assertEqual(result["sources"][0]["source"], "https://example.com/a")
        self.assertIn("example.com", result["answer"])

    async def test_web_failure_is_reported_not_raised(self):
        from app.web_search import WebSearchError

        search = AsyncMock(side_effect=WebSearchError("Tavily 额度已用完"))
        result = await self._run(self._patches(search))

        self.assertTrue(result["web_searched"])
        self.assertIn("额度", result["web_error"])
        self.assertIn("额度", result["answer"])


class ParseRouteTests(unittest.TestCase):
    def test_parse_route(self):
        from app.rag_chain import _parse_route

        self.assertEqual(
            _parse_route("路由：web\n问题：今天北京天气"),
            ("web", "今天北京天气"),
        )
        self.assertEqual(
            _parse_route("路由: LOCAL\n问题: 什么是 RRF"),
            ("local", "什么是 RRF"),
        )
        self.assertEqual(_parse_route("随便输出"), (None, None))


if __name__ == "__main__":
    unittest.main()
