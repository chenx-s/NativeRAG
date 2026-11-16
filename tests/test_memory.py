import os
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.context import fit_documents, render_conversation
from app.memory import (
    ConversationMemory,
    InMemoryMemoryStore,
    MemoryManager,
    SqliteMemoryStore,
)


# ================================================================ 上下文工程


class RenderConversationTests(unittest.TestCase):
    def test_empty_memory_renders_nothing(self):
        self.assertEqual(
            render_conversation("", [], max_turns=3, max_chars=1000, answer_chars=100), ""
        )

    def test_selects_most_recent_turns_within_budget(self):
        turns = [(f"问题{i}", "回答" * 50) for i in range(10)]
        text = render_conversation("", turns, max_turns=3, max_chars=10_000, answer_chars=20)

        self.assertIn("问题9", text)
        self.assertIn("问题7", text)
        self.assertNotIn("问题6", text)  # 超出 max_turns
        # 回答被截断到 answer_chars
        self.assertNotIn("回答" * 15, text)

    def test_budget_drops_older_turns_first(self):
        turns = [("旧问题", "x" * 200), ("新问题", "y" * 200)]
        text = render_conversation("", turns, max_turns=5, max_chars=260, answer_chars=200)

        self.assertIn("新问题", text)
        self.assertNotIn("旧问题", text)
        self.assertLessEqual(len(text), 260 + len("【最近对话】\n"))

    def test_summary_is_included_and_capped(self):
        text = render_conversation(
            "摘" * 1000, [("q", "a")], max_turns=3, max_chars=400, answer_chars=50
        )
        self.assertIn("【较早对话摘要】", text)
        self.assertIn("【最近对话】", text)
        self.assertLess(text.count("摘"), 260)


class FitDocumentsTests(unittest.TestCase):
    def test_keeps_documents_in_order_until_budget(self):
        docs = [Document(page_content="a" * 400, metadata={"i": i}) for i in range(5)]
        kept = fit_documents(docs, 1000)

        self.assertEqual([d.metadata["i"] for d in kept], [0, 1, 2])
        self.assertTrue(kept[-1].page_content.endswith("（已截断）"))
        # 原对象不被修改
        self.assertEqual(len(docs[2].page_content), 400)

    def test_single_oversized_document_is_truncated_not_dropped(self):
        kept = fit_documents([Document(page_content="b" * 5000)], 100)
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0].page_content.startswith("b" * 100))


# ================================================================ 存储后端


class SqliteStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_trip_and_delete(self):
        with _tmpdir() as tmp:
            store = SqliteMemoryStore(os.path.join(tmp, "m.sqlite3"))
            memory = ConversationMemory("s1", "摘要", [("问", "答")], time.time())

            await store.asave(memory)
            loaded = await store.aload("s1")
            self.assertEqual(loaded.summary, "摘要")
            self.assertEqual(loaded.turns, [("问", "答")])

            # 新建实例（模拟进程重启）仍然能读到
            reopened = SqliteMemoryStore(os.path.join(tmp, "m.sqlite3"))
            self.assertEqual((await reopened.aload("s1")).turns, [("问", "答")])

            await store.adelete("s1")
            self.assertIsNone(await store.aload("s1"))

    async def test_expired_session_is_ignored(self):
        with _tmpdir() as tmp:
            store = SqliteMemoryStore(os.path.join(tmp, "m.sqlite3"), ttl_seconds=60)
            await store.asave(ConversationMemory("old", "", [("q", "a")], time.time() - 3600))
            self.assertIsNone(await store.aload("old"))


def _tmpdir():
    # Windows 上 SQLite 的 WAL 文件偶尔释放得慢，清理失败不应让测试失败
    return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)


# ================================================================ 记忆管理器


class MemoryManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_compacts_old_turns_into_summary(self):
        manager = MemoryManager(
            InMemoryMemoryStore(),
            llm_factory=lambda: FakeListChatModel(responses=["用户在了解 RRF 与精排"]),
        )
        with (
            patch("app.memory.settings.MEMORY_WINDOW_TURNS", 4),
            patch("app.memory.settings.MEMORY_KEEP_TURNS", 2),
        ):
            for i in range(5):
                await manager.aappend("s", f"q{i}", f"a{i}")

        memory = await manager.aload("s")
        self.assertEqual(memory.summary, "用户在了解 RRF 与精排")
        self.assertEqual(memory.turns, [("q3", "a3"), ("q4", "a4")])

    async def test_summary_failure_keeps_turns(self):
        def broken_llm():
            raise RuntimeError("LLM down")

        manager = MemoryManager(InMemoryMemoryStore(), llm_factory=broken_llm)
        with (
            patch("app.memory.settings.MEMORY_WINDOW_TURNS", 2),
            patch("app.memory.settings.MEMORY_KEEP_TURNS", 1),
        ):
            for i in range(3):
                await manager.aappend("s", f"q{i}", f"a{i}")

        memory = await manager.aload("s")
        self.assertEqual(memory.summary, "")
        self.assertEqual(len(memory.turns), 3)  # 没丢信息，仅受 2 倍窗口硬上限约束

    async def test_disabled_manager_is_noop(self):
        manager = MemoryManager(None)
        self.assertIsNone(await manager.aappend("s", "q", "a"))
        self.assertEqual((await manager.aload("s")).turns, [])

    async def test_store_failure_does_not_raise(self):
        store = InMemoryMemoryStore()
        store.asave = AsyncMock(side_effect=OSError("disk full"))
        manager = MemoryManager(store)
        await manager.aappend("s", "q", "a")  # 不抛异常即可


# ================================================================ 端到端：记忆进入状态图


class SessionMemoryGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_turn_sees_first_turn(self):
        from app.rag import RAGChain

        store = InMemoryMemoryStore()
        rag = RAGChain(warm=False, memory=MemoryManager(store))
        llm = FakeListChatModel(responses=["路由：local\n问题：RRF 融合的公式是什么"])

        with (
            patch("app.config.settings.QUERY_TRANSFORM", "none"),
            patch("app.config.settings.MAX_RETRIES", 0),
            patch("app.config.settings.WEB_SEARCH_ENABLED", False),
            patch("app.vector_store.amulti_query_search", new=AsyncMock(return_value=[])),
            patch("app.rag.llm.get_llm", new=lambda **_: llm),
        ):
            first = await rag.aanswer("什么是 RRF", session_id="s1")
            second = await rag.aanswer("它的公式呢", session_id="s1")

        self.assertEqual(first["memory_turns"], 0)
        self.assertEqual(second["memory_turns"], 1)
        # 第二轮借助记忆把追问补全成了独立问题
        self.assertEqual(second["question"], "RRF 融合的公式是什么")

        memory = await store.aload("s1")
        # 写入记忆的是补全后的独立问题
        self.assertEqual([q for q, _ in memory.turns], ["什么是 RRF", "RRF 融合的公式是什么"])

    async def test_without_session_nothing_is_stored(self):
        from app.rag import RAGChain

        store = InMemoryMemoryStore()
        rag = RAGChain(warm=False, memory=MemoryManager(store))
        with (
            patch("app.config.settings.QUERY_TRANSFORM", "none"),
            patch("app.config.settings.MAX_RETRIES", 0),
            patch("app.config.settings.WEB_SEARCH_ENABLED", False),
            patch("app.vector_store.amulti_query_search", new=AsyncMock(return_value=[])),
        ):
            await rag.aanswer("单轮问题")

        self.assertEqual(store._data, {})


if __name__ == "__main__":
    unittest.main()
