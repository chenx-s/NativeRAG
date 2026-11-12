import asyncio
import os
import unittest

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from research_agent.agent import ResearchAgent


class DummyRAG:
    def __init__(self):
        self.loop_ids = []
        self.session_ids = []

    async def aanswer(self, question, include_documents=False, history=None, session_id=None):
        self.loop_ids.append(id(asyncio.get_running_loop()))
        self.session_ids.append(session_id)
        return {"answer": f"ok:{question}", "steps": ["route_question", "generate"]}

    @staticmethod
    async def aclose():
        return None


class ResearchAgentTests(unittest.TestCase):
    def test_sync_chat_reuses_one_event_loop(self):
        agent = ResearchAgent(thread_id="test", rag=DummyRAG())
        agent.aclose = DummyRAG.aclose  # 测试里不碰真实 Redis/Tavily

        try:
            self.assertEqual(agent.chat("one", verbose=False)["answer"], "ok:one")
            self.assertEqual(agent.chat("two", verbose=False)["answer"], "ok:two")
            self.assertEqual(len(set(agent.rag.loop_ids)), 1)
        finally:
            agent.close()

    def test_thread_id_is_used_as_memory_session(self):
        rag = DummyRAG()
        agent = ResearchAgent(thread_id="a", rag=rag)
        agent.aclose = DummyRAG.aclose

        try:
            agent.chat("q1", verbose=False)
            agent.new_session("b")
            agent.chat("q2", verbose=False)
            # 记忆的读写交给 RAGChain，按 session_id 隔离
            self.assertEqual(rag.session_ids, ["a", "b"])
        finally:
            agent.close()


if __name__ == "__main__":
    unittest.main()
