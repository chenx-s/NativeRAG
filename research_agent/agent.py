"""研究助手（命令行会话层）

================================ 架构变化 ================================

旧版本是"两层 Agent"：

    ReAct Agent（create_agent）
        ├─ 工具 search_local_knowledge  →  内部再跑一整条 LangGraph RAG
        └─ 工具 search_web              →  Tavily

    问题：
      · 外层 LLM 每轮都要推理"调哪个工具"，而联网的时机其实是确定的
        （需要实时信息 / 本地查不到），用 LLM 推理既慢又不稳定；
      · 两层循环嵌套，一次提问的 LLM 调用次数和耗时都没有上限可言；
      · 内层 RAG 生成的答案又被外层 LLM 重新改写一遍，出处和忠实性检查结果被稀释。

新版本只有一层 —— 联网搜索被并入 app/rag 的状态图（CRAG）：

    route_question ─(需要实时信息)──────────────────────────► web_search ─┐
        │                                                       ▲         │
        └─(local)→ transform → retrieve → rerank → grade ─(无相关文档)─┘    │
                                                    │                     ▼
                                                    └─(有相关文档)──► generate → check_groundedness

本文件只负责：会话管理（thread_id 即 session_id，记忆由 RAGChain 持久化）、
执行轨迹展示、命令行交互。所有检索/联网/生成/记忆逻辑都在 app/ 里。
"""

import asyncio
import logging
import sys
import uuid
from typing import Optional

from app.rag.graph import STEP_LABELS  # 节点中文名，与 HTTP 流式进度事件共用一份
from research_agent.settings import agent_settings

# Windows 终端默认可能是 GBK，打印中文/emoji 会 UnicodeEncodeError。
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("research_agent")


class ResearchAgent:
    """研究助手的会话封装。

    相对直接调用 RAGChain，它多做了三件事：
      ① 管理 thread_id：它就是记忆的 session_id。默认 ID 固定，
         所以重启 CLI 后会自动接上上次的对话（记忆持久化在 MEMORY_BACKEND 里）
      ② 懒加载 RAGChain；本地知识库初始化失败时降级为"只联网"，不影响启动
      ③ 把状态图的执行路径整理成人类可读的轨迹
    """

    def __init__(self, thread_id: str = agent_settings.DEFAULT_SESSION_ID, rag=None):
        self.rag = rag
        self.thread_id = thread_id
        self._sync_runner: Optional[asyncio.Runner] = None

    # ------------------------------------------------------------------ 内部
    async def _aget_rag(self):
        """懒加载 RAGChain。预热失败时退化为不预热（本地检索失败会自动转联网）。"""
        if self.rag is None:
            from app.rag import RAGChain

            try:
                self.rag = await asyncio.to_thread(RAGChain, True)
            except Exception as error:  # noqa: BLE001
                logger.warning("本地知识库预热失败，将只依赖联网搜索：%s", error)
                self.rag = RAGChain(warm=False)
        return self.rag

    def _memory_manager(self):
        """读/清记忆不需要加载 Embedding 模型：RAG 还没初始化时直接按配置建一个管理器。"""
        if self.rag is not None:
            return self.rag.memory
        from app.memory import MemoryManager

        return MemoryManager.from_settings()

    async def amemory(self):
        """读取当前会话的持久化记忆（ConversationMemory）。"""
        return await self._memory_manager().aload(self.thread_id)

    async def aforget(self) -> None:
        """清除当前会话的持久化记忆。"""
        await self._memory_manager().aclear(self.thread_id)

    # ------------------------------------------------------------------ 对外
    def chat(self, question: str, verbose: bool = True) -> dict:
        """同步兼容入口；异步应用应使用 :meth:`achat`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Runner 持有一个长期 event loop，使 Redis/httpx 连接池可以跨多轮
            # 同步 chat() 调用安全复用。
            if self._sync_runner is None:
                self._sync_runner = asyncio.Runner()
            return self._sync_runner.run(self.achat(question, verbose))
        raise RuntimeError("事件循环中不能调用 chat()，请改用 await achat()")

    async def achat(self, question: str, verbose: bool = True) -> dict:
        """问一个问题，返回 {"answer": 答案, "trace": 执行轨迹, "result": 完整结果}。"""
        try:
            rag = await self._aget_rag()
            # 记忆的加载、注入、写回、压缩都由 RAGChain 按 session_id 完成
            result = await rag.aanswer(question, session_id=self.thread_id)
        except Exception as error:  # noqa: BLE001
            logger.error("问答执行失败：%s: %s", type(error).__name__, error)
            return {
                "answer": f"抱歉，处理这个问题时出错了：{type(error).__name__}: {error}",
                "trace": [],
                "error": error,
            }

        answer = (result.get("answer") or "").strip() or "（没有生成答案，请换个问法再试）"

        trace = result.get("steps", [])
        if verbose:
            self._print_trace(result)

        return {"answer": answer, "trace": trace, "result": result}

    @staticmethod
    def _print_trace(result: dict) -> None:
        """打印执行路径与来源。"""
        steps = result.get("steps") or []
        if steps:
            print("  🧭 路径：" + " → ".join(STEP_LABELS.get(s, s) for s in steps))

        if result.get("memory_turns") or result.get("memory_summary_used"):
            print(
                f"  🧠 记忆：注入最近 {result.get('memory_turns', 0)} 轮"
                + ("，含较早对话摘要" if result.get("memory_summary_used") else "")
            )

        if result.get("question") and result.get("question") != result.get("original_question"):
            print(f"  ✏️ 补全/改写后的问题：{result['question']}")

        if result.get("web_error"):
            print(f"  ⚠️ 联网搜索失败：{result['web_error']}")

        sources = result.get("sources") or []
        if sources:
            label = "互联网" if result.get("source_type") == "web" else "本地知识库"
            print(f"  📚 来源（{label}）：")
            for i, item in enumerate(sources[:5], 1):
                if item.get("source_type") == "web":
                    print(f"     [{i}] {item.get('filename')}  {item.get('source')}")
                else:
                    page = item.get("page")
                    location = item.get("filename") or item.get("source") or "未知文件"
                    print(f"     [{i}] {location}" + (f" 第 {page} 页" if page else ""))

        verdict = result.get("groundedness")
        if verdict and verdict not in ("yes", "skipped", "off"):
            print(f"  ⚠️ 忠实性检查：{verdict}")

    def new_session(self, thread_id: Optional[str] = None) -> None:
        """开启新会话（换一个 thread_id；旧会话的记忆仍保存在存储里，直到过期）。"""
        self.thread_id = thread_id or f"research-{uuid.uuid4().hex[:8]}"
        logger.info("已切换到新会话：%s", self.thread_id)

    async def aclose(self) -> None:
        """释放 Tavily 与 Redis 连接池。"""
        from app.rag import RAGChain

        await RAGChain.aclose()

    def close(self) -> None:
        """释放同步 ``chat`` 创建的长期事件循环。"""
        if self._sync_runner is not None:
            self._sync_runner.run(self.aclose())
            self._sync_runner.close()
            self._sync_runner = None


# ===========================================================================
# 命令行交互
# ===========================================================================


HELP_TEXT = """
可用命令：
  /new          开启新会话（旧会话记忆保留在存储里）
  /session ID   切换到指定会话（可接上之前任意一次对话）
  /history      查看当前会话保存的记忆（摘要 + 最近几轮）
  /forget       清除当前会话的记忆
  /warmup       预热本地知识库（首次提问前跑一次，避免干等 30 秒）
  /quiet        切换是否显示执行路径
  /help         显示本帮助
  /quit         退出
"""


async def _print_memory(agent: "ResearchAgent") -> None:
    memory = await agent.amemory()
    print(f"\n会话：{agent.thread_id}")
    if not memory.summary and not memory.turns:
        print("  （这个会话还没有记忆）")
        return
    if memory.summary:
        print("  【较早对话摘要】")
        print("    " + memory.summary.replace("\n", "\n    "))
    if memory.turns:
        print(f"  【最近 {len(memory.turns)} 轮】")
        for i, (q, a) in enumerate(memory.turns, 1):
            print(f"    {i}. 你：{q}")
            print(f"       助手：{a[:120]}{'…' if len(a) > 120 else ''}")


async def amain() -> int:
    """命令行入口：交互式研究助手。"""
    from app.config import settings

    print("=" * 72)
    print(" 研究助手  ——  本地知识库 + 联网搜索（LangGraph 单层编排）")
    print("=" * 72)

    if not settings.OPENAI_API_KEY:
        print("❌ 未配置 LLM_API_KEY，无法启动。请检查项目根目录的 .env。")
        return 1

    print(f"  模型          : {settings.LLM_MODEL_NAME}")
    print(f"  联网搜索      : {'✅ 已配置' if settings.web_search_ready else '❌ 未配置 TAVILY_API_KEY'}")
    print(f"  入口路由      : {settings.WEB_ROUTING}")
    print()
    print(HELP_TEXT)

    agent = ResearchAgent()

    if agent_settings.WARMUP_ON_START:
        print("  ⏳ 正在预热本地知识库（加载 Embedding 模型 + 打开 Milvus）...")
        await agent._aget_rag()
        print("  ✅ 就绪。")

    memory = await agent.amemory()
    print(f"  会话          : {agent.thread_id}（记忆存储：{settings.MEMORY_BACKEND}）")
    if memory.turns or memory.summary:
        print(
            f"  🧠 已恢复上次对话：{len(memory.turns)} 轮"
            + ("（另有较早对话摘要）" if memory.summary else "")
            + "。输入 /history 查看，/new 开启新会话。"
        )

    verbose = True

    while True:
        try:
            question = (await asyncio.to_thread(input, "\n你 > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            await agent.aclose()
            return 0

        if not question:
            continue

        if question in ("/quit", "/exit", "q", "exit"):
            print("再见。")
            await agent.aclose()
            return 0
        if question == "/new":
            agent.new_session()
            print(f"✅ 已开启新会话：{agent.thread_id}")
            continue
        if question.startswith("/session"):
            target = question[len("/session"):].strip()
            if not target:
                print(f"当前会话：{agent.thread_id}。用法：/session 会话ID")
                continue
            agent.new_session(target)
            await _print_memory(agent)
            continue
        if question == "/history":
            await _print_memory(agent)
            continue
        if question == "/forget":
            await agent.aforget()
            print(f"✅ 会话 {agent.thread_id} 的记忆已清除。")
            continue
        if question == "/warmup":
            await agent._aget_rag()
            print("✅ 本地知识库已就绪。")
            continue
        if question == "/quiet":
            verbose = not verbose
            print(f"✅ 执行路径显示：{'开' if verbose else '关'}")
            continue
        if question == "/help":
            print(HELP_TEXT)
            continue

        print("\n🤔 思考中 ...")
        outcome = await agent.achat(question, verbose=verbose)

        print("\n" + "─" * 72)
        print("助手 >")
        print(outcome["answer"])
        print("─" * 72)


def main() -> int:
    """同步命令行入口；内部始终复用同一个事件循环。"""
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
