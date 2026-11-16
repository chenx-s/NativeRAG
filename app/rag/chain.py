"""RAGChain：状态图的对外封装（记忆加载/写回、流式事件、结果整理）。

调用方（FastAPI、CLI、RAGAs 评估）只和这个类打交道：

    rag = RAGChain()
    result = await rag.aanswer("问题", session_id="u1")           # 一次性返回
    async for event in rag.astream_events("问题", session_id="u1"):  # 流式
        ...

aanswer 内部就是把 astream_events 跑完取最后的 done 事件 —— 流式和非流式只有一条代码路径。
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Sequence, Tuple

from app.config import settings
from app.memory import MemoryManager
from app.rag import llm
from app.rag.graph import STEP_LABELS, build_graph, required_recursion_limit
from app.rag.prompts import FALLBACK_ANSWER
from app.vector_store import warmup
from app.web_search import aclose_web_client

Turn = Tuple[str, str]


class RAGChain:
    """封装编译后的 LangGraph 应用。"""

    def __init__(self, warm: bool = True, memory: Optional[MemoryManager] = None):
        """构建状态图，并按需预热。

        参数：
            warm: 是否在初始化时就预热（加载 Embedding 模型、打开 Milvus、
                  从语料副本重建 BM25 倒排索引）。预热很慢，放在启动时做，
                  代价是"启动慢一次"；放到第一次提问时做，代价是"第一个用户请求超时"。
            memory: 会话记忆管理器；不传则按 MEMORY_BACKEND 配置创建
                    （存储是懒连接的，不传 session_id 的调用不会触碰存储）。
        """
        self.app = build_graph()
        # 用 lambda 延迟取 llm.get_llm：测试里 patch("app.rag.llm.get_llm") 对摘要压缩同样生效
        self.memory = memory if memory is not None else MemoryManager.from_settings(
            lambda **kwargs: llm.get_llm(**kwargs)
        )
        if warm:
            warmup()

    # ------------------------------------------------------------------ 同步入口 / 资源

    def answer(
        self,
        question: str,
        include_documents: bool = False,
        history: Optional[Sequence[Turn]] = None,
        session_id: Optional[str] = None,
        allow_web: bool = True,
    ) -> dict:
        """同步兼容入口（脚本、RAGAs 评估用）；异步代码请使用 :meth:`aanswer`。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            async def run_once() -> dict:
                try:
                    return await self.aanswer(question, include_documents, history, session_id, allow_web)
                finally:
                    # asyncio.run 每次都会创建新 loop，不能把连接池留给下一轮。
                    await self.aclose()

            return asyncio.run(run_once())
        raise RuntimeError("事件循环中不能调用 answer()，请改用 await aanswer()")

    @staticmethod
    async def aclose() -> None:
        """释放 Tavily 与 Redis 连接池（服务关闭或事件循环结束前调用）。"""
        from app.redis_client import close_redis

        try:
            await aclose_web_client()
        finally:
            await close_redis()

    # ------------------------------------------------------------------ 问答

    @staticmethod
    def _initial_state(
        question: str,
        history: Sequence[Turn],
        memory_summary: str = "",
        allow_web: bool = True,
    ) -> dict:
        """一次请求的初始状态。每个字段都显式给出，避免节点读到上一轮残留的值。"""
        return {
            "question": question,
            "original_question": question,
            "standalone_question": question,
            "history": list(history),
            "memory_summary": memory_summary,
            "allow_web": allow_web,
            "route": "local",
            "source_type": "local",
            "web_searched": False,
            "web_error": "",
            "queries": [],
            "documents": [],
            "candidates": 0,
            "reranked": False,
            "transform_mode": settings.QUERY_TRANSFORM,
            "rewritten": False,
            "retries": 0,
            "rewrite_unchanged": False,
            "generation": "",
            "skip_check": False,
            "groundedness_retries": 0,
            "strict_grounding": False,
            "regenerate": False,
            "groundedness_verdict": "",
        }

    async def aanswer(
        self,
        question: str,
        include_documents: bool = False,
        history: Optional[Sequence[Turn]] = None,
        session_id: Optional[str] = None,
        allow_web: bool = True,
    ) -> dict:
        """执行完整问答，返回最终结果。

        两种多轮方式（二选一）：
            session_id  服务端记忆：自动加载该会话的摘要与最近几轮，答完自动写回并按需压缩
            history     客户端自带历史 [(问题, 回答), ...]，服务端不保存
        都不传就是单轮无状态问答。
        allow_web=False 时本次请求不联网（只查本地知识库）。
        include_documents=True 时额外返回完整检索上下文（RAGAs 评估需要）。
        """
        payload: dict = {}
        async for event in self.astream_events(question, include_documents, history, session_id, allow_web):
            if event["event"] == "done":
                payload = event["result"]
        return payload

    async def astream_events(
        self,
        question: str,
        include_documents: bool = False,
        history: Optional[Sequence[Turn]] = None,
        session_id: Optional[str] = None,
        allow_web: bool = True,
    ):
        """流式执行问答，逐个产出事件（异步生成器）。

        事件格式（都是 dict）：
            {"event": "step",  "node": 节点名, "label": 中文名}   每个图节点执行完毕
            {"event": "token", "text": 一段答案文本}               答案逐段输出
            {"event": "reset", "reason": "regenerate" | "replaced"}
                已输出的答案作废：regenerate = 草稿没通过忠实性检查，下面是严格模式重写的版本；
                replaced = 最终答案被替换成了拒答 / 兜底文本，下面会给出替换后的全文
            {"event": "done",  "result": 与 aanswer 返回值相同的 dict}  最后一个事件

        约定：最后一次 reset 之后的 token 拼起来，一定等于最终答案。
        兜底答案不经过 LLM，没有逐段输出，会在 done 之前作为一个完整 token 补发。
        """
        if session_id:
            memory = await self.memory.aload(session_id)
            summary, turns = memory.summary, list(memory.turns)
        else:
            summary, turns = "", list(history or [])

        steps: List[str] = []
        result: dict = {}
        streamed = ""
        recursion_limit = max(settings.RECURSION_LIMIT, required_recursion_limit())

        # updates：每个节点执行完的增量 -> 进度事件
        # custom ：generate 节点经 stream writer 推送的 token
        # values ：每步后的完整状态，最后一帧就是最终状态（等价于 ainvoke 的返回值）
        async for mode, chunk in self.app.astream(
            self._initial_state(question, turns, summary, allow_web),
            config={"recursion_limit": recursion_limit},
            stream_mode=["updates", "custom", "values"],
        ):
            if mode == "updates":
                for node in chunk.keys():
                    steps.append(node)
                    yield {"event": "step", "node": node, "label": STEP_LABELS.get(node, node)}
            elif mode == "custom":
                if chunk.get("type") == "generation_start":
                    if streamed:
                        streamed = ""
                        yield {"event": "reset", "reason": "regenerate"}
                elif chunk.get("type") == "token":
                    streamed += chunk.get("text", "")
                    yield {"event": "token", "text": chunk.get("text", "")}
            else:
                result = chunk

        payload = self._build_payload(result, question, steps, session_id, turns, summary, include_documents)

        # 最终答案和已输出的不一致（兜底答案 / 忠实性检查后被替换）-> 补发全文
        final_answer = payload["answer"]
        if final_answer.strip() != streamed.strip():
            if streamed:
                yield {"event": "reset", "reason": "replaced"}
            yield {"event": "token", "text": final_answer}

        # 写回记忆：问题存补全后的独立问题，脱离上下文也能读懂
        if session_id:
            await self.memory.aappend(session_id, result.get("standalone_question") or question, final_answer)

        yield {"event": "done", "result": payload}

    # ------------------------------------------------------------------ 结果整理

    @staticmethod
    def _build_payload(
        result: dict,
        question: str,
        steps: List[str],
        session_id: Optional[str],
        turns: Sequence[Turn],
        summary: str,
        include_documents: bool,
    ) -> dict:
        """把最终状态整理成对外返回的结果（答案、来源、观测字段）。"""
        documents = result.get("documents", [])

        sources = []
        seen = set()
        for document in documents:
            key = (document.metadata.get("source"), document.metadata.get("page"), document.page_content[:80])
            if key in seen:
                continue
            seen.add(key)

            page = document.metadata.get("page")
            sources.append({
                "source": document.metadata.get("source", ""),
                "filename": document.metadata.get("filename", ""),
                "page": page + 1 if page is not None else None,
                "snippet": document.page_content[:200],
                # 精排分数（未精排时为 None；联网结果为 Tavily 相关度）
                "relevance_score": document.metadata.get("relevance_score"),
                "rrf_score": document.metadata.get("rrf_score"),
                # local=本地知识库 / web=互联网
                "source_type": document.metadata.get("source_type", "local"),
                # 本地文档的 ID，前端可用 GET /documents/{doc_id}/file 打开原文
                "doc_id": document.metadata.get("doc_id", ""),
            })

        payload = {
            "answer": result.get("generation") or FALLBACK_ANSWER,
            "question": result.get("question", question),
            "original_question": result.get("original_question", question),
            "rewritten": bool(result.get("rewritten")),
            "rewrite_unchanged": bool(result.get("rewrite_unchanged")),
            "num_documents": len(documents),
            "sources": sources,
            # ---- 检索观测 ----
            "queries": result.get("queries", []),
            "transform_mode": result.get("transform_mode", "none"),
            "candidates": result.get("candidates", 0),
            "reranked": bool(result.get("reranked")),
            # 忠实性自检：yes=通过 no=未通过(已按配置处理) skipped=兜底答案 off=已关闭 unknown=检查失败
            "groundedness": result.get("groundedness_verdict", ""),
            "groundedness_retries": result.get("groundedness_retries", 0),
            # ---- 联网 ----
            "route": result.get("route", "local"),
            "source_type": result.get("source_type", "local"),
            "web_searched": bool(result.get("web_searched")),
            "web_error": result.get("web_error", ""),
            "steps": steps,
            # ---- 记忆 ----
            "session_id": session_id or "",
            "memory_turns": len(turns),
            "memory_summary_used": bool(summary),
        }

        if include_documents:
            payload["documents"] = [doc.page_content for doc in documents]

        return payload
