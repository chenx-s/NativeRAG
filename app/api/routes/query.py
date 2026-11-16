"""问答接口：/query、/query/stream、/retrieve、/sessions。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi import Path as PathParam
from fastapi.responses import StreamingResponse

from app.api.runtime import SESSION_ID_PATTERN, check_can_answer, get_rag
from app.api.schemas import (
    QueryRequest,
    QueryResponse,
    RetrieveRequest,
    RetrieveResponse,
    SessionResponse,
    SourceInfo,
)
from app.api.security import AUTH
from app.api.sse import SSE_HEADERS, sse
from app.config import settings
from app.rag import RAGChain
from app.rag.nodes import rerank_documents, retrieve, transform_query
from app.vector_store import count

router = APIRouter(tags=["问答"], dependencies=AUTH)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Agentic RAG 问答完整流程，一次性返回。

    入口路由（需要实时信息 -> 直接联网）
        -> 查询变换 -> 混合检索 -> 精排 -> 文档评分
        -> (无相关文档时 联网搜索 / 改写查询 / 兜底)
        -> 生成答案 -> 忠实性自检 -> (未通过时 严格模式重生成) -> 返回
    """
    rag = get_rag()
    check_can_answer(request.allow_web)

    try:
        result = await rag.aanswer(
            request.question.strip(),
            history=request.history_pairs(),
            session_id=request.session_id,
            allow_web=request.allow_web,
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"RAG 链执行失败：{error}")

    return QueryResponse(**{**result, "sources": [SourceInfo(**s) for s in result["sources"]]})


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """流式问答（Server-Sent Events）。

    事件类型：
        step   {"node": 节点名, "label": 中文名}            每个处理步骤完成时
        token  {"text": 一段答案}                            答案逐段输出
        reset  {"reason": "regenerate" | "replaced"}        已输出的答案作废，前端应清空后接着显示
        done   与 POST /query 相同结构的完整结果              最后一个事件
        error  {"detail": 错误信息}                          处理失败

    浏览器端请用 fetch + ReadableStream 读取（EventSource 不支持 POST）。
    """
    rag = get_rag()
    check_can_answer(request.allow_web)

    async def event_stream():
        try:
            async for event in rag.astream_events(
                request.question.strip(),
                history=request.history_pairs(),
                session_id=request.session_id,
                allow_web=request.allow_web,
            ):
                kind = event["event"]
                if kind == "step":
                    yield sse({"node": event["node"], "label": event["label"]}, event="step")
                elif kind == "token":
                    yield sse({"text": event["text"]}, event="token")
                elif kind == "reset":
                    yield sse({"reason": event["reason"]}, event="reset")
                elif kind == "done":
                    yield sse(event["result"], event="done")
        except Exception as error:  # noqa: BLE001
            # 响应头已发出，无法再返回 500，用 error 事件通知前端
            yield sse({"detail": f"{type(error).__name__}: {error}"}, event="error")

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)


def _to_source(document) -> SourceInfo:
    page = document.metadata.get("page")
    return SourceInfo(
        source=document.metadata.get("source", ""),
        filename=document.metadata.get("filename", ""),
        page=page + 1 if page is not None else None,
        snippet=document.page_content[:200],
        relevance_score=document.metadata.get("relevance_score"),
        rrf_score=document.metadata.get("rrf_score"),
        doc_id=document.metadata.get("doc_id", ""),
    )


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_debug(request: RetrieveRequest):
    """只做检索调试：查询变换 -> 混合检索 -> 精排，不生成答案。

    用来对比不同配置下的召回质量（开关 skip_rerank、调 DENSE/SPARSE_WEIGHT），
    除 multi_query/HyDE 的查询变换外不产生 LLM 调用。
    """
    if count() == 0:
        raise HTTPException(status_code=400, detail="向量库为空，请先上传文档入库")

    question = request.question.strip()
    state = RAGChain._initial_state(question, [])
    # 节点函数都是 async 的，必须 await
    state.update(await transform_query(state))
    state.update(await retrieve(state))
    candidates = state["candidates"]
    if not request.skip_rerank:
        state.update(await rerank_documents(state))

    documents = state["documents"]
    if request.top_k:
        documents = documents[: request.top_k]

    return RetrieveResponse(
        question=question,
        queries=state["queries"],
        candidates=candidates,
        reranked=state["reranked"],
        results=[_to_source(doc) for doc in documents],
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str = PathParam(..., pattern=SESSION_ID_PATTERN)):
    """查看某个会话的持久化记忆（滚动摘要 + 最近几轮原文）。"""
    rag = get_rag()
    if not rag.memory.enabled:
        raise HTTPException(status_code=404, detail="会话记忆未启用（MEMORY_BACKEND=off）")

    memory = await rag.memory.aload(session_id)
    if not memory.turns and not memory.summary:
        raise HTTPException(status_code=404, detail=f"会话不存在或已过期：{session_id}")

    return SessionResponse(
        session_id=session_id,
        backend=settings.MEMORY_BACKEND,
        summary=memory.summary,
        turns=[list(turn) for turn in memory.turns],
        updated_at=memory.updated_at,
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str = PathParam(..., pattern=SESSION_ID_PATTERN)):
    """清除某个会话的记忆（用户主动"忘掉这段对话"）。"""
    await get_rag().memory.aclear(session_id)
    return {"status": "ok", "detail": f"会话 {session_id} 的记忆已清除"}
