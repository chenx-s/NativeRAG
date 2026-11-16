"""系统接口：健康检查。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter

from app.api.runtime import llm_configured, runtime
from app.config import settings
from app.documents import FAILED, PENDING, PROCESSING, READY
from app.vector_store import stats

router = APIRouter(tags=["系统"])


@router.get("/health")
async def health():
    """健康检查：暴露运行时配置，方便排查"到底生效的是哪套参数"。不需要鉴权。"""
    documents = {}
    if runtime.registry is not None:
        records = await asyncio.to_thread(runtime.registry.list_all)
        documents = {
            "total": len(records),
            **{s: sum(1 for r in records if r.status == s) for s in (READY, PENDING, PROCESSING, FAILED)},
            "queue_pending": runtime.ingestion.pending if runtime.ingestion else 0,
        }

    return {
        "status": "ok",
        "llm_model": settings.LLM_MODEL_NAME,
        "llm_base_url": settings.OPENAI_BASE_URL,
        "api_key_configured": llm_configured(),
        "embedding": {"model": settings.EMBEDDING_MODEL, "dimension_check": runtime.embedding_check},
        "vector_store": "milvus",
        "retrieval": {
            "hybrid_enabled": settings.HYBRID_ENABLED,
            "dense_weight": settings.DENSE_WEIGHT,
            "sparse_weight": settings.SPARSE_WEIGHT,
            "dense_top_k": settings.DENSE_TOP_K,
            "sparse_top_k": settings.SPARSE_TOP_K,
            "fusion_top_k": settings.FUSION_TOP_K,
        },
        "query_transform": {"mode": settings.QUERY_TRANSFORM, "num_queries": settings.NUM_QUERIES},
        "rerank": {
            "enabled": settings.RERANK_ENABLED,
            "provider": settings.RERANK_PROVIDER,
            "model": settings.COHERE_RERANK_MODEL,
            "top_n": settings.RERANK_TOP_N,
            "ready": settings.rerank_ready,
        },
        "web_search": {
            "enabled": settings.WEB_SEARCH_ENABLED,
            "ready": settings.web_search_ready,
            "routing": settings.WEB_ROUTING,
            "fallback_after_retries": settings.WEB_FALLBACK_AFTER_RETRIES,
            "max_results": settings.WEB_SEARCH_MAX_RESULTS,
            "depth": settings.WEB_SEARCH_DEPTH,
        },
        "memory": {
            "backend": settings.MEMORY_BACKEND,
            "ttl_seconds": settings.MEMORY_TTL_SECONDS,
            "window_turns": settings.MEMORY_WINDOW_TURNS,
            "keep_turns": settings.MEMORY_KEEP_TURNS,
        },
        "context_budget": {
            "history_turns": settings.HISTORY_TURNS,
            "memory_context_chars": settings.MEMORY_CONTEXT_CHARS,
            "documents_max_chars": settings.CONTEXT_MAX_CHARS,
        },
        "documents": documents,
        "vector_store_stats": await asyncio.to_thread(stats),
        "openai_compat": {"models": [settings.OPENAI_COMPAT_MODEL, f"{settings.OPENAI_COMPAT_MODEL}-local"]},
        "auth_enabled": bool(settings.API_TOKEN),
        "metrics": {"enabled": settings.METRICS_ENABLED, "path": settings.METRICS_PATH},
    }
