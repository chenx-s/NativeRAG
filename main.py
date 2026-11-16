"""FastAPI 入口：创建应用、管理生命周期、挂载路由。

具体接口按业务分在 app/api/routes/ 下：

    健康与监控   GET /health、GET /metrics                        app/api/routes/system.py
    问答         POST /query、/query/stream、/retrieve、/sessions  app/api/routes/query.py
    文档管理     /documents/*、DELETE /index、旧版 /ingest/*        app/api/routes/documents.py
    OpenAI 兼容  GET /v1/models、POST /v1/chat/completions         app/api/openai_compat.py

除 /health、/metrics 外，配置了 API_TOKEN 时都需要请求头 Authorization: Bearer <token>。

启动：
    python main.py
    或 uvicorn main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from redis_fastapi import AsyncRedisDep, FastAPIRedis

from app.api.openai_compat import build_router as build_openai_router
from app.api.routes import documents, query, system
from app.api.runtime import runtime
from app.api.security import AUTH
from app.config import settings
from app.documents import DocumentRegistry, IngestionQueue
from app.observability import setup_metrics
from app.rag import RAGChain
from app.vector_store import check_embedding_dimension, count, warmup
from app.web_search import aclose_web_client

logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._page").setLevel(logging.ERROR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动：预热检索组件 -> 创建 RAG 链 -> 检查向量维度 -> 启动后台入库队列。
    关闭：停止入库队列 -> 释放联网搜索连接池。
    """
    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

    # 预热：加载 Embedding 模型 -> 打开 Milvus -> 构建 BM25 倒排索引
    warmup()
    runtime.rag = RAGChain(warm=False)

    runtime.embedding_check = check_embedding_dimension()
    if not runtime.embedding_check.get("ok", True):
        print(f"[warn] {runtime.embedding_check['detail']}")

    runtime.registry = DocumentRegistry()
    runtime.ingestion = IngestionQueue(runtime.registry)
    await runtime.ingestion.start(resume=True)

    _print_startup_summary()

    yield

    print("[shutdown] FastAPI 退出，清理资源")
    await runtime.ingestion.stop()
    await aclose_web_client()


def _print_startup_summary() -> None:
    print(f"[startup] 向量库就绪，当前索引块数：{count()} | Embedding={settings.EMBEDDING_MODEL}")
    print(
        f"[startup] 混合检索={settings.HYBRID_ENABLED} "
        f"| 查询变换={settings.QUERY_TRANSFORM} "
        f"| 精排={settings.RERANK_ENABLED}/{settings.rerank_ready} "
        f"| 联网={'可用' if settings.web_search_ready else '未启用'}（路由 {settings.WEB_ROUTING}）"
        f"| 记忆={settings.MEMORY_BACKEND}"
    )
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("[warn] 未配置 LLM_API_KEY，无法调用 LLM")
    if not settings.API_TOKEN:
        print("[warn] 未设置 API_TOKEN，接口不鉴权 —— 只适合本机使用，对外暴露前务必设置")


app = FastAPI(
    title="RAG 文档问答 API",
    description=(
        "FastAPI + LangChain + LangGraph 实现的 Agentic RAG 服务：混合检索 + 精排 + 联网纠错 + "
        "忠实性自检 + 会话记忆。提供 SSE 流式接口和 OpenAI 兼容接口（/v1），可直接接入 Open WebUI。"
    ),
    version="3.0.0",
    lifespan=lifespan,
)

setup_metrics(app, enabled=settings.METRICS_ENABLED, endpoint=settings.METRICS_PATH)
FastAPIRedis(app).lifespan()

app.include_router(system.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(build_openai_router(lambda: runtime.rag))


@app.get("/items", dependencies=AUTH, include_in_schema=False)
async def get_item(redis: AsyncRedisDep):
    """fastapi-redis-sdk 的连通性示例接口。"""
    return {"items": await redis.get("items")}


if __name__ == "__main__":
    import os

    import uvicorn

    # 以 .env 的 LANGSMITH_TRACING 为准（旧代码这里强制写 true，会覆盖用户的关闭设置）
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
