"""运行时依赖：服务启动后创建的单例（RAG 链、文档登记表、入库队列），以及取用它们的辅助函数。

为什么单独一个模块：
    路由分散在多个文件里，都要访问同一组单例。放在这里由 main.py 的 lifespan 赋值，
    各路由通过 get_rag() / get_registry() 取用 —— 未初始化时统一返回 503。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from fastapi import HTTPException

from app.config import settings
from app.documents import DocumentRegistry, IngestionQueue
from app.rag import RAGChain
from app.vector_store import count

SESSION_ID_PATTERN = r"^[A-Za-z0-9_.:\-]{1,128}$"
DOC_ID_PATTERN = r"^[0-9a-f]{16}$"


@dataclass
class Runtime:
    rag: Optional[RAGChain] = None
    registry: Optional[DocumentRegistry] = None
    ingestion: Optional[IngestionQueue] = None
    # 启动时的 Embedding 维度检查结果（见 vector_store.check_embedding_dimension）
    embedding_check: dict = field(default_factory=dict)


runtime = Runtime()


def get_rag() -> RAGChain:
    if runtime.rag is None:
        raise HTTPException(status_code=503, detail="RAG 链未初始化")
    return runtime.rag


def get_registry() -> DocumentRegistry:
    if runtime.registry is None or runtime.ingestion is None:
        raise HTTPException(status_code=503, detail="文档服务未初始化")
    return runtime.registry


def get_ingestion() -> IngestionQueue:
    get_registry()
    return runtime.ingestion


def llm_configured() -> bool:
    return bool(settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "YOUR_OPENAI_API_KEY")


def check_can_answer(allow_web: bool) -> None:
    """问答前置检查：LLM 已配置；知识库为空时必须能联网。"""
    if not llm_configured():
        raise HTTPException(status_code=503, detail="未配置 LLM_API_KEY，无法调用 LLM")
    if not (allow_web and settings.web_search_ready) and count() == 0:
        raise HTTPException(
            status_code=400,
            detail="向量库为空，请先通过 POST /documents 上传文档（或配置 TAVILY_API_KEY 启用联网搜索）",
        )
