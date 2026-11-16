"""向量库门面：对外统一暴露检索层能力（实现已按职责拆分到 app/retrieval/）。

设计说明
--------
1. 稠密检索：Milvus 存 BGE 向量，负责"语义相近"。
2. 稀疏检索：BM25 负责"关键词精确命中"（专有名词、型号、错误码、人名）。
3. 融合：Reciprocal Rank Fusion（RRF）——两路分数量纲不同，只按排名融合。

    实现位置                         内容
    app/retrieval/embeddings.py     Embedding 模型
    app/retrieval/milvus.py         Milvus 连接 / 诊断 / 计数 / 维度检查
    app/retrieval/corpus.py         BM25 语料副本
    app/retrieval/bm25.py           分词 + BM25 索引
    app/retrieval/fusion.py         RRF 融合与去重
    app/retrieval/search.py         各种检索入口
    app/retrieval/store.py          入库 / 删除 / 清库 / 预热 / 统计

为什么保留这个门面：
    · 调用方（RAG 节点、文档管理、API）只认 `app.vector_store` 一个入口，不关心内部怎么拆；
    · 节点通过 `vector_store.amulti_query_search(...)` 调用，测试只需在这里 patch 一处；
    · 下方的异步版本统一把同步的 Milvus / Embedding / BM25 调用移到工作线程，
      让 Web/Agent 的事件循环可以继续服务其他请求。
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

from app.config import settings  # noqa: F401  （兼容：旧代码/测试通过 vector_store.settings 访问配置）
from app.retrieval.bm25 import _invalidate_bm25, _tokenize, get_bm25_retriever  # noqa: F401
from app.retrieval.corpus import (  # noqa: F401
    _append_corpus,
    _clear_corpus,
    _remove_from_corpus,
    load_corpus,
)
from app.retrieval.embeddings import _detect_device, embedding_dim, get_embeddings  # noqa: F401
from app.retrieval.fusion import RRF_C, _doc_key, rrf_fuse  # noqa: F401
from app.retrieval.milvus import (  # noqa: F401
    _connection_args,
    _exception_chain,
    _explain_milvus_open_failure,
    _local_db_locked,
    check_embedding_dimension,
    count,
    get_vector_store,
)
from app.retrieval.search import dense_search, multi_query_search, search, sparse_search  # noqa: F401
from app.retrieval.store import (  # noqa: F401
    add_documents,
    clear,
    delete_documents_by_doc_id,
    stats,
    warmup,
)

# ================================================================ 异步版本


async def amulti_query_search(
    queries: Sequence[str],
    k: Optional[int] = None,
    strategy: str = "rrf",
) -> List[Document]:
    return await asyncio.to_thread(multi_query_search, queries, k, strategy)


async def aadd_documents(docs: List[Document]) -> int:
    return await asyncio.to_thread(add_documents, docs)


async def acount() -> int:
    return await asyncio.to_thread(count)


async def aclear() -> None:
    await asyncio.to_thread(clear)


async def awarmup() -> None:
    await asyncio.to_thread(warmup)


async def astats() -> Dict[str, object]:
    return await asyncio.to_thread(stats)
