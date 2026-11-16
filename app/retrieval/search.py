"""检索入口：稠密检索、稀疏检索、单查询混合检索、多查询融合检索（RAG-Fusion）。

    稠密检索（Milvus 向量）负责"语义相近"——问"如何提升检索质量"能命中讲"recall 优化"的段落；
    稀疏检索（BM25）负责"关键词精确命中"——专有名词、型号、错误码、人名这类
    语义模型容易漂移的东西，BM25 往往一把命中；两路结果用 RRF 融合。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

from app.config import settings
from app.retrieval.bm25 import get_bm25_retriever
from app.retrieval.fusion import _doc_key, rrf_fuse
from app.retrieval.milvus import get_vector_store


def dense_search(query: str, k: Optional[int] = None) -> List[Document]:
    """稠密（向量）检索。"""
    return get_vector_store().similarity_search(query, k=k or settings.DENSE_TOP_K)


def sparse_search(query: str, k: Optional[int] = None) -> List[Document]:
    """稀疏（BM25 关键词）检索；BM25 不可用时返回空列表。"""
    retriever = get_bm25_retriever()
    if retriever is None:
        return []

    retriever.k = k or settings.SPARSE_TOP_K
    try:
        return retriever.invoke(query)
    except Exception as error:
        print(f"    [warn] BM25 检索失败：{error}")
        return []


def search(query: str, k: Optional[int] = None) -> List[Document]:
    """单查询混合检索：稠密 ⊕ 稀疏，RRF 融合。HYBRID_ENABLED=false 时退化为纯向量检索。"""
    limit = k or settings.FUSION_TOP_K

    if not settings.HYBRID_ENABLED:
        return dense_search(query, k=limit)

    dense_docs = dense_search(query, k=settings.DENSE_TOP_K)
    sparse_docs = sparse_search(query, k=settings.SPARSE_TOP_K)

    # BM25 不可用（没装/语料为空）时直接返回稠密结果，不让整条链路挂掉
    if not sparse_docs:
        return dense_docs[:limit]

    return rrf_fuse(
        [dense_docs, sparse_docs],
        top_k=limit,
        weights=[settings.DENSE_WEIGHT, settings.SPARSE_WEIGHT],
    )


def multi_query_search(
    queries: Sequence[str],
    k: Optional[int] = None,
    strategy: str = "rrf",
) -> List[Document]:
    """多查询融合检索（RAG-Fusion）：对每条查询各做一次混合检索，再用 RRF 跨查询融合。

    为什么有效：一个问题的多种问法会命中不同的文档块，融合后召回面明显变宽；
    RRF 又能把"多路都排前面"的块提上来，等于用排名投票做了一次无需训练的过滤。
    后面还有 Reranker 兜底精排，所以这里召回宁宽勿窄。
    """
    effective = [q.strip() for q in queries if q and q.strip()]
    if not effective:
        return []

    # 单条查询没必要走融合
    if len(effective) == 1:
        return search(effective[0], k=k)

    if strategy == "rrf":
        per_query = [search(q, k=settings.FUSION_TOP_K) for q in effective]
        return rrf_fuse(per_query, top_k=k or settings.FUSION_TOP_K)

    # 备用策略：按最好名次排序
    pooled: Dict[str, Document] = {}
    best_rank: Dict[str, int] = {}
    for query in effective:
        for rank, doc in enumerate(search(query, k=settings.FUSION_TOP_K)):
            key = _doc_key(doc)
            pooled.setdefault(key, doc)
            best_rank[key] = min(best_rank.get(key, 10**6), rank)

    ordered = sorted(pooled.keys(), key=lambda key: best_rank[key])
    return [pooled[key] for key in ordered[: (k or settings.FUSION_TOP_K)]]
