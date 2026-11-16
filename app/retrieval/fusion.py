"""Reciprocal Rank Fusion（RRF）：把多路召回结果按排名融合成一路。

不用加权求和而用 RRF，是因为稠密相似度（余弦，0~1）和 BM25 分数（无上界，
可能是 0.3 也可能是 27）**量纲完全不同**，直接加权毫无意义；RRF 只看"排名第几"，
天然免疫量纲问题：

    score(doc) = Σ  weight_i * 1 / (c + rank_i(doc))      c 取 60

这正是 Milvus 原生 hybrid search 和 LlamaIndex QueryFusionRetriever 的默认融合策略。
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

# RRF 公式里的常数 c。c 越大，排名靠后的文档被"拉平"得越厉害；
# 60 是 Cormack 等人提出 RRF 时给出的经验值，也是 Milvus / LlamaIndex 的默认值。
RRF_C = 60


def _doc_key(doc: Document) -> str:
    """文档去重用的稳定 key。

    【踩坑记录】这里一开始写的是"优先用 Milvus 主键 pk"，跑测试才发现是错的：

        稠密检索走 Milvus，返回的 metadata 里带 pk 字段；
        BM25 走的是本地语料副本，那份 metadata 里没有 pk。
        于是同一块文档，两路算出来的 key 不一样，RRF 融合时被当成两个不同结果，
        最终返回的 Top-K 里出现了重复文档。

    现在统一用「来源 + 内容哈希」：
    - 与检索通路无关，稠密/稀疏/多查询三条路径算出的 key 一定一致
    - 用完整内容的 md5 而不是前 N 个字符，避免"长尾相同前缀"的块被误判为同一块
    - 顺带把重复入库产生的完全相同的块也合并掉
    """
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{doc.metadata.get('source', '')}|{digest}"


def rrf_fuse(
    rank_lists: Sequence[Sequence[Document]],
    top_k: int,
    weights: Optional[Sequence[float]] = None,
) -> List[Document]:
    """RRF 融合。

    rank_lists : 每一路的召回结果，已按相关性从高到低排序
    top_k      : 融合后保留多少条
    weights    : 每一路的权重，缺省等权

    返回融合后的文档列表（降序），每条的 metadata 里写入了 rrf_score 与命中路数 rrf_hits。
    """
    if not rank_lists:
        return []

    scores: Dict[str, float] = {}
    hits: Dict[str, int] = {}
    payload: Dict[str, Document] = {}

    for index, docs in enumerate(rank_lists):
        weight = 1.0 if not weights else float(weights[index])
        for rank, doc in enumerate(docs):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (RRF_C + rank + 1))
            hits[key] = hits.get(key, 0) + 1
            payload.setdefault(key, doc)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    fused: List[Document] = []
    for key, score in ordered:
        doc = payload[key]
        # 复制 metadata，避免缓存的 Document 对象被反复覆写
        doc.metadata = {**doc.metadata, "rrf_score": round(score, 6), "rrf_hits": hits[key]}
        fused.append(doc)

    return fused
