"""写入与维护：入库、按文档删除、清库、预热、运行时统计。

每个写操作都同时维护两份数据：Milvus 向量 + BM25 语料副本，并让 BM25 缓存失效。
"""

from __future__ import annotations

import re
from typing import Dict, List

from langchain_core.documents import Document

from app.config import settings
from app.retrieval.bm25 import _invalidate_bm25, get_bm25_retriever
from app.retrieval.corpus import _append_corpus, _clear_corpus, _remove_from_corpus, load_corpus
from app.retrieval.embeddings import _detect_device
from app.retrieval.milvus import _exception_chain, count, drop_collection, get_vector_store


def add_documents(docs: List[Document]) -> int:
    """把文本块写入 Milvus，同时落一份 BM25 语料副本。返回写入数量。"""
    if not docs:
        return 0

    try:
        get_vector_store().add_documents(list(docs))
    except Exception as error:
        # 最常见的失败是"旧 collection 的表结构和当前数据对不上"：
        # 早期用 enable_dynamic_field=False 建的库会把某个文件的 metadata 字段固化，
        # 之后再插别的文件就报 DataNotMatchException；换过 Embedding 模型也是类似症状。
        text = _exception_chain(error)
        if "DataNotMatchException" in text or "missed an field" in text or "dimension" in text.lower():
            raise RuntimeError(
                f"写入 Milvus 失败：数据和现有 collection「{settings.INDEX_NAME}」的表结构对不上"
                "（旧版本建的库，或换过 Embedding 模型导致向量维度变化）。\n"
                "处理办法：调用 POST /documents/reindex，用当前模型重建全部索引。\n\n"
                f"底层报错：{text}"
            ) from error
        raise

    # 向量写完再写语料副本，并让 BM25 下次检索时重建
    _append_corpus(docs)
    _invalidate_bm25()
    return len(docs)


def delete_documents_by_doc_id(doc_id: str) -> int:
    """删除某个文档的全部块（Milvus 向量 + BM25 语料），返回 BM25 侧删除的块数。

    chunk 的 metadata 里带 doc_id（入库时由 app/documents.py 写入），
    开启了动态字段，所以可以直接用 `doc_id == "..."` 作为过滤表达式。
    doc_id 是内容哈希的十六进制前缀，先校验格式再拼进表达式，避免注入。
    """
    if not re.fullmatch(r"[0-9a-f]{8,64}", doc_id or ""):
        raise ValueError(f"非法的 doc_id：{doc_id!r}")

    try:
        client = get_vector_store().client
        if client.has_collection(settings.INDEX_NAME):
            client.delete(collection_name=settings.INDEX_NAME, filter=f'doc_id == "{doc_id}"')
    except Exception as error:
        raise RuntimeError(f"从 Milvus 删除文档 {doc_id} 失败：{_exception_chain(error)}") from error

    removed = _remove_from_corpus(doc_id)
    _invalidate_bm25()
    return removed


def clear() -> None:
    """清空向量库：删除 collection、BM25 语料副本，并重置全部单例。"""
    drop_collection()
    _invalidate_bm25()
    _clear_corpus()
    print("[vector_store] 向量库已清空")


def warmup() -> None:
    """预热：加载 Embedding 模型、打开 Milvus、构建 BM25 索引。

    放在 FastAPI lifespan 里调用，避免第一个请求承担全部冷启动开销。
    """
    get_vector_store()
    get_bm25_retriever()


def stats() -> Dict[str, object]:
    """给 /health 用的运行时概览。"""
    bm25_docs = len(load_corpus())
    return {
        "milvus_uri": settings.MILVUS_URI,
        "collection": settings.INDEX_NAME,
        "index_count": count(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_device": _detect_device(),
        "hybrid_enabled": settings.HYBRID_ENABLED,
        "hybrid_weights": {"dense": settings.DENSE_WEIGHT, "sparse": settings.SPARSE_WEIGHT},
        "bm25_corpus_chunks": bm25_docs,
        "bm25_ready": bm25_docs > 0,
    }
