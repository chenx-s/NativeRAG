"""BM25 稀疏检索：中文分词器 + 倒排索引缓存。"""

from __future__ import annotations

import re
import threading
from typing import List

from app.config import settings
from app.retrieval.corpus import load_corpus

_BM25_LOCK = threading.Lock()
_BM25_RETRIEVER = None  # 模块级缓存，避免每次检索都重建倒排索引

# CJK 统一汉字区间 U+4E00 ~ U+9FFF
_CJK_PATTERN = re.compile(r"[一-鿿]")
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")


def _tokenize(text: str) -> List[str]:
    """BM25 分词器。

    BM25Retriever 默认用 `text.split()`，对中文等于整句一个 token，完全失效。
    这里优先用 jieba；没装 jieba 就退化为"英文按词 + 中文按字 + 相邻字二元组"，
    二元组能把"向量数据库"拆成 向量/量数/数据/据库，召回效果接近分词。
    """
    text = (text or "").lower()

    try:
        import jieba  # type: ignore

        return [token for token in jieba.lcut(text) if token.strip()]
    except Exception:
        pass

    tokens = _TOKEN_PATTERN.findall(text)

    # 给中文补相邻二元组，弥补无分词器时的召回损失
    bigrams: List[str] = []
    for left, right in zip(tokens, tokens[1:]):
        if _CJK_PATTERN.match(left) and _CJK_PATTERN.match(right):
            bigrams.append(left + right)

    return tokens + bigrams


def _invalidate_bm25() -> None:
    """清空 BM25 缓存（写入/删除/清库后调用，下次检索时按最新语料重建）。"""
    global _BM25_RETRIEVER
    with _BM25_LOCK:
        _BM25_RETRIEVER = None


def get_bm25_retriever():
    """构建（并缓存）BM25 检索器；语料为空或构建失败时返回 None（退化为纯向量检索）。"""
    global _BM25_RETRIEVER
    with _BM25_LOCK:
        if _BM25_RETRIEVER is not None:
            return _BM25_RETRIEVER

        documents = load_corpus()
        if not documents:
            return None

        try:
            from langchain_community.retrievers import BM25Retriever

            retriever = BM25Retriever.from_documents(
                documents,
                preprocess_func=_tokenize,
                k=settings.SPARSE_TOP_K,
            )
        except Exception as error:
            print(f"    [warn] BM25 检索器构建失败，将退化为纯向量检索：{error}")
            return None

        _BM25_RETRIEVER = retriever
        return _BM25_RETRIEVER
