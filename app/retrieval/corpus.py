"""BM25 语料副本（JSONL）。

Milvus 只存向量，BM25 需要原始文本；每次入库同时追加一份到这里，服务重启后据此重建倒排索引。
追加与重写（删除文档时）共用一把锁，否则并发入库/删除会丢行。
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document

from app.config import settings

_CORPUS_LOCK = threading.Lock()


def _append_corpus(docs: Sequence[Document]) -> None:
    """把文本块追加写入 JSONL。"""
    if not docs:
        return

    path = Path(settings.bm25_corpus_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _CORPUS_LOCK, open(path, "a", encoding="utf-8") as handle:
        for doc in docs:
            record = {"page_content": doc.page_content, "metadata": doc.metadata}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_corpus() -> List[Document]:
    """读回 BM25 语料副本。文件不存在时返回空列表；损坏的行跳过。"""
    path = Path(settings.bm25_corpus_path)
    if not path.exists():
        return []

    documents: List[Document] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = record.get("page_content", "")
            if content:
                documents.append(Document(page_content=content, metadata=record.get("metadata") or {}))

    return documents


def _clear_corpus() -> None:
    """删除 BM25 语料副本。"""
    path = Path(settings.bm25_corpus_path)
    if path.exists():
        try:
            path.unlink()
        except OSError as error:
            print(f"    [warn] BM25 语料副本删除失败：{error}")


def _remove_from_corpus(doc_id: str) -> int:
    """从语料副本里删掉某个文档的全部块，返回删除的行数。

    先写临时文件再 os.replace 原子替换：中途崩溃也不会留下写了一半的语料文件。
    """
    path = Path(settings.bm25_corpus_path)
    if not path.exists():
        return 0

    removed = 0
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with _CORPUS_LOCK:
        with open(path, "r", encoding="utf-8") as src, open(tmp_path, "w", encoding="utf-8") as dst:
            for line in src:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    dst.write(stripped + "\n")  # 解析不了的行原样保留，不擅自丢数据
                    continue
                if (record.get("metadata") or {}).get("doc_id") == doc_id:
                    removed += 1
                    continue
                dst.write(stripped + "\n")
        os.replace(tmp_path, path)

    return removed
