"""文档管理：登记表 + 内容去重 + 版本替换 + 后台入库队列。

================================ 为什么需要这一层 ================================

旧版入库是"读文件 -> 切块 -> 追加写入"，没有任何"文档"的概念：
    · 同一个文件传两次，库里就有两份重复块，检索结果被重复内容占满；
    · 文件更新了只能整库清空重来；
    · 删除某一份文档做不到；
    · 大 PDF 入库要几十秒，HTTP 请求一直挂着，前端很容易超时。

现在每个文件都是一条登记记录（SQLite），每个 chunk 的 metadata 都带 doc_id：

    doc_id = 文件内容 SHA-256 的前 16 位
        -> 内容相同 = doc_id 相同 = 天然去重（换个文件名再传也能认出来）
    source_key = 文档的"身份"（上传：upload:<文件名>；目录扫描：dir:<相对路径>）
        -> source_key 相同但内容变了 = 新版本：新版本入库成功后，自动删除旧版本

入库走后台队列（单工作协程串行执行）：
    · 上传接口立即返回 doc_id 和 pending 状态，前端轮询 GET /documents/{doc_id}；
    · 串行是刻意的：Milvus Lite 只允许单写，Embedding 是 CPU 密集，并发入库只会互相拖慢；
    · 服务重启时，未完成（pending/processing）的任务会自动重新排队。

================================ 状态流转 ================================

    pending ──► processing ──► ready
                    │
                    └────────► failed（error 字段有原因，重新上传即可重试）
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from contextlib import closing, suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, List, Optional, Tuple

from app.config import settings
from app.document_processer import SUPPORTED_EXTENSIONS, load_document, split_documents
from app.vector_store import add_documents, clear as clear_vector_store, delete_documents_by_doc_id

logger = logging.getLogger(__name__)

PENDING = "pending"
PROCESSING = "processing"
READY = "ready"
FAILED = "failed"

ORIGIN_UPLOAD = "upload"
ORIGIN_DIRECTORY = "directory"

# 同一时刻只允许一个入库任务写向量库（后台队列 + 兼容旧接口的同步入库共用）
_INGEST_LOCK = threading.Lock()


# ================================================================ 数据结构


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    source_key: str
    path: str
    sha256: str
    size: int
    origin: str
    status: str = PENDING
    chunks: int = 0
    error: str = ""
    # 本文档是哪个旧版本的替代（入库成功后删除旧版本）
    replaces: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


_COLUMNS = (
    "doc_id, filename, source_key, path, sha256, size, origin, status, "
    "chunks, error, replaces, created_at, updated_at"
)


def _row_to_record(row) -> DocumentRecord:
    return DocumentRecord(*row)


# ================================================================ 登记表


class DocumentRegistry:
    """文档登记表（SQLite）。全部是同步方法，异步代码里请经 asyncio.to_thread 调用。"""

    def __init__(self, path: Optional[str] = None):
        self.path = path or settings.DOCUMENTS_DB_PATH
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10)
        if not self._schema_ready:
            with self._schema_lock:
                if not self._schema_ready:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS documents (
                            doc_id     TEXT PRIMARY KEY,
                            filename   TEXT NOT NULL,
                            source_key TEXT NOT NULL,
                            path       TEXT NOT NULL,
                            sha256     TEXT NOT NULL,
                            size       INTEGER NOT NULL,
                            origin     TEXT NOT NULL,
                            status     TEXT NOT NULL,
                            chunks     INTEGER NOT NULL DEFAULT 0,
                            error      TEXT NOT NULL DEFAULT '',
                            replaces   TEXT NOT NULL DEFAULT '',
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL
                        )
                        """
                    )
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_key)")
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
                    conn.commit()
                    self._schema_ready = True
        return conn

    def get(self, doc_id: str) -> Optional[DocumentRecord]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        return _row_to_record(row) if row else None

    def find_by_source(self, source_key: str) -> List[DocumentRecord]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE source_key = ? ORDER BY created_at",
                (source_key,),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def list(
        self, status: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> Tuple[List[DocumentRecord], int]:
        where, params = ("WHERE status = ?", [status]) if status else ("", [])
        with closing(self._connect()) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM documents {where}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM documents {where} ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [_row_to_record(r) for r in rows], int(total)

    def list_all(self) -> List[DocumentRecord]:
        with closing(self._connect()) as conn:
            rows = conn.execute(f"SELECT {_COLUMNS} FROM documents").fetchall()
        return [_row_to_record(r) for r in rows]

    def list_unfinished(self) -> List[DocumentRecord]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM documents WHERE status IN (?, ?) ORDER BY created_at",
                (PENDING, PROCESSING),
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def upsert(self, record: DocumentRecord) -> None:
        now = time.time()
        record.created_at = record.created_at or now
        record.updated_at = now
        values = (
            record.doc_id, record.filename, record.source_key, record.path, record.sha256,
            record.size, record.origin, record.status, record.chunks, record.error,
            record.replaces, record.created_at, record.updated_at,
        )
        with closing(self._connect()) as conn, conn:
            conn.execute(
                f"INSERT OR REPLACE INTO documents ({_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )

    def update_status(
        self,
        doc_id: str,
        status: str,
        *,
        chunks: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        sets, params = ["status = ?", "updated_at = ?"], [status, time.time()]
        if chunks is not None:
            sets.append("chunks = ?")
            params.append(chunks)
        if error is not None:
            sets.append("error = ?")
            params.append(error[:2000])
        with closing(self._connect()) as conn, conn:
            conn.execute(f"UPDATE documents SET {', '.join(sets)} WHERE doc_id = ?", [*params, doc_id])

    def delete(self, doc_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def clear_all(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM documents")


# ================================================================ 文件与登记


def _sha256_of(path: str) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def safe_filename(filename: str) -> str:
    """只保留文件名部分，防止 ../ 之类的路径穿越。"""
    name = Path(filename or "").name.strip().replace("\x00", "")
    return name or "unnamed"


def register_file(
    registry: DocumentRegistry,
    path: str,
    *,
    filename: str,
    origin: str,
    source_key: str,
    sha256: Optional[str] = None,
    size: Optional[int] = None,
) -> Tuple[str, DocumentRecord]:
    """登记一个文件，返回 (动作, 记录)。动作为 queued（需要入库）或 duplicate（已存在，跳过）。"""
    if sha256 is None or size is None:
        sha256, size = _sha256_of(path)
    doc_id = sha256[:16]

    existing = registry.get(doc_id)
    if existing and existing.status in (READY, PENDING, PROCESSING):
        return "duplicate", existing

    # 同一个身份（同名上传 / 同一路径）的其他版本 -> 本次是新版本，入库成功后替换旧版本
    previous = [r for r in registry.find_by_source(source_key) if r.doc_id != doc_id]
    record = DocumentRecord(
        doc_id=doc_id,
        filename=filename,
        source_key=source_key,
        path=str(path),
        sha256=sha256,
        size=int(size),
        origin=origin,
        status=PENDING,
        replaces=previous[-1].doc_id if previous else "",
        created_at=existing.created_at if existing else 0.0,
    )
    registry.upsert(record)
    return "queued", record


class UploadTooLargeError(ValueError):
    pass


def save_upload(
    registry: DocumentRegistry, fileobj: BinaryIO, filename: str
) -> Tuple[str, DocumentRecord]:
    """保存上传文件并登记（同步，含哈希计算）。

    边写临时文件边算哈希、边检查大小上限，不需要把整个文件读进内存。
    确定 doc_id 后再移动到 uploads/<doc_id>/<文件名>：同名不同内容的文件互不覆盖。
    """
    filename = safe_filename(filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型：{suffix}，支持：{sorted(SUPPORTED_EXTENSIONS)}")

    upload_root = Path(settings.UPLOAD_DIR)
    tmp_dir = upload_root / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{os.getpid()}-{threading.get_ident()}-{time.time_ns()}{suffix}"

    limit = max(1, settings.MAX_UPLOAD_MB) * 1024 * 1024
    digest, size = hashlib.sha256(), 0
    try:
        with open(tmp_path, "wb") as out:
            for block in iter(lambda: fileobj.read(1024 * 1024), b""):
                size += len(block)
                if size > limit:
                    raise UploadTooLargeError(f"文件超过大小上限 {settings.MAX_UPLOAD_MB} MB")
                digest.update(block)
                out.write(block)

        sha256 = digest.hexdigest()
        doc_id = sha256[:16]
        existing = registry.get(doc_id)
        if existing and existing.status in (READY, PENDING, PROCESSING):
            return "duplicate", existing

        dest_dir = upload_root / doc_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        os.replace(tmp_path, dest)

        return register_file(
            registry, str(dest), filename=filename, origin=ORIGIN_UPLOAD,
            source_key=f"upload:{filename}", sha256=sha256, size=size,
        )
    finally:
        with suppress(OSError):
            tmp_path.unlink()


def _is_managed_upload_path(path: Path, upload_root: Path) -> bool:
    """是否位于新版上传目录结构 uploads/<doc_id>/ 或 uploads/.tmp/ 下（这些已单独登记）。

    旧版 /ingest/file 把文件平铺存在 uploads/ 根目录，这类文件不算，扫描时照常登记，
    这样升级后执行一次 /documents/scan 就能把旧上传的文件重新纳入管理。
    """
    try:
        relative = path.resolve().relative_to(upload_root)
    except ValueError:
        return False
    if len(relative.parts) < 2:
        return False
    first = relative.parts[0]
    return first == ".tmp" or bool(re.fullmatch(r"[0-9a-f]{16}", first))


def scan_directory(registry: DocumentRegistry, dir_path: Optional[str] = None) -> List[Tuple[str, DocumentRecord]]:
    """扫描数据目录（递归），登记所有支持的文件；跳过新版上传目录下已单独登记的文件。"""
    root = Path(dir_path or settings.DATA_DIR)
    if not root.exists():
        return []

    upload_root = Path(settings.UPLOAD_DIR).resolve()
    results: List[Tuple[str, DocumentRecord]] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if _is_managed_upload_path(file_path, upload_root):
            continue
        relative = file_path.relative_to(root).as_posix()
        try:
            results.append(register_file(
                registry, str(file_path), filename=file_path.name,
                origin=ORIGIN_DIRECTORY, source_key=f"dir:{relative}",
            ))
        except OSError as error:
            logger.warning("跳过无法读取的文件 %s：%s", file_path, error)
    return results


# ================================================================ 入库 / 删除


def _is_managed_upload(record: DocumentRecord) -> bool:
    """文件是否位于 uploads/<doc_id>/ 下（只有这类文件允许被我们删除）。"""
    try:
        expected = (Path(settings.UPLOAD_DIR) / record.doc_id).resolve()
        return Path(record.path).resolve().parent == expected
    except OSError:
        return False


def delete_document(registry: DocumentRegistry, doc_id: str, *, remove_file: bool = True) -> bool:
    """删除文档：向量 + BM25 语料 + 登记记录（+ 上传的源文件）。不存在时返回 False。

    目录扫描来的文档不会删除用户的源文件；下次扫描目录时它会被重新登记。
    """
    record = registry.get(doc_id)
    if record is None:
        return False

    with _INGEST_LOCK:
        delete_documents_by_doc_id(doc_id)
        if remove_file and record.origin == ORIGIN_UPLOAD and _is_managed_upload(record):
            shutil.rmtree(Path(record.path).parent, ignore_errors=True)
        registry.delete(doc_id)
    return True


def ingest_document(registry: DocumentRegistry, doc_id: str) -> Optional[DocumentRecord]:
    """执行一个入库任务（同步，在工作线程中运行）。失败不抛异常，结果写进登记表。"""
    record = registry.get(doc_id)
    if record is None or record.status == READY:
        return record

    with _INGEST_LOCK:
        registry.update_status(doc_id, PROCESSING, error="")
        try:
            documents = load_document(record.path)
            chunks = split_documents(documents)
            if not chunks:
                raise ValueError("文档里没有可检索的文本（可能是扫描版 PDF、纯图片或空文件）")

            for chunk in chunks:
                chunk.metadata["doc_id"] = doc_id
                chunk.metadata["filename"] = record.filename

            written = add_documents(chunks)
            registry.update_status(doc_id, READY, chunks=written, error="")
            logger.info("文档入库完成：%s（%s，%d 块）", record.filename, doc_id, written)
        except Exception as error:  # noqa: BLE001
            message = f"{type(error).__name__}: {error}"
            logger.warning("文档入库失败：%s（%s）：%s", record.filename, doc_id, message)
            # 写了一半的块要清掉，否则失败的文档仍会被检索到
            with suppress(Exception):
                delete_documents_by_doc_id(doc_id)
            registry.update_status(doc_id, FAILED, chunks=0, error=message)
            return registry.get(doc_id)

    # 新版本入库成功后再删除旧版本：替换过程中检索始终有内容可用
    if record.replaces and record.replaces != doc_id:
        with suppress(Exception):
            delete_document(registry, record.replaces)

    return registry.get(doc_id)


def prepare_reindex(registry: DocumentRegistry) -> List[str]:
    """清空向量库，把所有源文件仍在的文档重置为 pending，返回需要重新入库的 doc_id。

    用于更换 Embedding 模型（向量维度变化）后的一键重建。
    """
    with _INGEST_LOCK:
        clear_vector_store()
        queued: List[str] = []
        for record in registry.list_all():
            if Path(record.path).exists():
                registry.update_status(record.doc_id, PENDING, chunks=0, error="")
                queued.append(record.doc_id)
            else:
                registry.update_status(record.doc_id, FAILED, chunks=0, error="源文件不存在，无法重建")
    return queued


def delete_all_documents(registry: DocumentRegistry) -> None:
    """清空向量库和登记表，并删除上传目录里的文件（目录扫描来的源文件不动）。"""
    with _INGEST_LOCK:
        records = registry.list_all()
        clear_vector_store()
        for record in records:
            if record.origin == ORIGIN_UPLOAD and _is_managed_upload(record):
                shutil.rmtree(Path(record.path).parent, ignore_errors=True)
        registry.clear_all()


# ================================================================ 后台队列


class IngestionQueue:
    """进程内的后台入库队列：一个工作协程，任务在线程池里串行执行。"""

    def __init__(self, registry: DocumentRegistry):
        self.registry = registry
        self._queue: Optional[asyncio.Queue] = None
        self._task: Optional[asyncio.Task] = None

    async def start(self, resume: bool = True) -> None:
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._worker(), name="ingestion-worker")
        if resume:
            unfinished = await asyncio.to_thread(self.registry.list_unfinished)
            for record in unfinished:
                await self.submit(record.doc_id)
            if unfinished:
                logger.info("恢复 %d 个未完成的入库任务", len(unfinished))

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def submit(self, doc_id: str) -> None:
        if self._queue is None:
            raise RuntimeError("入库队列尚未启动")
        await self._queue.put(doc_id)

    async def join(self) -> None:
        """等待队列中所有任务完成（测试 / 同步兼容接口用）。"""
        if self._queue is not None:
            await self._queue.join()

    @property
    def pending(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0

    async def _worker(self) -> None:
        while True:
            doc_id = await self._queue.get()
            try:
                await asyncio.to_thread(ingest_document, self.registry, doc_id)
            except Exception:  # noqa: BLE001
                logger.exception("入库任务异常：%s", doc_id)
            finally:
                self._queue.task_done()
