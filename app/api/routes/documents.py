"""文档管理接口：/documents/*、/index，以及兼容旧版的 /ingest/*。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi import Path as PathParam
from fastapi.responses import FileResponse

from app.api.runtime import DOC_ID_PATTERN, get_ingestion, get_registry, runtime
from app.api.schemas import (
    DocumentInfo,
    DocumentListResponse,
    IngestResponse,
    ScanResponse,
    UploadResponse,
)
from app.api.security import AUTH
from app.config import settings
from app.documents import (
    READY,
    UploadTooLargeError,
    delete_all_documents,
    delete_document,
    ingest_document,
    prepare_reindex,
    save_upload,
    scan_directory,
)
from app.vector_store import count

router = APIRouter(tags=["文档管理"], dependencies=AUTH)


async def _save(file: UploadFile):
    """保存上传文件并登记，把业务异常翻译成 HTTP 状态码。"""
    try:
        return await asyncio.to_thread(save_upload, get_registry(), file.file, file.filename or "")
    except UploadTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


# ================================================================ 新接口


@router.post("/documents", response_model=UploadResponse, status_code=202)
async def upload_document(file: UploadFile = File(...)):
    """上传一个文档并排队入库，立即返回（不等待解析和向量化）。

    - 内容与已有文档完全相同 -> action=duplicate，直接返回已有文档；
    - 与已上传的文档同名但内容不同 -> 视为新版本，入库成功后自动删除旧版本；
    - 之后用 GET /documents/{doc_id} 轮询 status，直到 ready 或 failed。
    """
    action, record = await _save(file)
    if action == "queued":
        await get_ingestion().submit(record.doc_id)
    return UploadResponse(action=action, document=DocumentInfo.from_record(record))


@router.post("/documents/scan", response_model=ScanResponse)
async def scan_documents():
    """扫描 data/ 目录（递归，不含上传目录），新文件和有变化的文件排队入库。"""
    results = await asyncio.to_thread(scan_directory, get_registry(), settings.DATA_DIR)
    queued = [record for action, record in results if action == "queued"]
    for record in queued:
        await get_ingestion().submit(record.doc_id)
    return ScanResponse(
        queued=len(queued),
        duplicates=len(results) - len(queued),
        documents=[DocumentInfo.from_record(record) for _, record in results],
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    status: Optional[str] = Query(None, pattern="^(pending|processing|ready|failed)$"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """文档列表（按最近更新时间倒序）。"""
    records, total = await asyncio.to_thread(get_registry().list, status, limit, offset)
    return DocumentListResponse(
        total=total,
        queue_pending=get_ingestion().pending,
        documents=[DocumentInfo.from_record(r) for r in records],
    )


@router.post("/documents/reindex")
async def reindex_documents():
    """用当前 Embedding 模型重建全部索引（换模型后必做）。

    清空向量库，把所有源文件仍存在的文档重新排队入库；期间检索结果会逐步恢复。
    """
    doc_ids = await asyncio.to_thread(prepare_reindex, get_registry())
    for doc_id in doc_ids:
        await get_ingestion().submit(doc_id)
    runtime.embedding_check = {"ok": True, "detail": "已触发重建，新索引将使用当前模型的维度"}
    return {"status": "ok", "queued": len(doc_ids), "detail": f"已清空向量库，{len(doc_ids)} 个文档排队重建"}


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(doc_id: str = PathParam(..., pattern=DOC_ID_PATTERN)):
    record = await asyncio.to_thread(get_registry().get, doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"文档不存在：{doc_id}")
    return DocumentInfo.from_record(record)


@router.get("/documents/{doc_id}/file")
async def get_document_file(doc_id: str = PathParam(..., pattern=DOC_ID_PATTERN)):
    """返回原文件（浏览器内联打开，PDF 可配合 #page=N 跳到引用页）。"""
    record = await asyncio.to_thread(get_registry().get, doc_id)
    if record is None or not Path(record.path).is_file():
        raise HTTPException(status_code=404, detail=f"文档或源文件不存在：{doc_id}")
    return FileResponse(record.path, filename=record.filename, content_disposition_type="inline")


@router.delete("/documents/{doc_id}")
async def remove_document(doc_id: str = PathParam(..., pattern=DOC_ID_PATTERN)):
    """删除文档：向量、BM25 语料、登记记录；上传的文件一并删除（目录扫描来的源文件保留）。"""
    deleted = await asyncio.to_thread(delete_document, get_registry(), doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"文档不存在：{doc_id}")
    return {"status": "ok", "detail": f"文档 {doc_id} 已删除"}


@router.delete("/index")
async def delete_index():
    """删除全部文档：清空向量库与登记表，并删除上传的文件（data/ 下的源文件保留）。会话记忆不受影响。"""
    await asyncio.to_thread(delete_all_documents, get_registry())
    return {"status": "ok", "detail": "全部文档已删除"}


# ================================================================ 旧接口（兼容）


@router.post("/ingest/file", response_model=IngestResponse, deprecated=True)
async def ingest_file(file: UploadFile = File(...)):
    """[兼容旧接口] 上传并**同步等待**入库完成。新代码请用 POST /documents。"""
    action, record = await _save(file)
    if action == "duplicate":
        return IngestResponse(
            status="duplicate", chunk_indexed=0, total_chunks=await asyncio.to_thread(count),
            detail=f"内容相同的文档已存在：{record.filename}（{record.doc_id}）",
        )

    final = await asyncio.to_thread(ingest_document, get_registry(), record.doc_id)
    if final is None or final.status != READY:
        raise HTTPException(status_code=500, detail=f"文档入库失败：{final.error if final else '未知原因'}")
    return IngestResponse(
        status="success", chunk_indexed=final.chunks, total_chunks=await asyncio.to_thread(count),
        detail=f"已入库 {final.filename}，{final.chunks} 个文档块",
    )


@router.post("/ingest/directory", response_model=IngestResponse, deprecated=True)
async def ingest_directory():
    """[兼容旧接口] 扫描 data/ 并**同步等待**全部入库完成。新代码请用 POST /documents/scan。"""
    registry = get_registry()
    results = await asyncio.to_thread(scan_directory, registry, settings.DATA_DIR)
    written, failed = 0, []
    for action, record in results:
        if action != "queued":
            continue
        final = await asyncio.to_thread(ingest_document, registry, record.doc_id)
        if final is not None and final.status == READY:
            written += final.chunks
        else:
            failed.append(record.filename)

    detail = f"扫描 {len(results)} 个文件，新入库 {written} 个文档块"
    if failed:
        detail += f"；失败 {len(failed)} 个：{', '.join(failed[:5])}"
    return IngestResponse(
        status="success" if not failed else "partial",
        chunk_indexed=written,
        total_chunks=await asyncio.to_thread(count),
        detail=detail,
    )
