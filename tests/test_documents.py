"""文档管理测试：登记表、去重、版本替换、删除、后台队列、BM25 语料删除。

向量库与文档解析都被替换掉，测试只验证文档管理本身的逻辑。
"""

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.documents import Document

from app import documents as docs_module
from app.documents import (
    FAILED,
    READY,
    DocumentRegistry,
    IngestionQueue,
    UploadTooLargeError,
    delete_document,
    ingest_document,
    save_upload,
)


def _fake_load(path):
    return [Document(page_content=Path(path).read_text(encoding="utf-8"), metadata={"source": path})]


class DocumentTestCase(unittest.TestCase):
    def setUp(self):
        # Windows 上 SQLite 的 WAL 文件偶尔释放得慢，清理失败不应让测试失败
        self._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self._tmp.name)
        self.registry = DocumentRegistry(str(root / "documents.sqlite3"))
        self.added = []
        self.deleted = []

        def fake_add(chunks):
            self.added.extend(chunks)
            return len(chunks)

        self.patches = [
            patch("app.config.settings.UPLOAD_DIR", str(root / "uploads")),
            patch.object(docs_module, "load_document", side_effect=_fake_load),
            patch.object(docs_module, "split_documents", side_effect=lambda d: d),
            patch.object(docs_module, "add_documents", side_effect=fake_add),
            patch.object(docs_module, "delete_documents_by_doc_id", side_effect=self.deleted.append),
            patch.object(docs_module, "clear_vector_store", MagicMock()),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self._tmp.cleanup()

    def upload(self, name, text):
        return save_upload(self.registry, io.BytesIO(text.encode("utf-8")), name)


class UploadAndIngestTests(DocumentTestCase):
    def test_same_content_is_duplicate(self):
        action1, rec1 = self.upload("a.txt", "hello")
        ingest_document(self.registry, rec1.doc_id)
        action2, rec2 = self.upload("renamed.txt", "hello")

        self.assertEqual((action1, action2), ("queued", "duplicate"))
        self.assertEqual(rec1.doc_id, rec2.doc_id)

    def test_chunks_are_tagged_with_doc_id(self):
        _, record = self.upload("a.txt", "hello")
        final = ingest_document(self.registry, record.doc_id)

        self.assertEqual(final.status, READY)
        self.assertEqual(final.chunks, 1)
        self.assertEqual(self.added[0].metadata["doc_id"], record.doc_id)
        self.assertEqual(self.added[0].metadata["filename"], "a.txt")

    def test_same_name_new_content_replaces_old_version(self):
        _, old = self.upload("a.txt", "version 1")
        ingest_document(self.registry, old.doc_id)
        old_dir = Path(old.path).parent

        _, new = self.upload("a.txt", "version 2")
        self.assertEqual(new.replaces, old.doc_id)
        ingest_document(self.registry, new.doc_id)

        self.assertIsNone(self.registry.get(old.doc_id))
        self.assertIn(old.doc_id, self.deleted)
        self.assertFalse(old_dir.exists())
        self.assertEqual(self.registry.get(new.doc_id).status, READY)

    def test_failed_ingest_records_error_and_cleans_up(self):
        _, record = self.upload("empty.txt", "x")
        with patch.object(docs_module, "split_documents", return_value=[]):
            final = ingest_document(self.registry, record.doc_id)

        self.assertEqual(final.status, FAILED)
        self.assertIn("没有可检索的文本", final.error)
        self.assertIn(record.doc_id, self.deleted)

    def test_delete_document_removes_uploaded_file(self):
        _, record = self.upload("a.txt", "hello")
        ingest_document(self.registry, record.doc_id)

        self.assertTrue(delete_document(self.registry, record.doc_id))
        self.assertFalse(Path(record.path).exists())
        self.assertIsNone(self.registry.get(record.doc_id))
        self.assertFalse(delete_document(self.registry, record.doc_id))

    def test_rejects_unsupported_and_oversized_files(self):
        with self.assertRaises(ValueError):
            self.upload("a.exe", "x")
        with patch("app.config.settings.MAX_UPLOAD_MB", 1):
            with self.assertRaises(UploadTooLargeError):
                save_upload(self.registry, io.BytesIO(b"x" * (2 * 1024 * 1024)), "big.txt")

    def test_scan_picks_up_legacy_flat_uploads_but_skips_managed_ones(self):
        from app.documents import scan_directory

        data_dir = Path(self._tmp.name)
        (data_dir / "notes.txt").write_text("目录里的文件", encoding="utf-8")
        legacy = data_dir / "uploads" / "old.txt"  # 旧版 /ingest/file 平铺保存的上传文件
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_text("旧版上传", encoding="utf-8")
        _, managed = self.upload("new.txt", "新版上传")  # uploads/<doc_id>/new.txt

        results = scan_directory(self.registry, str(data_dir))
        keys = {record.source_key for _, record in results}

        self.assertIn("dir:notes.txt", keys)
        self.assertIn("dir:uploads/old.txt", keys)
        self.assertNotIn(f"dir:uploads/{managed.doc_id}/new.txt", keys)


class IngestionQueueTests(DocumentTestCase):
    def test_queue_processes_submitted_documents(self):
        import asyncio

        _, record = self.upload("a.txt", "hello")

        async def run():
            queue = IngestionQueue(self.registry)
            await queue.start(resume=False)
            await queue.submit(record.doc_id)
            await queue.join()
            await queue.stop()

        asyncio.run(run())
        self.assertEqual(self.registry.get(record.doc_id).status, READY)

    def test_unfinished_documents_are_resumed(self):
        import asyncio

        _, record = self.upload("a.txt", "hello")  # 状态 pending，模拟服务重启前未处理

        async def run():
            queue = IngestionQueue(self.registry)
            await queue.start(resume=True)
            await queue.join()
            await queue.stop()

        asyncio.run(run())
        self.assertEqual(self.registry.get(record.doc_id).status, READY)


class CorpusRemovalTests(unittest.TestCase):
    def test_remove_from_corpus_keeps_other_documents(self):
        from app import vector_store

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with patch("app.config.settings.VECTOR_DB_PATH", tmp):
                path = Path(vector_store.settings.bm25_corpus_path)
                rows = [
                    {"page_content": "a1", "metadata": {"doc_id": "aaaaaaaaaaaaaaaa"}},
                    {"page_content": "b1", "metadata": {"doc_id": "bbbbbbbbbbbbbbbb"}},
                    {"page_content": "a2", "metadata": {"doc_id": "aaaaaaaaaaaaaaaa"}},
                ]
                path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

                removed = vector_store._remove_from_corpus("aaaaaaaaaaaaaaaa")
                remaining = [d.page_content for d in vector_store.load_corpus()]

        self.assertEqual(removed, 2)
        self.assertEqual(remaining, ["b1"])


if __name__ == "__main__":
    unittest.main()
