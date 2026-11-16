"""Milvus 向量库：连接、打开失败时的中文诊断、实体计数、维度检查。"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.retrieval.embeddings import embedding_dim, get_embeddings

# ================================================================ 连接


def _connection_args() -> Dict[str, str]:
    """组装 Milvus 连接参数。

    - 本地文件（Milvus Lite）：./vector_db/milvus_rag.db
    - 单机服务：http://localhost:19530
    - Zilliz Cloud：https://xxx.zillizcloud.com   + token
    """
    args: Dict[str, str] = {"uri": settings.MILVUS_URI}

    # 本地文件模式要先保证父目录存在，否则 Milvus Lite 打不开库
    if not settings.MILVUS_URI.startswith(("http://", "https://")):
        parent = Path(settings.MILVUS_URI).expanduser().resolve().parent
        parent.mkdir(parents=True, exist_ok=True)

    if settings.MILVUS_TOKEN:
        args["token"] = settings.MILVUS_TOKEN

    return args


def _exception_chain(error: BaseException) -> str:
    """把整条异常链（__cause__ / __context__）拼成一个字符串，用于关键字判断。

    第三方库里真正的原因常常被包装了一层，只看最外层异常会漏掉关键信息。
    """
    parts: List[str] = []
    seen = set()
    current: Optional[BaseException] = error

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__

    return " | ".join(parts)


# ================================================================ 打开失败诊断


def _local_db_locked(db_path: str) -> bool:
    """探测本地 Milvus 库的独占锁是否已被别的进程持有。"""
    if not os.path.isdir(db_path):
        return False  # 目录都还没建，不是锁的问题

    try:
        fd = os.open(os.path.join(db_path, "LOCK"), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return False

    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        return False  # 我们加锁成功了，说明没人占用
    except OSError:
        return True  # 加锁失败，说明别的进程占着
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _explain_milvus_open_failure(error: BaseException) -> None:
    """把 pymilvus 那句含糊的报错翻译成能直接照做的提示，然后抛出 RuntimeError。"""
    text = _exception_chain(error)

    # 情况一：没装 milvus-lite，只有远程客户端
    if "milvus-lite is required" in text or "No module named 'milvus_lite'" in text:
        raise RuntimeError(
            "本地文件模式需要 milvus-lite，当前环境没装：\n"
            "    pip install milvus-lite\n"
            "或改用远程 Milvus：把 .env 的 RAG_MILVUS_URI 设为 http://localhost:19530"
        ) from error

    # 情况二：地址格式不对（例如误用了保留变量名 MILVUS_URI 被提前解析）
    if "Illegal uri" in text:
        raise RuntimeError(
            f"Milvus 地址格式不对：{settings.MILVUS_URI}\n"
            "本地文件模式应形如 ./vector_db/milvus_rag.db，"
            "远程服务应形如 http://localhost:19530。\n"
            "注意：配置项必须叫 RAG_MILVUS_URI。使用 MILVUS_URI 这个变量名会让 "
            "pymilvus 在 import 阶段就去解析它，从而直接失败。"
        ) from error

    # 情况三：本地文件模式打开失败。pymilvus 的报错里没有有效信息，
    # 所以用 _local_db_locked() 主动探测一把，把真正的原因找出来。
    is_local_file = (
        settings.MILVUS_URI.endswith(".db")
        and not settings.MILVUS_URI.startswith(("http://", "https://"))
    )

    if is_local_file and "Open local milvus failed" in text:
        db_path = os.path.abspath(os.path.expanduser(settings.MILVUS_URI))
        parent = os.path.dirname(db_path)

        if not os.path.isdir(parent):
            raise RuntimeError(
                f"Milvus 本地库的上级目录不存在：{parent}\n"
                f"请先创建该目录，或确认 .env 里 RAG_MILVUS_URI（当前值 {settings.MILVUS_URI}）"
                "配置正确。"
            ) from error

        if _local_db_locked(db_path):
            raise RuntimeError(
                f"Milvus Lite 本地库已被另一个进程占用，无法打开：\n    {db_path}\n\n"
                "Milvus Lite 对同一个库文件加【独占锁】，同一时刻只允许一个进程打开。\n"
                "最常见的场景：已经有一个 `python main.py` 在跑，又启动了一次。\n\n"
                "排查与处理：\n"
                "    netstat -ano | findstr :8000          查看 8000 端口被哪个进程占用\n"
                "    taskkill /PID <上面查到的进程号> /F    停掉那个进程后再启动\n\n"
                "如果你确实需要多个实例共用同一个库，请改用远程 Milvus：\n"
                "    docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest\n"
                "然后把 .env 里 RAG_MILVUS_URI 改成 http://localhost:19530\n"
                "（远程模式下多进程共用同一个库不会冲突）"
            ) from error

        raise RuntimeError(
            f"Milvus 本地库打开失败：{db_path}\n"
            "锁没有被占用、上级目录也存在，可能是库文件损坏或权限不足。\n"
            "可以试着删掉该目录让它重建（注意会丢失已入库的数据）：\n"
            f"    rmdir /s /q \"{db_path}\""
        ) from error

    # 其他情况：保留原始异常，但把异常链打印出来方便定位
    print(f"    [warn] Milvus 打开失败，异常链：{text}")


# ================================================================ 向量库单例


@lru_cache(maxsize=1)
def get_vector_store():
    """Milvus 向量库（collection 名为 INDEX_NAME）。

    【混合 metadata 踩坑 —— enable_dynamic_field 必须为 True】
        langchain-milvus 默认 `enable_dynamic_field=False`，此时它会拿**第一次入库
        时扫到的全部 metadata key** 去建固定字段。之后再插 metadata 形状不同的文件
        （比如 PDF 带 producer、txt 没有）就会报 DataNotMatchException。
        开启动态字段后，所有 metadata 统一进 Milvus 的 `$meta` JSON 字段，
        任意形状的 metadata 都能写入，检索时也会自动回填，还能按 doc_id 过滤删除。
    """
    from langchain_milvus import Milvus

    # 先在 try 外加载 Embedding：模型下载失败有自己的中文提示（见 embeddings.py），
    # 不能被下面的 except 误报成"Milvus 打开失败"
    embeddings = get_embeddings()

    try:
        return Milvus(
            embedding_function=embeddings,
            collection_name=settings.INDEX_NAME,
            connection_args=_connection_args(),
            # 主键交给 Milvus 自动生成，避免重复插入时主键冲突
            auto_id=True,
            # True 时每次启动重建 collection（改了向量维度/切了 embedding 模型才需要）
            drop_old=settings.MILVUS_DROP_OLD,
            # 统一用文本字段存原文，检索结果回填 metadata
            text_field="text",
            vector_field="vector",
            enable_dynamic_field=True,
        )
    except Exception as error:
        _explain_milvus_open_failure(error)
        raise


def drop_collection() -> None:
    """删除 collection 并让单例失效（清库用）。"""
    try:
        client = getattr(get_vector_store(), "client", None)
        if client is not None and client.has_collection(settings.INDEX_NAME):
            client.drop_collection(collection_name=settings.INDEX_NAME)
    except Exception as error:
        print(f"    [warn] 通过 langchain-milvus 删除 collection 失败，改用原生客户端：{error}")
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(**_connection_args())
            if client.has_collection(settings.INDEX_NAME):
                client.drop_collection(settings.INDEX_NAME)
        except Exception as inner:
            print(f"    [warn] 原生客户端删除失败：{inner}")

    # 单例必须失效，否则下次拿到的是指向旧 collection 的空壳对象
    get_vector_store.cache_clear()


# ================================================================ 统计与检查


def count() -> int:
    """当前向量库中的实体数量。

    用 count(*) 查询而不是 get_collection_stats：后者的 row_count 是 Milvus 的
    统计缓存，刚写入的数据可能还没算进去（实测会滞后），清库后也不一定立刻归零。
    """
    try:
        client = get_vector_store().client  # langchain-milvus 0.4 起是 @property
    except Exception:
        return 0

    # 必须先判断 collection 是否存在：langchain-milvus 是**惰性建表**的，
    # 没有文档时直接 query 会抛 MilvusException(code=100)，还会刷一整屏 ERROR 堆栈。
    try:
        if not client.has_collection(settings.INDEX_NAME):
            return 0
    except Exception:
        return 0

    try:
        rows = client.query(collection_name=settings.INDEX_NAME, filter="", output_fields=["count(*)"])
        if rows:
            return int(rows[0].get("count(*)", 0))
    except Exception:
        pass

    # 兜底：统计缓存（可能滞后，但聊胜于无）
    try:
        stats = client.get_collection_stats(collection_name=settings.INDEX_NAME)
        return int(stats.get("row_count", 0))
    except Exception:
        return 0


def check_embedding_dimension() -> Dict[str, object]:
    """检查现有 collection 的向量维度和当前 Embedding 模型是否一致。

    换了 Embedding 模型（例如 bge-small 384 维 -> bge-m3 1024 维）却没重建索引时，
    检索会在 Milvus 侧报一条含义模糊的错误。启动时主动检查，把问题和解决办法说清楚。
    返回 {"ok": bool, "collection_dim": int|None, "model_dim": int|None, "detail": str}
    """
    result: Dict[str, object] = {"ok": True, "collection_dim": None, "model_dim": None, "detail": ""}
    try:
        client = get_vector_store().client
        if not client.has_collection(settings.INDEX_NAME):
            return result  # 还没建表，第一次写入时会按当前模型的维度建

        description = client.describe_collection(settings.INDEX_NAME)
        for field_info in description.get("fields", []):
            params = field_info.get("params") or {}
            if field_info.get("name") == "vector" and "dim" in params:
                result["collection_dim"] = int(params["dim"])
                break

        result["model_dim"] = embedding_dim()
        if result["collection_dim"] and result["collection_dim"] != result["model_dim"]:
            result["ok"] = False
            result["detail"] = (
                f"向量库维度 {result['collection_dim']} 与当前模型 {settings.EMBEDDING_MODEL} "
                f"的维度 {result['model_dim']} 不一致（换过 Embedding 模型）。"
                "检索会失败，请调用 POST /documents/reindex 用新模型重建索引。"
            )
    except Exception as error:  # noqa: BLE001
        result["detail"] = f"维度检查未完成：{type(error).__name__}: {error}"
    return result
