"""对话记忆：按 session_id 持久化"滚动摘要 + 最近 N 轮原文"。

================================ 存什么 ================================

    ConversationMemory
        session_id   会话 ID（CLI 的 thread_id / HTTP 请求里的 session_id）
        summary      滚动摘要：被压缩掉的较早轮次的浓缩（用户目标、已得结论、未决问题）
        turns        最近若干轮原文 [(独立问题, 回答), ...]
        updated_at   最近一次写入时间（用于 TTL 过期）

    问题存的是 route_question 补全后的**独立问题**，而不是用户原话：
    "那它呢？"这种原话脱离上下文就没有意义，存独立问题能让之后的摘要和补全都更准。

================================ 怎么压缩 ================================

    每轮追加后，原文超过 MEMORY_WINDOW_TURNS 轮时，把较早的轮次和旧摘要一起交给 LLM
    合并成新摘要，只保留最近 MEMORY_KEEP_TURNS 轮原文。
    批量压缩（例如 8 轮压到 4 轮）而不是每轮都压，摘要调用次数降为约 1/4。
    摘要失败不影响问答：保留原文并按 2 倍窗口硬截断，下次再试。

================================ 存在哪 ================================

    sqlite  标准库 sqlite3，零依赖，默认文件 vector_db/chat_memory.sqlite3
    redis   复用 REDIS_URL，多实例部署时共享会话
    memory  进程内存，测试用

    为什么不直接用 LangGraph checkpointer？
        checkpointer 会在**每个超步**保存完整图状态（包括检索到的文档列表），
        适合"中断恢复 / 时间回溯"，用来存聊天记忆太重；而且需要额外安装
        langgraph-checkpoint-sqlite。这里只持久化真正需要跨请求保留的两样东西。

记忆是增强能力：存储读写失败只打印警告，问答照常进行。
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings

logger = logging.getLogger(__name__)

Turn = Tuple[str, str]


# ================================================================ 数据结构


@dataclass
class ConversationMemory:
    session_id: str
    summary: str = ""
    turns: List[Turn] = field(default_factory=list)
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "summary": self.summary,
            "turns": [list(turn) for turn in self.turns],
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationMemory":
        turns = [
            (str(item[0]), str(item[1]))
            for item in data.get("turns") or []
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]
        return cls(
            session_id=str(data.get("session_id", "")),
            summary=str(data.get("summary") or ""),
            turns=turns,
            updated_at=float(data.get("updated_at") or 0.0),
        )


def _is_expired(memory: ConversationMemory, ttl_seconds: int) -> bool:
    return ttl_seconds > 0 and memory.updated_at > 0 and (
        time.time() - memory.updated_at > ttl_seconds
    )


# ================================================================ 存储后端


class MemoryStore(Protocol):
    async def aload(self, session_id: str) -> Optional[ConversationMemory]: ...
    async def asave(self, memory: ConversationMemory) -> None: ...
    async def adelete(self, session_id: str) -> None: ...


class InMemoryMemoryStore:
    """进程内存存储（测试 / MEMORY_BACKEND=memory）。"""

    def __init__(self, ttl_seconds: int = 0):
        self.ttl_seconds = ttl_seconds
        self._data: Dict[str, dict] = {}

    async def aload(self, session_id: str) -> Optional[ConversationMemory]:
        data = self._data.get(session_id)
        if data is None:
            return None
        memory = ConversationMemory.from_dict(data)
        if _is_expired(memory, self.ttl_seconds):
            self._data.pop(session_id, None)
            return None
        return memory

    async def asave(self, memory: ConversationMemory) -> None:
        self._data[memory.session_id] = memory.to_dict()

    async def adelete(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class SqliteMemoryStore:
    """SQLite 存储：一会话一行，turns 以 JSON 保存。

    sqlite3 是同步 API，统一通过 asyncio.to_thread 执行，不阻塞事件循环；
    每次操作新开连接，避免跨线程共享连接（sqlite3 默认禁止）。
    开启 WAL：读写可以并发，CLI 和 API 同时打开同一个库也不会互相锁死。
    """

    def __init__(self, path: str, ttl_seconds: int = 0):
        self.path = path
        self.ttl_seconds = ttl_seconds
        self._schema_ready = False
        self._schema_lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=5)
        if not self._schema_ready:
            with self._schema_lock:
                if not self._schema_ready:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_memory (
                            session_id TEXT PRIMARY KEY,
                            summary    TEXT NOT NULL DEFAULT '',
                            turns      TEXT NOT NULL DEFAULT '[]',
                            updated_at REAL NOT NULL
                        )
                        """
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_chat_memory_updated "
                        "ON chat_memory(updated_at)"
                    )
                    conn.commit()
                    self._schema_ready = True
        return conn

    def _load(self, session_id: str) -> Optional[ConversationMemory]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT summary, turns, updated_at FROM chat_memory WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None

        try:
            turns = json.loads(row[1] or "[]")
        except json.JSONDecodeError:
            logger.warning("会话 %s 的历史记录损坏，已忽略", session_id)
            turns = []

        memory = ConversationMemory.from_dict({
            "session_id": session_id,
            "summary": row[0],
            "turns": turns,
            "updated_at": row[2],
        })
        if _is_expired(memory, self.ttl_seconds):
            self._delete(session_id)
            return None
        return memory

    def _save(self, memory: ConversationMemory) -> None:
        payload = json.dumps([list(turn) for turn in memory.turns], ensure_ascii=False)
        with closing(self._connect()) as conn, conn:
            conn.execute(
                """
                INSERT INTO chat_memory (session_id, summary, turns, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary = excluded.summary,
                    turns = excluded.turns,
                    updated_at = excluded.updated_at
                """,
                (memory.session_id, memory.summary, payload, memory.updated_at),
            )
            # 顺手清理过期会话（有索引，代价很小），防止库文件无限增长
            if self.ttl_seconds > 0:
                conn.execute(
                    "DELETE FROM chat_memory WHERE updated_at < ?",
                    (time.time() - self.ttl_seconds,),
                )

    def _delete(self, session_id: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM chat_memory WHERE session_id = ?", (session_id,))

    async def aload(self, session_id: str) -> Optional[ConversationMemory]:
        return await asyncio.to_thread(self._load, session_id)

    async def asave(self, memory: ConversationMemory) -> None:
        await asyncio.to_thread(self._save, memory)

    async def adelete(self, session_id: str) -> None:
        await asyncio.to_thread(self._delete, session_id)


class RedisMemoryStore:
    """Redis 存储：一会话一个 JSON 键，TTL 交给 Redis 的 EX 处理。"""

    def __init__(self, prefix: str, ttl_seconds: int = 0):
        self.prefix = prefix.rstrip(":")
        self.ttl_seconds = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"{self.prefix}:{session_id}"

    async def aload(self, session_id: str) -> Optional[ConversationMemory]:
        from app.redis_client import get_redis

        raw = await (await get_redis()).get(self._key(session_id))
        if not raw:
            return None
        return ConversationMemory.from_dict(json.loads(raw))

    async def asave(self, memory: ConversationMemory) -> None:
        from app.redis_client import get_redis

        await (await get_redis()).set(
            self._key(memory.session_id),
            json.dumps(memory.to_dict(), ensure_ascii=False),
            ex=self.ttl_seconds or None,
        )

    async def adelete(self, session_id: str) -> None:
        from app.redis_client import get_redis

        await (await get_redis()).delete(self._key(session_id))


def build_store_from_settings() -> Optional[MemoryStore]:
    backend = settings.MEMORY_BACKEND
    ttl = max(0, settings.MEMORY_TTL_SECONDS)
    if backend == "off":
        return None
    if backend == "memory":
        return InMemoryMemoryStore(ttl)
    if backend == "redis":
        return RedisMemoryStore(settings.MEMORY_REDIS_PREFIX, ttl)
    if backend != "sqlite":
        logger.warning("未知的 MEMORY_BACKEND=%s，改用 sqlite", backend)
    return SqliteMemoryStore(settings.MEMORY_SQLITE_PATH, ttl)


# ================================================================ 摘要压缩

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你负责维护一段对话的长期记忆摘要。请把【已有摘要】和【新增对话】合并成一段新的摘要。

    要求：
    1. 保留：用户的身份、目标与偏好；讨论过的主题；已经得出的关键结论、数字、名称与出处；尚未解决的问题。
    2. 删除：寒暄、重复内容、与后续对话无关的细节。
    3. 只写对话中确实出现过的信息，不要推断或编造。
    4. 用第三人称、中文书写，不超过 {max_chars} 字。只输出摘要正文。""",
    ),
    (
        "human",
        "【已有摘要】\n{summary}\n\n【新增对话】\n{turns}",
    ),
])


def _format_turns_for_summary(turns: List[Turn]) -> str:
    return "\n".join(f"用户：{q}\n助手：{a[:800]}" for q, a in turns)


# ================================================================ 记忆管理器


class MemoryManager:
    """会话记忆的读写入口（加载 / 追加并压缩 / 清除）。

    llm_factory: 返回 ChatModel 的函数（由 RAGChain 注入 get_llm，避免循环导入）。
    store 为 None 表示记忆已关闭：读返回空记忆，写什么都不做。
    """

    def __init__(
        self,
        store: Optional[MemoryStore],
        llm_factory: Optional[Callable] = None,
    ):
        self.store = store
        self.llm_factory = llm_factory
        self._locks: Dict[Tuple[int, str], asyncio.Lock] = {}

    @classmethod
    def from_settings(cls, llm_factory: Optional[Callable] = None) -> "MemoryManager":
        return cls(build_store_from_settings(), llm_factory)

    @property
    def enabled(self) -> bool:
        return self.store is not None

    @asynccontextmanager
    async def _session_lock(self, session_id: str):
        """同一进程内对同一会话的"读-改-写"串行化，防止并发请求互相覆盖。

        锁按 (事件循环, 会话) 区分：asyncio.Lock 会绑定首次等待它的事件循环，
        同步入口每次 asyncio.run 都是新循环，共用一把锁会报错。
        """
        key = (id(asyncio.get_running_loop()), session_id)
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            yield

    async def aload(self, session_id: str) -> ConversationMemory:
        if not self.enabled or not session_id:
            return ConversationMemory(session_id=session_id)
        try:
            memory = await self.store.aload(session_id)
        except Exception as error:  # noqa: BLE001
            logger.warning("读取会话记忆失败，按新会话处理：%s: %s", type(error).__name__, error)
            memory = None
        return memory or ConversationMemory(session_id=session_id)

    async def _summarize(self, previous: str, turns: List[Turn]) -> str:
        if self.llm_factory is None:
            raise RuntimeError("未配置摘要模型")
        chain = SUMMARY_PROMPT | self.llm_factory() | StrOutputParser()
        text = await chain.ainvoke({
            "summary": previous or "（无）",
            "turns": _format_turns_for_summary(turns),
            "max_chars": settings.MEMORY_SUMMARY_MAX_CHARS,
        })
        text = (text or "").strip()
        if not text:
            raise ValueError("摘要模型返回空内容")
        return text[: settings.MEMORY_SUMMARY_MAX_CHARS * 2]

    async def _compact(self, memory: ConversationMemory) -> None:
        window = max(2, settings.MEMORY_WINDOW_TURNS)
        keep = max(1, min(settings.MEMORY_KEEP_TURNS, window - 1))
        if len(memory.turns) <= window:
            return

        folded, recent = memory.turns[:-keep], memory.turns[-keep:]
        try:
            memory.summary = await self._summarize(memory.summary, folded)
            memory.turns = recent
            logger.info("会话 %s：%d 轮已压缩进摘要", memory.session_id, len(folded))
        except Exception as error:  # noqa: BLE001
            # 摘要失败时不丢信息，只做硬上限保护；下一轮会再次尝试压缩
            logger.warning("对话摘要失败，暂保留原文：%s: %s", type(error).__name__, error)
            memory.turns = memory.turns[-(window * 2):]

    async def aappend(self, session_id: str, question: str, answer: str) -> Optional[ConversationMemory]:
        """追加一轮问答；超出窗口时压缩；写回存储。失败只告警，不抛异常。"""
        if not self.enabled or not session_id:
            return None

        async with self._session_lock(session_id):
            memory = await self.aload(session_id)
            memory.turns.append((
                (question or "").strip(),
                (answer or "").strip()[: settings.MEMORY_STORE_ANSWER_CHARS],
            ))
            await self._compact(memory)
            memory.updated_at = time.time()
            try:
                await self.store.asave(memory)
            except Exception as error:  # noqa: BLE001
                logger.warning("保存会话记忆失败：%s: %s", type(error).__name__, error)
            return memory

    async def aclear(self, session_id: str) -> None:
        if not self.enabled or not session_id:
            return
        async with self._session_lock(session_id):
            await self.store.adelete(session_id)
