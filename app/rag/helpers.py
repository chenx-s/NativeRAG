"""节点共用的纯函数：LLM 输出解析、上下文拼装、联网判定、流式写入。

这里的函数都不调用 LLM、不访问外部服务，可以直接单测。
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.config import settings
from app.context import render_conversation

# ================================================================ LLM 输出解析


def normalize_question(question: str) -> str:
    """规范化问题，用于判断查询改写前后是否发生实际变化。

    处理规则：
    1. 使用 NFKC 统一全角和半角字符。
    2. 忽略空格、换行符等空白字符。
    3. 忽略英文字符的大小写。
    4. 忽略句末的常见中英文标点。
    """
    normalized = unicodedata.normalize("NFKC", question)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.casefold()
    normalized = normalized.rstrip("。！？? ! ;  ； ， ,")
    return normalized


def parse_yes_no(text: str, default: bool) -> bool:
    """把 LLM 输出解析为布尔值；解析不出时返回 default。

    说明：GLM 对 with_structured_output 的 JSON 模式遵循不稳定
    （会输出裸 yes/no 而非 JSON），因此评分器统一使用
    纯文本 yes/no + 字符串解析，兼容性最好。
    """
    normalized = (text or "").strip().lower()
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False
    return default


_LIST_PREFIX = re.compile(r"^\s*(?:\d+\s*[\.\)、:：]|[-*•·])\s*")


def parse_query_list(text: str, limit: int) -> List[str]:
    """把 LLM 输出的多行查询解析成列表。

    容错处理：兼容 "1. xxx" / "1) xxx" / "- xxx" / "• xxx" 等编号形式，
    以及模型偶尔多写一句开场白的情况（长度异常的整句会被丢掉）。
    """
    queries: List[str] = []
    seen = set()

    for raw_line in (text or "").splitlines():
        line = _LIST_PREFIX.sub("", raw_line).strip().strip('"“”')
        if not line or len(line) > 200:
            continue
        # 模型有时会把整个问题再抄一遍，去重时用规范化结果
        key = normalize_question(line)
        if not key or key in seen:
            continue
        seen.add(key)
        queries.append(line)
        if len(queries) >= limit:
            break

    return queries


_ROUTE_LINE = re.compile(r"路由\s*[:：]\s*(web|local)", re.IGNORECASE)
_QUESTION_LINE = re.compile(r"问题\s*[:：]\s*(.+)")


def parse_route(text: str) -> Tuple[Optional[str], Optional[str]]:
    """解析路由器输出的两行文本 "路由：xxx / 问题：xxx"；解析不出的部分返回 None。"""
    text = text or ""
    route_match = _ROUTE_LINE.search(text)
    question_match = _QUESTION_LINE.search(text)
    route = route_match.group(1).lower() if route_match else None
    question = question_match.group(1).strip() if question_match else None
    if question and len(question) > 500:
        question = None
    return route, question


# ================================================================ 上下文拼装


def format_context(documents: List[Document]) -> str:
    """把检索到的文档块拼成带出处的上下文（本地片段带文件名/页码，网页片段带标题/URL）。"""
    parts = []
    for index, document in enumerate(documents):
        metadata = document.metadata
        if metadata.get("source_type") == "web":
            title = metadata.get("filename") or "无标题网页"
            url = metadata.get("source", "")
            parts.append(f"[片段{index + 1} | 网页：{title} | {url}]\n{document.page_content}")
            continue

        filename = metadata.get("filename", metadata.get("source", "未知"))
        page = metadata.get("page")
        page_text = f",第{page + 1}页" if page is not None else ""
        parts.append(f"[片段{index + 1} | {filename}{page_text}]\n{document.page_content}")
    return "\n\n".join(parts)


def render_conversation_for(state: dict) -> str:
    """上下文工程 —— 选择 + 压缩：把记忆渲染成受预算约束的对话背景块（可能为空）。"""
    return render_conversation(
        state.get("memory_summary", ""),
        state.get("history") or [],
        max_turns=settings.HISTORY_TURNS,
        max_chars=settings.MEMORY_CONTEXT_CHARS,
        answer_chars=settings.MEMORY_ANSWER_CHARS,
    )


# ================================================================ 联网判定


def web_allowed(state: dict) -> bool:
    """联网是否可用：全局已配置 Tavily，且本次请求没有禁用联网（allow_web=False）。"""
    return settings.web_search_ready and state.get("allow_web", True) is not False


def can_web_search(state: dict) -> bool:
    """联网可用，且本轮还没联网过（保证每次请求最多联网一次）。"""
    return web_allowed(state) and not state.get("web_searched")


def local_rewrite_budget(state: dict) -> int:
    """本地"改写重检索"的次数上限。

    联网可用时，本地失败应尽快转向联网（默认不改写直接联网）；
    联网不可用时，改写重检索是唯一的补救手段，按 MAX_RETRIES 来。
    """
    if web_allowed(state):
        return max(0, min(settings.MAX_RETRIES, settings.WEB_FALLBACK_AFTER_RETRIES))
    return max(0, settings.MAX_RETRIES)


# ================================================================ 流式输出


def stream_writer():
    """取 LangGraph 的自定义流写入器。

    不在流式上下文里（普通 ainvoke / 单测直接调用节点）时返回空操作，
    所以节点代码可以无条件调用 writer(...)。
    """
    try:
        from langgraph.config import get_stream_writer

        return get_stream_writer()
    except Exception:  # noqa: BLE001
        return lambda _chunk: None
