"""联网搜索（Tavily）—— 作为 LangGraph 状态图里的一个检索节点使用。

【为什么不再做成 Agent 工具】
    旧版本由一个 ReAct Agent 在"本地知识库"和"联网搜索"两个工具之间自由选择，
    但什么时候该联网，其实只有两种确定的情况：
        ① 问题依赖实时/最新信息（新闻、价格、天气……）—— 入口路由一次判断即可
        ② 本地检索 + 精排 + 评分之后没有相关文档    —— 纯规则就能判断
    这正是 Corrective RAG（CRAG）的做法：联网是检索失败后的**纠错分支**，
    用条件边表达即可，不需要让 LLM 每轮都推理"要不要调工具"。

【本模块只做三件事】
    1. 调 Tavily 搜索（异步客户端，懒加载，进程内复用连接池）
    2. 故障分类：临时故障（超时/限流/5xx）退避重试，永久故障（无 key/配额/参数错）直接失败
    3. 把搜索结果转成 LangChain Document，和本地检索结果走同一套生成 / 忠实性检查
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)


class WebSearchError(Exception):
    """联网搜索失败（已按策略重试过）。message 是可直接展示给用户的中文原因。"""


# ================================================================ 故障分类

_TRANSIENT_HINTS = (
    "timeout", "timed out", "connection", "connect", "temporarily",
    "unavailable", "rate limit", "too many requests", "429",
    "502", "503", "504", "overloaded", "reset by peer", "ssl",
)
_PERMANENT_HINTS = (
    "invalid api key", "unauthorized", "authentication", "api key",
    "quota", "usage limit", "exceeded", "forbidden", "not permitted",
    "400", "401", "403", "bad request", "invalid parameter",
)


def classify_exception(exc: BaseException) -> Tuple[bool, str]:
    """判断异常是否值得重试。返回 (是否可重试, 中文原因)。

    先看明确的异常类型，判断不出来再退化为关键词匹配；
    认不出来的一律不重试（保守选择，避免白白等待多个超时周期）。
    """
    exc_type = type(exc).__name__

    try:
        from tavily.errors import (
            BadRequestError,
            ForbiddenError,
            InvalidAPIKeyError,
            MissingAPIKeyError,
            TimeoutError as TavilyTimeoutError,
            UsageLimitExceededError,
        )

        if isinstance(exc, TavilyTimeoutError):
            return True, "联网搜索超时（Tavily 响应太慢）"
        if isinstance(exc, (InvalidAPIKeyError, MissingAPIKeyError)):
            return False, "Tavily API key 无效或未配置，请检查 .env 里的 TAVILY_API_KEY"
        if isinstance(exc, UsageLimitExceededError):
            return False, "Tavily 额度已用完（下月重置或升级套餐）"
        if isinstance(exc, ForbiddenError):
            return False, "Tavily 拒绝访问（key 权限或地区限制）"
        if isinstance(exc, BadRequestError):
            return False, f"搜索请求参数不合法：{exc}"
    except ImportError:
        pass

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True, f"网络连接异常：{exc_type}"

    try:
        import httpx

        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True, f"网络请求失败：{exc_type}"
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status == 429 or status >= 500:
                return True, f"搜索服务暂时不可用（HTTP {status}）"
            return False, f"搜索请求被拒绝（HTTP {status}）"
    except ImportError:
        pass

    message = f"{exc_type}: {exc}".lower()
    if any(hint in message for hint in _PERMANENT_HINTS):
        return False, f"联网搜索失败（{exc_type}）：{exc}"
    if any(hint in message for hint in _TRANSIENT_HINTS):
        return True, f"联网搜索暂时失败（{exc_type}）：{exc}"
    return False, f"联网搜索失败（{exc_type}）：{exc}"


# ================================================================ 客户端


_client = None


def _get_client():
    """懒加载 AsyncTavilyClient：没用到联网时不 import、不建连接池。"""
    global _client
    if _client is None:
        from tavily import AsyncTavilyClient

        _client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
    return _client


async def aclose_web_client() -> None:
    """关闭 Tavily 的 HTTP 连接池。

    异步客户端绑定创建它的事件循环；每次 asyncio.run() 结束前都要关掉，
    否则下一个事件循环复用旧连接池会报 "Event loop is closed"。
    """
    global _client
    client, _client = _client, None
    if client is None:
        return
    close = getattr(client, "close", None) or getattr(client, "aclose", None)
    if close is None:
        return
    try:
        result = close()
        if asyncio.iscoroutine(result):
            await result
    except Exception as error:  # noqa: BLE001
        logger.debug("关闭 Tavily 客户端失败：%s", error)


# ================================================================ 结果转换


def results_to_documents(payload: dict) -> List[Document]:
    """把 Tavily 返回转成 Document 列表。

    每条正文截断到 WEB_SNIPPET_CHARS：几条网页全文能轻松吃掉几千 token，
    还会稀释关键信息。Tavily 自带的 AI 摘要（answer）不进上下文 ——
    它不是原始网页，放进去会让忠实性检查失去"可核对的出处"。
    """
    limit = max(100, settings.WEB_SNIPPET_CHARS)
    documents: List[Document] = []

    for item in payload.get("results") or []:
        content = (item.get("content") or "").strip()
        if not content:
            continue
        if len(content) > limit:
            content = content[:limit] + " …（已截断）"

        url = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip() or url or "无标题网页"
        score = item.get("score")

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": url,
                    "filename": title,
                    "source_type": "web",
                    "relevance_score": float(score) if isinstance(score, (int, float)) else None,
                },
            )
        )

    return documents


# ================================================================ 对外接口


async def asearch_web(query: str, max_results: Optional[int] = None) -> List[Document]:
    """联网搜索，返回 Document 列表；失败时抛 WebSearchError（原因已转成中文）。"""
    if not settings.web_search_ready:
        raise WebSearchError(
            "联网搜索未启用：请在 .env 中配置 TAVILY_API_KEY（免费申请 https://tavily.com），"
            "并确认 WEB_SEARCH_ENABLED=true"
        )

    safe_max = max(1, min(int(max_results or settings.WEB_SEARCH_MAX_RESULTS), 10))
    max_attempts = max(1, settings.WEB_SEARCH_MAX_ATTEMPTS)

    for attempt in range(1, max_attempts + 1):
        try:
            client = _get_client()
            payload = await client.search(
                query=query,
                max_results=safe_max,
                search_depth=settings.WEB_SEARCH_DEPTH,
                timeout=settings.WEB_SEARCH_TIMEOUT,
            )
            documents = results_to_documents(payload or {})
            logger.info("[联网搜索] query=%r 返回 %d 条", query, len(documents))
            return documents
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001
            retryable, reason = classify_exception(error)
            if not retryable or attempt >= max_attempts:
                logger.warning("[联网搜索] 第 %d/%d 次失败，放弃：%s", attempt, max_attempts, reason)
                raise WebSearchError(reason) from error

            delay = min(10.0, 1.0 * (2 ** (attempt - 1))) + random.uniform(0, 0.3)
            logger.warning("[联网搜索] 第 %d/%d 次失败，%.1fs 后重试：%s", attempt, max_attempts, delay, reason)
            await asyncio.sleep(delay)

    raise WebSearchError("联网搜索失败（未知原因）")
