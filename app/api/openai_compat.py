"""OpenAI 兼容接口：让 Open WebUI、Chatbox、LobeChat 等现成客户端直接当前端用。

    GET  /v1/models             列出模型
    POST /v1/chat/completions   对话（支持 stream=true）

================================ 映射规则 ================================

    messages 最后一条 user 消息            -> 本轮问题
    之前的 user/assistant 消息成对组成历史   -> history（客户端每次都会带全量历史，服务端不保存）
    system 消息                           -> 忽略（行为由本服务的提示词决定）
    model = <OPENAI_COMPAT_MODEL>         -> 本地知识库 + 按需联网
    model = <OPENAI_COMPAT_MODEL>-local   -> 只查本地知识库，不联网

    可选请求头 X-Session-Id：使用服务端持久化记忆（含滚动摘要），此时忽略 messages 里的历史。

================================ 流式输出 ================================

    · 进度（"检索知识库""联网搜索"……）放在 delta.reasoning_content：
      Open WebUI 等客户端会折叠显示为"思考过程"，不认识这个字段的客户端会直接忽略，
      也不会混进下一轮请求带回来的历史里；
    · 答案逐段放在 delta.content；
    · 草稿没通过忠实性检查时，OpenAI 协议无法撤回已发出的内容，
      所以插入一行醒目的分隔说明，再输出修订后的答案；
      下一轮请求带回历史时，会自动只保留分隔线之后的最终版本。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Callable, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.api.security import require_api_token
from app.api.sse import SSE_HEADERS
from app.config import settings

REGENERATE_NOTICE = "\n\n> ⚠️ 上面的草稿未通过出处核对，以下是严格依据资料重新生成的回答：\n\n"
REPLACED_NOTICE = "\n\n> ⚠️ 上面的回答未通过出处核对，已撤回：\n\n"
SOURCES_HEADER = "\n\n---\n**参考来源**\n"

_NOTICE_PATTERN = re.compile(r"\n*> ⚠️ 上面的(?:草稿|回答)未通过出处核对[^\n]*\n+")
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


# ================================================================ 请求模型


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    # content 可能是字符串，也可能是多模态分段 [{"type":"text","text":...}, ...]
    content: Union[str, List[Any], None] = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # temperature 等参数照单全收但不使用

    model: str = Field(default_factory=lambda: settings.OPENAI_COMPAT_MODEL)
    messages: List[ChatMessage]
    stream: bool = False


# ================================================================ 消息解析


def message_text(content: Union[str, List[Any], None]) -> str:
    """把 OpenAI 消息内容统一成纯文本（多模态里只取文字部分）。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text", "")))
        elif isinstance(part, str):
            parts.append(part)
    return "\n".join(parts)


def clean_assistant_text(text: str) -> str:
    """去掉本服务在回答里附加的内容，只留正文，作为下一轮的历史。

    去掉：<think> 块、参考来源列表、被撤回的草稿（只保留最后一条分隔说明之后的部分）。
    """
    text = _THINK_PATTERN.sub("", text or "")
    head, sep, _ = text.partition(SOURCES_HEADER.strip("\n"))
    if sep:
        text = head
    pieces = _NOTICE_PATTERN.split(text)
    return pieces[-1].strip()


def parse_messages(messages: List[ChatMessage]) -> Tuple[str, List[Tuple[str, str]]]:
    """返回 (本轮问题, 历史问答对)。没有 user 消息时问题为空字符串。"""
    question_index = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            question_index = index
            break
    if question_index is None:
        return "", []

    question = message_text(messages[question_index].content).strip()

    history: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
    for message in messages[:question_index]:
        if message.role == "user":
            pending_user = message_text(message.content).strip()
        elif message.role == "assistant" and pending_user is not None:
            answer = clean_assistant_text(message_text(message.content))
            if pending_user and answer:
                history.append((pending_user, answer))
            pending_user = None
    return question, history


def format_sources(sources: List[dict], limit: int = 6) -> str:
    """把来源列表渲染成 markdown，追加在回答末尾。"""
    lines = []
    for index, item in enumerate(sources[:limit], 1):
        if item.get("source_type") == "web":
            title = item.get("filename") or item.get("source") or "网页"
            url = item.get("source", "")
            lines.append(f"{index}. [{title}]({url})" if url else f"{index}. {title}")
        else:
            name = item.get("filename") or item.get("source") or "未知文件"
            page = item.get("page")
            lines.append(f"{index}. {name}" + (f"，第 {page} 页" if page else ""))
    return SOURCES_HEADER + "\n".join(lines) if lines else ""


# ================================================================ 路由


def model_ids() -> List[str]:
    base = settings.OPENAI_COMPAT_MODEL
    return [base, f"{base}-local"]


def resolve_model(model: str) -> bool:
    """返回本次是否允许联网；未知模型名报 404（与 OpenAI 行为一致）。"""
    if model == settings.OPENAI_COMPAT_MODEL:
        return True
    if model == f"{settings.OPENAI_COMPAT_MODEL}-local":
        return False
    raise HTTPException(status_code=404, detail=f"模型不存在：{model}，可用：{model_ids()}")


def _chunk(completion_id: str, created: int, model: str, delta: dict, finish: Optional[str] = None) -> str:
    body = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(body, ensure_ascii=False)}\n\n"


def build_router(get_rag: Callable[[], Any]) -> APIRouter:
    """构建 /v1 路由。get_rag 返回当前的 RAGChain（未初始化时返回 None）。

    以函数参数注入 RAG 链（而不是直接 import 全局单例），测试时可以传入假对象。
    """
    router = APIRouter(prefix="/v1", tags=["OpenAI 兼容接口"], dependencies=[Depends(require_api_token)])

    @router.get("/models")
    async def list_models():
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {"id": model_id, "object": "model", "created": created, "owned_by": "rag-agent"}
                for model_id in model_ids()
            ],
        }

    @router.post("/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        x_session_id: Optional[str] = Header(None, pattern=r"^[A-Za-z0-9_.:\-]{1,128}$"),
    ):
        rag = get_rag()
        if rag is None:
            raise HTTPException(status_code=503, detail="RAG 链未初始化")

        allow_web = resolve_model(request.model)
        question, history = parse_messages(request.messages)
        if not question:
            raise HTTPException(status_code=400, detail="messages 里没有 user 消息")

        kwargs: dict = {"allow_web": allow_web}
        if x_session_id:
            kwargs["session_id"] = x_session_id
        else:
            kwargs["history"] = history

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if not request.stream:
            try:
                result = await rag.aanswer(question, **kwargs)
            except Exception as error:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"RAG 链执行失败：{error}")
            content = result["answer"]
            if settings.OPENAI_COMPAT_APPEND_SOURCES:
                content += format_sources(result.get("sources") or [])
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
                # 一次问答内部有多次 LLM 调用，这里不做 token 统计
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }

        async def event_stream():
            yield _chunk(completion_id, created, request.model, {"role": "assistant", "content": ""})
            try:
                async for event in rag.astream_events(question, **kwargs):
                    kind = event["event"]
                    if kind == "step" and settings.OPENAI_COMPAT_SHOW_PROGRESS:
                        delta = {"reasoning_content": f"✓ {event['label']}\n"}
                        yield _chunk(completion_id, created, request.model, delta)
                    elif kind == "token":
                        yield _chunk(completion_id, created, request.model, {"content": event["text"]})
                    elif kind == "reset":
                        notice = REGENERATE_NOTICE if event["reason"] == "regenerate" else REPLACED_NOTICE
                        yield _chunk(completion_id, created, request.model, {"content": notice})
                    elif kind == "done" and settings.OPENAI_COMPAT_APPEND_SOURCES:
                        sources = format_sources(event["result"].get("sources") or [])
                        if sources:
                            yield _chunk(completion_id, created, request.model, {"content": sources})
            except Exception as error:  # noqa: BLE001
                # 响应头已经发出，不能再改状态码；把错误作为内容告诉用户
                message = f"\n\n> ❌ 处理失败：{type(error).__name__}: {error}"
                yield _chunk(completion_id, created, request.model, {"content": message})
            yield _chunk(completion_id, created, request.model, {}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)

    return router
