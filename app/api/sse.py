"""Server-Sent Events（SSE）工具。"""

from __future__ import annotations

import json
from typing import Any, Optional


def sse(data: Any, event: Optional[str] = None) -> str:
    """格式化一条 SSE 消息。data 为 dict/list 时序列化为 JSON（ensure_ascii=False 保留中文）。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    lines = [f"event: {event}"] if event else []
    # SSE 规范：data 里的换行要拆成多行 data:
    lines.extend(f"data: {line}" for line in payload.split("\n"))
    return "\n".join(lines) + "\n\n"


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # 关闭 Nginx 等反向代理的响应缓冲，否则流式输出会被攒成一大块才下发
    "X-Accel-Buffering": "no",
}
