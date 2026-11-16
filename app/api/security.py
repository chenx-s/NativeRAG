"""接口鉴权：可选的 Bearer Token。"""

from __future__ import annotations

import hmac
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from app.config import settings


async def require_api_token(authorization: Optional[str] = Header(None)) -> None:
    """配置了 API_TOKEN 时，要求请求头 `Authorization: Bearer <API_TOKEN>`。

    用 hmac.compare_digest 做常数时间比较，避免通过响应时间逐字符猜出 token。
    未配置 API_TOKEN 时不做校验（仅适合本机开发）。
    """
    expected = settings.API_TOKEN
    if not expected:
        return

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(token.strip(), expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少或错误的 API Token（请求头 Authorization: Bearer <API_TOKEN>）",
            headers={"WWW-Authenticate": "Bearer"},
        )


# 挂在路由上：APIRouter(dependencies=AUTH) 或 @router.get(..., dependencies=AUTH)
AUTH = [Depends(require_api_token)]
