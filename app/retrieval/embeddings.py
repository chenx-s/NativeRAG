"""本地 Embedding 模型（HuggingFace，离线运行）。"""

from __future__ import annotations

import os
from functools import lru_cache

from app.config import settings

# 下载失败时异常链里常见的关键字 -> 判定为网络问题，给出可操作的中文提示
_NETWORK_HINTS = (
    "407", "proxy", "connecterror", "connection", "timed out", "timeout",
    "localentrynotfound", "offlinemode", "name resolution", "ssl", "max retries",
)


def _detect_device() -> str:
    """自动选择推理设备：有 CUDA 用 GPU，否则 CPU。

    注意：原代码写的是 {"device": " cpu"}——引号里有空格，
    sentence-transformers 拿到 " cpu" 这种带空格的字符串会直接报错。
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _exception_text(error: BaseException) -> str:
    parts, current, seen = [], error, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


def _explain_download_failure(error: BaseException) -> None:
    """模型下载失败时，把底层的网络异常翻译成能照着做的中文提示并抛出。"""
    text = _exception_text(error)
    if not any(hint in text.lower() for hint in _NETWORK_HINTS):
        return

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY")
    proxy_line = f"当前代理：{proxy}" if proxy else "环境变量里没有设置代理（可能来自系统代理设置）"
    is_proxy_auth = "407" in text

    raise RuntimeError(
        f"Embedding 模型 {settings.EMBEDDING_MODEL} 加载失败：本机没有缓存，下载时网络出错。\n"
        + ("原因：代理服务器要求身份验证（HTTP 407），下载请求没有带用户名和密码。\n" if is_proxy_auth else "")
        + f"{proxy_line}；HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'https://huggingface.co（默认）')}\n\n"
        "任选一种解决办法：\n"
        "  1) 代理需要认证：把账号写进代理地址，例如在 .env 里加\n"
        "        HTTPS_PROXY=http://用户名:密码@代理地址:端口\n"
        "  2) 不需要走这个代理：在 .env 里加 NO_PROXY=huggingface.co,hf-mirror.com，\n"
        "     或在当前终端清掉代理：$env:HTTPS_PROXY=\"\"; $env:HTTP_PROXY=\"\"\n"
        "  3) 改用国内镜像：在 .env 里加 HF_ENDPOINT=https://hf-mirror.com（镜像同样会经过代理）\n"
        "  4) 离线使用：在能联网的机器上下载模型目录，拷到本机，把 EMBEDDING_MODEL 改成本地路径，\n"
        "     例如 EMBEDDING_MODEL=./models/bge-m3，并加 HF_HUB_OFFLINE=1\n"
        "  5) 暂时换回已缓存的旧模型：EMBEDDING_MODEL=BAAI/bge-small-en-v1.5\n\n"
        f"底层报错：{text[:500]}"
    ) from error


@lru_cache(maxsize=1)
def get_embeddings():
    """本地 HuggingFace Embedding（单例，避免重复加载模型权重）。

    normalize_embeddings=True：向量归一化后，内积就等于余弦相似度。
    首次使用会从 HuggingFace 下载模型（受 .env 里的 HF_ENDPOINT / HTTPS_PROXY / HF_HUB_OFFLINE 影响）。
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    try:
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": _detect_device()},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as error:
        _explain_download_failure(error)
        raise


def embedding_dim() -> int:
    """探测当前 Embedding 模型的向量维度（建表、维度检查用）。"""
    return len(get_embeddings().embed_query("dimension probe"))
