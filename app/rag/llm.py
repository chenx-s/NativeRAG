"""LLM 与精排器的构建。

整个 RAG 链路里所有 LLM 调用都通过 get_llm() 创建，便于统一配置（模型、温度、缓存）
和在测试中统一替换（patch("app.rag.llm.get_llm", ...)）。
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from app.config import settings
from app.llm_cache import get_llm_cache


def get_llm(**kwargs) -> ChatOpenAI:
    """创建大模型客户端（智谱 GLM，走 OpenAI 兼容协议）。

    cache 挂的是 Redis LLM 缓存：提示词和模型配置完全相同的调用直接命中，
    Redis 不可用时自动旁路。thinking 关掉是因为 GLM 的思考模式会拖慢
    评分/路由这类只需要一个词的调用，且思考内容会混进输出。
    """
    return ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        cache=get_llm_cache(),
        extra_body={"thinking": {"type": "disabled"}},
        **kwargs,
    )


def get_reranker():
    """构建精排器（Cross-Encoder Reranker）。

    精排和向量检索的本质区别：
        向量检索（双塔/Bi-Encoder）把问题和文档**分别**编码成向量再算相似度，
        快，但两者从未"见面"；
        精排（交叉编码器/Cross-Encoder）把 [问题, 文档] **拼在一起**送进模型，
        每个词都能和对方的词做交互注意力，准确率高得多，代价是只能逐条打分。
        所以标准打法是：向量检索负责"从百万里捞出一百"，精排负责"从一百里挑出五"。

    未启用 / 没配 Key / 初始化失败时返回 None，调用方据此跳过精排。
    """
    if not settings.RERANK_ENABLED:
        return None

    if settings.RERANK_PROVIDER != "cohere":
        print(f"    [warn] 未知的 RERANK_PROVIDER={settings.RERANK_PROVIDER}，跳过精排")
        return None

    if not settings.COHERE_API_KEY:
        print(
            "    [warn] 未配置 COHERE_API_KEY，跳过精排。"
            "申请地址：https://dashboard.cohere.com/api-keys（Rerank 有免费额度）"
        )
        return None

    # langchain-cohere 各版本读取 Key 的方式不一致，先把环境变量兜住
    os.environ.setdefault("COHERE_API_KEY", settings.COHERE_API_KEY)

    try:
        try:
            from langchain_cohere import CohereRerank
        except ImportError:
            from langchain_cohere.rerank import CohereRerank  # 兜底旧路径

        try:
            return CohereRerank(
                model=settings.COHERE_RERANK_MODEL,
                top_n=settings.RERANK_TOP_N,
            )
        except TypeError:
            # 老版本需要显式传 cohere_api_key
            return CohereRerank(
                model=settings.COHERE_RERANK_MODEL,
                top_n=settings.RERANK_TOP_N,
                cohere_api_key=settings.COHERE_API_KEY,
            )
    except Exception as error:
        print(f"    [warn] 精排器初始化失败，跳过精排：{error}")
        return None
