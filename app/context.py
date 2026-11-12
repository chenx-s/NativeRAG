"""上下文工程（Context Engineering）：决定每次调用 LLM 时，上下文里放什么、放多少、怎么放。

================================ 为什么要单独一层 ================================

一次问答会往提示词里塞三类东西：用户问题、检索资料、对话历史。
不加管理时会出现几种典型问题：
    · 历史越聊越长，token 成本线性增长，最终撑爆上下文窗口；
    · 本地片段 + 网页片段一多，关键信息被稀释（"lost in the middle"）；
    · 历史和资料混在一起，模型把上一轮的说法当成"资料"复述，忠实性检查也分不清依据。

本模块对应上下文工程的四个动作：

    写入（Write）    长期记忆用"滚动摘要 + 最近 N 轮原文"保存，见 app/memory.py
    选择（Select）   只取最近 HISTORY_TURNS 轮 + 摘要；资料按相关度取前若干条
    压缩（Compress） 历史回答截断、资料按字符预算裁剪，预算内优先保留最新/最相关的
    隔离（Isolate）  历史放在独立的【对话背景】分区，提示词里明确它只用于理解指代、
                     保持连贯，新的事实必须来自【资料】分区

预算用字符数而不是 token 数：不依赖分词器，中英文混排时也足够稳定，
对"别撑爆窗口、别稀释重点"这个目标来说精度已经够用。
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from langchain_core.documents import Document

NO_CONVERSATION = "（无）"


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def render_conversation(
    summary: str,
    turns: Sequence[Tuple[str, str]],
    *,
    max_turns: int,
    max_chars: int,
    answer_chars: int,
) -> str:
    """把"滚动摘要 + 最近若干轮"渲染成一个受预算约束的对话背景块。

    选择策略：
      1. 摘要最多占一半预算（它浓缩了更早的全部轮次，信息密度最高）；
      2. 剩余预算从**最新一轮往前**装，装不下就停 —— 越近的轮次对理解追问越关键；
      3. 每条回答先截断到 answer_chars，避免一条长答案吃掉全部预算。
    返回空字符串表示没有任何对话背景。
    """
    if max_chars <= 0:
        return ""

    budget = max_chars
    parts: List[str] = []

    summary = (summary or "").strip()
    if summary:
        summary_block = "【较早对话摘要】\n" + _truncate(summary, max(50, max_chars // 2))
        parts.append(summary_block)
        budget -= len(summary_block)

    recent = list(turns or [])[-max_turns:] if max_turns > 0 else []
    selected: List[str] = []
    for question, answer in reversed(recent):
        block = f"用户：{_truncate(question, 300)}\n助手：{_truncate(answer, answer_chars)}"
        if len(block) > budget:
            # 最新一轮至少保留一部分，否则追问补全会完全失去依据
            if not selected and budget > 40:
                selected.append(_truncate(block, budget))
            break
        selected.append(block)
        budget -= len(block) + 1

    if selected:
        parts.append("【最近对话】\n" + "\n".join(reversed(selected)))

    return "\n\n".join(parts)


def fit_documents(documents: Sequence[Document], max_chars: int) -> List[Document]:
    """按字符预算裁剪检索资料（资料已按相关度从高到低排序）。

    - 预算内的文档原样保留；
    - 第一条放不下的文档：剩余预算足够（≥200 字）或者一条都还没放时，截断后保留；
    - 之后的全部丢弃。
    截断时复制 Document，不修改原对象（它可能来自缓存，也会被用作返回给前端的 sources）。
    """
    if max_chars <= 0:
        return list(documents)

    kept: List[Document] = []
    used = 0
    for document in documents:
        length = len(document.page_content)
        if used + length <= max_chars:
            kept.append(document)
            used += length
            continue

        remaining = max_chars - used
        if remaining >= 200 or (not kept and remaining > 0):
            kept.append(
                Document(
                    page_content=document.page_content[:remaining] + " …（已截断）",
                    metadata=dict(document.metadata),
                )
            )
        break

    return kept


def conversation_or_placeholder(conversation: str) -> str:
    """提示词里统一用"（无）"表示没有对话背景，避免出现空分区让模型困惑。"""
    return conversation.strip() or NO_CONVERSATION
