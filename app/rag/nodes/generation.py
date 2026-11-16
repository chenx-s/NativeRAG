"""生成与自检：generate（流式生成答案）-> check_groundedness（忠实性检查）。"""

from __future__ import annotations

from typing import List

from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.context import conversation_or_placeholder, fit_documents
from app.rag import llm
from app.rag.helpers import format_context, parse_yes_no, render_conversation_for, stream_writer
from app.rag.prompts import (
    CONVERSATION_RULES,
    FALLBACK_ANSWER,
    GENERATE_PROMPT,
    GENERATE_STRICT_PROMPT,
    GROUNDEDNESS_PROMPT,
    WEB_FALLBACK_ANSWER,
    WEB_GENERATE_PROMPT,
)
from app.rag.state import GraphState


async def generate(state: GraphState) -> dict:
    """根据相关文档生成答案；没有相关文档时直接返回兜底答案。

    提示词三选一：本地资料 / 联网结果 / 严格模式（上一版没通过忠实性检查）。
    答案通过 astream 逐段生成，并经 LangGraph 的 custom 流推给调用方（见 RAGChain.astream_events）。
    """
    if not state["documents"]:
        print("--->[节点]generate: 无相关文档，返回兜底答案")
        fallback = WEB_FALLBACK_ANSWER if state.get("web_searched") else FALLBACK_ANSWER
        if state.get("web_error"):
            fallback += f"\n（联网搜索不可用：{state['web_error']}）"
        return {"generation": fallback, "skip_check": True}

    # strict_grounding 为 True 表示：上一版答案没通过忠实性检查，
    # 这次改用严格提示词重生成（只准照抄上下文，不准补充背景知识）。
    strict = bool(state.get("strict_grounding"))
    is_web = state.get("source_type") == "web"
    if strict:
        prompt = GENERATE_STRICT_PROMPT
    else:
        prompt = WEB_GENERATE_PROMPT if is_web else GENERATE_PROMPT
    mode = "严格模式重生成" if strict else "正常生成"

    print(
        f"--->[节点]generate（{mode}）: 基于{len(state['documents'])}个"
        f"{'网页' if is_web else '文档'}片段生成"
    )

    # 上下文工程 —— 压缩：资料按字符预算裁剪（已按相关度排序，从尾部裁）；
    # 忠实性检查用同一个函数裁剪，保证"生成看到的"和"核对依据的"是同一份资料
    documents = fit_documents(state["documents"], settings.CONTEXT_MAX_CHARS)
    conversation = render_conversation_for(state)

    inputs = {
        "question": state["question"],
        "context": format_context(documents),
        "conversation": conversation_or_placeholder(conversation),
        # 隔离规则只在确实有对话背景时注入
        "conversation_rules": CONVERSATION_RULES if conversation else "",
    }

    # 非流式调用时 writer 是空操作，行为与 ainvoke 完全一致。
    # generation_start 带上第几次生成：严格模式重生成时，调用方据此丢弃上一版草稿。
    writer = stream_writer()
    writer({"type": "generation_start", "attempt": state.get("groundedness_retries", 0), "strict": strict})

    try:
        generate_chain = prompt | llm.get_llm() | StrOutputParser()
        pieces: List[str] = []
        async for piece in generate_chain.astream(inputs):
            if piece:
                pieces.append(piece)
                writer({"type": "token", "text": piece})
        answer = "".join(pieces)
    except Exception as error:
        print(f"    [warn]生成失败：{error}")
        return {"generation": FALLBACK_ANSWER, "skip_check": True, "regenerate": False}

    return {
        "generation": answer.strip(),
        "skip_check": False,
        # 必须显式复位，否则上一轮的 True 会一直留在状态里
        "regenerate": False,
    }


async def check_groundedness(state: GraphState) -> dict:
    """检查生成答案是否由检索文档支撑（防幻觉自检）。

    失败后的处理由 GROUNDEDNESS_ACTION 控制（见 app/config.py）：

        retry  失败 -> 置 strict_grounding=True 走严格模式重生成一次
                      -> 仍失败才替换为拒答【默认】
        warn   失败 -> 只打印警告，保留原答案
        off    完全不检查

    为什么默认不是"直接拒答"：
        这个判断本身并不总是可靠，最典型的是跨语言数字换算 ——
        事实写 "7 billion to 70 billion"，中文答"70 亿到 700 亿"是完全正确的，
        但模型自己会算错（把 70 billion 当成 70 亿），从而把正确答案判成幻觉。
        所以先给它一次"重说一遍"的机会：严格模式重生成出来的答案更贴原文，
        通常能通过；还不通过才是真有问题。
    """
    # ---- 分支 1：兜底答案不需要检查 ----
    if state.get("skip_check"):
        print("--->[节点]check_groundedness: 跳过兜底答案")
        return {"generation": state["generation"], "regenerate": False, "groundedness_verdict": "skipped"}

    # ---- 分支 2：检查被关闭 ----
    if settings.GROUNDEDNESS_ACTION == "off":
        print("--->[节点]check_groundedness: 已关闭（GROUNDEDNESS_ACTION=off）")
        return {"generation": state["generation"], "regenerate": False, "groundedness_verdict": "off"}

    print("--->[节点]check_groundedness")

    documents = fit_documents(state["documents"], settings.CONTEXT_MAX_CHARS)
    document_text = "\n\n".join(document.page_content for document in documents)

    # 答案允许复述此前回答（用户明确问到时），这部分的依据是对话背景，
    # 一并交给评分器，避免把"总结你刚才说的"误判成幻觉
    conversation = render_conversation_for(state)
    if conversation:
        document_text += "\n\n【此前对话（助手之前已给出的回答）】\n" + conversation

    try:
        grader = GROUNDEDNESS_PROMPT | llm.get_llm() | StrOutputParser()
        score_text = await grader.ainvoke({"documents": document_text, "generation": state["generation"]})
    except Exception as error:
        # 检查服务本身出问题时，保留原答案 —— 不能因为"校验跑挂了"就不给用户答案
        print(f"    [warn]忠实性检查调用失败，保留原答案：{error}")
        return {"generation": state["generation"], "regenerate": False, "groundedness_verdict": "unknown"}

    # 把评分器的原始输出打出来：判 no 时不看原始输出，分不清是"真的在编"还是"评分器误判"。
    passed = parse_yes_no(score_text, default=True)  # 解析失败时保守放行
    print(f"   评分器原始输出：{score_text.strip()[:60]!r} -> {'通过' if passed else '未通过'}")

    # ---- 分支 3：通过 ----
    if passed:
        return {"generation": state["generation"], "regenerate": False, "groundedness_verdict": "yes"}

    # ---- 分支 4：未通过，且还有重生成额度 -> 走严格模式重来一次 ----
    retries = state.get("groundedness_retries", 0)
    if settings.GROUNDEDNESS_ACTION == "retry" and retries < settings.MAX_GROUNDEDNESS_RETRIES:
        print(f"   [warn]答案未通过忠实性检查，第 {retries + 1} 次改用严格模式重新生成")
        return {
            "groundedness_retries": retries + 1,
            "strict_grounding": True,
            "regenerate": True,
            "groundedness_verdict": "no",
        }

    # ---- 分支 5：未通过，且没有重生成额度了 ----
    if settings.GROUNDEDNESS_ACTION == "warn":
        print("   [warn]答案未通过忠实性检查，但 GROUNDEDNESS_ACTION=warn，保留原答案")
        return {"generation": state["generation"], "regenerate": False, "groundedness_verdict": "no"}

    print("   [warn]答案未通过忠实性检查，替换为拒答")
    return {
        "generation": WEB_FALLBACK_ANSWER if state.get("source_type") == "web" else FALLBACK_ANSWER,
        "regenerate": False,
        "groundedness_verdict": "no",
    }
