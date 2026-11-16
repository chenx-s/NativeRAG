"""入口节点：补全追问 + 决定先查本地还是直接联网。"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser

from app import vector_store
from app.config import settings
from app.context import conversation_or_placeholder
from app.rag import llm
from app.rag.helpers import parse_route, render_conversation_for, web_allowed
from app.rag.prompts import ROUTE_PROMPT
from app.rag.state import GraphState


async def route_question(state: GraphState) -> dict:
    """入口节点：补全追问 + 决定先查本地还是直接联网。

    只在真正需要时才调 LLM：
      · 有对话背景                          -> 需要把追问补全成独立问题
      · 联网可用、知识库非空、WEB_ROUTING=llm -> 需要判断是否依赖实时信息
    两者都不满足时零 LLM 调用，直接走本地检索。
    知识库为空且联网可用时，直接联网（本地查了也是白查）。
    拿不准时一律走本地：本地检索失败后图会自动降级为联网，不会漏答。
    """
    question = state["question"]
    conversation = render_conversation_for(state)
    web_ready = web_allowed(state)

    kb_empty = False
    if web_ready:
        try:
            kb_empty = (await vector_store.acount()) == 0
        except Exception:  # noqa: BLE001
            kb_empty = False

    need_routing = web_ready and not kb_empty and settings.WEB_ROUTING == "llm"
    route = "local"
    standalone = question

    if need_routing or conversation:
        try:
            chain = ROUTE_PROMPT | llm.get_llm() | StrOutputParser()
            raw = await chain.ainvoke({
                "question": question,
                "history": conversation_or_placeholder(conversation),
            })
            parsed_route, parsed_question = parse_route(raw)
            if parsed_question:
                standalone = parsed_question
            if need_routing and parsed_route:
                route = parsed_route
        except Exception as error:  # noqa: BLE001
            print(f"    [warn] 路由判断失败，默认先查本地知识库：{error}")

    if web_ready and kb_empty:
        route = "web"

    print(f"--->[节点]route_question: route={route}" + (
        f"，补全后问题：{standalone}" if standalone != question else ""
    ))
    return {"question": standalone, "standalone_question": standalone, "route": route}
