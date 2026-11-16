"""组装并编译 LangGraph 状态图。

                      route_question（补全追问 + 判断是否需要实时信息）
                        |                        |
                   (local)                    (web)
                        v                        |
    transform_query -> retrieve -> rerank_documents -> grade_document
           ^                                        |        |
           |                    (无相关文档且可重试)  |        | (无相关文档、改写额度用尽)
      rewrite_query <-------------------------------+        v
           |                                             web_search  <---+
           +--(改写无效)--------------------------------------^          |
                                                             |          |
                     generate <----(有相关文档 / 联网结果)-----+----------+
                       ^   |
                       |   v
                       |  check_groundedness
                       +---+ (未通过 + 还有额度 -> 严格模式重生成)
                           |
                           +--(通过 / 额度用尽)--> END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.config import settings
from app.rag.edges import (
    decide_after_grading,
    decide_after_groundedness,
    decide_after_rewrite,
    decide_after_route,
)
from app.rag.nodes import (
    check_groundedness,
    generate,
    grade_documents,
    rerank_documents,
    retrieve,
    rewrite_query,
    route_question,
    transform_query,
    web_search,
)
from app.rag.state import GraphState

# 节点名 -> 中文展示名（流式进度事件、CLI 轨迹共用）
STEP_LABELS = {
    "route_question": "理解问题",
    "transform_query": "查询变换",
    "retrieve": "检索知识库",
    "rerank_documents": "精排",
    "grade_document": "相关性评估",
    "rewrite_query": "改写查询",
    "web_search": "联网搜索",
    "generate": "生成回答",
    "check_groundedness": "核对出处",
}


def required_recursion_limit() -> int:
    """按当前配置估算图最多会执行多少步，用来兜底 recursion_limit。

    route(1) + 每轮本地检索 transform/retrieve/rerank/grade/rewrite(5) × (MAX_RETRIES+1)
    + web_search(1) + (generate + check) × (MAX_GROUNDEDNESS_RETRIES+1)，再留一点余量。
    """
    local_rounds = 5 * (max(0, settings.MAX_RETRIES) + 1)
    generation = 2 * (max(0, settings.MAX_GROUNDEDNESS_RETRIES) + 1)
    return 1 + local_rounds + 1 + generation + 5


def build_graph():
    """构建并编译 Agentic RAG 状态图。"""
    workflow = StateGraph(GraphState)

    workflow.add_node("route_question", route_question)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("grade_document", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("check_groundedness", check_groundedness)

    workflow.set_entry_point("route_question")

    # 入口路由：需要实时信息 -> 直接联网；否则先查本地
    workflow.add_conditional_edges(
        "route_question",
        decide_after_route,
        {"local": "transform_query", "web": "web_search"},
    )

    # 主链路：查询变换 -> 混合检索 -> 精排 -> 相关度评分
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("retrieve", "rerank_documents")
    workflow.add_edge("rerank_documents", "grade_document")

    workflow.add_conditional_edges(
        "grade_document",
        decide_after_grading,
        {"rewrite": "rewrite_query", "web": "web_search", "generate": "generate"},
    )

    # 改写成功时回到 transform_query（用新问题重新生成多条查询）；
    # 改写无变化时联网（可用时）或直接返回兜底答案。
    workflow.add_conditional_edges(
        "rewrite_query",
        decide_after_rewrite,
        {"transform": "transform_query", "web": "web_search", "generate": "generate"},
    )

    # 联网结果不再做 LLM 逐条评分（Tavily 已按相关度排序），直接生成，
    # 生成后仍然要过忠实性检查
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "check_groundedness")

    # 忠实性检查没通过时，回到 generate 用严格模式重生成一次（Self-RAG 的反思闭环）；
    # 通过或额度用尽则结束。
    workflow.add_conditional_edges(
        "check_groundedness",
        decide_after_groundedness,
        {"regenerate": "generate", "end": END},
    )

    return workflow.compile()
