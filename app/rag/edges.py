"""条件路由（条件边）：每个函数读取状态，返回下一步走哪条边。

全部是纯函数、不调用 LLM —— 这正是"确定性 Agentic 流程"的含义：
LLM 只负责判断（评分、路由），走哪条路由代码说了算，可以单测、可以穷举。
"""

from __future__ import annotations

from app.rag.helpers import can_web_search, local_rewrite_budget
from app.rag.state import GraphState


def decide_after_route(state: GraphState) -> str:
    """入口路由：需要实时信息 -> 直接联网；否则先查本地。"""
    return "web" if state.get("route") == "web" and can_web_search(state) else "local"


def decide_after_grading(state: GraphState) -> str:
    """文档评分完成后的条件路由（CRAG）。

    1. 有相关文档：进入答案生成。
    2. 无相关文档，还有本地改写额度：改写查询后重新检索。
    3. 无相关文档，改写额度用尽，联网可用：联网搜索纠错。
    4. 都不行：生成兜底答案。
    """
    if state["documents"]:
        return "generate"

    if state.get("retries", 0) < local_rewrite_budget(state):
        return "rewrite"

    if can_web_search(state):
        print("    本地知识库没有相关文档，转为联网搜索")
        return "web"

    print("    [warn]已达到最大查询改写次数")
    return "generate"


def decide_after_rewrite(state: GraphState) -> str:
    """查询改写完成后的条件路由。

    改写生效 -> 回到 transform_query，用新问题重新做查询变换 + 混合检索。
    改写无效 -> 继续检索相同问题不会有新结果：能联网就联网，否则交给 generate 兜底。
    """
    if state.get("rewrite_unchanged"):
        return "web" if can_web_search(state) else "generate"
    return "transform"


def decide_after_groundedness(state: GraphState) -> str:
    """忠实性检查后的条件路由。

    检查未通过、且还有重生成额度时 -> 回到 generate 用严格模式重写一遍。
    其余情况（通过 / 已用尽额度 / 被关闭） -> 结束。
    """
    return "regenerate" if state.get("regenerate") else "end"
