"""联网搜索节点（CRAG 的纠错检索）。

为什么联网是图里的节点，而不是交给 Agent 选择的工具？
    "什么时候联网"只有两种确定的情况：问题依赖实时信息（入口路由判断一次），
    或本地检索评分后没有相关文档（纯规则判断）。用条件边表达，路径确定、
    LLM 调用次数有上限、可单测；不需要再套一层 ReAct Agent 让 LLM 每轮推理。
"""

from __future__ import annotations

from app import web_search as web_api
from app.rag.state import GraphState


async def web_search(state: GraphState) -> dict:
    """联网搜索。

    两个入口：route_question 判定需要实时信息；或本地检索没有相关文档。
    搜索失败不抛异常 —— 记录原因、返回空文档，由 generate 给出兜底答案。
    """
    question = state["question"]
    print(f"--->[节点]web_search: {question}")

    try:
        documents = await web_api.asearch_web(question)
        error = ""
    except web_api.WebSearchError as exc:
        print(f"    [warn] 联网搜索失败：{exc}")
        documents, error = [], str(exc)

    print(f"   联网结果：{len(documents)} 条")
    return {
        "documents": documents,
        "source_type": "web",
        "web_searched": True,
        "web_error": error,
        # 联网结果不经过精排，避免沿用本地检索的标记造成误读
        "reranked": False,
    }
