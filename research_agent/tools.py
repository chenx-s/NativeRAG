"""[已废弃] 旧版 ReAct Agent 的工具定义，可以直接删除本文件。

联网搜索已并入 app/rag_chain.py 的 LangGraph 状态图（web_search 节点），
Tavily 调用与故障分类见 app/web_search.py。
"""

raise ImportError(
    "research_agent.tools 已废弃：联网搜索现在是状态图节点，见 app/web_search.py 与 app/rag_chain.py"
)
