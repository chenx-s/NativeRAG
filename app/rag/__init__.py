"""Agentic RAG 核心：LangGraph 状态图及其对外封装。

包结构（按职责拆分，每个文件只做一件事）：

    chain.py      RAGChain —— 对外入口：记忆加载/写回、流式事件、结果整理
    graph.py      组装状态图（节点 + 边），节点中文名
    edges.py      条件路由（纯函数，决定下一步走哪条边）
    nodes/        节点实现：route / retrieval / web / generation
    state.py      GraphState —— 节点间共享的状态定义
    prompts.py    全部提示词模板与兜底文案
    helpers.py    解析 LLM 输出、拼上下文、联网判定等纯函数
    llm.py        LLM 与精排器的构建

使用：
    from app.rag import RAGChain
"""

from app.rag.chain import RAGChain
from app.rag.graph import STEP_LABELS, build_graph, required_recursion_limit

__all__ = ["RAGChain", "STEP_LABELS", "build_graph", "required_recursion_limit"]
