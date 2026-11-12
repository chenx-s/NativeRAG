"""研究助手 —— 本地知识库 + 联网搜索的命令行会话层。

检索、联网、生成全部由 app/rag_chain.py 的 LangGraph 状态图完成，
本包只负责多轮会话与命令行交互：

    settings.py    CLI 会话层配置
    agent.py       ResearchAgent 会话封装 + 命令行入口

运行：
    python -m research_agent.agent
"""

from research_agent.agent import ResearchAgent

__all__ = ["ResearchAgent"]
