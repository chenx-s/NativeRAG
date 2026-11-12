"""研究助手 CLI 的配置。

联网搜索（TAVILY_API_KEY、WEB_SEARCH_* 等）和大模型配置已统一收归 app/config.py：
联网现在是状态图里的一个节点，HTTP 服务和 CLI 共用同一套参数。
这里只保留命令行会话层自己的开关。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class AgentSettings:
    """CLI 会话层配置。读取方式：`from research_agent.settings import agent_settings`"""

    # 启动时是否预热本地知识库（首次加载 BGE 模型要 30+ 秒）。
    # 只想用联网搜索时设成 false，可以秒启动。
    WARMUP_ON_START: bool = os.getenv(
        "AGENT_WARMUP_ON_START", "true"
    ).strip().lower() in ("true", "1", "yes", "on")

    # CLI 默认会话 ID。固定 ID 意味着重启 CLI 会自动接上上次的对话；
    # 记忆本身由 app/memory.py 持久化（窗口、压缩、TTL 见 app/config.py 的 MEMORY_*）
    DEFAULT_SESSION_ID: str = os.getenv("AGENT_SESSION_ID", "research-default")


agent_settings = AgentSettings()
