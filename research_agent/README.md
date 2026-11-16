# 研究助手（命令行会话层）

> `python -m research_agent.agent` 启动的命令行多轮对话。它与 HTTP 接口
> **共用同一张 LangGraph 状态图和同一个记忆库**，本包只负责会话管理与命令行交互。

## 快速开始

1. 项目根目录 `.env` 填好 `LLM_API_KEY`（必填），`TAVILY_API_KEY` 可选（不填则不联网）。
2. 运行：

```bash
python -m research_agent.agent
```

启动时会预热本地知识库（加载 Embedding 模型 + 打开 Milvus），首次较慢。
不想等可在 `.env` 设 `AGENT_WARMUP_ON_START=false`。

## 可用命令

| 命令 | 作用 |
|---|---|
| `/history` | 查看当前会话的记忆（摘要 + 最近几轮） |
| `/forget` | 清除当前会话的记忆 |
| `/new` | 开启一个新会话 |
| `/session ID` | 切换到指定会话 |
| `/warmup` | 预热本地知识库 |
| `/quiet` | 开关执行路径的显示 |
| `/quit` | 退出 |

默认会话 ID 固定（`AGENT_SESSION_ID`），所以**重启 CLI 会自动接上上次对话**（记忆持久化在 `MEMORY_BACKEND` 里）。

## 它长什么样

```
你 > SAR图像相干斑抑制有哪些深度学习方法？

🤔 思考中 ...
  🧭 路径：理解问题 → 查询变换 → 检索知识库 → 精排 → 相关性评估 → 生成回答 → 核对出处
  📚 来源（本地知识库）：
     [1] 基于深度学习的SAR图像相干斑抑制网络研究_姚同钰.pdf 第 3 页
     [2] 复杂环境下雷达图像相干斑噪声抑制...pdf 第 12 页

────────────────────────────────────────────────────────────────
助手 >
主要有三类方法：……

────────────────────────────────────────────────────────────────
```

`🧭 路径` 就是状态图的节点轨迹；`📚 来源` 是最终答案依据的文档片段。

## 架构

```
用户提问
   │
   ▼
ResearchAgent.achat()  ──►  RAGChain.aanswer(question, session_id=thread_id)
   │                               │
   │                     ┌─────────┴──────────┐
   │                     ▼                    ▼
   │              app/rag/（状态图）      app/memory.py（记忆）
   │              route → 检索 → 精排     滚动摘要 + 最近 N 轮
   │              → 评分 → (联网) → 生成   SQLite / Redis / 内存
   │              → 忠实性自检
   │                               │
   └── 打印轨迹 + 答案 + 来源 ◄─────┘
```

- **thread_id 就是记忆的 session_id**：切换会话只是换一个 thread_id。
- **懒加载 RAGChain**：本地知识库预热失败时降级为"只联网"，不影响启动。
- 联网搜索不是 Agent 自由选择的工具，而是状态图里的一个纠错节点（Corrective RAG），
  时机由入口路由和本地检索结果决定。

## 文件结构

```
research_agent/
├── __init__.py     导出 ResearchAgent
├── settings.py     CLI 会话层配置（默认会话 ID、是否预热）
└── agent.py        ResearchAgent 会话封装 + 命令行入口
```

所有检索、联网、生成、记忆逻辑都在 `app/` 里，本包只做会话与交互。
