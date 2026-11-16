# RAG_Agent

一个面向文档问答的 **Agentic RAG 服务**：基于 FastAPI 提供 HTTP 接口，用 LangGraph 编排问答流程，用 LangChain 对接模型与检索组件。

一次提问会经过下面这条完整链路：

1. 先判断问题需要查本地知识库还是需要实时信息；
2. 在本地知识库上做混合检索和精排；
3. 本地找不到相关内容时，自动转为联网搜索；
4. 生成答案并做忠实性自检，防止模型编造；
5. 按会话记住上下文，多轮追问能被正确理解，记忆持久化保存，服务重启后不丢。

| 能力 | 实现 |
|---|---|
| 流程编排 | LangGraph `StateGraph`：9 个节点、4 组条件路由，全部为确定性路径 |
| 检索 | Milvus 稠密检索 ⊕ BM25 稀疏检索，RRF 融合；可选 multi-query / HyDE 查询变换 |
| 精排 | Cohere Rerank `rerank-v3.5`，未配置时自动跳过 |
| 联网 | Tavily 搜索，作为状态图中的纠错检索节点（Corrective RAG） |
| 防幻觉 | LLM 逐条评估文档相关性 + 答案忠实性自检，未通过时改用严格模式重新生成 |
| 记忆 | 滚动摘要加最近 N 轮原文，持久化到 SQLite 或 Redis |
| 上下文工程 | 对历史和资料做选择、压缩，并分区隔离，所有内容都在字符预算内 |
| 流式输出 | SSE 推送节点进度与逐段答案；草稿未通过核对时自动撤回并重写 |
| 前端接入 | OpenAI 兼容接口 `/v1/chat/completions`，可直接接 Open WebUI、Chatbox、LobeChat |
| 文档管理 | 按内容哈希去重、同名文件自动替换旧版本、按文档删除、后台入库队列、一键重建索引 |
| LLM | 智谱 GLM `glm-4.5-air`，走 OpenAI 兼容协议，可替换为任意兼容端点 |
| Embedding | 本地运行 `BAAI/bge-m3`（多语言，中文效果好），不依赖外部 API |
| 工程化 | 接入 Redis LLM 响应缓存（Redis 故障时自动绕过）、Prometheus 指标、Docker Compose、RAGAs 离线评估 |

---

## 目录

- [架构总览](#架构总览)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [使用方式](#使用方式)
- [接口一览](#接口一览)
- [核心设计](#核心设计)
- [配置速查](#配置速查)
- [可观测性](#可观测性)
- [测试与评估](#测试与评估)
- [故障排查](#故障排查)
- [注意事项](#注意事项)

---

## 架构总览

```text
                    ┌───────────────── FastAPI (/query) · CLI (research_agent) ─────────────────┐
                    │                                                                          │
  session_id ──►  加载记忆（SQLite / Redis）──► 上下文工程：选择 · 压缩 · 隔离                     │
                    │                                                                          │
                    ▼                                                                          │
  ┌──────────────────────────────── LangGraph 状态图 ───────────────────────────────┐          │
  │                                                                                 │          │
  │  route_question ──(需要实时信息 / 知识库为空)───────────────────► web_search ──┐ │          │
  │        │                                                            ▲          │ │          │
  │        └─(local)─► transform_query → retrieve → rerank → grade ─────┘(无相关)  │ │          │
  │                          ▲                                   │                 │ │          │
  │                          └──────── rewrite_query ◄─(可改写)───┤                 │ │          │
  │                                                              ▼(有相关)          ▼ │          │
  │                                   generate ◄──────────────────────────────────┘ │          │
  │                                     │  ▲                                        │          │
  │                                     ▼  └─(未通过：严格模式重生成)               │          │
  │                              check_groundedness ──► END                         │          │
  └─────────────────────────────────────────────────────────────────────────────────┘          │
                    │                                                                          │
                    ▼                                                                          │
              写回记忆（超出窗口时压缩为摘要）──► 返回答案 + 来源 + 执行路径 ─────────────────┘
```

---

## 目录结构

```text
.
├── main.py                   # FastAPI 入口：创建应用、生命周期、挂载路由（约 100 行）
├── app/
│   ├── config.py             # 全局配置：唯一的配置来源（读取 .env）
│   ├── rag/                  # ★ Agentic RAG 状态图
│   │   ├── chain.py          #   RAGChain：对外入口（记忆、流式事件、结果整理）
│   │   ├── graph.py          #   组装状态图、节点中文名
│   │   ├── edges.py          #   条件路由（纯函数）
│   │   ├── nodes/            #   节点：route / retrieval / web / generation
│   │   ├── state.py          #   GraphState
│   │   ├── prompts.py        #   全部提示词
│   │   ├── helpers.py        #   解析、上下文拼装、联网判定等纯函数
│   │   └── llm.py            #   LLM / 精排器
│   ├── retrieval/            # ★ 检索层：embeddings / milvus / corpus / bm25 / fusion / search / store
│   ├── api/                  # ★ HTTP 接口层
│   │   ├── routes/           #   system（健康）/ query（问答）/ documents（文档管理）
│   │   ├── openai_compat.py  #   OpenAI 兼容接口 /v1
│   │   ├── schemas.py        #   请求 / 响应模型
│   │   ├── runtime.py        #   运行时单例
│   │   ├── security.py       #   Bearer Token 鉴权
│   │   └── sse.py            #   SSE 工具
│   ├── vector_store.py       # 检索层门面（统一入口 + 异步版本）
│   ├── documents.py          # 文档登记、去重、版本替换、后台入库队列
│   ├── memory.py             # 会话记忆：滚动摘要压缩、SQLite / Redis 持久化
│   ├── context.py            # 上下文工程：对话背景渲染、资料预算裁剪
│   ├── web_search.py         # Tavily 联网搜索
│   ├── document_processer.py # 文档加载与分块
│   └── llm_cache.py · redis_client.py · observability.py
├── research_agent/           # 命令行多轮对话
├── tests/                    # 单元测试（mock 网络 / Milvus / LLM，可离线运行）
├── eval_ragas.py · eval_set.sample.json · _verify_after_merge.py
├── requirements.txt · requirements.runtime.txt · Dockerfile · compose.yaml · .env.example
├── data/                     # 知识库源文档；上传的文件在 data/uploads/<doc_id>/
└── vector_db/                # 自动生成：Milvus 库、BM25 语料、文档登记表、会话记忆
```

代码按职责拆分，单个文件不超过 600 行（`_verify_after_merge.py` 会检查）。

---

## 快速开始

### 1. 准备 Python 环境

Python 版本需要 **3.10 ~ 3.12**，推荐 **3.11**。LangChain 1.x 要求 Python ≥ 3.10；`torch`、`milvus-lite` 在 Windows 上对 3.13 及以上版本的预编译包支持还不完整。

```bash
conda create -n LearnLC311 python=3.11
conda activate LearnLC311
pip install -r requirements.txt
```

> **VS Code 终端里找不到 `conda`？**
>
> 原因是 Miniconda 装成了"所有用户"模式，没有加入 PATH。解决步骤：
>
> 1. 执行一次 `conda init`，执行完彻底重启 VS Code：
>
>    ```powershell
>    & "C:\ProgramData\miniconda3\Scripts\conda.exe" init powershell cmd.exe
>    ```
>
> 2. 如果 PowerShell 报"禁止运行脚本"，再执行：
>
>    ```powershell
>    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
>    ```

> **国内网络下载 Embedding 模型慢？**
>
> 先设置 HuggingFace 镜像再启动：
>
> - CMD：`set HF_ENDPOINT=https://hf-mirror.com`
> - PowerShell：`$env:HF_ENDPOINT="https://hf-mirror.com"`

### 2. 配置 `.env`

```bash
copy .env.example .env      # Windows；Linux / macOS 用 cp
```

至少需要填写 `LLM_API_KEY`。其余配置项都是可选的，不填时对应功能会自动降级：

| 配置项 | 不填会怎样 |
|---|---|
| `LLM_API_KEY` | **必填**，不填无法回答问题 |
| `COHERE_API_KEY` | 跳过精排，混合检索照常工作 |
| `TAVILY_API_KEY` | 不联网，只用本地知识库 |
| `REDIS_URL` 指向的 Redis | Redis 不可用时自动绕过缓存，请求直接调用 LLM |
| `LANGSMITH_API_KEY` | 不做链路追踪；建议同时设置 `LANGSMITH_TRACING=false` |

### 3. 启动服务

```bash
# 可选：启动 Redis，作为 LLM 缓存（MEMORY_BACKEND=redis 时也用于存会话记忆）
docker run -d --name rag-redis -p 6379:6379 redis:7-alpine

python main.py
# 或 uvicorn main:app --reload --port 8000
```

启动后打开 <http://127.0.0.1:8000/docs>，可以在 Swagger UI 里直接调试所有接口。首次启动会下载 Embedding 模型（`bge-m3` 约 2.2GB），并构建 BM25 倒排索引。

### 使用 Docker Compose

```bash
docker compose up --build -d     # 构建并启动 API + Redis
docker compose logs -f app       # 查看日志（首次会下载模型，耗时较长）
docker compose ps
```

Compose 会自动完成以下配置：

- Redis 地址设为 `redis://redis:6379/0`；
- `data/` 和 `vector_db/` 挂载到宿主机，所以文档、Milvus 数据、**会话记忆**都不会随容器删除；
- Redis 数据和 HuggingFace 模型缓存存放在 Docker Volume 中；
- `.env` 只在启动时注入，不会打进镜像。

镜像只安装 `requirements.runtime.txt`，不包含离线评估用的 `ragas`。

```bash
docker compose restart app       # 重启应用
docker compose down              # 停止并删除容器，保留数据卷
docker compose down -v           # 同时删除 Redis / 模型缓存卷（谨慎）
```

---

## 使用方式

> 以下 `curl` 示例采用 Git Bash / Linux 写法。在 PowerShell 中请使用 `curl.exe`，并注意 JSON 引号的转义；或者直接在 Swagger UI 里操作。

### 入库与单轮问答

```bash
# 1) 查看生效配置和向量库状态
curl http://127.0.0.1:8000/health

# 2) 上传文档：立即返回 doc_id，后台排队入库
curl -X POST http://127.0.0.1:8000/documents -F "file=@data/paul_graham_essay.txt"
#    或扫描 data/ 目录（新文件、有变化的文件才会入库）
curl -X POST http://127.0.0.1:8000/documents/scan

# 3) 查看入库进度（status: pending → processing → ready / failed）
curl http://127.0.0.1:8000/documents

# 4) 提问
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"作者为什么要学习画画？"}'
```

文档管理规则：

- **内容去重**：内容完全相同的文件（哪怕换了文件名）只入库一次，第二次上传返回 `action: duplicate`。
- **版本替换**：同名上传、内容变了，视为新版本；新版本入库成功后自动删除旧版本，替换过程中检索不中断。
- **删除**：`DELETE /documents/{doc_id}` 会同时删除向量、BM25 语料和上传的文件。
- **查看原文**：`GET /documents/{doc_id}/file` 在浏览器里打开原文件；PDF 可加 `#page=N` 跳到引用页。`/query` 返回的 `sources[].doc_id` 就是这里用的 ID。
- 设置了 `API_TOKEN` 时，所有请求都要加上 `-H "Authorization: Bearer <API_TOKEN>"`。

### 流式问答（SSE）

```bash
curl -N -X POST http://127.0.0.1:8000/query/stream -H "Content-Type: application/json" \
  -d '{"question":"RRF 是什么？","session_id":"user-42"}'
```

会依次收到以下事件：

| 事件 | 数据 | 前端怎么处理 |
|---|---|---|
| `step` | `{"node","label"}` | 显示进度，如"检索知识库""联网搜索" |
| `token` | `{"text"}` | 追加到答案 |
| `reset` | `{"reason"}` | 清空已显示的答案，后续 token 是新版本（草稿未通过出处核对） |
| `done` | 与 `/query` 相同的完整结果 | 渲染来源、观测信息 |
| `error` | `{"detail"}` | 显示错误 |

浏览器端请用 `fetch` 加 `ReadableStream` 读取，因为 `EventSource` 不支持 POST 请求。

### 接入 Open WebUI（零前端开发）

服务提供 OpenAI 兼容接口，任何支持"自定义 OpenAI 地址"的聊天客户端都能直接当前端用。

```bash
# 方式一：和本服务一起用 Docker Compose 启动（Open WebUI 地址 http://127.0.0.1:3000）
docker compose --profile ui up -d

# 方式二：已有 Open WebUI 时，在 管理员设置 → 外部连接 → OpenAI API 里添加：
#   URL：http://<本服务地址>:8000/v1
#   Key：.env 里的 API_TOKEN
```

接入后可以选择两个模型：

- `rag-agent`：查本地知识库，需要时联网；
- `rag-agent-local`：只查本地，不联网。

接入效果：

- 进度以"思考过程"的形式折叠显示；
- 回答末尾附参考来源；
- 多轮历史由 Open WebUI 每次随请求带上；
- 如果想使用服务端记忆（含滚动摘要），可以在请求头里加 `X-Session-Id`。

> **重要：** Open WebUI 默认会用当前模型自动生成对话标题、标签和追问建议。每一次这样的生成都会跑一遍完整的 RAG 流程，浪费调用。
> 请在"管理员设置 → 界面"里，把"任务模型"设为另一个普通 LLM，或者关闭标题、标签、追问建议的自动生成。

### 多轮对话（服务端持久化记忆）

请求里带上同一个 `session_id`，服务端就会自动加载、注入、保存这个会话的记忆，服务重启后仍然有效：

```bash
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question":"Paul Graham 早年学过什么？","session_id":"user-42"}'

# 追问里的"他"会结合记忆补全为完整问题
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" \
  -d '{"question":"他后来为什么放弃了？","session_id":"user-42"}'

curl http://127.0.0.1:8000/sessions/user-42             # 查看记忆
curl -X DELETE http://127.0.0.1:8000/sessions/user-42   # 清除记忆
```

不想让服务端保存会话时，可以由客户端自己传 `history: [["问题","回答"], ...]`。两者同时传时以 `session_id` 为准。

### 调试检索（不生成答案）

```bash
# 只走"查询变换 → 混合检索 → 精排"，用来调参
curl -X POST http://127.0.0.1:8000/retrieve -H "Content-Type: application/json" \
  -d '{"question":"混合检索为什么用 RRF？","top_k":5}'

# 跳过精排，对比精排前后的差异
curl -X POST http://127.0.0.1:8000/retrieve -H "Content-Type: application/json" \
  -d '{"question":"混合检索为什么用 RRF？","skip_rerank":true}'
```

### 命令行多轮对话

```bash
python -m research_agent.agent
```

CLI 与 HTTP 接口共用同一张状态图和同一个记忆库。默认会话 ID 是固定的（`AGENT_SESSION_ID`），所以**重启后会自动接上次的对话**。可用命令：

| 命令 | 作用 |
|---|---|
| `/history` | 查看当前会话的记忆（摘要 + 最近几轮） |
| `/forget` | 清除当前会话的记忆 |
| `/new` | 开启一个新会话 |
| `/session ID` | 切换到指定会话 |
| `/warmup` | 预热本地知识库 |
| `/quiet` | 开关执行路径的显示 |
| `/quit` | 退出 |

---

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查：生效参数、文档统计、Embedding 维度检查（无需鉴权） |
| GET | `/metrics` | Prometheus 指标（无需鉴权，不在 Swagger 中显示） |
| POST | `/query` | 问答，一次性返回；传 `session_id` 启用持久化记忆，`allow_web=false` 不联网 |
| POST | `/query/stream` | 问答，SSE 流式返回 |
| POST | `/retrieve` | 只检索不生成，用于调参 |
| GET | `/sessions/{id}` | 查看某个会话保存的记忆 |
| DELETE | `/sessions/{id}` | 清除某个会话的记忆 |
| POST | `/documents` | 上传文档（去重 / 同名替换），后台入库，返回 202 |
| POST | `/documents/scan` | 扫描 `data/` 目录并入库新文件和有变化的文件 |
| GET | `/documents` | 文档列表，可按 `status` 过滤 |
| GET | `/documents/{id}` | 文档详情与入库状态 |
| GET | `/documents/{id}/file` | 打开原文件 |
| DELETE | `/documents/{id}` | 删除文档 |
| POST | `/documents/reindex` | 用当前 Embedding 模型重建全部索引 |
| DELETE | `/index` | 删除全部文档（`data/` 下的源文件保留，会话记忆不受影响） |
| GET | `/v1/models` | OpenAI 兼容：模型列表 |
| POST | `/v1/chat/completions` | OpenAI 兼容：对话（支持 `stream`） |
| POST | `/ingest/file`、`/ingest/directory` | 旧接口，保留兼容，会同步等待入库完成 |

设置了 `API_TOKEN` 时，除 `/health`、`/metrics` 外的所有接口都需要请求头 `Authorization: Bearer <API_TOKEN>`。

### `/query` 响应字段

```json
{
  "answer": "……[paul_graham_essay.txt]",
  "question": "补全/改写后的问题",
  "rewritten": false,
  "num_documents": 5,
  "sources": [
    {"source": "data/paul_graham_essay.txt", "filename": "paul_graham_essay.txt", "page": null,
     "snippet": "...", "relevance_score": 0.87, "rrf_score": 0.0164, "source_type": "local"}
  ],
  "queries": ["原问题", "变换出的查询2", "变换出的查询3"],
  "transform_mode": "multi_query(3)",
  "candidates": 20,
  "reranked": true,
  "groundedness": "yes",
  "groundedness_retries": 0,
  "route": "local",
  "source_type": "local",
  "web_searched": false,
  "web_error": "",
  "steps": ["route_question", "transform_query", "retrieve", "rerank_documents",
            "grade_document", "generate", "check_groundedness"],
  "session_id": "user-42",
  "memory_turns": 1,
  "memory_summary_used": false
}
```

答案不对时，可以按下面的顺序沿着这些字段逐级定位问题出在哪一环：

| 现象 | 看哪个字段 | 可能原因 |
|---|---|---|
| 检索没找到 | `queries`、`candidates` | 问法不好，或候选太少 |
| 检索到了但被筛掉 | `candidates` 有值，但 `sources` 为空 | 精排或相关性评分把文档筛掉了 |
| 有文档但答案被拒 | `groundedness` | 是否为 `no` |
| 不确定走了哪条路径 | `steps`、`route`、`source_type`、`web_error` | 查看实际执行的节点、是否联网、联网是否失败 |
| 追问没被理解 | `memory_turns`、`question` | 历史有没有注入，问题有没有被正确补全 |

字段取值说明：

- `relevance_score`：本地结果为 Cohere 精排分，网页结果为 Tavily 相关度。
- `groundedness`：`yes` 通过；`no` 未通过（已按配置处理）；`skipped` 兜底答案无需检查；`off` 检查已关闭；`unknown` 检查调用失败。

---

## 核心设计

### 1. 状态图节点

| 节点 | 做什么 | 解决什么问题 |
|---|---|---|
| `route_question` | 结合记忆补全追问；判断是否依赖实时信息 | 追问能被理解；实时类问题不必先白查一遍本地库 |
| `transform_query` | 查询变换：multi_query / HyDE | 问法不好时，后面的环节再强也救不回来 |
| `retrieve` | 稠密(Milvus) ⊕ 稀疏(BM25)，RRF 融合 | 语义匹配与关键词精确命中互补 |
| `rerank_documents` | Cohere 交叉编码器精排，20 条 → 5 条 | 排得准；同时减少后续 LLM 评分的调用次数 |
| `grade_document` | LLM 逐条评估文档相关性（并发执行） | 过滤漏网的无关文档 |
| `rewrite_query` | 改写查询后重新检索 | 联网不可用时的自我修正手段 |
| `web_search` | Tavily 联网搜索，结果转为 Document | 本地没有的内容、实时信息 |
| `generate` | 按资料生成答案（本地、网页、严格模式各有一套提示词） | 带出处，不编造 |
| `check_groundedness` | 忠实性自检；未通过时改用严格模式重新生成一次 | 拦截幻觉，同时减少对正确答案的误判拒答 |

`recursion_limit` 会自动取配置值与按当前配置估算出的最大步数中的较大者。调大 `MAX_RETRIES` 后不需要再手动同步这个值。

### 2. 联网搜索：状态图中的节点，而非 Agent 工具

需要联网的情况只有两种，而且都是确定的：

- 问题依赖实时信息：入口处判断一次即可；
- 本地检索、精排、评分之后没有相关文档：用规则就能判断。

这正是 Corrective RAG 的模式，用条件边就能表达，不需要再套一层 ReAct Agent，让 LLM 每一轮都推理"要不要调工具"。具体好处如下：

- **路径确定**：每次请求最多联网一次，LLM 调用次数有上限，可以写单元测试覆盖。
- **出处不被稀释**：网页结果直接带 URL 进入生成环节，不会被外层 LLM 再改写一遍；网页答案同样要经过忠实性检查。
- **省调用**：拿不准时先查本地，本地失败再自动联网，所以入口路由只需要对"必须联网"的问题判断准确即可。`WEB_FALLBACK_AFTER_RETRIES=0`（默认）时，本地检索失败后直接联网，不再做无谓的查询改写。
- **故障可见**：联网失败时，超时、限流、5xx 等临时故障会退避重试；key 无效、额度用尽等永久故障不重试。失败原因写入 `web_error`，不抛异常。

### 3. 检索层：混合检索 + 精排

```text
                 ┌─ 稠密：Milvus 向量检索 (Top 20) ─┐
   查询 ─────────┤                                  ├─ RRF 融合 ─► Top 20 ─► Rerank ─► Top 5
                 └─ 稀疏：BM25 关键词检索 (Top 20) ─┘
```

- **稠密检索负责"语义相近"**。例如问"如何提升检索质量"，能命中讲"recall 优化"的段落。
- **稀疏检索负责"关键词精确命中"**。专有名词、型号、人名这类内容，语义模型容易漂移，BM25 往往能直接命中。中文用 jieba 分词；没装 jieba 时退化为单字加二元组切分。
- **RRF 融合**：余弦相似度在 0~1 之间，BM25 分数没有上界，两者量纲不同，直接加权没有意义。RRF 只看排名，公式为 `score(doc) = Σ weight_i × 1/(60 + rank_i(doc))`。
- **精排放在相关性评分之前**：评分要对每个文档调用一次 LLM，先用精排把 20 条砍到 5 条，LLM 调用次数直接降为原来的 1/4。

查询变换由 `QUERY_TRANSFORM` 控制：

| 模式 | 做法 | 适用场景 |
|---|---|---|
| `none` | 直接用原问题检索 | 作为调试基线 |
| `multi_query` | RAG-Fusion：把问题裂变成多条不同角度的查询，分别检索后用 RRF 融合 | 默认推荐 |
| `hyde` | 先让 LLM 写一段假想答案，再用这段答案去检索（答案与文档属于同分布匹配） | 问题短、术语少 |

### 4. 忠实性自检

LLM 裁判也会出错。最典型的例子是跨语言的数字换算：原文是 `70 billion`，中文答案"700 亿"本来是正确的，但裁判会误判为错误。为此做了三层防护：

1. 生成提示词要求数字原样引用；
2. 裁判提示词里加入了中英文数量级的换算对照表；
3. 第一次判为 `no` 时不直接拒答，而是改用严格模式重新生成一次，再检查一次。

失败后的处理方式由 `GROUNDEDNESS_ACTION` 控制，可选 `retry`、`warn`、`off`。

### 5. 上下文工程与会话记忆

每次调用 LLM 时，上下文里都有三类内容：问题、资料、历史。`app/context.py` 负责管理"放什么、放多少、怎么放"：

| 动作 | 做法 |
|---|---|
| **写入** | 长期记忆采用"滚动摘要 + 最近 N 轮原文"的形式；存的是补全后的独立问题，脱离上下文也能读懂 |
| **选择** | 只注入最近 `HISTORY_TURNS` 轮原文和摘要；资料已按相关度排好序 |
| **压缩** | 历史受 `MEMORY_CONTEXT_CHARS` 预算约束，超出时先丢最旧的轮次；资料受 `CONTEXT_MAX_CHARS` 约束，从尾部裁剪；生成和忠实性检查用的是同一份资料 |
| **隔离** | 提示词分为【对话背景】和【资料】两个区；背景只用于理解指代、保持前后一致，新的事实必须来自资料 |

记忆的生命周期（`app/memory.py`）：

```text
加载（摘要 + 最近几轮）→ 注入状态图 → 问答 → 写回本轮
                                              │
                         原文超过 MEMORY_WINDOW_TURNS（8）轮？
                                              │ 是
                         较早的轮次 + 旧摘要 ─LLM─► 新摘要，只保留最近 MEMORY_KEEP_TURNS（4）轮
```

- **存储后端**：用 `MEMORY_BACKEND` 选择。默认 `sqlite`（标准库自带，零部署，文件为 `vector_db/chat_memory.sqlite3`）；多实例部署时用 `redis`；另有 `memory`（进程内存）和 `off`。
- **压缩成本**：批量压缩，约每 4 轮才调用一次摘要；摘要失败时保留原文，下一轮再试。
- **过期**：会话超过 `MEMORY_TTL_SECONDS`（默认 7 天）没有新消息即过期，SQLite 中的过期数据会被顺带清理。
- **容错**：记忆读写失败只打印告警，不影响问答。
- **为什么不用 LangGraph checkpointer**：它会在每个超步保存完整的图状态，包括检索到的文档，适合"中断恢复、时间回溯"场景，用来存聊天记忆太重，而且还需要额外安装依赖。

### 6. LLM 调用次数

| 环节 | 次数 |
|---|---|
| 入口路由 | 0~1：有历史、或需要判断是否联网时才调用 |
| 查询变换 | 1：`multi_query` / `hyde` 模式；`none` 模式为 0 |
| 相关性评分 | ≤ `RERANK_TOP_N`，并发执行 |
| 查询改写 | 0~`MAX_RETRIES`：仅在联网不可用时 |
| 生成 + 忠实性检查 | 2；严格模式重新生成时再加 2 |
| 记忆摘要 | 约每 4 轮 1 次 |

直接联网的路径更短：`route → web_search → generate → check`，通常只需 3 次 LLM 调用。想省成本，可以调小 `NUM_QUERIES`、`RERANK_TOP_N`，或者设 `WEB_ROUTING=off`。所有 LLM 调用都会经过 Redis 缓存：缓存键包含完整提示词和模型配置，因此不会误命中。

---

## 配置速查

完整配置项及注释见 [.env.example](.env.example)，最常调整的几项如下：

```ini
# 检索
HYBRID_ENABLED=true
DENSE_WEIGHT=0.5            # 知识库专有名词、型号多时调低，多给 BM25 权重
SPARSE_WEIGHT=0.5
FUSION_TOP_K=20             # 送给精排的候选数
RERANK_TOP_N=5              # 最终喂给 LLM 的条数，同时决定评分次数
QUERY_TRANSFORM=multi_query # none / multi_query / hyde
NUM_QUERIES=3               # 2~3 性价比最高

# 联网
TAVILY_API_KEY=
WEB_ROUTING=llm             # llm：入口判断是否需要实时信息；off：总是先查本地
WEB_FALLBACK_AFTER_RETRIES=0

# 防幻觉
GROUNDEDNESS_ACTION=retry   # retry / warn / off

# 上下文与记忆
HISTORY_TURNS=3
MEMORY_CONTEXT_CHARS=1500
CONTEXT_MAX_CHARS=6000
MEMORY_BACKEND=sqlite       # sqlite / redis / memory / off
MEMORY_TTL_SECONDS=604800
```

`GET /health` 会回显所有实际生效的参数，调参前先看一眼，确认改动已经生效。

---

## 可观测性

`/metrics` 以 Prometheus 文本格式暴露指标，由 `METRICS_ENABLED` 和 `METRICS_PATH` 控制。

| 指标 | 标签 | 用途 |
|---|---|---|
| `http_requests_total` | `handler`、`method`、`status` | 请求总数，状态码按 2xx/4xx/5xx 聚合 |
| `http_request_duration_seconds` | `handler`、`method` | 延迟直方图，桶覆盖 50ms～300s |
| `http_requests_inprogress` | `handler`、`method` | 当前正在处理的请求数 |

```promql
sum by (handler, method) (rate(http_requests_total[5m]))                       # 各接口 QPS
histogram_quantile(0.95, sum by (le, handler, method)
  (rate(http_request_duration_seconds_bucket[5m])))                            # P95 延迟
sum(rate(http_requests_total{status="5xx"}[5m])) / sum(rate(http_requests_total[5m]))  # 5xx 错误率
```

生产环境建议只允许 Prometheus 从内网访问 `/metrics`。如果需要逐节点的链路追踪，可以在 `.env` 中开启 LangSmith（`LANGSMITH_TRACING=true`）。

---

## 测试与评估

```bash
# 单元测试：网络、Milvus、LLM 都已 mock，不需要 API Key
python -m unittest discover -s tests -v

# 只读验收：模块导入、配置、图节点与关键边、提示词规则
python _verify_after_merge.py
```

测试覆盖的内容：

- 联网降级路径、空知识库直接联网、联网失败时的提示、`allow_web=false` 时不联网；
- 流式事件契约：逐段输出、草稿未通过核对时的 `reset`、兜底答案补发；
- OpenAI 兼容接口：消息解析、流式格式、`-local` 模型、`X-Session-Id`、Token 鉴权；
- 文档管理：内容去重、同名替换旧版本、入库失败清理、删除、后台队列与重启恢复、BM25 语料删除；
- 上下文预算裁剪；
- SQLite 读写，以及模拟进程重启后的恢复、TTL 过期；
- 记忆摘要压缩，以及摘要失败时的降级；
- 两轮对话通过记忆补全追问的端到端流程。

> 在 Windows PowerShell 中运行验收脚本前，先执行 `$env:PYTHONIOENCODING='utf-8'`，否则输出里的 emoji 可能触发 GBK 编码错误。

RAGAs 离线评估（会调用真实模型，消耗 token）：

```bash
python eval_ragas.py --dataset eval_set.sample.json --limit 3              # 先用 3 条试跑
python eval_ragas.py --dataset eval_set.sample.json --out eval_report.csv  # 全量评估并导出报告
```

评估时会**自动关闭联网**：网页结果随时间变化，结果不可复现，会污染指标。评估集格式为 `[{"question": "...", "reference": "可选的标准答案"}]`；不提供 `reference` 时只能计算两项指标。

| 指标偏低 | 说明什么 | 调整方向 |
|---|---|---|
| `context_recall` | 该找的证据没找全 | 调整 chunk 大小、提高 `FUSION_TOP_K`、开启 `multi_query` |
| `context_precision` | 召回了很多无关内容 | 调整 `RERANK_TOP_N`、`DENSE/SPARSE_WEIGHT` |
| `faithfulness` | 模型在编造 | 收紧生成提示词，`GROUNDEDNESS_ACTION=retry` |
| `answer_relevancy` | 答案跑题 | 调整生成模板 |

---

## 故障排查

### 启动时报 `Open local milvus failed`

最常见的原因是**已经有一个服务实例在运行**。Milvus Lite 会对本地库文件加独占锁，同一时刻只允许一个进程打开（`python main.py` 和 CLI 也不能同时打开同一个库）。

```bash
netstat -ano | findstr :8000          # 查看占用 8000 端口的进程 PID
taskkill /PID <进程号> /F
```

`app/vector_store.py` 会主动探测锁冲突，并直接打印真实原因和处理步骤。如果需要多个进程共用同一个库，可以改用 Milvus 服务（`docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest`），并把 `RAG_MILVUS_URI` 改为 `http://localhost:19530`。

> 会话记忆所在的 SQLite 开启了 WAL 模式，API 和 CLI 可以同时读写，不受上面这个限制。

### 联网搜索没有生效

1. 看 `/health` 中 `web_search.ready` 是否为 `true`，并确认已配置 `TAVILY_API_KEY`、`WEB_SEARCH_ENABLED=true`；
2. 看 `/query` 响应中的 `web_error`，那里有中文失败原因，例如 key 无效、额度用尽、超时；
3. 如果 `WEB_ROUTING=off`，只有本地检索不到相关文档时才会联网，这是预期行为。

### 追问没有被理解

1. 确认请求里带了**同一个** `session_id`；
2. 看响应中的 `memory_turns` 是否大于 0；
3. 用 `GET /sessions/{id}` 确认记忆确实已保存；
4. `MEMORY_BACKEND=off` 时不会保存任何记忆。

### `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`

这是 ragas 0.4.x 与新版 langchain-community 之间的兼容问题。`eval_ragas.py` 已内置补丁，不需要额外处理。

### 精排分数普遍很低（0.0x 量级）

先确认知识库里到底有没有相关内容：用 `/health` 查看 `index_count`，用 `/retrieve` 查看返回结果的 `filename` 分布。如果库里本来就没有相关文档，分数低、本地检索后转为联网或拒答都是**正确行为**。

### 中文问题检索效果差

先确认 `/health` 里的 `embedding.model` 是 `BAAI/bge-m3`。如果你的 `.env` 还是从旧模板复制来的 `BAAI/bge-small-en-v1.5`（纯英文模型），中文检索排序会偏。改成 `bge-m3` 后需要重建索引，见下方注意事项第 1 条。

### 启动日志提示"向量库维度与当前模型不一致"

说明换过 Embedding 模型，但索引还是旧模型建的。这时检索会失败，系统会自动降级为联网或拒答。调用 `POST /documents/reindex` 重建即可。

### Open WebUI 里每发一条消息，后台跑了好几次问答

这是 Open WebUI 在用当前模型生成标题、标签和追问建议。处理方法见上文"接入 Open WebUI"一节：设置任务模型，或关闭这些自动生成功能。

---

## 注意事项

1. **更换 Embedding 模型后必须重建索引。** 向量维度变了，collection 结构就对不上。修改 `EMBEDDING_MODEL` 后重启服务，再调用 `POST /documents/reindex`：系统会清空向量库，把所有已登记的文档用新模型重新入库。
   - `bge-m3` 约 2.2GB，首次下载较慢，CPU 上入库速度也比 `bge-small` 慢；
   - 旧版本直接写入、没有登记的数据，重建后需要再执行一次 `POST /documents/scan`。

2. **Milvus 地址的变量名是 `RAG_MILVUS_URI`，不是 `MILVUS_URI`。** pymilvus 3.x 在 import 阶段就会读取 `MILVUS_URI` 并按 `http[s]://` 格式解析，如果值是本地文件路径，会直接报错。

3. **不要单独删除 `vector_db/` 下的文件。** `bm25_corpus.jsonl` 是 BM25 的数据源，删除后 BM25 会**静默失效**；`documents.sqlite3` 是文档登记表，删除后去重和按文档删除都会失效。需要清空时请调用 `DELETE /index`。

4. **`DELETE /index` 会删除全部文档**，包括向量、登记表和上传的文件；`data/` 下的源文件保留，会话记忆也不受影响。

5. **`milvus-lite` 的 PyPI 元数据标注为不支持 Windows**，但在本项目的 Windows 环境中实测可用。如果某台机器装不上，改用 Docker 启动 Milvus 服务，并设置 `RAG_MILVUS_URI`。

6. **对外提供服务前必须设置 `API_TOKEN`**，并把 `session_id` 与用户身份绑定，避免会话被他人猜中并读取。会话记忆包含用户的对话内容：按合规要求设置 `MEMORY_TTL_SECONDS`，并通过 `DELETE /sessions/{id}` 提供"删除对话"的能力。
