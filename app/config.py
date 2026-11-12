# 配置（模型， 路径， 超参数）
#   1) Milvus 向量库（替代 Chroma/FAISS）
#   2) 混合检索 Hybrid Search（稠密向量 + BM25 稀疏，RRF 融合）
#   3) 查询变换 Query Transformations（multi-query / HyDE）
#   4) Reranker（Cohere Rerank 精排）
# 另外补上了原文件缺失的 RECURSION_LIMIT —— 没有它 rag_chain.answer() 会直接 AttributeError。

import os
from dotenv import load_dotenv

load_dotenv()

_reserved_uri = os.environ.get("MILVUS_URI", "")
if _reserved_uri and not _reserved_uri.startswith(("http://", "https://")):
    os.environ.pop("MILVUS_URI", None)
    print(
        "[config] 检测到环境变量 MILVUS_URI 不是 http(s) 地址，已临时移除。"
        "它会让 import pymilvus 直接失败；请改用 RAG_MILVUS_URI。"
    )

os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_GO_LOG_SEVERITY_LEVEL", "ERROR")
os.environ.setdefault("GRPC_GO_LOG_VERBOSITY_LEVEL", "ERROR")


def _env_bool(key: str, default: str = "false") -> bool:
    """把 .env 里的字符串解析成布尔值。"""
    return os.getenv(key, default).strip().lower() in ("true", "1", "yes", "on")


class Settings:
    """全局配置的唯一来源。

    【属性分组】
        LLM            —— 大模型（智谱 GLM，走 OpenAI 兼容协议）
        Embedding      —— 本地向量模型（HuggingFace，离线跑）
        向量库          —— Milvus 连接与 collection 配置
        RAG 参数        —— 分块大小、召回条数等基础参数
        混合检索        —— 稠密/稀疏双路检索与 RRF 融合
        查询变换        —— multi_query / hyde
        Reranker       —— Cohere 精排
        忠实性检查      —— 防幻觉（Self-RAG 反思闭环）
        Agentic RAG    —— 循环上限、图步数上限
        LangSmith      —— 链路追踪
        派生属性        —— 由上面计算得出，不需要配置
    """

    # ------------------------------------------------------------ LLM
    OPENAI_API_KEY: str = os.getenv("LLM_API_KEY")
    OPENAI_BASE_URL: str = os.getenv(
        "LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"
    )
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL", "glm-4.5-air")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.0))

    # ------------------------------------------------------------ Embedding
    # 本次 embedding 模型（本地 HuggingFace，384 维）
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # ------------------------------------------------------------ 向量库（Milvus）
    # 注意环境变量名叫 RAG_MILVUS_URI 而不是 MILVUS_URI，原因见文件开头的踩坑警告。
    # 三种写法都支持：
    #   ./vector_db/milvus_rag.db        -> Milvus Lite 本地文件（学习/单机，零部署）
    #   http://localhost:19530           -> Milvus 单机服务
    #   https://xxx.api.zilliz.com:443   -> Zilliz Cloud
    MILVUS_URI: str = os.getenv("RAG_MILVUS_URI", "./vector_db/milvus_rag.db")
    # 有鉴权时填 "用户名:密码" 或 Zilliz 的 API Key；本地 Lite 留空
    MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "")
    # 本地数据总目录（上传文件、BM25 语料副本都放这里）
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./vector_db")
    # Milvus collection 名
    INDEX_NAME: str = os.getenv("INDEX_NAME", "rag_index")
    # 启动时是否重建 collection（结构变更后需要，平时保持 false）
    MILVUS_DROP_OLD: bool = _env_bool("MILVUS_DROP_OLD", "false")

    # ------------------------------------------------------------ RAG 参数
    TOP_K: int = int(os.getenv("TOP_K", 5))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))

    # ------------------------------------------------------------ 混合检索
    HYBRID_ENABLED: bool = _env_bool("HYBRID_ENABLED", "true")
    # RRF 融合时稠密/稀疏两路各自的权重（0.5/0.5 表示等权）
    DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", 0.5))
    SPARSE_WEIGHT: float = float(os.getenv("SPARSE_WEIGHT", 0.5))
    # 每一路先各召回多少条（粗排），再融合 + 精排
    DENSE_TOP_K: int = int(os.getenv("DENSE_TOP_K", 20))
    SPARSE_TOP_K: int = int(os.getenv("SPARSE_TOP_K", 20))
    # 融合后保留的候选数（送给 Reranker 的条数）
    FUSION_TOP_K: int = int(os.getenv("FUSION_TOP_K", 20))
    # BM25 语料副本文件名（Milvus 只存向量，BM25 需要原始文本，重启后从这里恢复）
    BM25_CORPUS_FILE: str = os.getenv("BM25_CORPUS_FILE", "bm25_corpus.jsonl")

    # ------------------------------------------------------------ 查询变换
    # none        关闭，直接用原问题检索
    # multi_query RAG-Fusion：一个问题裂变成多条查询，各自检索后 RRF 融合
    # hyde        让 LLM 先写一段"假想答案"，用这段假想答案去检索
    QUERY_TRANSFORM: str = os.getenv("QUERY_TRANSFORM", "multi_query").strip().lower()
    # multi_query 模式下额外生成几条查询（不含原问题）
    NUM_QUERIES: int = int(os.getenv("NUM_QUERIES", 3))

    # ------------------------------------------------------------ Reranker（精排）
    RERANK_ENABLED: bool = _env_bool("RERANK_ENABLED", "true")
    RERANK_PROVIDER: str = os.getenv("RERANK_PROVIDER", "cohere").strip().lower()
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    # rerank-v3.5（多语言，推荐）/ rerank-multilingual-v3.0
    COHERE_RERANK_MODEL: str = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    # 精排后最终喂给 LLM 的文档块数
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", 5))

    # ------------------------------------------------------------ 忠实性检查（防幻觉）
    # 生成完答案后，会再让 LLM 判断"答案是否由检索到的文档支撑"。
    # 这个判断不总是可靠，最典型的是**跨语言数字换算**：
    #   英文事实 "7 billion to 70 billion" 对应中文 "70亿到700亿"，
    #   模型有时会把 70 billion 误读成 70亿，从而把正确答案判成幻觉。
    # 所以失败后不要立刻拒答，而是先用更严格的提示词重生成一次。
    #
    # retry  失败 -> 严格模式重生成 -> 仍失败才拒答【默认，推荐】
    # warn   失败 -> 只打印警告，保留原答案（宁可多给答案，也不要误拒）
    # off    完全不检查（最省 token，但失去防幻觉能力）
    GROUNDEDNESS_ACTION: str = os.getenv("GROUNDEDNESS_ACTION", "retry").strip().lower()
    # retry 模式下最多重新生成几次（每次多一次 LLM 调用）
    MAX_GROUNDEDNESS_RETRIES: int = int(os.getenv("MAX_GROUNDEDNESS_RETRIES", 1))

    # ------------------------------------------------------------ Agentic RAG 参数
    # 最大"检索-评分-改写"循环轮数（防止死循环）
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 2))
    # LangGraph 单次调用的最大步数上限（图节点执行次数）。
    # 实际生效值会取它与 rag_chain.required_recursion_limit() 的较大者，
    # 调大 MAX_RETRIES 时不会再因为忘了同步调这里而抛 GraphRecursionError
    RECURSION_LIMIT: int = int(os.getenv("RECURSION_LIMIT", 25))
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")

    # ------------------------------------------------------------ 联网搜索（Tavily，CRAG 纠错检索）
    # 联网搜索是状态图里的一个**节点**，不是交给 Agent 自由选择的工具：
    #   入口路由判定"需要实时信息" -> 直接联网
    #   本地检索评分后没有相关文档   -> 确定性地降级为联网（Corrective RAG）
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    WEB_SEARCH_ENABLED: bool = _env_bool("WEB_SEARCH_ENABLED", "true")
    # 入口路由策略：
    #   llm  用一次 LLM 判断问题是否依赖实时/最新信息，是则跳过本地检索直接联网【默认】
    #   off  总是先查本地，本地没有相关文档时再联网（省一次 LLM 调用）
    WEB_ROUTING: str = os.getenv("WEB_ROUTING", "llm").strip().lower()
    # 本地检索失败后，先做几次"查询改写重检索"再联网。
    # 默认 0：multi_query 已经做过多角度检索 + 精排，评分仍判无关，
    # 基本说明知识库里没有，继续改写只会白烧 LLM 调用、拖慢响应。
    # 实际改写次数取 min(MAX_RETRIES, 本值)；联网不可用时仍按 MAX_RETRIES 改写。
    WEB_FALLBACK_AFTER_RETRIES: int = int(os.getenv("WEB_FALLBACK_AFTER_RETRIES", 0))
    WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", 3))
    # basic 快且便宜；advanced 抓取更深，适合需要细节的问题
    WEB_SEARCH_DEPTH: str = os.getenv("WEB_SEARCH_DEPTH", "basic")
    # 每条网页正文喂给 LLM 前截断到多少字符，防止上下文膨胀
    WEB_SNIPPET_CHARS: int = int(os.getenv("WEB_SNIPPET_CHARS", 800))
    WEB_SEARCH_TIMEOUT: float = float(
        os.getenv("WEB_SEARCH_TIMEOUT", os.getenv("TOOL_TIMEOUT", 30))
    )
    # 超时/限流/5xx 等临时故障的最大尝试次数（鉴权、配额类错误不重试）
    WEB_SEARCH_MAX_ATTEMPTS: int = int(
        os.getenv("WEB_SEARCH_MAX_ATTEMPTS", os.getenv("TOOL_MAX_ATTEMPTS", 2))
    )

    # ------------------------------------------------------------ 多轮对话 / 上下文工程
    # 注入提示词的最近对话轮数（更早的轮次以"滚动摘要"形式出现）
    HISTORY_TURNS: int = int(os.getenv("HISTORY_TURNS", 3))
    # 对话背景块（摘要 + 最近几轮）注入提示词的字符预算
    MEMORY_CONTEXT_CHARS: int = int(os.getenv("MEMORY_CONTEXT_CHARS", 1500))
    # 对话背景里每条历史回答最多保留多少字符
    MEMORY_ANSWER_CHARS: int = int(os.getenv("MEMORY_ANSWER_CHARS", 300))
    # 生成答案时检索资料（本地片段 / 网页片段）的总字符预算；
    # 资料已按相关度排好序，超出预算时从尾部裁掉，生成与忠实性检查使用同一份资料
    CONTEXT_MAX_CHARS: int = int(os.getenv("CONTEXT_MAX_CHARS", 6000))

    # ------------------------------------------------------------ 记忆持久化
    # sqlite  本地文件，零部署【默认】
    # redis   复用 REDIS_URL，适合多实例部署
    # memory  进程内存（重启即丢，测试用）
    # off     不保存任何对话
    MEMORY_BACKEND: str = os.getenv("MEMORY_BACKEND", "sqlite").strip().lower()
    MEMORY_SQLITE_PATH: str = os.getenv(
        "MEMORY_SQLITE_PATH", os.path.join(VECTOR_DB_PATH, "chat_memory.sqlite3")
    )
    MEMORY_REDIS_PREFIX: str = os.getenv("MEMORY_REDIS_PREFIX", "rag:memory:v1")
    # 会话多久没有新消息就过期（秒），0 表示永不过期；默认 7 天
    MEMORY_TTL_SECONDS: int = int(os.getenv("MEMORY_TTL_SECONDS", 7 * 24 * 3600))
    # 原文最多保留多少轮；超过后把较早的轮次压缩进摘要，只保留最近 MEMORY_KEEP_TURNS 轮
    MEMORY_WINDOW_TURNS: int = int(os.getenv("MEMORY_WINDOW_TURNS", 8))
    MEMORY_KEEP_TURNS: int = int(os.getenv("MEMORY_KEEP_TURNS", 4))
    # 滚动摘要的长度上限（字符）
    MEMORY_SUMMARY_MAX_CHARS: int = int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", 600))
    # 每条回答写入存储时的长度上限（字符），防止单条超长答案撑大存储
    MEMORY_STORE_ANSWER_CHARS: int = int(os.getenv("MEMORY_STORE_ANSWER_CHARS", 2000))

    # ------------------------------------------------------------ LangSmith
    LANGSMITH_TRACING: bool = _env_bool("LANGSMITH_TRACING", "true")
    LANGSMITH_ENDPOINT: str = os.getenv(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    )
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "MyFirstProject")


    # ------------------------------------------------------------ Redis / LLM 缓存
    # 缓存挂在 LangChain 的模型层，因此 RAG 的查询变换、评分、答案生成以及
    # ResearchAgent 的推理调用都会自动复用；Redis 故障时会旁路，不阻断问答。
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    LLM_CACHE_ENABLED: bool = _env_bool("LLM_CACHE_ENABLED", "true")
    LLM_CACHE_TTL_SECONDS: int = int(os.getenv("LLM_CACHE_TTL_SECONDS", "3600"))
    LLM_CACHE_PREFIX: str = os.getenv("LLM_CACHE_PREFIX", "rag:llm-cache:v1")
    REDIS_SOCKET_TIMEOUT_SECONDS: float = float(
        os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS", "1.5")
    )
    LLM_CACHE_FAILURE_COOLDOWN_SECONDS: float = float(
        os.getenv("LLM_CACHE_FAILURE_COOLDOWN_SECONDS", "30")
    )

    # ------------------------------------------------------------ 派生属性
    @property
    def bm25_corpus_path(self) -> str:
        """BM25 语料副本的完整路径（随 VECTOR_DB_PATH 走）。"""
        return os.path.join(self.VECTOR_DB_PATH, self.BM25_CORPUS_FILE)

    @property
    def rerank_ready(self) -> bool:
        """精排是否真的可用（开关打开 + 有 Key）。"""
        if not self.RERANK_ENABLED:
            return False
        if self.RERANK_PROVIDER == "cohere":
            return bool(self.COHERE_API_KEY)
        return False

    @property
    def web_search_ready(self) -> bool:
        """联网搜索是否真的可用（开关打开 + 配了 Tavily Key）。"""
        return self.WEB_SEARCH_ENABLED and bool((self.TAVILY_API_KEY or "").strip())


settings = Settings()
