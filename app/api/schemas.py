"""HTTP 接口的请求 / 响应模型（Pydantic）。集中定义，Swagger 文档也由这里生成。"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.api.runtime import SESSION_ID_PATTERN

# ================================================================ 问答


class QueryRequest(BaseModel):
    """`POST /query`、`POST /query/stream` 的请求体。

    多轮对话两种用法（二选一，同时传时以 session_id 为准）：
        session_id  服务端记忆：同一个 ID 的历史自动加载、自动保存（持久化，重启不丢）
        history     客户端自带历史，服务端不保存
    """

    question: str = Field(..., min_length=1, description="用户问题")
    session_id: Optional[str] = Field(
        None,
        pattern=SESSION_ID_PATTERN,
        description="会话 ID（字母/数字/_.:-，最长 128）；传了就启用服务端持久化记忆",
    )
    history: List[List[str]] = Field(
        default_factory=list,
        description="可选的多轮历史 [[问题, 回答], ...]，用于补全追问；不传则单轮无状态",
    )
    allow_web: bool = Field(True, description="是否允许联网搜索（false = 只查本地知识库）")

    def history_pairs(self) -> List[tuple]:
        return [(turn[0], turn[1]) for turn in self.history if len(turn) >= 2]


class SourceInfo(BaseModel):
    """单条引用来源。

    本地知识库：source=文件路径，filename=文件名，page=页码，doc_id=文档 ID；
    联网搜索：  source=网页 URL，filename=网页标题，page=None。
    """

    source: str
    filename: str
    page: Optional[int] = None
    snippet: str
    # 精排分数（未启用精排时为 None；联网结果为 Tavily 相关度）；RRF 融合分（单路检索时为 None）
    relevance_score: Optional[float] = None
    rrf_score: Optional[float] = None
    source_type: str = Field("local", description="local=本地知识库 / web=互联网")
    doc_id: str = Field("", description="本地文档 ID，可用 GET /documents/{doc_id}/file 打开原文")


class QueryResponse(BaseModel):
    """`POST /query` 的响应体。

    `answer` / `sources` 是给用户看的；其余是观测字段，答案不对时用来定位是哪一级出了问题。
    """

    answer: str
    question: str
    rewritten: bool
    num_documents: int
    sources: List[SourceInfo]

    queries: List[str] = Field(default_factory=list, description="本轮实际用于检索的查询")
    transform_mode: str = Field("none", description="查询变换策略与产出条数")
    candidates: int = Field(0, description="混合检索融合后的候选数（精排前）")
    reranked: bool = Field(False, description="本轮是否执行了精排")

    groundedness: str = Field("unknown", description="忠实性自检结果")
    groundedness_retries: int = Field(0, description="触发严格模式重生成的次数")

    route: str = Field("local", description="入口路由：local=先查本地 / web=直接联网")
    source_type: str = Field("local", description="答案依据的来源：local / web")
    web_searched: bool = Field(False, description="本轮是否执行了联网搜索")
    web_error: str = Field("", description="联网搜索失败原因（为空表示没失败）")
    steps: List[str] = Field(default_factory=list, description="本轮实际执行的图节点序列")

    session_id: str = Field("", description="本轮使用的会话 ID（未启用服务端记忆时为空）")
    memory_turns: int = Field(0, description="本轮注入的历史轮数（原文部分）")
    memory_summary_used: bool = Field(False, description="本轮是否使用了较早对话的滚动摘要")


class RetrieveRequest(BaseModel):
    """`POST /retrieve` 的请求体（调试接口，只检索不生成）。"""

    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=50, description="返回条数，默认取 FUSION_TOP_K")
    skip_rerank: bool = Field(False, description="跳过精排，用于对比精排前后的差异")


class RetrieveResponse(BaseModel):
    """`POST /retrieve` 的响应体：只有检索结果，没有答案，用来单独检验检索层。"""

    question: str
    queries: List[str]
    candidates: int
    reranked: bool
    results: List[SourceInfo]


class SessionResponse(BaseModel):
    """`GET /sessions/{session_id}` 的响应体：查看某个会话当前保存的记忆。"""

    session_id: str
    backend: str
    summary: str
    turns: List[List[str]]
    updated_at: float


# ================================================================ 文档管理


class DocumentInfo(BaseModel):
    """一份文档的登记信息与入库状态。"""

    doc_id: str
    filename: str
    source_key: str = Field(..., description="文档身份：upload:<文件名> / dir:<相对路径>")
    origin: str = Field(..., description="upload=上传 / directory=目录扫描")
    sha256: str
    size: int
    status: str = Field(..., description="pending / processing / ready / failed")
    chunks: int
    error: str = ""
    replaces: str = Field("", description="本文档替换的旧版本 doc_id（入库成功后旧版本被删除）")
    created_at: float
    updated_at: float

    @classmethod
    def from_record(cls, record) -> "DocumentInfo":
        data = record.to_dict()
        data.pop("path", None)  # 不对外暴露服务器上的文件路径
        return cls(**data)


class UploadResponse(BaseModel):
    action: str = Field(..., description="queued=已排队入库 / duplicate=内容相同的文档已存在，跳过")
    document: DocumentInfo


class DocumentListResponse(BaseModel):
    total: int
    queue_pending: int = Field(..., description="后台队列中等待处理的任务数")
    documents: List[DocumentInfo]


class ScanResponse(BaseModel):
    queued: int
    duplicates: int
    documents: List[DocumentInfo]


class IngestResponse(BaseModel):
    """旧接口 `/ingest/file`、`/ingest/directory` 的响应体（保持兼容）。"""

    status: str
    chunk_indexed: int
    total_chunks: int
    detail: str = ""
