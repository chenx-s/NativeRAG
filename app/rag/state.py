"""状态图里流转的共享状态 GraphState。"""

from __future__ import annotations

from typing import List, Tuple, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """LangGraph 状态图里流转的「共享状态」。

    【这是什么】
        它不是普通类，而是 TypedDict —— 运行时就是**一个普通 dict**，
        这里的类型标注只给编辑器和类型检查器看，不会真的强制类型。

    【为什么用它】
        LangGraph 的每个节点都是 `def node(state) -> dict` 的形式：
          · 输入：当前完整的状态字典；
          · 输出：**只返回要更新的字段**（不用返回全部），
                  LangGraph 会自动把返回值合并回状态里，传给下一个节点。
        比如 generate 节点只 `return {"generation": 答案}`，
        其余字段原样保留 —— 这就是为什么节点函数不需要处理"其他字段怎么办"。

    【一个容易踩的坑】
        正因为是"合并"而不是"替换"，你在节点里**改了 state 的某个字段却没返回**，
        这个修改会在节点结束时被丢弃。想改什么就必须 `return` 什么。

    【字段分组】
        · 问题：question / original_question / standalone_question
        · 记忆：history / memory_summary
        · 路由与联网：allow_web / route / source_type / web_searched / web_error
        · 检索：queries / documents / candidates / reranked / transform_mode
        · 改写：rewritten / retries / rewrite_unchanged
        · 生成与忠实性：generation / skip_check / groundedness_* / strict_grounding / regenerate
    """

    # ---- 问题 ----
    question: str
    original_question: str
    # route_question 补全后的独立问题（写入记忆用；question 之后可能被改写成检索友好的形式）
    standalone_question: str

    # ---- 记忆（来自 app/memory.py）----
    # 多轮对话历史 [(用户问题, 助手回答), ...]（最近若干轮原文）
    history: List[Tuple[str, str]]
    # 更早轮次的滚动摘要
    memory_summary: str

    # ---- 路由与联网 ----
    # 本次请求是否允许联网（OpenAI 兼容接口的 "-local" 模型、RAGAs 评估会关掉）
    allow_web: bool
    # 入口路由结果：local=先查本地知识库 / web=直接联网
    route: str
    # 最终用于生成答案的上下文来源：local / web
    source_type: str
    # 本轮是否已经联网搜索过（保证最多联网一次）
    web_searched: bool
    # 联网失败原因（为空表示没失败）
    web_error: str

    # ---- 检索 ----
    # 本轮检索实际使用的查询集合（查询变换的产物，至少 1 条）
    queries: List[str]
    documents: List[Document]
    # 混合检索融合后的候选数量（精排前）
    candidates: int
    # 本轮是否真的执行了精排
    reranked: bool
    # 本轮使用的查询变换策略
    transform_mode: str

    # ---- 改写 ----
    # 是否执行过查询改写，用于对外返回执行信息
    rewritten: bool
    # 已执行的查询改写次数，用于限制最大重试次数
    retries: int
    # 本次改写结果是否和改写前的问题相同
    rewrite_unchanged: bool

    # ---- 生成与忠实性检查 ----
    generation: str
    # 兜底答案不需要进行忠实性检查
    skip_check: bool
    # 已触发"严格模式重生成"的次数
    groundedness_retries: int
    # 本次生成是否使用严格模式（check_groundedness 未通过时会置 True）
    strict_grounding: bool
    # 本次是否需要重新生成（供条件路由读取）
    regenerate: bool
    # 最后一次忠实性判定结果：yes / no / skipped / off / unknown
    groundedness_verdict: str
