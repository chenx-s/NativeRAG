"""Agentic RAG 问答链（LangGraph 实现，内置联网搜索 / CRAG 纠错检索）
完整流程图：

                      route_question（补全追问 + 判断是否需要实时信息）
                        |                        |
                   (local)                    (web)
                        v                        |
    transform_query -> retrieve -> rerank_documents -> grade_document
           ^                                        |        |
           |                    (无相关文档且可重试)  |        | (无相关文档、改写额度用尽)
      rewrite_query <-------------------------------+        v
           |                                             web_search  <---+
           +--(改写无效)--------------------------------------^          |
                                                             |          |
                     generate <----(有相关文档 / 联网结果)-----+----------+
                       ^   |
                       |   v
                       |  check_groundedness
                       +---+ (未通过 + 还有额度 -> 严格模式重生成)
                           |
                           +--(通过 / 额度用尽)--> END

为什么联网搜索是节点而不是 Agent 工具？
    "什么时候联网"只有两种确定的情况：问题依赖实时信息（入口路由判断一次），
    或本地检索评分后没有相关文档（纯规则判断）。用条件边表达，路径确定、
    LLM 调用次数有上限、可单测；不需要再套一层 ReAct Agent 让 LLM 每轮推理。

为什么是这个顺序？
    查询变换放在最前面：问法不好，后面再怎么精排也救不回来。
    精排放在文档评分之前：评分要对每个文档调一次 LLM，先精排把 20 条砍到 5 条，
    评分调用次数直接降 4 倍，整条链路的延迟和 token 成本都跟着降。
    这也正是 LlamaIndex "先宽召回、再精排、后处理" 的标准分层。

"""

from __future__ import annotations

import asyncio
import os
import re
import unicodedata
from typing import List, Optional, Sequence, Tuple, TypedDict

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.config import settings
from app.context import conversation_or_placeholder, fit_documents, render_conversation
from app.llm_cache import get_llm_cache
from app.memory import MemoryManager
from app.vector_store import acount, amulti_query_search, warmup
from app.web_search import WebSearchError, aclose_web_client, asearch_web

# ================================================================ LLM


def get_llm(**kwargs) -> ChatOpenAI:

    return ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=settings.LLM_TEMPERATURE,
        cache=get_llm_cache(),
        extra_body={"thinking": {"type": "disabled"}},
        **kwargs,
    )


# ================================================================ 图状态


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
        · 问题相关：question / original_question
        · 检索相关：queries / documents / candidates / reranked
        · 生成与改写：generation / rewritten / retries / rewrite_unchanged
        · 查询变换：transform_mode
        · 忠实性检查：groundedness_verdict / groundedness_retries /
                      strict_grounding / regenerate / skip_check

    字段含义见下方逐行注释。
    """

    question: str
    original_question: str

    # 多轮对话历史 [(用户问题, 助手回答), ...]（最近若干轮原文）
    history: List[Tuple[str, str]]
    # 更早轮次的滚动摘要（来自持久化记忆，见 app/memory.py）
    memory_summary: str
    # route_question 补全后的独立问题（写入记忆用；question 之后可能被改写成检索友好的形式）
    standalone_question: str
    # 入口路由结果：local=先查本地知识库 / web=直接联网
    route: str
    # 最终用于生成答案的上下文来源：local / web
    source_type: str
    # 本轮是否已经联网搜索过（保证最多联网一次）
    web_searched: bool
    # 联网失败原因（为空表示没失败）
    web_error: str

    # [新增] 本轮检索实际使用的查询集合（查询变换的产物，至少 1 条）
    queries: List[str]

    documents: List[Document]
    generation: str

    # 是否执行过查询改写，用于对外返回执行信息
    rewritten: bool
    # 已执行的查询改写次数，用于限制最大重试次数
    retries: int
    # 本次改写结果是否和改写前的问题相同
    rewrite_unchanged: bool

    # [新增] 混合检索融合后的候选数量（精排前）
    candidates: int
    # [新增] 本轮是否真的执行了精排
    reranked: bool
    # [新增] 本轮使用的查询变换策略
    transform_mode: str

    # 兜底答案不需要进行忠实性检查
    skip_check: bool

    # [新增] 忠实性检查相关
    # 已触发"严格模式重生成"的次数
    groundedness_retries: int
    # 本次生成是否使用严格模式（check_groundedness 未通过时会置 True）
    strict_grounding: bool
    # 本次是否需要重新生成（供条件路由读取）
    regenerate: bool
    # 最后一次忠实性判定结果：yes / no / skipped / off / unknown
    groundedness_verdict: str


# ================================================================ Prompt

DOC_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个评估检索文档与用户问题相关性的评分器。
    不需要严格的测试，目标是过滤掉明显无关的检索结果。
    如果文档包含与问题相关的关键词或语义信息，请评级为 yes；
    否则评级为 no。
    你只能输出 yes 或 no 两个单词之一，不要输出任何其他内容。""",
    ),
    (
        "human",
        "检索的文档：\n\n{document}\n\n用户问题:{question}",
    ),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个查询改写器。将输入问题改写为更合适向量检索的版本：
    补全关键词、消除指代模糊、保留原意。
    只输出改写后的问题，不要输出其他内容。""",
    ),
    (
        "human",
        "初始问题:\n\n{question}\n\n改写后的问题:",
    ),
])

# [新增] RAG-Fusion 的查询裂变器
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个检索查询生成器。请针对用户问题，生成 {num_queries} 个不同角度的检索查询，
    用于从知识库中召回尽可能全面的相关文档。

    要求：
    1. 每个查询覆盖问题的不同侧面（同义词、不同表述、上位概念、具体细节）。
    2. 每个查询必须能独立成句、可单独用于检索，不要出现"它""这个"等指代。
    3. 使用与问题相同的语言。
    4. 每行一个查询，用阿拉伯数字编号，不要输出任何解释、标题或多余文字。""",
    ),
    (
        "human",
        "用户问题：{question}",
    ),
])

# [新增] HyDE：让模型先"编"一段答案，再用这段假设答案去检索
HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """请针对下面的问题，写一段大约 150 字的段落，像知识库中真实的资料那样直接陈述答案。

    要求：
    1. 使用与问题相同的语言，使用陈述句、专业表述。
    2. 不要写"我不知道""可能""根据资料"这类推脱或元话语。
    3. 直接输出段落本身，不要任何标题、解释或前后缀。""",
    ),
    (
        "human",
        "问题：{question}",
    ),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个问答助手，请仅使用下方【资料】中的上下文片段回答问题。
    如果上下文不足以回答问题，请直接说“根据知识库内容无法回答该问题”，不要编造。
    请用中文回答，保持简洁准确。
    在合适的位置标注出处，例如[文件名,第x页]。
    涉及数字、单位、比例时，请**原样引用**上下文中的数值，不要自行换算或改写
    （例如上下文写 70 billion，就在括号里保留原值，不要换算成中文数量级）。
{conversation_rules}""",
    ),
    (
        "human",
        "【对话背景】\n{conversation}\n\n问题：{question}\n\n【资料】\n\n{context}\n\n回答：",
    ),
])

# [新增] 严格模式生成：忠实性检查没通过时，用这个提示词重生成一次。
# 思路来自 rag-from-scratch 第 18 课（Self-RAG）：不让模型重答一遍，
# 而是明确告诉它"上一版没通过校验"，并要求它只保留能在原文中找到依据的内容。
GENERATE_STRICT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个严格依据资料作答的问答助手。

    要求：
    1. 只输出下方上下文中**明确写出**的内容。上下文没写的信息，一律不要写。
    2. 不要补充你的背景知识，不要做任何推断、联想或举例。
    3. 数字、单位、专有名词一律**原样照抄**，不要换算、不要改写数量级。
       如果中文表达会引起歧义，就用「中文说明（原文英文）」，例如
       “参数量从 70 亿到 700 亿（原文：7 billion to 70 billion）”。
    4. 上下文只能部分回答问题时，就只回答能回答的部分，并说明其余部分资料未提及。
    5. 上下文完全无法回答时，直接说“根据现有资料无法回答该问题”。
    6. 用中文回答，在合适位置标注出处：知识库片段用[文件名,第x页]，网页片段用[网页标题](URL)。
{conversation_rules}""",
    ),
    (
        "human",
        "【对话背景】\n{conversation}\n\n问题：{question}\n\n【资料】\n\n{context}\n\n回答：",
    ),
])

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个检查答案是否由一组事实支撑的评分器。

评级为 yes 的情况（以下都算"被支撑"，不要因此判 no）：
1. 答案的核心事实内容能够由给定事实推导、翻译或包含于其中。
2. 答案与事实**语言不同**。中文答案对应英文事实是正常情况，
   翻译、转述、同义改写一律视为被支撑，语言差异本身不是问题。
3. 答案对多条事实做了归纳、合并、排序或总结 —— 这是正常的信息加工。
4. 答案带有出处标注（如 [文件名,第x页]）、格式符号、语气词、礼貌性措辞、
   过渡句和结构性说明文字。

只有在答案中出现**与事实矛盾**，或**事实中毫无依据的具体信息**
（具体数字、人名、机构名、时间、结论、因果关系）时，才评级为 no。

【关于数字 —— 这是最容易误判的地方，请特别小心】
比较数字前，先把两边换算到同一单位，再判断是否矛盾。中英数量级换算关系：

    1 thousand = 1 千          1 million  = 1 百万（100万）
    1 billion  = 10 亿         1 trillion = 1 万亿

对照示例（事实为英文 "7 billion to 70 billion parameters"）：

    “参数量从 70 亿到 700 亿”      -> 正确    （7 billion=70亿，70 billion=700亿）
    “参数量从 7 billion 到 70 billion” -> 正确（原样照抄）
    “参数量从 70 亿到 70 亿”        -> 错误    （上限算错了一个数量级）

换算后数值一致就判 yes，**不要因为中英文写法不同、或数量级看上去"变大了"而判 no**。

请只输出 yes 或 no 一个单词，不要输出任何其他内容。""",
    ),
    (
        "human",
        "事实:\n\n{documents}\n\n答案:\n\n{generation}",
    ),
])

FALLBACK_ANSWER = (
    "抱歉，在当前知识库中没有找到与该问题相关的信息。"
    "请先上传文档，或换个问法试试。"
)

WEB_FALLBACK_ANSWER = (
    "抱歉，我查了本地知识库和互联网，暂时没有找到与该问题相关的可靠信息。"
    "可以换个问法，或补充更具体的关键词再试。"
)

# [新增] 入口路由：一次 LLM 调用同时完成"追问补全"和"是否需要实时信息"两件事。
# 拿不准时选 local 是安全的 —— 本地检索失败后图会自动降级为联网，不会漏答。
ROUTE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是问答系统的入口路由器，需要完成两件事：

    1. 补全问题：如果有对话历史，且当前问题含有指代或省略（如"它""那个""再详细点""第二点呢"），
       结合历史把它改写成一个可以独立理解的完整问题；否则原样保留，不要改动。
    2. 选择数据源：
       - web：回答必须依赖实时或最新的公开信息，例如新闻、天气、股价、汇率、比赛结果、
              "今天/现在/最近/最新"的动态、最新版本号或发布信息。
       - local：概念解释、专业知识、文档/论文/笔记内容、人物或事件的背景资料等，
                以及所有拿不准的情况。

    严格按下面两行格式输出，不要输出任何其他内容：
    路由：web 或 local
    问题：补全后的独立问题""",
    ),
    (
        "human",
        "对话历史：\n{history}\n\n当前问题：{question}",
    ),
])

# [新增] 基于联网搜索结果生成答案
WEB_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一个问答助手，请仅使用下方提供的【联网搜索结果】回答问题。
    要求：
    1. 先用一两句话给出结论，再展开必要细节，用中文回答，保持简洁准确。
    2. 每条关键信息后标注来源，格式为[网页标题](URL)。
    3. 搜索结果不足以回答时，直接说“联网搜索也没有找到可靠答案”，不要编造，
       也不要用你的背景知识补全没有依据的细节。
    4. 多个来源互相矛盾时，指出矛盾并说明各自出处。
    5. 涉及时间敏感的信息（价格、排名、版本等），说明这是搜索结果中的信息，可能随时间变化。
{conversation_rules}""",
    ),
    (
        "human",
        "【对话背景】\n{conversation}\n\n问题：{question}\n\n【联网搜索结果】\n\n{context}\n\n回答：",
    ),
])

# [新增] 上下文工程 —— 隔离规则：对话背景和资料分区放置，并明确各自的用途。
# 只有存在对话背景时才注入这段规则，单轮问答的提示词保持精简。
CONVERSATION_RULES = """
    关于【对话背景】：
    - 它只用于理解指代和省略（如"它""上面说的第二点"）、保持前后回答一致、避免重复已经说过的内容。
    - 新的事实性信息必须来自【资料】，不能把对话背景当作资料来源。
    - 只有当用户明确问到此前回答的内容（如"总结一下你刚才说的"）时，才可以复述对话背景里的回答。"""


# ================================================================ Reranker


def get_reranker():
    """构建精排器（Cross-Encoder Reranker）。

    精排和向量检索的本质区别：
        向量检索（双塔/Bi-Encoder）把问题和文档**分别**编码成向量再算相似度，
        快，但两者从未"见面"；
        精排（交叉编码器/Cross-Encoder）把 [问题, 文档] **拼在一起**送进模型，
        每个词都能和对方的词做交互注意力，准确率高得多，代价是只能逐条打分。
        所以标准打法是：向量检索负责"从百万里捞出一百"，精排负责"从一百里挑出五"。

    """
    if not settings.RERANK_ENABLED:
        return None

    if settings.RERANK_PROVIDER != "cohere":
        print(f"    [warn] 未知的 RERANK_PROVIDER={settings.RERANK_PROVIDER}，跳过精排")
        return None

    if not settings.COHERE_API_KEY:
        print(
            "    [warn] 未配置 COHERE_API_KEY，跳过精排。"
            "申请地址：https://dashboard.cohere.com/api-keys（Rerank 有免费额度）"
        )
        return None

    # langchain-cohere 各版本读取 Key 的方式不一致，先把环境变量兜住
    os.environ.setdefault("COHERE_API_KEY", settings.COHERE_API_KEY)

    try:
        try:
            from langchain_cohere import CohereRerank
        except ImportError:
            from langchain_cohere.rerank import CohereRerank  # 兜底旧路径

        try:
            return CohereRerank(
                model=settings.COHERE_RERANK_MODEL,
                top_n=settings.RERANK_TOP_N,
            )
        except TypeError:
            # 老版本需要显式传 cohere_api_key
            return CohereRerank(
                model=settings.COHERE_RERANK_MODEL,
                top_n=settings.RERANK_TOP_N,
                cohere_api_key=settings.COHERE_API_KEY,
            )
    except Exception as error:
        print(f"    [warn] 精排器初始化失败，跳过精排：{error}")
        return None


# ================================================================ 辅助函数


def normalize_question(question: str) -> str:
    """规范化问题，用于判断查询改写前后是否发生实际变化。

    处理规则：
    1. 使用 NFKC 统一全角和半角字符。
    2. 忽略空格、换行符等空白字符。
    3. 忽略英文字符的大小写。
    4. 忽略句末的常见中英文标点。

    """
    normalized = unicodedata.normalize("NFKC", question)
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.casefold()
    normalized = normalized.rstrip("。！？? ! ;  ； ， ,")
    return normalized


def _parse_yes_no(text: str, default: bool) -> bool:
    """把 LLM 输出解析为布尔值；解析不出时返回 default。

    说明：GLM 对 with_structured_output 的 JSON 模式遵循不稳定
    （会输出裸 yes/no 而非 JSON），因此评分器统一使用
    纯文本 yes/no + 字符串解析，兼容性最好。
    """
    normalized = (text or "").strip().lower()
    if normalized.startswith("yes"):
        return True
    if normalized.startswith("no"):
        return False
    return default


_LIST_PREFIX = re.compile(r"^\s*(?:\d+\s*[\.\)、:：]|[-*•·])\s*")


def _parse_query_list(text: str, limit: int) -> List[str]:
    """把 LLM 输出的多行查询解析成列表。

    容错处理：兼容 "1. xxx" / "1) xxx" / "- xxx" / "• xxx" 等编号形式，
    以及模型偶尔多写一句开场白的情况（长度异常的整句会被丢掉）。
    """
    queries: List[str] = []
    seen = set()

    for raw_line in (text or "").splitlines():
        line = _LIST_PREFIX.sub("", raw_line).strip().strip('"“”')
        if not line or len(line) > 200:
            continue
        # 模型有时会把整个问题再抄一遍，去重时用规范化结果
        key = normalize_question(line)
        if not key or key in seen:
            continue
        seen.add(key)
        queries.append(line)
        if len(queries) >= limit:
            break

    return queries


def _doc_grader():
    """创建文档相关性评分链。"""
    return DOC_GRADE_PROMPT | get_llm() | StrOutputParser()


def _groundedness_grader():
    """创建答案忠实性评分链。"""
    return GROUNDEDNESS_PROMPT | get_llm() | StrOutputParser()


def _format_context(documents: List[Document]) -> str:
    """把检索到的文档块拼成带出处的上下文（本地片段带文件名/页码，网页片段带标题/URL）。"""
    parts = []
    for index, document in enumerate(documents):
        metadata = document.metadata
        if metadata.get("source_type") == "web":
            title = metadata.get("filename") or "无标题网页"
            url = metadata.get("source", "")
            parts.append(
                f"[片段{index + 1} | 网页：{title} | {url}]\n{document.page_content}"
            )
            continue

        filename = metadata.get("filename", metadata.get("source", "未知"))
        page = metadata.get("page")
        page_text = f",第{page + 1}页" if page is not None else ""
        parts.append(
            f"[片段{index + 1} | {filename}{page_text}]\n{document.page_content}"
        )
    return "\n\n".join(parts)


def _render_conversation_for(state: dict) -> str:
    """上下文工程 —— 选择 + 压缩：把记忆渲染成受预算约束的对话背景块（可能为空）。"""
    return render_conversation(
        state.get("memory_summary", ""),
        state.get("history") or [],
        max_turns=settings.HISTORY_TURNS,
        max_chars=settings.MEMORY_CONTEXT_CHARS,
        answer_chars=settings.MEMORY_ANSWER_CHARS,
    )


_ROUTE_LINE = re.compile(r"路由\s*[:：]\s*(web|local)", re.IGNORECASE)
_QUESTION_LINE = re.compile(r"问题\s*[:：]\s*(.+)")


def _parse_route(text: str) -> Tuple[Optional[str], Optional[str]]:
    """解析路由器输出的两行文本；解析不出的部分返回 None。"""
    text = text or ""
    route_match = _ROUTE_LINE.search(text)
    question_match = _QUESTION_LINE.search(text)
    route = route_match.group(1).lower() if route_match else None
    question = question_match.group(1).strip() if question_match else None
    if question and len(question) > 500:
        question = None
    return route, question


def _local_rewrite_budget() -> int:
    """本地"改写重检索"的次数上限。

    联网可用时，本地失败应尽快转向联网（默认不改写直接联网）；
    联网不可用时，改写重检索是唯一的补救手段，按 MAX_RETRIES 来。
    """
    if settings.web_search_ready:
        return max(0, min(settings.MAX_RETRIES, settings.WEB_FALLBACK_AFTER_RETRIES))
    return max(0, settings.MAX_RETRIES)


def _can_web_search(state: dict) -> bool:
    """联网可用，且本轮还没联网过（保证每次请求最多联网一次）。"""
    return settings.web_search_ready and not state.get("web_searched")


def required_recursion_limit() -> int:
    """按当前配置估算图最多会执行多少步，用来兜底 recursion_limit。

    route(1) + 每轮本地检索 transform/retrieve/rerank/grade/rewrite(5) × (MAX_RETRIES+1)
    + web_search(1) + (generate + check) × (MAX_GROUNDEDNESS_RETRIES+1)，再留一点余量。
    """
    local_rounds = 5 * (max(0, settings.MAX_RETRIES) + 1)
    generation = 2 * (max(0, settings.MAX_GROUNDEDNESS_RETRIES) + 1)
    return 1 + local_rounds + 1 + generation + 5


# ================================================================ 节点函数


async def route_question(state: GraphState) -> dict:
    """入口节点：补全追问 + 决定先查本地还是直接联网。

    只在真正需要时才调 LLM：
      · 有对话历史                         -> 需要把追问补全成独立问题
      · 联网可用、知识库非空、WEB_ROUTING=llm -> 需要判断是否依赖实时信息
    两者都不满足时零 LLM 调用，直接走本地检索。
    知识库为空且联网可用时，直接联网（本地查了也是白查）。
    """
    question = state["question"]
    conversation = _render_conversation_for(state)
    web_ready = settings.web_search_ready

    kb_empty = False
    if web_ready:
        try:
            kb_empty = (await acount()) == 0
        except Exception:  # noqa: BLE001
            kb_empty = False

    need_routing = web_ready and not kb_empty and settings.WEB_ROUTING == "llm"
    route = "local"
    standalone = question

    if need_routing or conversation:
        try:
            chain = ROUTE_PROMPT | get_llm() | StrOutputParser()
            raw = await chain.ainvoke({
                "question": question,
                "history": conversation_or_placeholder(conversation),
            })
            parsed_route, parsed_question = _parse_route(raw)
            if parsed_question:
                standalone = parsed_question
            if need_routing and parsed_route:
                route = parsed_route
        except Exception as error:  # noqa: BLE001
            print(f"    [warn] 路由判断失败，默认先查本地知识库：{error}")

    if web_ready and kb_empty:
        route = "web"

    print(f"--->[节点]route_question: route={route}" + (
        f"，补全后问题：{standalone}" if standalone != question else ""
    ))
    return {"question": standalone, "standalone_question": standalone, "route": route}


async def web_search(state: GraphState) -> dict:
    """联网搜索节点（CRAG 的纠错检索）。

    两个入口：route_question 判定需要实时信息；或本地检索没有相关文档。
    搜索失败不抛异常 —— 记录原因、返回空文档，由 generate 给出兜底答案。
    """
    question = state["question"]
    print(f"--->[节点]web_search: {question}")

    try:
        documents = await asearch_web(question)
        error = ""
    except WebSearchError as exc:
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


async def transform_query(state: GraphState) -> dict:
    """节点 0：查询变换。

    - none        直接用原问题（等价于升级前的行为）
    - multi_query RAG-Fusion：把一个问题裂变成 N 个不同角度的查询。
                  单一问法只能命中一种表述的文档；多问法各自检索后 RRF 融合，
                  召回率显著上升。这就是 RAG-Fusion 论文的核心思想。
    - hyde        Hypothetical Document Embeddings：让 LLM 先"编造"一段
                  假想答案，再用假想答案去检索。原理是"答案和答案"的语义距离，
                  比"问题和答案"更近——用问题去检索是跨分布匹配，
                  用假设答案去检索是同分布匹配。

    """
    question = state["question"]
    mode = settings.QUERY_TRANSFORM
    print(f"--->[节点]transform_query: mode={mode}")

    if mode == "none":
        return {"queries": [question], "transform_mode": "none"}

    if mode == "hyde":
        try:
            chain = HYDE_PROMPT | get_llm() | StrOutputParser()
            hypothetical = (await chain.ainvoke({"question": question})).strip()
        except Exception as error:
            print(f"    [warn] HyDE 生成失败，退回原问题：{error}")
            hypothetical = ""

        if not hypothetical:
            return {"queries": [question], "transform_mode": "none"}

        print(f"    HyDE 假设文档（前 60 字）：{hypothetical[:60]}...")
        # 纯 HyDE 只用假设文档检索。若发现效果不稳，可改成 [question, hypothetical]
        # 让原问题和假设文档并集检索，RRF 会自己挑出更有效的那一路。
        return {"queries": [hypothetical], "transform_mode": "hyde"}

    # 默认：multi_query（RAG-Fusion）
    try:
        chain = MULTI_QUERY_PROMPT | get_llm() | StrOutputParser()
        raw = await chain.ainvoke(
            {"question": question, "num_queries": settings.NUM_QUERIES}
        )
        variants = _parse_query_list(raw, limit=settings.NUM_QUERIES)
    except Exception as error:
        print(f"    [warn] 多查询生成失败，退回原问题：{error}")
        variants = []

    if not variants:
        return {"queries": [question], "transform_mode": "none"}

    # 原问题永远保留：模型生成的查询可能有偏，原问题是唯一"保底"的那一路
    queries = [question] + [v for v in variants if normalize_question(v) != normalize_question(question)]
    for index, query in enumerate(queries):
        print(f"   [{index + 1}] {query}")

    return {"queries": queries, "transform_mode": f"multi_query({len(queries)})"}


async def retrieve(state: GraphState) -> dict:
    """节点 1：混合检索（稠密向量 ⊕ BM25 稀疏，RRF 融合）。

    这里刻意"召回宁宽勿窄"（默认 20 条），因为后面紧跟精排。
    粗排负责别漏，精排负责排准，两者分工明确。
    """
    queries = state.get("queries") or [state["question"]]
    print(f"--->[节点]retrieve: {len(queries)} 条查询 × 混合检索")

    try:
        documents = await amulti_query_search(queries, k=settings.FUSION_TOP_K)
    except Exception as error:  # noqa: BLE001
        # 向量库打不开 / Embedding 加载失败时不让整条链路崩掉：
        # 返回空结果，后续路由会自动降级为联网搜索（若已配置）
        print(f"    [warn] 本地检索失败，按无结果处理：{error}")
        documents = []
    print(f"   融合后候选：{len(documents)} 条（精排前）")

    return {
        "documents": documents,
        "question": state["question"],
        "candidates": len(documents),
        "source_type": "local",
    }


async def rerank_documents(state: GraphState) -> dict:
    """节点 2：交叉编码器精排。

    Cohere Rerank（也是 LlamaIndex 官方推荐的 Node Postprocessor）：
    输入 [问题, 20 条候选]，输出按真实相关度重排后的 top_n 条。

    为什么值得单独加一个模型？
        RRF 融合只用了"排名"信息，是启发式的；
        精排是真正把问题和文档放在一起算过一次，是模型层面的判断。
        实践中"向量检索 + Rerank"通常能把 Top-5 命中率提升 10~30 个百分点，
        是投入产出比最高的一步。

    失败降级：没配 Key / 网络不通 / 依赖缺失 -> 原样透传，链路不受影响。
    """
    documents = state.get("documents") or []
    question = state["question"]

    if not documents:
        return {"reranked": False}

    if not settings.rerank_ready:
        print("--->[节点]rerank_documents: 精排未启用，跳过")
        return {"reranked": False}

    if len(documents) <= settings.RERANK_TOP_N:
        print("--->[节点]rerank_documents: 候选数已不超过 top_n，跳过")
        return {"reranked": False}

    compressor = get_reranker()
    if compressor is None:
        return {"reranked": False}

    print(f"--->[节点]rerank_documents: {len(documents)} 条候选 -> top {settings.RERANK_TOP_N}")

    try:
        # 注意：精排的 query 用"当前问题"而不是变换出来的查询。
        # 变换查询是为了"捞得全"，精排要判断的是"跟用户真正想问的有多相关"。
        reranked = await asyncio.to_thread(
            compressor.compress_documents, documents, question
        )
    except Exception as error:
        print(f"    [warn] 精排调用失败，保留融合结果：{error}")
        return {"reranked": False}

    if not reranked:
        print("    [warn] 精排返回空结果，保留融合结果")
        return {"reranked": False}

    kept = list(reranked)[: settings.RERANK_TOP_N]
    print("   精排后 top 分数：" + ", ".join(
        f"{doc.metadata.get('relevance_score', 0):.3f}" for doc in kept
    ))

    return {"documents": kept, "reranked": True}


async def grade_documents(state: GraphState) -> dict:
    """节点 3：逐条评估检索文档，过滤明显无关的内容。

    精排已经砍到 5 条左右，这里的 LLM 调用次数因此大幅下降。
    """
    print(f"--->[节点]grade_documents: 共{len(state['documents'])}个文档块")
    if not state["documents"]:
        return {"documents": []}

    question = state["question"]
    grader = _doc_grader()
    async def grade_one(document: Document) -> tuple[Document, bool]:
        try:
            score_text = await grader.ainvoke({
                "question": question,
                "document": document.page_content[:2000],
            })
            return document, _parse_yes_no(score_text, default=True)
        except Exception as error:
            # 评分服务发生异常时，保守地保留文档，
            # 避免因为评分失败而丢失可能有用的上下文。
            print(f"    [warn]评分失败，保留该文档：{error}")
            return document, True

    # 文档评分互不依赖，并发请求可显著降低 RERANK_TOP_N > 1 时的总延迟。
    graded = await asyncio.gather(*(grade_one(doc) for doc in state["documents"]))
    filtered_documents = [document for document, keep in graded if keep]

    print(f"   相关文档数:{len(filtered_documents)}")

    return {"documents": filtered_documents}


async def rewrite_query(state: GraphState) -> dict:
    """节点 4：将当前问题改写为更合适向量检索的形式。

    如果改写结果与当前问题相同，则设置 rewrite_unchanged=True。
    后续路由会停止重新检索，避免对相同问题反复执行检索。
    """
    print("--->[节点]rewrite_query")

    current_question = state["question"]
    rewrite_chain = REWRITE_PROMPT | get_llm() | StrOutputParser()

    try:
        rewritten_question = (await rewrite_chain.ainvoke({
            "question": current_question,
        })).strip()
        # 当模型返回空字符串时，当作没有改写成功。
        if not rewritten_question:
            rewritten_question = current_question
    except Exception as error:
        print(f"    [warn]查询改写失败，沿用原问题：{error}")
        rewritten_question = current_question

    rewrite_unchanged = (
        normalize_question(rewritten_question)
        == normalize_question(current_question)
    )
    if rewrite_unchanged:
        print("    [warn]改写结果没有变化，停止重复检索")
        rewritten_question = current_question
    else:
        print(f"   改写前：{current_question}")
        print(f"   改写后：{rewritten_question}")

    return {
        "question": rewritten_question,
        "rewritten": True,
        "rewrite_unchanged": rewrite_unchanged,
        "retries": state.get("retries", 0) + 1,
        # 问题变了，之前那批查询作废；
        # 路由会回到 transform_query 用新问题重新生成查询。
        "queries": [],
    }


async def generate(state: GraphState) -> dict:
    """节点 5：根据相关文档生成答案。

    如果没有相关文档，则直接返回兜底拒答。
    """
    if not state["documents"]:
        print("--->[节点]generate: 无相关文档，返回兜底答案")
        fallback = WEB_FALLBACK_ANSWER if state.get("web_searched") else FALLBACK_ANSWER
        if state.get("web_error"):
            fallback += f"\n（联网搜索不可用：{state['web_error']}）"
        return {
            "generation": fallback,
            "skip_check": True,
        }

    # strict_grounding 为 True 表示：上一版答案没通过忠实性检查，
    # 这次改用严格提示词重生成（只准照抄上下文，不准补充背景知识）。
    strict = bool(state.get("strict_grounding"))
    is_web = state.get("source_type") == "web"
    if strict:
        prompt = GENERATE_STRICT_PROMPT
    else:
        prompt = WEB_GENERATE_PROMPT if is_web else GENERATE_PROMPT
    mode = "严格模式重生成" if strict else "正常生成"

    print(
        f"--->[节点]generate（{mode}）: 基于{len(state['documents'])}个"
        f"{'网页' if is_web else '文档'}片段生成"
    )

    # 上下文工程 —— 压缩：资料按字符预算裁剪（已按相关度排序，从尾部裁）；
    # 忠实性检查用同一个函数裁剪，保证"生成看到的"和"核对依据的"是同一份资料
    documents = fit_documents(state["documents"], settings.CONTEXT_MAX_CHARS)
    conversation = _render_conversation_for(state)

    context = _format_context(documents)
    generate_chain = prompt | get_llm() | StrOutputParser()

    try:
        answer = await generate_chain.ainvoke({
            "question": state["question"],
            "context": context,
            "conversation": conversation_or_placeholder(conversation),
            # 隔离规则只在确实有对话背景时注入
            "conversation_rules": CONVERSATION_RULES if conversation else "",
        })
    except Exception as error:
        print(f"    [warn]生成失败：{error}")
        return {
            "generation": FALLBACK_ANSWER,
            "skip_check": True,
            "regenerate": False,
        }

    return {
        "generation": answer.strip(),
        "skip_check": False,
        # 必须显式复位，否则上一轮的 True 会一直留在状态里
        "regenerate": False,
    }


async def check_groundedness(state: GraphState) -> dict:
    """节点 6：检查生成答案是否由检索文档支撑（防幻觉自检）。

    失败后的处理由 GROUNDEDNESS_ACTION 控制（见 app/config.py）：

        retry  失败 -> 置 strict_grounding=True 走严格模式重生成一次
                      -> 仍失败才替换为拒答【默认】
        warn   失败 -> 只打印警告，保留原答案
        off    完全不检查

    为什么默认不是"直接拒答"：
        这个判断本身并不总是可靠，最典型的是跨语言数字换算 ——
        事实写 "7 billion to 70 billion"，中文答"70 亿到 700 亿"是完全正确的，
        但模型自己会算错（把 70 billion 当成 70 亿），从而把正确答案判成幻觉。
        实测中这会导致明明有答案却一直拒答。所以先给它一次"重说一遍"的机会：
        严格模式重生成出来的答案更贴原文，通常能通过；还不通过才是真有问题。
    """
    # ---- 分支 1：兜底答案不需要检查 ----
    if state.get("skip_check"):
        print("--->[节点]check_groundedness: 跳过兜底答案")
        return {
            "generation": state["generation"],
            "regenerate": False,
            "groundedness_verdict": "skipped",
        }

    # ---- 分支 2：检查被关闭 ----
    if settings.GROUNDEDNESS_ACTION == "off":
        print("--->[节点]check_groundedness: 已关闭（GROUNDEDNESS_ACTION=off）")
        return {
            "generation": state["generation"],
            "regenerate": False,
            "groundedness_verdict": "off",
        }

    print("--->[节点]check_groundedness")

    documents = fit_documents(state["documents"], settings.CONTEXT_MAX_CHARS)
    document_text = "\n\n".join(document.page_content for document in documents)

    # 答案允许复述此前回答（用户明确问到时），这部分的依据是对话背景，
    # 一并交给评分器，避免把"总结你刚才说的"误判成幻觉
    conversation = _render_conversation_for(state)
    if conversation:
        document_text += "\n\n【此前对话（助手之前已给出的回答）】\n" + conversation

    try:
        grader = _groundedness_grader()
        score_text = await grader.ainvoke({
            "documents": document_text,
            "generation": state["generation"],
        })
    except Exception as error:
        # 检查服务本身出问题时，保留原答案 —— 不能因为"校验跑挂了"就不给用户答案
        print(f"    [warn]忠实性检查调用失败，保留原答案：{error}")
        return {
            "generation": state["generation"],
            "regenerate": False,
            "groundedness_verdict": "unknown",
        }

    # 把评分器的原始输出打出来。这一步很关键：
    # 判 no 时如果不看原始输出，你分不清是"真的在编"还是"评分器误判"。
    passed = _parse_yes_no(score_text, default=True)  # 解析失败时保守放行
    print(f"   评分器原始输出：{score_text.strip()[:60]!r} -> {'通过' if passed else '未通过'}")

    # ---- 分支 3：通过 ----
    if passed:
        return {
            "generation": state["generation"],
            "regenerate": False,
            "groundedness_verdict": "yes",
        }

    # ---- 分支 4：未通过，且还有重生成额度 -> 走严格模式重来一次 ----
    retries = state.get("groundedness_retries", 0)
    if (
        settings.GROUNDEDNESS_ACTION == "retry"
        and retries < settings.MAX_GROUNDEDNESS_RETRIES
    ):
        print(
            f"   [warn]答案未通过忠实性检查，"
            f"第 {retries + 1} 次改用严格模式重新生成"
        )
        return {
            "groundedness_retries": retries + 1,
            "strict_grounding": True,
            "regenerate": True,
            "groundedness_verdict": "no",
        }

    # ---- 分支 5：未通过，且没有重生成额度了 ----
    if settings.GROUNDEDNESS_ACTION == "warn":
        print(
            "   [warn]答案未通过忠实性检查，"
            "但 GROUNDEDNESS_ACTION=warn，保留原答案"
        )
        return {
            "generation": state["generation"],
            "regenerate": False,
            "groundedness_verdict": "no",
        }

    print("   [warn]答案未通过忠实性检查，替换为拒答")
    return {
        "generation": (
            WEB_FALLBACK_ANSWER if state.get("source_type") == "web" else FALLBACK_ANSWER
        ),
        "regenerate": False,
        "groundedness_verdict": "no",
    }


# ================================================================ 条件路由


def decide_after_route(state: GraphState) -> str:
    """入口路由：需要实时信息 -> 直接联网；否则先查本地。"""
    return "web" if state.get("route") == "web" and _can_web_search(state) else "local"


def decide_after_grading(state: GraphState) -> str:
    """文档评分完成后的条件路由（CRAG）。

    1. 有相关文档：进入答案生成。
    2. 无相关文档，还有本地改写额度：改写查询后重新检索。
    3. 无相关文档，改写额度用尽，联网可用：联网搜索纠错。
    4. 都不行：生成兜底答案。
    """
    if state["documents"]:
        return "generate"

    if state.get("retries", 0) < _local_rewrite_budget():
        return "rewrite"

    if _can_web_search(state):
        print("    本地知识库没有相关文档，转为联网搜索")
        return "web"

    print("    [warn]已达到最大查询改写次数")
    return "generate"


def decide_after_rewrite(state: GraphState) -> str:
    """查询改写完成后的条件路由。

    改写生效 -> 回到 transform_query，用新问题重新做查询变换 + 混合检索。
    改写无效 -> 继续检索相同问题不会有新结果：能联网就联网，否则交给 generate 兜底。
    """
    if state.get("rewrite_unchanged"):
        return "web" if _can_web_search(state) else "generate"

    return "transform"


def decide_after_groundedness(state: GraphState) -> str:
    """忠实性检查后的条件路由。

    检查未通过、且还有重生成额度时 -> 回到 generate 用严格模式重写一遍。
    其余情况（通过 / 已用尽额度 / 被关闭） -> 结束。
    """
    if state.get("regenerate"):
        return "regenerate"

    return "end"


# ================================================================ 构建图


def build_graph():
    """构建并编译 Agentic RAG 状态图。"""
    workflow = StateGraph(GraphState)

    workflow.add_node("route_question", route_question)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank_documents", rerank_documents)
    workflow.add_node("grade_document", grade_documents)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("check_groundedness", check_groundedness)

    workflow.set_entry_point("route_question")

    # 入口路由：需要实时信息 -> 直接联网；否则先查本地
    workflow.add_conditional_edges(
        "route_question",
        decide_after_route,
        {
            "local": "transform_query",
            "web": "web_search",
        },
    )

    # 主链路：查询变换 -> 混合检索 -> 精排 -> 相关度评分
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("retrieve", "rerank_documents")
    workflow.add_edge("rerank_documents", "grade_document")

    workflow.add_conditional_edges(
        "grade_document",
        decide_after_grading,
        {
            "rewrite": "rewrite_query",
            "web": "web_search",
            "generate": "generate",
        },
    )

    # 改写成功时回到 transform_query（用新问题重新生成多条查询）；
    # 改写无变化时联网（可用时）或直接返回兜底答案。
    workflow.add_conditional_edges(
        "rewrite_query",
        decide_after_rewrite,
        {
            "transform": "transform_query",
            "web": "web_search",
            "generate": "generate",
        },
    )

    # 联网结果不再做 LLM 逐条评分（Tavily 已按相关度排序），直接生成，
    # 生成后仍然要过忠实性检查
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "check_groundedness")

    # 忠实性检查没通过时，回到 generate 用严格模式重生成一次（Self-RAG 的反思闭环）；
    # 通过或额度用尽则结束。
    workflow.add_conditional_edges(
        "check_groundedness",
        decide_after_groundedness,
        {
            "regenerate": "generate",
            "end": END,
        },
    )

    return workflow.compile()


# ================================================================ 对外接口


class RAGChain:
    """封装编译后的 LangGraph 应用，供 FastAPI 调用。"""

    def __init__(self, warm: bool = True, memory: Optional[MemoryManager] = None):
        """构建状态图，并按需预热。

        参数：
            warm: 是否在初始化时就预热（加载 Embedding 模型、打开 Milvus、
                  从语料副本重建 BM25 倒排索引）。
            memory: 会话记忆管理器；不传则按 MEMORY_BACKEND 配置创建
                    （存储是懒连接的，不传 session_id 的调用不会触碰存储）。

        为什么默认 True：
            预热很慢（本地要加载 BGE 权重，首次还要从 HuggingFace 下载），
            放在 `__init__` 里做，代价是**启动慢一次**；
            如果放到第一次 `/query` 里做，代价是**第一个用户请求超时**。
            FastAPI 的 lifespan 里会调用它，所以服务起来时就已就绪。

            RAGAs 评估脚本里传 warm=False —— 因为那时图已经建好了，
            没必要为每个样本重复预热。
        """
        self.app = build_graph()
        self.memory = memory if memory is not None else MemoryManager.from_settings(get_llm)
        if warm:
            # 启动时预热：加载 Embedding、打开 Milvus、构建 BM25 倒排索引
            warmup()

    def answer(
        self,
        question: str,
        include_documents: bool = False,
        history: Optional[Sequence[Tuple[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """同步兼容入口；异步服务代码应使用 :meth:`aanswer`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            async def run_once() -> dict:
                try:
                    return await self.aanswer(question, include_documents, history, session_id)
                finally:
                    # asyncio.run 每次都会创建新 loop，不能把连接池留给下一轮。
                    await self.aclose()

            return asyncio.run(run_once())
        raise RuntimeError("事件循环中不能调用 answer()，请改用 await aanswer()")

    @staticmethod
    async def aclose() -> None:
        """释放 Tavily 与 Redis 连接池（服务关闭或事件循环结束前调用）。"""
        from app.redis_client import close_redis

        try:
            await aclose_web_client()
        finally:
            await close_redis()

    @staticmethod
    def _initial_state(
        question: str,
        history: Sequence[Tuple[str, str]],
        memory_summary: str = "",
    ) -> dict:
        return {
            "question": question,
            "original_question": question,
            "history": list(history),
            "memory_summary": memory_summary,
            "standalone_question": question,
            "route": "local",
            "source_type": "local",
            "web_searched": False,
            "web_error": "",
            "queries": [],
            "documents": [],
            "generation": "",
            "rewritten": False,
            "retries": 0,
            "rewrite_unchanged": False,
            "candidates": 0,
            "reranked": False,
            "transform_mode": settings.QUERY_TRANSFORM,
            "skip_check": False,
            "groundedness_retries": 0,
            "strict_grounding": False,
            "regenerate": False,
            "groundedness_verdict": "",
        }

    async def aanswer(
        self,
        question: str,
        include_documents: bool = False,
        history: Optional[Sequence[Tuple[str, str]]] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """执行完整的 Agentic RAG 流程（本地检索 + 按需联网 + 会话记忆）。

        两种多轮方式（二选一）：
            session_id  服务端记忆：自动加载该会话的摘要与最近几轮，答完自动写回并按需压缩
            history     客户端自带历史 [(问题, 回答), ...]，服务端不保存
        都不传就是单轮无状态问答。
        include_documents=True 时额外返回完整检索上下文（RAGAs 评估需要）。
        返回值里的 steps 是本轮实际执行过的节点序列，用于展示/调试执行路径。
        """
        if session_id:
            memory = await self.memory.aload(session_id)
            summary, turns = memory.summary, list(memory.turns)
        else:
            summary, turns = "", list(history or [])

        steps: List[str] = []
        result: dict = {}
        recursion_limit = max(settings.RECURSION_LIMIT, required_recursion_limit())

        # stream_mode=["updates","values"]：updates 记录执行了哪些节点，
        # values 的最后一帧就是最终状态，等价于 ainvoke 的返回值
        async for mode, chunk in self.app.astream(
            self._initial_state(question, turns, summary),
            config={"recursion_limit": recursion_limit},
            stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                steps.extend(chunk.keys())
            else:
                result = chunk

        documents = result.get("documents", [])

        sources = []
        seen = set()
        for document in documents:
            key = (
                document.metadata.get("source"),
                document.metadata.get("page"),
                document.page_content[:80],
            )
            if key in seen:
                continue
            seen.add(key)

            page = document.metadata.get("page")
            sources.append({
                "source": document.metadata.get("source", ""),
                "filename": document.metadata.get("filename", ""),
                "page": page + 1 if page is not None else None,
                "snippet": document.page_content[:200],
                # 精排分数（未精排时为 None），方便前端展示"这条有多相关"
                "relevance_score": document.metadata.get("relevance_score"),
                "rrf_score": document.metadata.get("rrf_score"),
                # local=本地知识库 / web=互联网
                "source_type": document.metadata.get("source_type", "local"),
            })

        payload = {
            "answer": result.get("generation") or FALLBACK_ANSWER,
            "question": result.get("question", question),
            "original_question": result.get("original_question", question),
            "rewritten": bool(result.get("rewritten")),
            "rewrite_unchanged": bool(result.get("rewrite_unchanged")),
            "num_documents": len(documents),
            "sources": sources,
            # ---- 观测字段 ----
            "queries": result.get("queries", []),
            "transform_mode": result.get("transform_mode", "none"),
            "candidates": result.get("candidates", 0),
            "reranked": bool(result.get("reranked")),
            # 忠实性自检结果：yes=通过 no=未通过(已按配置处理)
            # skipped=兜底答案无需检查 off=已关闭 unknown=检查调用失败
            "groundedness": result.get("groundedness_verdict", ""),
            "groundedness_retries": result.get("groundedness_retries", 0),
            # ---- 联网相关 ----
            "route": result.get("route", "local"),
            "source_type": result.get("source_type", "local"),
            "web_searched": bool(result.get("web_searched")),
            "web_error": result.get("web_error", ""),
            "steps": steps,
            # ---- 记忆相关 ----
            "session_id": session_id or "",
            "memory_turns": len(turns),
            "memory_summary_used": bool(summary),
        }

        if include_documents:
            payload["documents"] = [doc.page_content for doc in documents]

        # 写回记忆：问题存补全后的独立问题，脱离上下文也能读懂
        if session_id:
            await self.memory.aappend(
                session_id,
                result.get("standalone_question") or question,
                payload["answer"],
            )

        return payload
