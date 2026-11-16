"""本地检索链路的五个节点：查询变换 -> 混合检索 -> 精排 -> 相关性评分 -> 查询改写。

为什么是这个顺序？
    查询变换放在最前面：问法不好，后面再怎么精排也救不回来。
    精排放在文档评分之前：评分要对每个文档调一次 LLM，先精排把 20 条砍到 5 条，
    评分调用次数直接降 4 倍，整条链路的延迟和 token 成本都跟着降。
    这也正是 LlamaIndex "先宽召回、再精排、后处理" 的标准分层。
"""

from __future__ import annotations

import asyncio

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from app import vector_store
from app.config import settings
from app.rag import llm
from app.rag.helpers import normalize_question, parse_query_list, parse_yes_no
from app.rag.prompts import DOC_GRADE_PROMPT, HYDE_PROMPT, MULTI_QUERY_PROMPT, REWRITE_PROMPT
from app.rag.state import GraphState


async def transform_query(state: GraphState) -> dict:
    """查询变换。

    - none        直接用原问题
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
            chain = HYDE_PROMPT | llm.get_llm() | StrOutputParser()
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
        chain = MULTI_QUERY_PROMPT | llm.get_llm() | StrOutputParser()
        raw = await chain.ainvoke({"question": question, "num_queries": settings.NUM_QUERIES})
        variants = parse_query_list(raw, limit=settings.NUM_QUERIES)
    except Exception as error:
        print(f"    [warn] 多查询生成失败，退回原问题：{error}")
        variants = []

    if not variants:
        return {"queries": [question], "transform_mode": "none"}

    # 原问题永远保留：模型生成的查询可能有偏，原问题是唯一"保底"的那一路
    queries = [question] + [
        v for v in variants if normalize_question(v) != normalize_question(question)
    ]
    for index, query in enumerate(queries):
        print(f"   [{index + 1}] {query}")

    return {"queries": queries, "transform_mode": f"multi_query({len(queries)})"}


async def retrieve(state: GraphState) -> dict:
    """混合检索（稠密向量 ⊕ BM25 稀疏，RRF 融合）。

    这里刻意"召回宁宽勿窄"（默认 20 条），因为后面紧跟精排。
    粗排负责别漏，精排负责排准，两者分工明确。
    """
    queries = state.get("queries") or [state["question"]]
    print(f"--->[节点]retrieve: {len(queries)} 条查询 × 混合检索")

    try:
        documents = await vector_store.amulti_query_search(queries, k=settings.FUSION_TOP_K)
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
    """交叉编码器精排。

    Cohere Rerank：输入 [问题, 20 条候选]，输出按真实相关度重排后的 top_n 条。

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

    compressor = llm.get_reranker()
    if compressor is None:
        return {"reranked": False}

    print(f"--->[节点]rerank_documents: {len(documents)} 条候选 -> top {settings.RERANK_TOP_N}")

    try:
        # 注意：精排的 query 用"当前问题"而不是变换出来的查询。
        # 变换查询是为了"捞得全"，精排要判断的是"跟用户真正想问的有多相关"。
        reranked = await asyncio.to_thread(compressor.compress_documents, documents, question)
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
    """逐条评估检索文档，过滤明显无关的内容。

    精排已经砍到 5 条左右，这里的 LLM 调用次数因此大幅下降；评分互不依赖，并发执行。
    评分服务异常时保守地保留文档，避免因为评分失败而丢失可能有用的上下文。
    """
    print(f"--->[节点]grade_documents: 共{len(state['documents'])}个文档块")
    if not state["documents"]:
        return {"documents": []}

    question = state["question"]
    try:
        grader = DOC_GRADE_PROMPT | llm.get_llm() | StrOutputParser()
    except Exception as error:  # noqa: BLE001
        print(f"    [warn]评分器初始化失败，保留全部文档：{error}")
        return {"documents": state["documents"]}

    async def grade_one(document: Document) -> tuple[Document, bool]:
        try:
            score_text = await grader.ainvoke({
                "question": question,
                "document": document.page_content[:2000],
            })
            return document, parse_yes_no(score_text, default=True)
        except Exception as error:
            print(f"    [warn]评分失败，保留该文档：{error}")
            return document, True

    graded = await asyncio.gather(*(grade_one(doc) for doc in state["documents"]))
    filtered_documents = [document for document, keep in graded if keep]

    print(f"   相关文档数:{len(filtered_documents)}")
    return {"documents": filtered_documents}


async def rewrite_query(state: GraphState) -> dict:
    """将当前问题改写为更合适向量检索的形式。

    如果改写结果与当前问题相同，则设置 rewrite_unchanged=True，
    后续路由会停止重新检索，避免对相同问题反复执行检索。
    """
    print("--->[节点]rewrite_query")

    current_question = state["question"]

    try:
        # 链在 try 里构建：LLM 配置有误（如没配 Key）时同样降级为"沿用原问题"，而不是让整张图崩掉
        rewrite_chain = REWRITE_PROMPT | llm.get_llm() | StrOutputParser()
        rewritten_question = (await rewrite_chain.ainvoke({"question": current_question})).strip()
        # 当模型返回空字符串时，当作没有改写成功。
        if not rewritten_question:
            rewritten_question = current_question
    except Exception as error:
        print(f"    [warn]查询改写失败，沿用原问题：{error}")
        rewritten_question = current_question

    rewrite_unchanged = normalize_question(rewritten_question) == normalize_question(current_question)
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
        # 问题变了，之前那批查询作废；路由会回到 transform_query 用新问题重新生成查询。
        "queries": [],
    }
