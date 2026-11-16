"""状态图的全部节点。每个文件对应链路中的一个阶段：

    route.py       入口：补全追问 + 选择数据源
    retrieval.py   本地检索：查询变换 / 混合检索 / 精排 / 相关性评分 / 查询改写
    web.py         联网搜索（CRAG 纠错检索）
    generation.py  生成答案 + 忠实性检查
"""

from app.rag.nodes.generation import check_groundedness, generate
from app.rag.nodes.retrieval import (
    grade_documents,
    rerank_documents,
    retrieve,
    rewrite_query,
    transform_query,
)
from app.rag.nodes.route import route_question
from app.rag.nodes.web import web_search

__all__ = [
    "route_question",
    "transform_query",
    "retrieve",
    "rerank_documents",
    "grade_documents",
    "rewrite_query",
    "web_search",
    "generate",
    "check_groundedness",
]
