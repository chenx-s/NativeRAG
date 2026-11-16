"""检索层：Milvus 稠密检索 ⊕ BM25 稀疏检索，RRF 融合。

    embeddings.py  本地 Embedding 模型
    milvus.py      Milvus 连接、打开失败诊断、计数、维度检查
    corpus.py      BM25 语料副本（JSONL）
    bm25.py        中文分词 + BM25 倒排索引缓存
    fusion.py      RRF 融合与跨通路去重
    search.py      稠密 / 稀疏 / 混合 / 多查询检索
    store.py       入库、按文档删除、清库、预热、统计

其他模块统一通过门面 app/vector_store.py 使用这些能力（含异步版本）。
"""
