"""HTTP 接口层（FastAPI）。

    security.py        Bearer Token 鉴权
    sse.py             SSE 格式化工具
    runtime.py         运行时单例（RAG 链、文档登记表、入库队列）与取用函数
    schemas.py         请求 / 响应模型
    openai_compat.py   OpenAI 兼容接口（/v1），接 Open WebUI 等客户端
    routes/
        system.py      /health
        query.py       /query、/query/stream、/retrieve、/sessions
        documents.py   /documents/*、/index、旧版 /ingest/*

应用的创建与生命周期在项目根目录的 main.py。
"""
