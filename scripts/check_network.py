"""网络诊断：检查 Embedding 模型下载能走哪条路。

用法（在运行 main.py 的同一个终端里）：
    python scripts/check_network.py

会逐条测试 直连 / 公司代理 × HuggingFace 官网 / 国内镜像，每条单独捕获错误，
最后给出 .env 该怎么配。只发 GET 请求读取模型信息，不下载任何文件。
"""

import os
import urllib.request

import httpx

MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CORP_PROXY = "http://hkproxy.jaguarmicro.com:3128"
ENDPOINTS = {
    "HuggingFace 官网": "https://huggingface.co",
    "国内镜像 hf-mirror": "https://hf-mirror.com",
}
ROUTES = {"直连": None, "公司代理": CORP_PROXY}


def probe(base: str, proxy):
    url = f"{base}/api/models/{MODEL}"
    try:
        response = httpx.get(url, proxy=proxy, timeout=10, trust_env=False)
        return response.status_code, ""
    except Exception as error:  # noqa: BLE001
        return None, f"{type(error).__name__}: {str(error)[:90]}"


def main() -> None:
    print("=" * 72)
    print(f" 模型：{MODEL}")
    print(f" 环境变量代理：{ {k: v for k, v in os.environ.items() if 'proxy' in k.lower()} or '无'}")
    print(f" 系统代理（urllib.getproxies）：{urllib.request.getproxies() or '无'}")
    print("=" * 72)

    ok = {}
    for route_name, proxy in ROUTES.items():
        for site_name, base in ENDPOINTS.items():
            status, error = probe(base, proxy)
            passed = status == 200
            ok[(route_name, site_name)] = passed
            result = f"HTTP {status}" if status else error
            print(f"  {'✅' if passed else '❌'} {route_name:<6} -> {site_name:<14} {result}")

    print("-" * 72)
    if ok[("直连", "国内镜像 hf-mirror")]:
        print(" 建议：在 .env 里加一行  HF_ENDPOINT=https://hf-mirror.com")
    elif ok[("直连", "HuggingFace 官网")]:
        print(" 建议：直连官网可用，无需额外配置（检查是否有别处设置了代理）。")
    elif ok[("公司代理", "国内镜像 hf-mirror")] or ok[("公司代理", "HuggingFace 官网")]:
        print(" 建议：只能走公司代理。在 .env 里加：")
        print(f"     HTTPS_PROXY={CORP_PROXY}")
        if ok[("公司代理", "国内镜像 hf-mirror")]:
            print("     HF_ENDPOINT=https://hf-mirror.com")
    else:
        print(" 四条路都不通：")
        print("   · 若公司代理返回 407：向 IT 申请代理账号，")
        print(f"     .env 写 HTTPS_PROXY=http://用户名:密码@{CORP_PROXY[7:]}")
        print("   · 或离线：在能上网的电脑下载模型目录，拷到 ./models/bge-m3，")
        print("     .env 写 EMBEDDING_MODEL=./models/bge-m3 和 HF_HUB_OFFLINE=1")
    print("=" * 72)


if __name__ == "__main__":
    main()
