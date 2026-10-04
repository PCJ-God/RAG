"""向量化工具：走 OpenAI 兼容模式接口（与 LLM 同一通道）。

注意：百炼的 workspace 类型 API Key（sk-ws- 开头）在 DashScope 原生 SDK 接口
会返回 InvalidApiKey，但在 OpenAI 兼容模式接口可用。因此这里统一走兼容模式，
与 LLMClient 保持一致。
"""
from typing import List

from openai import OpenAI

from src.config import BASE_URL

EMBED_MODEL = "text-embedding-v3"
EMBED_BATCH_SIZE = 10


def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def embed_texts(api_key: str, texts: List[str]) -> List[List[float]]:
    """批量向量化，返回与输入顺序一致的向量列表。"""
    resp = _client(api_key).embeddings.create(model=EMBED_MODEL, input=texts)
    results: List[List[float]] = [[] for _ in texts]
    for item in resp.data:
        results[item.index] = item.embedding
    return results


def embed_query(api_key: str, query: str) -> List[float]:
    """单个问题向量化。"""
    resp = _client(api_key).embeddings.create(model=EMBED_MODEL, input=query)
    return resp.data[0].embedding
