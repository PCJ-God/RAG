"""LlamaIndex 基座上的向量化模型：走 DashScope OpenAI 兼容模式。

text-embedding-v3 是 DashScope 专有模型名，OpenAIEmbedding 的模型枚举不接受它，
因此这里用 BaseEmbedding 自定义实现——它仍是 LlamaIndex 的 embed 模型，可以挂到
Settings.embed_model / VectorStoreIndex 上正常参与构建与检索。
"""
from typing import List

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from openai import OpenAI

from src.config import BASE_URL, settings

# 兼容模式单次请求最多 10 条文本
EMBED_BATCH_SIZE = 10


class CompatibleEmbedding(BaseEmbedding):
    """DashScope 兼容模式的 Embedding 模型（text-embedding-v3，1024 维）。"""

    _client: OpenAI = PrivateAttr()
    _model: str = PrivateAttr()

    def __init__(
        self,
        api_key: str = "",
        base_url: str = BASE_URL,
        model: str = "text-embedding-v3",
        batch_size: int = EMBED_BATCH_SIZE,
    ):
        super().__init__(model_name=model, embed_batch_size=batch_size)
        self._client = OpenAI(
            api_key=api_key or settings.api_key,
            base_url=base_url,
        )
        self._model = model

    @classmethod
    def class_name(cls) -> str:
        return "CompatibleEmbedding"

    def _embed(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        data = sorted(resp.data, key=lambda d: d.index)
        return [d.embedding for d in data]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)
