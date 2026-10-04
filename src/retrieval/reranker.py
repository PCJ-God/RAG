"""重排序：把粗排结果精排到 top_n。

优先使用 DashScope 专用 rerank 端点（qwen3-rerank），
未安装对应包时回退到 LLM 打分精排（LLMRerank）。
"""
from typing import Any, List, Optional

from llama_index.core.schema import NodeWithScore, QueryBundle

from src.config import Config, settings


class Reranker:
    """精排器。"""

    def __init__(self, config: Optional[Config] = None, top_n: Optional[int] = None):
        self.config = config or settings
        self.top_n = top_n or self.config.rerank_top_n
        self._postprocessor = self._build()

    def _build(self) -> Any:
        try:
            from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
        except ImportError:
            return self._build_llm_rerank()
        return DashScopeRerank(top_n=self.top_n, model=self.config.rerank_model)

    def _build_llm_rerank(self):
        from llama_index.core.postprocessor.llm_rerank import LLMRerank

        from src.generation.llm import LLMClient

        return LLMRerank(llm=LLMClient(self.config).get_llm(), top_n=self.top_n)

    def rerank(
        self, nodes: List[NodeWithScore], query: str
    ) -> List[NodeWithScore]:
        if not nodes:
            return nodes
        return self._postprocessor.postprocess_nodes(
            nodes, query_bundle=QueryBundle(query)
        )
