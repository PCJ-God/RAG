"""
重排序模块
使用 LLM 对检索结果进行精排（LLMRerank）
"""
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.schema import NodeWithScore, QueryBundle
from typing import List, Optional
from src.generation.llm_client import LLMClient


class Reranker:
    """重排序器（工厂类）"""

    def __init__(self, top_n: int = 3, use_rerank: bool = True, llm_client=None):
        """
        Args:
            top_n: Rerank 后保留的文档数量
            use_rerank: 是否启用 Rerank
            llm_client: LLM 客户端实例（用于 LLMRerank）
        """
        self.top_n = top_n
        self.use_rerank = use_rerank
        self.llm_client = llm_client

    def get_postprocessor(self, use_rerank: bool = None, llm=None) -> list:
        """
        获取后处理器列表

        Args:
            use_rerank: 覆盖构造函数的 use_rerank 设置
            llm: LLM 实例（可选，不传则自动创建）

        Returns:
            后处理器列表
        """
        postprocessors = []
        enabled = use_rerank if use_rerank is not None else self.use_rerank

        if enabled:
            # 使用 LlamaIndex 内置的 LLMRerank
            rerank_llm = llm or (self.llm_client.get_llm() if self.llm_client else None)
            if rerank_llm:
                postprocessors.append(
                    LLMRerank(
                        llm=rerank_llm,
                        top_n=self.top_n,
                    )
                )
            else:
                # 没有 LLM，fallback 到相似度过滤
                postprocessors.append(
                    SimilarityPostprocessor(similarity_cutoff=0.2)
                )
        else:
            # fallback：相似度过滤
            postprocessors.append(
                SimilarityPostprocessor(similarity_cutoff=0.2)
            )

        return postprocessors
