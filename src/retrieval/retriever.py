"""
基础检索器
支持 Rerank、HyDE、QueryRewrite 等策略
"""
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from typing import Optional, List
from src.retrieval.reranker import Reranker


class Retriever:
    """基础检索器"""

    def __init__(self, index: VectorStoreIndex, similarity_top_k: int = 5,
                 similarity_cutoff: float = 0.0):
        """
        Args:
            index: 向量索引
            similarity_top_k: 返回最相似的文档数量
            similarity_cutoff: 相似度阈值
        """
        self.index = index
        self.similarity_top_k = similarity_top_k
        self.similarity_cutoff = similarity_cutoff
        self.query_engine = None
        self.reranker = None
        self.llm_client = None
        self.use_rerank = False
        self.use_hyde = False

    def enable_rerank(self, top_n: int = 3, llm_client=None):
        """启用 Rerank"""
        self.reranker = Reranker(top_n=top_n, use_rerank=True, llm_client=llm_client)
        self.llm_client = llm_client
        self.use_rerank = True

    def enable_hyde(self):
        """启用 HyDE（假设文档嵌入）"""
        self.use_hyde = True

    def create_query_engine(self, streaming: bool = True, llm=None,
                           node_postprocessors: Optional[List] = None):
        """
        创建查询引擎

        Args:
            streaming: 是否使用流式输出
            llm: 大语言模型
            node_postprocessors: 后处理器列表（如 Rerank）

        Returns:
            查询引擎实例
        """
        postprocessors = node_postprocessors or []

        # 如果启用了 Rerank 且没有外部传入 postprocessor
        if self.use_rerank and not node_postprocessors:
            postprocessors = self.reranker.get_postprocessor(llm=llm)
        elif self.similarity_cutoff > 0:
            # 没开 Rerank 但有 cutoff 阈值，添加相似度过滤
            postprocessors.append(
                SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)
            )

        base_query_engine = self.index.as_query_engine(
            streaming=streaming,
            similarity_top_k=self.similarity_top_k,
            llm=llm,
            node_postprocessors=postprocessors
        )

        # 如果启用了 HyDE，包装为 TransformQueryEngine
        if self.use_hyde:
            hyde_transform = HyDEQueryTransform(
                include_original=True,
                llm=llm
            )
            self.query_engine = TransformQueryEngine(
                query_engine=base_query_engine,
                query_transform=hyde_transform
            )
        else:
            self.query_engine = base_query_engine

        return self.query_engine

    def retrieve(self, question: str) -> List:
        """
        检索相关文档

        Args:
            question: 用户问题

        Returns:
            检索到的文档节点列表
        """
        if self.query_engine is None:
            self.create_query_engine()

        response = self.query_engine.query(question)
        return response.source_nodes

    def query(self, question: str):
        """
        执行查询

        Args:
            question: 用户问题

        Returns:
            查询响应对象
        """
        if self.query_engine is None:
            self.create_query_engine()

        return self.query_engine.query(question)
