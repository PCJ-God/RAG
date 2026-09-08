"""
基础检索器
"""
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from typing import Optional, List


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
    
    def create_query_engine(self, streaming: bool = True, llm=None, 
                           node_postprocessors: Optional[List] = None):
        """
        创建查询引擎
        
        Args:
            streaming: 是否使用流式输出
            llm: 大语言模型
            node_postprocessors: 后处理器列表（如Rerank）
        """
        postprocessors = node_postprocessors or []
        
        # 添加相似度过滤
        if self.similarity_cutoff > 0:
            postprocessors.append(
                SimilarityPostprocessor(similarity_cutoff=self.similarity_cutoff)
            )
        
        self.query_engine = self.index.as_query_engine(
            streaming=streaming,
            similarity_top_k=self.similarity_top_k,
            llm=llm,
            node_postprocessors=postprocessors
        )
        
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
