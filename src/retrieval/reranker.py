"""
重排序模块
使用Rerank模型对检索结果进行精排
"""
from llama_index.core.postprocessor import SimilarityPostprocessor
from typing import List, Optional


class Reranker:
    """重排序器"""
    
    def __init__(self, rerank_client=None, top_n: int = 3):
        """
        Args:
            rerank_client: Rerank客户端（暂时返回None，后续可实现）
            top_n: 最终返回的文档数量
        """
        self.rerank_client = rerank_client
        self.top_n = top_n
    
    def get_postprocessor(self):
        """
        获取后处理器
        
        Returns:
            后处理器列表
        """
        postprocessors = []
        
        # TODO: 实现DashScope Rerank
        # if self.rerank_client:
        #     postprocessors.append(
        #         DashScopeRerankPostprocessor(
        #             rerank_client=self.rerank_client,
        #             top_n=self.top_n
        #         )
        #     )
        
        # 暂时只添加相似度过滤
        postprocessors.append(
            SimilarityPostprocessor(similarity_cutoff=0.2)
        )
        
        return postprocessors
