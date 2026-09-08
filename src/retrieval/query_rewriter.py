"""
查询改写器
支持多种查询优化策略
"""
from llama_index.core import PromptTemplate
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
    HyDEQueryTransform,
)
from llama_index.core.query_engine import (
    MultiStepQueryEngine,
    TransformQueryEngine,
)


class QueryRewriter:
    """查询改写器"""
    
    def __init__(self, llm):
        """
        Args:
            llm: 大语言模型实例
        """
        self.llm = llm
    
    def rewrite_query(self, query: str) -> str:
        """
        使用LLM改写查询，扩展上下文
        
        Args:
            query: 原始查询
            
        Returns:
            改写后的查询
        """
        prompt = PromptTemplate("""\
系统角色设定:
你是一个专业的问题改写助手。你的任务是将用户的原始问题扩充为一个更完整、更全面的问题。

规则：
1. 将可能的歧义、相关概念和上下文信息整合到一个完整的问题中
2. 使用括号对歧义概念进行补充说明
3. 添加关键的限定词和修饰语
4. 确保改写后的问题清晰且语义完整
5. 对于模糊概念，在括号中列举主要可能性

原始问题:
{query}

请生成一个综合的改写问题，确保：
- 包含原始问题的核心意图
- 涵盖可能的歧义解释
- 使用清晰的逻辑关系词连接不同方面
- 必要时使用括号补充说明

输出格式：
[综合改写] - 改写后的问题
""")
        
        response = self.llm.predict(prompt, query=query)
        return response
    
    def create_multistep_query_engine(self, query_engine, index_summary: str = ""):
        """
        创建多步查询引擎
        
        Args:
            query_engine: 基础查询引擎
            index_summary: 索引摘要
            
        Returns:
            MultiStepQueryEngine实例
        """
        step_decompose_transform = StepDecomposeQueryTransform(verbose=True)
        
        return MultiStepQueryEngine(
            query_engine=query_engine,
            query_transform=step_decompose_transform,
            index_summary=index_summary
        )
    
    def create_hyde_query_engine(self, query_engine, include_original: bool = True):
        """
        创建HyDE查询引擎（假设文档嵌入）
        
        Args:
            query_engine: 基础查询引擎
            include_original: 是否包含原始查询
            
        Returns:
            TransformQueryEngine实例
        """
        hyde = HyDEQueryTransform(include_original=include_original)
        
        return TransformQueryEngine(
            query_engine=query_engine,
            query_transform=hyde
        )
