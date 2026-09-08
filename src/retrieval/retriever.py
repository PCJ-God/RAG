"""
基础检索器
支持 Rerank、HyDE、QueryRewrite(扩展/多步分解)、QueryRewrite 等策略
"""
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import TransformQueryEngine, MultiStepQueryEngine
from llama_index.core.indices.query.query_transform.base import (
    HyDEQueryTransform,
    StepDecomposeQueryTransform,
    BaseQueryTransform,
)
from typing import Optional, List
from src.retrieval.reranker import Reranker


class Retriever:
    """基础检索器"""

    # LLM 问题扩展 Prompt
    QUERY_EXPAND_PROMPT = """\
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
"""

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

        # 策略开关
        self.use_rerank = False
        self.use_hyde = False
        self.use_query_expand = False  # LLM 问题扩展
        self.use_multistep = False     # 多步骤查询分解

    def enable_rerank(self, top_n: int = 3, llm_client=None):
        """启用 Rerank"""
        self.reranker = Reranker(top_n=top_n, use_rerank=True, llm_client=llm_client)
        self.llm_client = llm_client
        self.use_rerank = True

    def enable_hyde(self):
        """启用 HyDE（假设文档嵌入）"""
        self.use_hyde = True

    def enable_query_expand(self):
        """启用 LLM 问题扩展"""
        self.use_query_expand = True

    def enable_multistep(self, index_summary: str = ""):
        """启用多步骤查询分解"""
        self.use_multistep = True
        self.index_summary = index_summary

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

        # 策略 1: LLM 问题扩展
        if self.use_query_expand:
            expand_prompt = PromptTemplate(self.QUERY_EXPAND_PROMPT)
            rewritten_query = llm.predict(expand_prompt, query="")  # 这里只是获取 prompt 结构

            # 实际的扩展在 query 时完成，用 TransformQueryEngine
            class QueryExpandTransform(BaseQueryTransform):
                """查询扩展 Transform"""
                def __init__(self, llm, prompt_template):
                    super().__init__()
                    self.llm = llm
                    self.prompt_template = prompt_template

                def _get_prompts(self):
                    return {"expand_prompt": self.prompt_template}

                def _update_prompts(self, prompts):
                    if "expand_prompt" in prompts:
                        self.prompt_template = prompts["expand_prompt"]

                def _run(self, query_bundle, metadata=None):
                    from llama_index.core.schema import QueryBundle
                    rewritten = self.llm.predict(
                        self.prompt_template,
                        query=query_bundle.query_str
                    )
                    if "[综合改写]" in rewritten:
                        rewritten = rewritten.split("[综合改写]")[-1].strip().lstrip("- ").strip()
                    return QueryBundle(query_str=rewritten)

            expand_transform = QueryExpandTransform(llm, expand_prompt)
            self.query_engine = TransformQueryEngine(
                query_engine=base_query_engine,
                query_transform=expand_transform
            )

        # 策略 2: 多步骤查询分解
        elif self.use_multistep:
            step_transform = StepDecomposeQueryTransform(verbose=True)
            self.query_engine = MultiStepQueryEngine(
                query_engine=base_query_engine,
                query_transform=step_transform,
                index_summary=getattr(self, 'index_summary', '')
            )

        # 策略 3: HyDE
        elif self.use_hyde:
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
