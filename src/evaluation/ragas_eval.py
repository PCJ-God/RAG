"""
Ragas自动化评测
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    context_recall,
    context_precision,
    faithfulness,
    answer_relevancy
)
from langchain_community.llms.tongyi import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
from src.config import LLM_MODEL, EMBEDDING_MODEL


class RagasEvaluator:
    """Ragas评测器"""
    
    def __init__(self, llm_model: str = None, embedding_model: str = None):
        """
        Args:
            llm_model: 评测用LLM模型
            embedding_model: 评测用Embedding模型
        """
        self.llm_model = llm_model or LLM_MODEL
        self.embedding_model = embedding_model or EMBEDDING_MODEL
        
        # 创建评测用LLM和Embedding
        self.llm = Tongyi(model_name=self.llm_model)
        self.embeddings = DashScopeEmbeddings(model=self.embedding_model)
    
    def evaluate_answer_quality(self, question: str, answer: str, 
                                ground_truth: str, contexts: list = None):
        """
        评测答案质量
        
        Args:
            question: 问题
            answer: 答案
            ground_truth: 标准答案
            contexts: 检索到的上下文
            
        Returns:
            评测结果DataFrame
        """
        data_samples = {
            'question': [question],
            'answer': [answer],
            'ground_truth': [ground_truth]
        }
        
        if contexts:
            data_samples['contexts'] = [contexts]
        
        dataset = Dataset.from_dict(data_samples)
        
        # 选择评测指标
        metrics = [answer_correctness]
        if contexts:
            metrics.extend([context_recall, context_precision])
        
        # 执行评测
        score = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        return score.to_pandas()
    
    def batch_evaluate(self, data_samples: dict):
        """
        批量评测
        
        Args:
            data_samples: 评测数据集，需包含question/answer/ground_truth字段
            
        Returns:
            评测结果DataFrame
        """
        dataset = Dataset.from_dict(data_samples)
        
        metrics = [answer_correctness, context_recall, context_precision]
        
        score = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        return score.to_pandas()
    
    @staticmethod
    def compare_splitter_results(results_dict: dict):
        """
        对比不同分块策略的结果
        
        Args:
            results_dict: {策略名称: 评测结果DataFrame}
        """
        import pandas as pd
        
        comparison = {}
        for strategy, df in results_dict.items():
            comparison[strategy] = {
                'answer_correctness': df['answer_correctness'].mean(),
                'context_recall': df['context_recall'].mean() if 'context_recall' in df.columns else 0,
                'context_precision': df['context_precision'].mean() if 'context_precision' in df.columns else 0
            }
        
        return pd.DataFrame(comparison).T
