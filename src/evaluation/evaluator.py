"""Ragas 评测：衡量检索召回与生成质量。

输入 rows 为列表，每项包含：
    question / answer / contexts / ground_truth
"""
from typing import List, Optional

from datasets import Dataset
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms.tongyi import Tongyi
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    context_precision,
    context_recall,
    faithfulness,
)

from src.config import Config, settings


class RagasEvaluator:
    """Ragas 评测器。"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or settings
        self.llm = Tongyi(model_name=self.config.llm_model)
        self.embeddings = DashScopeEmbeddings(model=self.config.embedding_model)

    def evaluate(self, rows: List[dict]):
        dataset = Dataset.from_dict(
            {
                "question": [r["question"] for r in rows],
                "answer": [r["answer"] for r in rows],
                "contexts": [r["contexts"] for r in rows],
                "ground_truth": [r["ground_truth"] for r in rows],
            }
        )

        result = evaluate(
            dataset,
            metrics=[
                answer_correctness,
                faithfulness,
                context_recall,
                context_precision,
            ],
            llm=self.llm,
            embeddings=self.embeddings,
        )
        return result.to_pandas()
