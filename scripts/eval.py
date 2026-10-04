"""运行 Ragas 评测。

用法：
    python scripts/eval.py
    python scripts/eval.py --qa-file data/eval/qa_pairs.json
"""
import sys
from pathlib import Path

# 让脚本从任意目录运行时都能找到项目根目录下的 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

from src.config import EVAL_DIR
from src.evaluation.evaluator import RagasEvaluator
from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ragas 评测")
    parser.add_argument("--domain", default="rag")
    parser.add_argument(
        "--qa-file", default=str(EVAL_DIR / "qa_pairs.json"), help="评测问答对文件"
    )
    args = parser.parse_args()

    with open(args.qa_file, encoding="utf-8") as f:
        qa_pairs = json.load(f)

    pipeline = RAGPipeline(domain=args.domain).load()
    rows = []
    for item in qa_pairs:
        result = pipeline.query(item["question"])
        rows.append(
            {
                "question": item["question"],
                "answer": result.answer,
                "contexts": [s.text for s in result.sources],
                "ground_truth": item["ground_truth"],
            }
        )
        print(f"[{len(rows)}/{len(qa_pairs)}] {item['question']}")

    evaluator = RagasEvaluator()
    df = evaluator.evaluate(rows)
    print("\n评测结果：")
    print(df.to_string())


if __name__ == "__main__":
    main()
