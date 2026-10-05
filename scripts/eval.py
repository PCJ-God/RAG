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

# 用 CLI 风格进度条替换 notebook 风格，评测进度显示更清爽（参考 ragas 官方示例）
from tqdm.cli import tqdm as tqdm_cli  # type: ignore[attr-defined]
import tqdm.auto

tqdm.auto.tqdm = tqdm_cli

import pandas as pd

from src.config import EVAL_DIR
from src.evaluation.evaluator import RagasEvaluator
from src.pipeline import RAGPipeline

# 单元格最大展示宽度：超出截断加省略号，避免长文本把表格撑爆
CELL_WIDTH = 40

# 中英文混排对齐，避免表格错位
pd.set_option("display.unicode.east_asian_width", True)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", CELL_WIDTH)


def _truncate(text: str) -> str:
    """截断长文本，超长加省略号。"""
    text = str(text)
    if len(text) <= CELL_WIDTH:
        return text
    return text[: CELL_WIDTH - 1] + "…"


def _cell(value) -> str:
    """把单元格转成适合表格展示的短文本。"""
    if isinstance(value, list):
        if not value:
            return "[]"
        preview = " ".join(str(x) for x in value[:2])
        return _truncate(f"[{len(value)} 段] {preview}")
    return _truncate(value)


def _display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """返回用于打印的副本：问题 + 评分列前置，长文本截断，分数保留 3 位小数。"""
    score_cols = [
        c for c in (
            "answer_correctness", "answer_relevancy", "faithfulness",
            "context_recall", "context_precision",
        ) if c in df.columns
    ]
    first = "question" if "question" in df.columns else df.columns[0]
    ordered = [first] + score_cols + [
        c for c in df.columns if c != first and c not in score_cols
    ]
    # round 只对数值列生效，其余列原样；map 统一转成短文本
    return df[ordered].round(3).map(_cell)


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
    print(_display_frame(df).to_string(index=False))


if __name__ == "__main__":
    main()
