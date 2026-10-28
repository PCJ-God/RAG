"""运行 Ragas 评测。

用法：
    python scripts/eval.py
    python scripts/eval.py --qa-file data/eval/qa_pairs.json
    python scripts/eval.py --output data/eval/results.csv

评测结果写入文件（不在终端打印结果表格）：
    - 完整结果：CSV（默认 data/eval/results.csv，UTF-8 BOM，Excel 可直接打开）
    - 可读摘要：Markdown 表格（同目录同名 .md，长文本截断，防止表格被撑爆）
"""
import sys
from pathlib import Path

# 让脚本从任意目录运行时都能找到项目根目录下的 src 包。
# notebook 单元格里没有 __file__，此时回退到当前工作目录（建议从项目根目录启动 jupyter）。
try:
    _ROOT = Path(__file__).resolve().parent.parent
    _RUN_AS_SCRIPT = True
except NameError:
    _ROOT = Path.cwd().resolve()
    _RUN_AS_SCRIPT = False
sys.path.insert(0, str(_ROOT))

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

# Markdown 摘要里单元格的最大宽度：超出截断加省略号，避免长文本把表格撑爆
CELL_WIDTH = 40

# 中英文混排对齐，避免表格错位（to_string 回退时生效）
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
    """可读摘要：问题 + 评分列前置，长文本截断，分数保留 3 位小数。"""
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
    return df[ordered].round(3).map(_cell)


def _file_frame(df: pd.DataFrame) -> pd.DataFrame:
    """完整结果：list 列（contexts）拼接成多行文本，其余列保留原样。"""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(
                lambda v: "\n\n".join(str(x) for x in v) if isinstance(v, list) else v
            )
    return out


def _to_markdown(show: pd.DataFrame) -> str:
    """把摘要 DataFrame 转成 Markdown 文本（tabulate 缺失时退回文本表格）。"""
    try:
        table = show.to_markdown(index=False)
    except ImportError:
        table = show.to_string(index=False)
    return "# 评测结果\n\n" + table + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ragas 评测")
    parser.add_argument("--domain", default="rag")
    parser.add_argument(
        "--qa-file", default=str(EVAL_DIR / "qa_pairs.json"), help="评测问答对文件"
    )
    parser.add_argument(
        "--output",
        default=str(EVAL_DIR / "results.csv"),
        help="完整结果输出文件（CSV），同目录会再生成一份 .md 可读摘要",
    )
    if argv is None and not _RUN_AS_SCRIPT:
        argv = []  # notebook 单元格粘贴运行：用默认参数，避免误解析内核启动参数
    args = parser.parse_args(argv)

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

    # 写入文件（不在终端打印结果表格）
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _file_frame(df).to_csv(out_path, index=False, encoding="utf-8-sig")

    md_path = out_path.with_suffix(".md")
    md_path.write_text(_to_markdown(_display_frame(df)), encoding="utf-8")

    print(f"\n评测完成，共 {len(rows)} 条")
    print(f"完整结果：{out_path}")
    print(f"可读摘要：{md_path}")


if __name__ == "__main__":
    main()
