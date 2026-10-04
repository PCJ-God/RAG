"""单轮问答 CLI。

用法：
    python scripts/ask.py "RAGAS 有几个常用指标？"
    python scripts/ask.py "RAG 建立索引包括哪几步？"
    python scripts/ask.py "什么是上下文工程？" --show-sources
    python scripts/ask.py                          # 进入交互模式
"""
import sys
from pathlib import Path

# 让脚本从任意目录运行时都能找到项目根目录下的 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.pipeline import RAGPipeline


def print_answer(result, show_sources: bool = False):
    print("\n" + "=" * 60)
    print("回答：")
    print(result.answer)
    if show_sources and result.sources:
        print("\n参考来源：")
        for s in result.sources:
            header = f" | {s.header_path}" if s.header_path else ""
            print(f"  [{s.index}] ({s.file_name}{header}) score={s.score:.4f}")
            print(f"      {s.text[:120]}...")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RAG 知识问答")
    parser.add_argument("question", nargs="?", help="要问的问题（留空进入交互模式）")
    parser.add_argument("--domain", default="rag")
    parser.add_argument("--top-k", type=int, default=None, help="最终用于生成的文档块数量")
    parser.add_argument("--no-rerank", action="store_true", help="关闭重排序")
    parser.add_argument("--no-tag", action="store_true", help="关闭标签检索")
    parser.add_argument("--show-sources", action="store_true", help="展示参考来源")
    args = parser.parse_args()

    pipeline = RAGPipeline(domain=args.domain).load()

    def ask(question: str):
        return pipeline.query(
            question,
            top_k=args.top_k,
            use_rerank=not args.no_rerank,
            use_tag=not args.no_tag,
        )

    if args.question:
        print_answer(ask(args.question), args.show_sources)
        return

    print("进入交互模式（输入 q 退出）")
    while True:
        try:
            question = input("\n问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("q", "quit", "exit"):
            break
        if not question:
            continue
        print_answer(ask(question), args.show_sources)


if __name__ == "__main__":
    main()
