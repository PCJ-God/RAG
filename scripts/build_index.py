"""构建知识库索引。

用法：
    python scripts/build_index.py                 # 构建默认域 rag
    python scripts/build_index.py --domain agent  # 构建其他知识域（后续自行添加）
    python scripts/build_index.py --rebuild       # 强制重建
"""
import sys
from pathlib import Path

# 让脚本从任意目录运行时都能找到项目根目录下的 src 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 知识库索引")
    parser.add_argument(
        "--domain", default="rag", help="知识域目录名（对应 data/knowledge/<domain>）"
    )
    parser.add_argument("--rebuild", action="store_true", help="强制重建索引")
    args = parser.parse_args()

    pipeline = RAGPipeline(domain=args.domain)

    if pipeline.indexer.store_type == "dashvector":
        where = f"DashVector collection「{pipeline.indexer.collection_name}」"
    else:
        where = str(pipeline.indexer.persist_dir)

    if not args.rebuild and pipeline.indexer.exists():
        print(f"索引已存在（{where}），如需重建请加 --rebuild")
        return

    count = pipeline.build(args.domain, rebuild=args.rebuild)
    print(f"完成：共 {count} 个文档块，已存入 {where}")


if __name__ == "__main__":
    main()
