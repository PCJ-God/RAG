#!/usr/bin/env python
"""
运行自动化评测脚本
支持策略对比：baseline / rerank / hyde / rerank+hyde
"""
import sys
from pathlib import Path
import json
import argparse

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import check_api_key, DATA_DIR, DOCS_DIR
from src.document.loader import DocumentLoader
from src.document.splitter import SplitterFactory
from src.storage.index_manager import IndexManager
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient
from src.evaluation.ragas_eval import RagasEvaluator


def run_eval(qa_pairs, use_rerank=False, use_hyde=False, label="baseline"):
    """
    运行一次评测

    Args:
        qa_pairs: 问答对列表
        use_rerank: 是否启用 Rerank
        use_hyde: 是否启用 HyDE
        label: 策略标签

    Returns:
        (result_df, answers_list)
    """
    print(f"\n{'=' * 60}")
    print(f"🔬 策略: {label}")
    if use_rerank:
        print("   🔀 Rerank: ON")
    if use_hyde:
        print("   🧪 HyDE: ON")
    print(f"{'=' * 60}")

    # 初始化 RAG 系统
    loader = DocumentLoader(str(DOCS_DIR))
    documents = loader.load()

    splitter = SplitterFactory.create_sentence_splitter()
    nodes = splitter.get_nodes_from_documents(documents)

    index_manager = IndexManager()
    try:
        index = index_manager.load_index()
    except FileNotFoundError:
        index = index_manager.build_index(nodes)

    llm_client = LLMClient()
    retriever = Retriever(index)

    if use_rerank:
        retriever.enable_rerank(llm_client=llm_client)
    if use_hyde:
        retriever.enable_hyde()

    retriever.create_query_engine(llm=llm_client.get_llm())

    # 收集回答
    print(f"\n正在提问并收集回答 ({len(qa_pairs)} 题)...")
    questions = []
    answers = []
    ground_truths = []
    contexts_list = []

    for i, qa in enumerate(qa_pairs, 1):
        question = qa['question']
        truth = qa['ground_truth']

        print(f"  [{i}/{len(qa_pairs)}] 正在回答: {question[:40]}...")

        response = retriever.query(question)
        answer = str(response)
        contexts = [node.text for node in response.source_nodes]

        questions.append(question)
        answers.append(answer)
        ground_truths.append(truth)
        contexts_list.append(contexts)

    # 构建评测数据集
    data_samples = {
        'question': questions,
        'answer': answers,
        'ground_truth': ground_truths,
        'contexts': contexts_list
    }

    # 运行评测
    print(f"\n正在对 [{label}] 运行 Ragas 评测...")
    evaluator = RagasEvaluator()
    result = evaluator.batch_evaluate(data_samples)

    return result, answers


def print_comparison(results_dict):
    """打印策略对比表"""
    print("\n" + "=" * 60)
    print("📊 策略对比结果")
    print("=" * 60)

    header = f"{'策略':<20}"
    cols = set()
    for df in results_dict.values():
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                cols.add(col)

    for col in sorted(cols):
        header += f" | {col:^12}"
    print(header)
    print("-" * len(header))

    for label, df in results_dict.items():
        row = f"{label:<20}"
        for col in sorted(cols):
            if col in df.columns:
                row += f" | {df[col].mean():12.4f}"
            else:
                row += f" | {'N/A':^12}"
        print(row)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RAG 自动化评测")
    parser.add_argument("--rerank", action="store_true", help="启用 Rerank")
    parser.add_argument("--hyde", action="store_true", help="启用 HyDE")
    parser.add_argument("--all", action="store_true",
                        help="运行所有策略对比: baseline / rerank / hyde / rerank+hyde")
    args = parser.parse_args()

    check_api_key()

    print("=" * 60)
    print("📊 开始自动化评测...")
    print("=" * 60)

    # 加载测试问答对
    qa_file = DATA_DIR / "qa_pairs.json"
    if not qa_file.exists():
        print(f"❌ 未找到测试数据: {qa_file}")
        return

    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)

    print(f"📝 加载了 {len(qa_pairs)} 个测试问答对")

    # 决定运行哪些策略
    if args.all:
        strategies = [
            (False, False, "baseline"),
            (True, False, "rerank"),
            (False, True, "hyde"),
            (True, True, "rerank+hyde"),
        ]
    else:
        strategies = [(args.rerank, args.hyde, "custom")]

    results_dict = {}
    all_answers = {}

    for use_rerank, use_hyde, label in strategies:
        result, answers = run_eval(qa_pairs, use_rerank, use_hyde, label)
        results_dict[label] = result
        all_answers[label] = answers

        # 保存单次结果
        output_file = DATA_DIR / f"eval_result_{label}.csv"
        result.to_csv(output_file, index=False)
        print(f"💾 [{label}] 结果已保存到 {output_file}")

    # 策略对比
    if len(results_dict) > 1:
        print_comparison(results_dict)

    # 打印最后一组的详细信息
    last_label = strategies[-1][2]
    last_answers = all_answers[last_label]
    print("\n" + "-" * 60)
    print(f"📋 逐题详情 ({last_label}):")
    print("-" * 60)
    for i, qa in enumerate(qa_pairs):
        print(f"\nQ: {qa['question']}")
        print(f"  A: {last_answers[i][:150]}...")
        print(f"  GT: {qa['ground_truth'][:100]}...")


if __name__ == "__main__":
    main()
