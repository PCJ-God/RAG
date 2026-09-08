#!/usr/bin/env python
"""
运行自动化评测脚本
"""
import sys
from pathlib import Path
import json

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import check_api_key, DATA_DIR, DOCS_DIR
from src.document.loader import DocumentLoader
from src.document.splitter import SplitterFactory
from src.storage.index_manager import IndexManager
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient
from src.evaluation.ragas_eval import RagasEvaluator


def main():
    """主函数"""
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

    # 初始化 RAG 系统
    print("\n正在初始化 RAG 系统...")
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
    retriever.create_query_engine(llm=llm_client.get_llm())

    # 收集回答和上下文
    print("\n正在向 RAG 系统提问并收集回答...")
    questions = []
    answers = []
    ground_truths = []
    contexts_list = []

    for i, qa in enumerate(qa_pairs, 1):
        question = qa['question']
        truth = qa['ground_truth']

        print(f"  [{i}/{len(qa_pairs)}] 正在回答: {question[:30]}...")

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
    print("\n正在运行 Ragas 评测...")
    evaluator = RagasEvaluator()
    result = evaluator.batch_evaluate(data_samples)

    # 打印结果
    print("\n" + "=" * 60)
    print("📈 评测结果")
    print("=" * 60)
    print(result.to_string(index=False))

    # 保存结果
    output_file = DATA_DIR / "eval_result.csv"
    result.to_csv(output_file, index=False)
    print(f"\n💾 评测结果已保存到 {output_file}")

    # 分指标打印平均值
    print("\n" + "-" * 60)
    print("📊 各指标平均分:")
    print("-" * 60)
    for col in result.columns:
        if result[col].dtype in ['float64', 'int64']:
            print(f"  {col}: {result[col].mean():.4f}")

    # 逐题详情
    print("\n" + "-" * 60)
    print("📋 逐题详情:")
    print("-" * 60)
    for i, qa in enumerate(qa_pairs):
        print(f"\nQ: {qa['question']}")
        print(f"  A: {answers[i][:150]}...")
        print(f"  GT: {qa['ground_truth'][:100]}...")


if __name__ == "__main__":
    main()
