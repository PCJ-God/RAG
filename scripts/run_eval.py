#!/usr/bin/env python
"""
运行自动化评测脚本
"""
import sys
from pathlib import Path
import json

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import check_api_key, DATA_DIR
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
    
    # 注意：这里需要实际的RAG系统回答
    # 在真实场景中，应该先运行RAG系统获取回答
    
    print("\n⚠️  自动化评测需要先获取RAG系统的回答")
    print("建议使用以下方式之一：")
    print("1. 使用 src/app/cli.py 运行问答并保存结果")
    print("2. 使用 Jupyter Notebook 交互式测试")
    print("3. 编写集成测试脚本自动收集回答")
    
    # 示例：如何运行评测
    print("\n" + "=" * 60)
    print("📖 评测示例代码：")
    print("=" * 60)
    print("""
from src.evaluation.ragas_eval import RagasEvaluator

evaluator = RagasEvaluator()

# 单个问题评测
result = evaluator.evaluate_answer_quality(
    question="张伟是哪个部门的？",
    answer="张伟在教研部。",
    ground_truth="公司有三名张伟，分别在教研部、课程开发部和IT部。",
    contexts=["张伟是教研部的成员"]
)
print(result)

# 批量评测
data_samples = {
    'question': ["问题1", "问题2"],
    'answer': ["回答1", "回答2"],
    'ground_truth': ["标准答案1", "标准答案2"],
    'contexts': [["上下文1"], ["上下文2"]]
}
result = evaluator.batch_evaluate(data_samples)
print(result)
""")


if __name__ == "__main__":
    main()
