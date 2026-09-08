#!/usr/bin/env python
"""
Meta Prompting CLI: 让大模型成为你的提示词教练
支持两种模式：
  1. 固定迭代模式（--iterations N）
  2. AI 裁判驱动模式（--auto，自动判断何时停止）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.prompt_coach import PromptCoach


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Meta Prompting - 提示词教练")
    parser.add_argument("--prompt", "-p", type=str, help="待优化的提示词")
    parser.add_argument("--analyze-only", action="store_true",
                        help="只分析不优化")
    parser.add_argument("--iterations", "-n", type=int, default=1,
                        help="固定迭代次数（默认1）")
    parser.add_argument("--auto", action="store_true",
                        help="AI 裁判驱动模式（自动判断何时停止）")
    parser.add_argument("--max-iterations", type=int, default=5,
                        help="AI 裁判模式的最大迭代次数（默认5）")
    parser.add_argument("--train-judge", action="store_true",
                        help="训练 AI 裁判模式（需提供标注样本文件）")
    parser.add_argument("--samples", type=str,
                        help="标注样本 JSON 文件路径（用于训练 AI 裁判）")

    args = parser.parse_args()

    coach = PromptCoach()

    # ---- 训练 AI 裁判 ----
    if args.train_judge:
        import json
        if not args.samples:
            print("❌ 训练 AI 裁判需要提供标注样本文件: --samples labels.json")
            return

        with open(args.samples, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        print("=" * 60)
        print("🎓 训练 AI 裁判")
        print("=" * 60)
        result = coach.train_judge(samples, max_iterations=args.max_iterations)
        print(f"\n✅ 训练完成！")
        print(f"   最终训练集准确率: {result['train_accuracy']:.2f}")
        print(f"\n训练后的裁判提示词:")
        print(result['judge_prompt'])
        return

    # ---- 普通模式 / AI 裁判模式 ----
    if args.prompt:
        current_prompt = args.prompt

        if args.analyze_only:
            analysis = coach.analyze(current_prompt)
            print(f"\n【分析结果】\n{analysis}")
            return

        if args.auto:
            # AI 裁判驱动模式
            print("=" * 60)
            print("🤖 AI 裁判驱动优化模式")
            print(f"最大迭代次数: {args.max_iterations}")
            print("=" * 60)

            result = coach.coach_auto(
                current_prompt,
                max_iterations=args.max_iterations
            )

            print(f"\n{'=' * 60}")
            print(f"最终结果（经过 {result['iterations']} 轮迭代，"
                  f"AI 裁判判定: {result['judge_verdict']}）:")
            print(f"{'=' * 60}\n{result['optimized_prompt']}")
        else:
            # 固定迭代模式
            for i in range(args.iterations):
                print(f"\n{'=' * 60}")
                print(f"第 {i + 1} 轮优化")
                print(f"{'=' * 60}")

                analysis = coach.analyze(current_prompt)
                print(f"\n【分析结果】\n{analysis}")

                optimized = coach.optimize(current_prompt, analysis)
                print(f"\n【优化后提示词】\n{optimized}")

                current_prompt = optimized

            print(f"\n{'=' * 60}")
            print(f"最终优化后的提示词（共 {args.iterations} 轮迭代）:")
            print(f"{'=' * 60}\n{current_prompt}")

    else:
        # 交互模式
        print("=" * 60)
        print("🎓 Meta Prompting - 提示词教练")
        print("模式: 1=固定迭代  2=AI 裁判驱动")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 60)

        while True:
            mode = input("\n选择模式 (1/2): ").strip()

            if mode.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break

            prompt = input("\n📝 请输入提示词: ").strip()
            if not prompt:
                continue

            if mode == "2":
                # AI 裁判模式
                result = coach.coach_auto(prompt, max_iterations=5)
                print(f"\n{'=' * 60}")
                print(f"AI 裁判判定: {result['judge_verdict']} "
                      f"(经过 {result['iterations']} 轮)")
                print(f"{'=' * 60}")
                print(result['optimized_prompt'])
            else:
                # 固定迭代
                result = coach.coach(prompt)
                print(f"\n{'=' * 60}")
                print("📊 分析结果:")
                print(f"{'=' * 60}")
                print(result['analysis'])
                print(f"\n{'=' * 60}")
                print("✨ 优化后的提示词:")
                print(f"{'=' * 60}")
                print(result['optimized_prompt'])
            print()


if __name__ == "__main__":
    main()
