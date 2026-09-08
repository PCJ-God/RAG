"""
Meta Prompting: 让大模型成为你的提示词教练（AI 裁判驱动版）
通过训练 AI 裁判自动判断提示词质量，实现自适应迭代，无需人为指定迭代次数
"""
from src.generation.llm_client import LLMClient
from src.config import LLM_MODEL
from typing import Optional


class PromptCoach:
    """提示词教练 —— 用 AI 裁判驱动提示词自动优化"""

    # ============ 阶段 1：训练 AI 裁判 ============

    JUDGE_PROMPT_V1 = """\
【角色】你是一位专业的提示词质量评估专家。
【任务】严格判断【待评估的提示词】是否合格。
【评判标准】
1. **角色定义 (必须满足)**: 是否清晰设定了 AI 的角色和职责边界？
2. **任务描述 (必须满足)**: 是否明确描述了需要完成的任务？
3. **输出格式 (必须满足)**: 是否指定了期望的输出格式或结构？
4. **约束条件 (必须满足)**: 是否包含了必要的限制条件？

【待评估的提示词】
---
{prompt_to_judge}
---
【输出要求】请只回答"通过"或"不通过"。
"""

    JUDGE_OPTIMIZER = """\
【角色】你是一位顶级的提示词工程专家。
【背景】我有一个"裁判提示词"，但它在以下案例上判断错误。
【当前裁判提示词】
---
{current_judge_prompt}
---
【判断错误的案例】
- 待评估内容: {misjudged_prompt}
- 裁判当前判断: {current_judgment}
- 正确判断应为: {correct_judgment}

【任务】重写"裁判提示词"，使其能正确判断此类案例。只返回优化后的裁判提示词。
"""

    # ============ 阶段 2：用裁判驱动提示词优化 ============

    ANALYZE_PROMPT = """\
你是一个专业的提示词教练（Prompt Coach）。请分析用户提供的提示词，指出其中的不足之处，并给出优化建议。

请从以下维度进行评价：
1. **角色定义**：是否清晰设定了 AI 的角色和职责？
2. **任务描述**：是否明确描述了需要完成的任务？
3. **输出格式**：是否指定了期望的输出格式或结构？
4. **约束条件**：是否包含了必要的限制条件？
5. **示例**：是否提供了 few-shot 示例？

请按以下格式回复：

【评分】X/10
【问题】
1. ...
2. ...
【建议】
1. ...
2. ...

用户提示词：
{prompt}
"""

    OPTIMIZE_PROMPT = """\
你是一个专业的提示词教练。请根据以下分析和原始提示词，生成一个优化后的版本。

优化原则：
1. 保持原始意图不变
2. 采用结构化排版（使用标题、分隔线、编号）
3. 补充缺失的角色定义、约束条件或输出格式
4. 如有必要，添加 few-shot 示例

【原始提示词】
{prompt}

【分析结果】
{analysis}

请直接输出优化后的提示词（不要加多余解释）：
"""

    GENERATOR_OPTIMIZER = """\
【角色】你是一位顶级的提示词工程专家。
【背景】我有一个"生成提示词"，但它生成的回答未能通过"AI 裁判"的审核。
【生成提示词】
---
{current_prompt}
---
【它生成的失败内容】
---
{failed_output}
---
【AI 裁判的规则书】
---
{judge_prompt}
---
【任务】重写"生成提示词"，确保新提示词能通过 AI 裁判审核。只返回优化后的提示词。
"""

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: LLM 客户端实例
        """
        self.llm_client = llm_client or LLMClient()
        self.judge_prompt = self.JUDGE_PROMPT_V1
        self.best_judge_prompt = self.JUDGE_PROMPT_V1

    # ============ 公共方法 ============

    def analyze(self, prompt: str) -> str:
        """分析提示词并给出评价和建议"""
        analysis_prompt = self.ANALYZE_PROMPT.format(prompt=prompt)
        return self.llm_client.invoke(analysis_prompt)

    def optimize(self, prompt: str, analysis: str = None) -> str:
        """优化提示词"""
        if analysis is None:
            analysis = self.analyze(prompt)
        optimize_prompt = self.OPTIMIZE_PROMPT.format(
            prompt=prompt,
            analysis=analysis
        )
        return self.llm_client.invoke(optimize_prompt)

    # ============ AI 裁判训练 ============

    def evaluate_judge_prompt(self, judge_prompt: str, dataset: list) -> tuple:
        """
        评估裁判提示词在数据集上的准确率

        Args:
            judge_prompt: 裁判提示词
            dataset: 标注样本列表，每项含 response/prompt_to_judge 和 label（"通过"/"不通过"）

        Returns:
            (accuracy, misjudged_cases)
        """
        correct = 0
        misjudged = []

        for sample in dataset:
            # 用当前裁判提示词判断
            text_to_judge = sample.get("prompt", sample.get("response", ""))
            filled = judge_prompt.format(prompt_to_judge=text_to_judge)
            predicted = self.llm_client.invoke(filled).strip().replace("。", "")

            # 提取判断结果（只保留"通过"或"不通过"）
            if "通过" in predicted and "不通过" not in predicted:
                predicted_cleaned = "通过"
            elif "不通过" in predicted:
                predicted_cleaned = "不通过"
            else:
                predicted_cleaned = "不通过"  # 默认为不通过

            label = sample["label"]
            if predicted_cleaned == label:
                correct += 1
            else:
                misjudged.append({
                    "content": text_to_judge,
                    "predicted": predicted_cleaned,
                    "correct": label
                })

        accuracy = correct / len(dataset) if dataset else 0
        return accuracy, misjudged

    def train_judge(self, labeled_samples: list, max_iterations: int = 5,
                   verbose: bool = True) -> dict:
        """
        训练 AI 裁判（用标注样本迭代优化裁判提示词）

        Args:
            labeled_samples: 标注样本列表，每项含 prompt/response 和 label（"通过"/"不通过"）
            max_iterations: 最大迭代轮数
            verbose: 是否打印训练过程

        Returns:
            包含 judge_prompt, train_accuracy, eval_accuracy, history 的字典
        """
        current_judge = self.JUDGE_PROMPT_V1
        best_judge = current_judge
        best_accuracy = 0
        history = []

        # 简单划分：前 70% 训练集，后 30% 评估集
        split = max(1, int(len(labeled_samples) * 0.7))
        train_set = labeled_samples[:split]
        eval_set = labeled_samples[split:]

        if verbose:
            print(f"训练集: {len(train_set)} 条, 评估集: {len(eval_set)} 条")

        for i in range(max_iterations):
            # 评估当前裁判
            train_acc, misjudged = self.evaluate_judge_prompt(current_judge, train_set)
            eval_acc, _ = self.evaluate_judge_prompt(current_judge, eval_set)

            history.append({
                "iteration": i + 1,
                "train_accuracy": train_acc,
                "eval_accuracy": eval_acc,
                "misjudged_count": len(misjudged)
            })

            if verbose:
                print(f"  轮次 {i+1}: 训练集准确率={train_acc:.2f}, "
                      f"评估集准确率={eval_acc:.2f}, 误判数={len(misjudged)}")

            # 更新最优裁判
            if eval_acc >= best_accuracy:
                best_accuracy = eval_acc
                best_judge = current_judge

            # 训练集无误判则停止
            if not misjudged:
                if verbose:
                    print("  训练集已无错误，训练完成。")
                break

            # 用误判案例优化裁判
            case = misjudged[0]
            opt_prompt = self.JUDGE_OPTIMIZER.format(
                current_judge_prompt=current_judge,
                misjudged_prompt=case["content"],
                current_judgment=case["predicted"],
                correct_judgment=case["correct"]
            )
            current_judge = self.llm_client.invoke(opt_prompt)

        self.judge_prompt = best_judge
        self.best_judge_prompt = best_judge

        return {
            "judge_prompt": best_judge,
            "train_accuracy": best_accuracy,
            "history": history
        }

    # ============ AI 裁判驱动优化 ============

    def judge(self, prompt: str) -> str:
        """
        用训练好的 AI 裁判判断提示词是否合格

        Args:
            prompt: 待评估的提示词

        Returns:
            "通过" 或 "不通过"
        """
        filled = self.judge_prompt.format(prompt_to_judge=prompt)
        result = self.llm_client.invoke(filled).strip().replace("。", "")
        if "不通过" in result:
            return "不通过"
        return "通过"

    def coach_auto(self, prompt: str, max_iterations: int = 5,
                   judge_prompt: str = None, verbose: bool = True) -> dict:
        """
        AI 裁判驱动的自动优化（无需人为指定迭代次数）

        Args:
            prompt: 原始提示词
            max_iterations: 最大迭代次数（安全上限）
            judge_prompt: 自定义裁判提示词（不传则用训练好的）
            verbose: 是否打印过程

        Returns:
            包含 analysis, optimized_prompt, iterations, judge_verdict 的字典
        """
        if judge_prompt:
            self.judge_prompt = judge_prompt

        current = prompt
        analysis = ""

        for i in range(max_iterations):
            # 1. 分析
            analysis = self.analyze(current)

            # 2. 优化
            current = self.optimize(current, analysis)

            # 3. 裁判判定
            verdict = self.judge(current)

            if verbose:
                print(f"  轮次 {i+1}: AI 裁判 → {verdict}")

            if verdict == "通过":
                if verbose:
                    print(f"  ✅ 优化成功！经过 {i+1} 轮迭代，AI 裁判判定通过。")
                return {
                    "original_prompt": prompt,
                    "analysis": analysis,
                    "optimized_prompt": current,
                    "iterations": i + 1,
                    "judge_verdict": verdict
                }

        if verbose:
            print(f"  ⚠️ 已达到最大迭代次数 ({max_iterations})，返回当前最优结果。")

        return {
            "original_prompt": prompt,
            "analysis": analysis,
            "optimized_prompt": current,
            "iterations": max_iterations,
            "judge_verdict": self.judge(current)
        }

    def coach(self, prompt: str) -> dict:
        """
        一键完成"分析 + 优化"流程（向后兼容，不带裁判）

        Args:
            prompt: 原始提示词

        Returns:
            包含 analysis, optimized_prompt 的字典
        """
        analysis = self.analyze(prompt)
        optimized = self.optimize(prompt, analysis)
        return {
            "analysis": analysis,
            "optimized_prompt": optimized
        }
