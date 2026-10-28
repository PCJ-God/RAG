"""意图识别：判断用户问题属于单意图、多意图，还是信息不足需要追问。

- 单意图（含模糊但可安全推断）走改写澄清；
- 多意图走拆解；
- 需追问则与用户交互补全问题（见 pipeline 中的澄清循环）。
"""
from typing import Optional

from src.config import Config, settings
from src.generation.llm import LLMClient


class IntentAnalyzer:
    """用 LLM 对用户问题做三分类。"""

    PROMPT = """\
你是意图识别助手。判断用户问题属于下面哪一类，只输出标签本身（单意图 / 多意图 / 需追问），不要任何解释或标点。

判断规则：
1. 单意图：问题只有一个核心意图，且信息足够明确（哪怕表述简短，也能安全推断出唯一意图）。例如"RAG"可推断为"什么是RAG"。
2. 多意图：问题包含两个及以上明确并列的独立意图（如"A 和 B 分别…""对比 A 与 B""A 的步骤以及 B 的步骤"）。
3. 需追问：问题模糊、有歧义或缺少必要信息，无法安全推断唯一意图（如"它是什么""那个步骤呢"且上下文无法消解，或一个词对应多种含义无法确定指哪个）。

{history_section}
问题：{question}
"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or settings
        self.llm_client = llm_client or LLMClient(self.config)

    def classify(self, question: str, history: str = "") -> str:
        """返回 单意图 / 多意图 / 需追问；解析失败默认按单意图（宁少拆）。"""
        history_section = ""
        if history:
            history_section = f"\n对话历史（用于消解指代）：\n{history}"
        prompt = self.PROMPT.format(
            question=question,
            history_section=history_section,
        )
        text = (self.llm_client.invoke(prompt) or "").strip()
        for label in ("单意图", "多意图", "需追问"):
            if label in text:
                return label
        return "单意图"
