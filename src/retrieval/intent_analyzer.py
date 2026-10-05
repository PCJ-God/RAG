"""意图识别：判断用户问题包含单一意图还是多个并列意图。

单意图（含模糊但单一）走改写澄清；多意图才走拆解，避免把单意图问题
人为扩成多意图。
"""
from typing import Optional

from src.config import Config, settings
from src.generation.llm import LLMClient


class IntentAnalyzer:
    """用 LLM 判断问题意图数量。"""

    PROMPT = """\
你是意图识别助手。判断用户问题包含几个核心意图。

规则：
1. 问题只有一个核心意图（哪怕表述模糊、简短），输出：单意图
2. 问题包含两个及以上并列的独立意图（如“A 和 B 分别…”“对比 A 与 B”“A 的步骤以及 B 的步骤”），输出：多意图
3. 只输出“单意图”或“多意图”四个字，不要任何解释或标点

问题：{question}
"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or settings
        self.llm_client = llm_client or LLMClient(self.config)

    def is_multi(self, question: str) -> bool:
        """返回 True 表示多意图。解析失败默认按单意图处理（宁少拆）。"""
        prompt = self.PROMPT.format(question=question)
        text = (self.llm_client.invoke(prompt) or "").strip()
        return "多意图" in text
