"""追问器：问题信息不足时，向用户提出一个最关键的具体追问。"""
from typing import Optional

from src.config import Config, settings
from src.generation.llm import LLMClient


CLARIFY_PROMPT = """\
你是追问助手。用户的问题不够明确，你需要通过追问补充关键信息。

规则：
1. 只问最关键的 1 个问题，这个问题的答案能直接决定能否回答
2. 追问要具体、可回答（例如"你说的『向量』是指文本向量化，还是向量数据库？"），不要问"请说清楚一点"这类空泛问题
3. 只输出追问本身，不要任何解释、编号或前缀

用户问题：{question}
"""


class Clarifier:
    """用 LLM 生成追问。"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or settings
        self.llm_client = llm_client or LLMClient(self.config)

    def ask(self, question: str) -> str:
        prompt = CLARIFY_PROMPT.format(question=question)
        text = (self.llm_client.invoke(prompt) or "").strip()
        return text or "能否再具体说明一下你的问题？"
