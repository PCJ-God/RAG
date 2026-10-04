"""全局记忆提取：每轮对话结束后，由 LLM 抽取用户的长期事实。"""
from typing import List, Optional

from src.config import Config, settings
from src.generation.llm import LLMClient


GLOBAL_MEMORY_PROMPT = """\
你是用户画像提取助手。请判断下面这轮对话中是否包含关于用户的长期事实。

长期事实指：身份、职业、技能、偏好、所属公司、学习目标等"长期稳定"的信息；
临时性、一次性的话题内容不算。

规则：
1. 一条事实一行，用简洁中文陈述（例如：用户是程序员 / 用户喜欢用 Python）
2. 只输出事实本身，不要编号、解释或多余文字
3. 如果没有长期事实，只输出：无

对话：
{conversation}
"""


class GlobalMemoryExtractor:
    """从每轮对话中抽取长期用户事实。"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or settings
        self.llm_client = llm_client or LLMClient(self.config)

    def extract(self, question: str, answer: str) -> List[str]:
        prompt = GLOBAL_MEMORY_PROMPT.format(
            conversation=f"用户：{question}\n助手：{answer}"
        )
        text = (self.llm_client.invoke(prompt) or "").strip()
        if not text or text == "无":
            return []

        facts = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line == "无":
                continue
            # 去掉可能的编号/符号前缀
            if line[:2] in ("- ", "· ", "* "):
                line = line[2:].strip()
            elif len(line) > 2 and line[0].isdigit() and line[1] in ".、)":
                line = line[2:].strip()
            if line:
                facts.append(line)
        return facts
