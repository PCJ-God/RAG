"""模糊问题改写：把模糊 / 含指代的问题扩充为完整、明确的检索问题。

改写结果只用于检索与召回，不改变最终对用户展示的原始问题。
"""
from typing import Optional

from src.config import Config, settings
from src.generation.llm import LLMClient


QUERY_REWRITE_PROMPT = """\
系统角色设定:
你是一个专业的问题改写助手。你的任务是把一个表述模糊、简短或含指代的问题，改写成一句意图单一、清晰完整的检索问题。

规则：
1. 澄清问题最可能指的那一个意图，不要扩写成多个意图
2. 使用括号对歧义概念做最小必要的补充，不要罗列多种可能性
3. 添加必要的限定词，使问题清晰且语义完整
4. 改写结果必须是一句话、不超过 50 字
{history_section}
原始问题:
{query}

输出格式：
[综合改写] - 改写后的问题
"""


class QueryRewriter:
    """用 LLM 改写问题，供检索使用。"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or settings
        self.llm_client = llm_client or LLMClient(self.config)

    def rewrite(self, query: str, history: str = "") -> str:
        history_section = ""
        if history:
            history_section = (
                "\n对话历史（用于消解“它”“第二步”等指代）:\n" + history
            )
        prompt = QUERY_REWRITE_PROMPT.format(
            query=query,
            history_section=history_section,
        )
        text = (self.llm_client.invoke(prompt) or "").strip()

        marker = "[综合改写]"
        if marker in text:
            text = text.split(marker, 1)[1].strip()
        text = text.lstrip("-：: ").strip()
        return text if text else query
