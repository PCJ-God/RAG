"""模糊问题改写：把模糊 / 含指代的问题扩充为完整、明确的检索问题。

改写结果只用于检索与召回，不改变最终对用户展示的原始问题。
"""
from typing import Optional

from src.config import Config, settings
from src.generation.llm import LLMClient


QUERY_REWRITE_PROMPT = """\
系统角色设定:
你是一个专业的问题改写助手。你的任务是将用户的原始问题扩充为一个更完整、更全面的问题。

规则：
1. 将可能的歧义、相关概念和上下文信息整合到一个完整的问题中
2. 使用括号对歧义概念进行补充说明
3. 添加关键的限定词和修饰语
4. 确保改写后的问题清晰且语义完整
5. 对于模糊概念，在括号中列举主要可能性
6. 改写结果必须控制在 1~2 句话、不超过 50 字
{history_section}
原始问题:
{query}

请生成一个综合的改写问题，确保：
- 包含原始问题的核心意图
- 涵盖可能的歧义解释
- 使用清晰的逻辑关系词连接不同方面
- 必要时使用括号补充说明
- 简洁：1~2 句话，不超过 50 字

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
