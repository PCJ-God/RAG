"""复杂问题拆解：把复杂问题拆成若干子问题，分别检索后再整合成最终回答。"""
from typing import List, Optional, Tuple

from src.config import Config, settings
from src.generation.llm import LLMClient


DECOMPOSE_PROMPT = """\
你是问题拆解助手。判断问题是否包含多个并列的独立意图。

规则：
1. 如果问题只有一个意图，原样输出该问题（一行）
2. 只有问题确实包含 2 个及以上独立意图时，才拆成对应的子问题，每个子问题一行（最多 4 个，宁少勿多）
3. 不要输出编号、解释或任何多余文字

问题：
{query}
"""

MERGE_PROMPT = """\
下面是把用户问题拆解后的子问题及其答案。请整合成一份连贯、准确的最终回答。

用户问题：{question}

子问题与答案：
{qa_text}

要求：
1. 直接回答用户问题，结构清晰
2. 覆盖所有子问题答案中的关键信息，不要遗漏
3. 不要编造子答案中没有的内容
"""


class QueryDecomposer:
    """复杂问题拆解与子答案整合。"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or settings
        self.llm_client = llm_client or LLMClient(self.config)

    def decompose(self, query: str) -> List[str]:
        """返回子问题列表；简单问题返回单个元素（原问题本身）。"""
        prompt = DECOMPOSE_PROMPT.format(query=query)
        text = (self.llm_client.invoke(prompt) or "").strip()

        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # 去掉可能的编号/符号前缀
            if line[:2] in ("- ", "· ", "* "):
                line = line[2:].strip()
            elif len(line) > 2 and line[0].isdigit() and line[1] in ".、)":
                line = line[2:].strip()
            if line:
                lines.append(line)
        if lines:
            lines = lines[:4]  # 硬上限：最多 4 个子问题
        return lines if lines else [query]

    def merge(self, question: str, qa_pairs: List[Tuple[str, str]]) -> str:
        """把各子问题答案整合成最终回答。"""
        qa_text = "\n\n".join(
            f"子问题：{q}\n答案：{a}" for q, a in qa_pairs
        )
        prompt = MERGE_PROMPT.format(question=question, qa_text=qa_text)
        return (self.llm_client.invoke(prompt) or "").strip()
