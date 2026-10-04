"""查询标签抽取：从用户问题中抽取一个知识点标签，用于标签检索。"""
from typing import List, Optional

from src.config import TAG_LIST, Config, settings
from src.generation.llm import LLMClient


class TagExtractor:
    """用 LLM 从问题中抽取知识点标签。"""

    PROMPT = """\
你是标签提取助手。请判断用户问题属于下面哪个知识点标签。

可选标签（必须且只能从中选一个）：
{tag_list}

如果问题与这些标签都无关，输出：无

只输出标签名称本身，不要输出任何解释、标点或多余文字。

问题：{question}
"""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or settings
        self.llm_client = llm_client or LLMClient(self.config)

    def extract(
        self, question: str, tag_list: Optional[List[str]] = None
    ) -> Optional[str]:
        tag_list = tag_list or TAG_LIST
        prompt = self.PROMPT.format(
            tag_list="、".join(tag_list),
            question=question,
        )
        tag = self.llm_client.invoke(prompt).strip()
        return tag if tag in tag_list else None
