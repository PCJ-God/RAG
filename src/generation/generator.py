"""答案生成器：把检索上下文拼进提示词，生成带引用的回答。"""
from dataclasses import dataclass, field
from typing import List, Optional

from llama_index.core.schema import NodeWithScore

from src.generation.llm import LLMClient
from src.generation.prompts import RAG_ANSWER_TEMPLATE


@dataclass
class Source:
    """一个被引用的文档片段。"""

    index: int
    text: str
    score: float
    header_path: str = ""
    file_name: str = ""


@dataclass
class Answer:
    """问答结果。"""

    answer: str
    sources: List[Source] = field(default_factory=list)


class AnswerGenerator:
    """答案生成器。"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        question: str,
        nodes: List[NodeWithScore],
        history: str = "",
        profile: str = "",
    ) -> Answer:
        contexts = []
        sources = []
        for i, item in enumerate(nodes, start=1):
            text = item.node.get_content().strip()
            meta = item.node.metadata
            contexts.append(f"{self._source_label(i, meta)}\n{text}")
            sources.append(
                Source(
                    index=i,
                    text=text,
                    score=item.score or 0.0,
                    header_path=meta.get("header_path", ""),
                    file_name=meta.get("file_name", ""),
                )
            )

        prompt = RAG_ANSWER_TEMPLATE.format(
            context_str="\n\n".join(contexts),
            query_str=question,
            history_str=history or "（无）",
        )
        system_prompt = None
        if profile:
            system_prompt = (
                "你是 RAG 知识助教，只依据参考资料回答。\n"
                f"以下是用户画像，请结合它回答：\n{profile}"
            )
        answer_text = self.llm_client.invoke(prompt, system_prompt=system_prompt)
        return Answer(answer=answer_text.strip(), sources=sources)

    @staticmethod
    def _source_label(index: int, meta) -> str:
        """把文件名和章节拼进编号标签，让引用能对应到具体文档。"""
        parts = [
            str(meta.get("file_name", "") or "").strip(),
            str(meta.get("header_path", "") or "").strip(),
        ]
        parts = [p for p in parts if p]
        return f"[{index}]（{' · '.join(parts)}）" if parts else f"[{index}]"
