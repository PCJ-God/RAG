"""高质量分块：结构感知 + 单块大小上限 + 文本清洗 + 打标签。

针对 Markdown 课程笔记（由 Notebook 导出）做了四点优化：
1. 先按标题层级切成小节，并把标题路径写入每个块的 header_path 元数据；
2. 超长小节再按句子二次切分，同时继承标题路径；
3. 清洗 HTML 标签等导出噪声，减少对向量化的干扰；
4. 按文档给每个块打上知识点标签（tag），用于后续标签检索。
"""
import html
import re
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import TextNode

from src.config import get_tag

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t]+")
_NEWLINE_RE = re.compile(r"\n{3,}")


class QualitySplitter:
    """结构感知分块器。"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._section_parser = MarkdownNodeParser()
        self._sub_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def get_nodes(self, documents: List[Document]) -> List[TextNode]:
        nodes: List[TextNode] = []
        for doc in documents:
            nodes.extend(self._split_document(doc))
        print(f"[splitter] 共切分 {len(nodes)} 个文档块")
        return nodes

    def _split_document(self, doc: Document) -> List[TextNode]:
        result: List[TextNode] = []
        tag = get_tag(doc.metadata.get("file_name", ""))
        sections = self._section_parser.get_nodes_from_documents([doc])

        for section in sections:
            text = self._clean(section.get_content())
            if not text:
                continue
            # 只保留后续检索/展示需要的字段，且统一转为字符串，
            # 兼容 DashVectorStore 的 flat_metadata 校验（值必须为 str/int/float/None）
            meta = {
                "tag": tag,
                "file_name": str(section.metadata.get("file_name") or ""),
                "header_path": str(section.metadata.get("header_path") or ""),
            }

            if len(text) <= self.chunk_size:
                section.set_content(text)
                section.metadata = meta
                result.append(section)
            else:
                # 超长小节再按句子切分，继承标题路径与标签等元数据
                sub_nodes = self._sub_splitter.get_nodes_from_documents(
                    [Document(text=text, metadata=meta)]
                )
                result.extend(sub_nodes)

        return result

    @staticmethod
    def _clean(text: str) -> str:
        text = _TAG_RE.sub(" ", text)
        text = html.unescape(text)
        text = text.replace("\u00a0", " ")
        text = _SPACE_RE.sub(" ", text)
        text = _NEWLINE_RE.sub("\n\n", text)
        return text.strip()
