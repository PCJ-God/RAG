"""文档加载：把知识库目录里的 Markdown 文档读进来。"""
from pathlib import Path
from typing import List

from llama_index.core import Document, SimpleDirectoryReader


class MarkdownLoader:
    """加载某个知识域目录下的 Markdown 文档。"""

    def __init__(self, docs_dir: str):
        self.docs_dir = Path(docs_dir)

    def load(self) -> List[Document]:
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"知识库目录不存在: {self.docs_dir}")

        documents = SimpleDirectoryReader(
            str(self.docs_dir),
            required_exts=[".md"],
            recursive=True,
        ).load_data()

        print(f"[loader] 加载 {len(documents)} 个文档")
        return documents
