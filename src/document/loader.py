"""
文档加载器
支持PDF/Markdown/DOCX等多种格式的文档加载
"""
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from typing import List, Optional


class DocumentLoader:
    """文档加载器"""
    
    def __init__(self, docs_dir: str, required_exts: Optional[List[str]] = None):
        """
        Args:
            docs_dir: 文档目录路径
            required_exts: 支持的文件扩展名列表，None表示使用默认列表
        """
        self.docs_dir = Path(docs_dir)
        self.required_exts = required_exts or ['.pdf', '.md', '.docx', '.txt']
    
    def load(self) -> List:
        """加载文档"""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"文档目录不存在: {self.docs_dir}")
        
        print(f"正在从 {self.docs_dir} 加载文档...")
        documents = SimpleDirectoryReader(
            str(self.docs_dir),
            required_exts=self.required_exts
        ).load_data()
        
        print(f"✅ 成功加载 {len(documents)} 个文档")
        return documents
    
    def load_single(self, file_path: str) -> List:
        """加载单个文档文件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        documents = SimpleDirectoryReader(
            input_files=[str(file_path)]
        ).load_data()
        
        return documents
