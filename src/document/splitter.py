"""
分块策略模块
支持多种文档分块方法
"""
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    SentenceWindowNodeParser,
    MarkdownNodeParser,
    TokenTextSplitter
)
from llama_index.core import Document
from typing import List, Optional


class SplitterFactory:
    """分块策略工厂"""
    
    SPLITTERS = {
        'sentence': 'sentence',
        'window': 'sentence_window',
        'semantic': 'semantic',
        'markdown': 'markdown',
        'token': 'token'
    }
    
    @classmethod
    def create(cls, splitter_type: str = 'sentence', **kwargs):
        """
        创建分块器
        
        Args:
            splitter_type: 分块类型
            **kwargs: 分块参数
            
        Returns:
            分块器实例
        """
        if splitter_type == 'sentence':
            return cls.create_sentence_splitter(**kwargs)
        elif splitter_type == 'window':
            return cls.create_sentence_window_splitter(**kwargs)
        elif splitter_type == 'semantic':
            return cls.create_semantic_splitter(**kwargs)
        elif splitter_type == 'markdown':
            return cls.create_markdown_splitter(**kwargs)
        elif splitter_type == 'token':
            return cls.create_token_splitter(**kwargs)
        else:
            raise ValueError(f"不支持的分块类型: {splitter_type}")
    
    @staticmethod
    def create_sentence_splitter(chunk_size: int = 512, chunk_overlap: int = 50):
        """句子分块器 - 默认策略"""
        return SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    @staticmethod
    def create_sentence_window_splitter(window_size: int = 3):
        """句子窗口分块器 - 保留上下文"""
        return SentenceWindowNodeParser.from_defaults(
            window_size=window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
    
    @staticmethod
    def create_semantic_splitter(embed_model, buffer_size: int = 1, percentile_threshold: int = 95):
        """语义分块器 - 按语义相关性分组"""
        return SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=percentile_threshold,
            embed_model=embed_model
        )
    
    @staticmethod
    def create_markdown_splitter():
        """Markdown分块器 - 按标题层级分割"""
        return MarkdownNodeParser()
    
    @staticmethod
    def create_token_splitter(chunk_size: int = 100, chunk_overlap: int = 20):
        """Token分块器 - 按Token数量分割"""
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    @classmethod
    def get_all_strategies(cls):
        """获取所有可用的分块策略"""
        return list(cls.SPLITTERS.keys())
