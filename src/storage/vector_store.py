"""
向量存储管理
支持内存存储、本地向量数据库、云服务等多种方案
"""
from llama_index.core import VectorStoreIndex
from typing import Optional


class VectorStoreManager:
    """向量存储管理器"""
    
    STORE_TYPES = {
        'memory': '内存存储',
        'milvus': 'Milvus本地',
        'qdrant': 'Qdrant本地',
        'dashvector': '阿里云DashVector'
    }
    
    def __init__(self, store_type: str = 'memory', persist_path: Optional[str] = None):
        """
        Args:
            store_type: 存储类型
            persist_path: 持久化路径（用于本地存储）
        """
        self.store_type = store_type
        self.persist_path = persist_path
        self.index = None
    
    def create_index(self, documents, embed_model) -> VectorStoreIndex:
        """
        创建向量索引
        
        Args:
            documents: 文档列表
            embed_model: Embedding模型
            
        Returns:
            VectorStoreIndex实例
        """
        print(f"正在使用 {self.STORE_TYPES[self.store_type]} 创建索引...")
        
        if self.store_type == 'memory':
            self.index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model
            )
        else:
            # TODO: 实现其他存储类型
            print(f"⚠️ {self.store_type} 存储暂未实现，使用内存存储")
            self.index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model
            )
        
        print("✅ 索引创建完成")
        return self.index
    
    def persist(self, persist_path: Optional[str] = None):
        """保存索引到本地"""
        path = persist_path or self.persist_path
        if path and self.index:
            self.index.storage_context.persist(str(path))
            print(f"✅ 索引已保存到: {path}")
    
    @classmethod
    def load_index(cls, persist_path: str, embed_model) -> VectorStoreIndex:
        """
        从本地加载索引
        
        Args:
            persist_path: 保存路径
            embed_model: Embedding模型
            
        Returns:
            VectorStoreIndex实例
        """
        from llama_index.core import StorageContext, load_index_from_storage
        
        print(f"正在从 {persist_path} 加载索引...")
        storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
        index = load_index_from_storage(storage_context, embed_model=embed_model)
        print("✅ 索引加载完成")
        
        return index
