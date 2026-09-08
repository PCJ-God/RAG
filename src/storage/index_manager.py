"""
索引管理器
负责索引的构建、保存、加载
"""
import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, Settings
from src.config import INDEX_DIR
from src.embedding.embedder import Embedder


class IndexManager:
    """索引管理器"""
    
    def __init__(self, persist_path: str = None):
        """
        Args:
            persist_path: 索引持久化路径
        """
        self.persist_path = persist_path or INDEX_DIR / "default"
        self.embedder = Embedder()
        self.index = None
    
    def build_index(self, documents, force_rebuild: bool = False) -> VectorStoreIndex:
        """
        构建索引
        
        Args:
            documents: 文档列表
            force_rebuild: 是否强制重建
            
        Returns:
            VectorStoreIndex实例
        """
        # 检查是否存在已保存的索引
        if not force_rebuild and os.path.exists(str(self.persist_path)):
            print(f"发现已存在的索引: {self.persist_path}")
            print("使用 load_index() 方法加载现有索引")
            return self.load_index()
        
        print("正在构建新索引...")
        
        # 设置全局Embed模型
        Settings.embed_model = self.embedder.get_model()
        
        # 创建索引
        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=Settings.embed_model
        )
        
        # 保存索引
        self.save_index()
        
        print("✅ 索引构建完成")
        return self.index
    
    def save_index(self):
        """保存索引到本地"""
        if self.index:
            self.index.storage_context.persist(str(self.persist_path))
            print(f"✅ 索引已保存到: {self.persist_path}")
    
    def load_index(self) -> VectorStoreIndex:
        """加载本地索引"""
        from llama_index.core import StorageContext, load_index_from_storage
        
        if not os.path.exists(str(self.persist_path)):
            raise FileNotFoundError(f"索引不存在: {self.persist_path}")
        
        Settings.embed_model = self.embedder.get_model()
        
        storage_context = StorageContext.from_defaults(persist_dir=str(self.persist_path))
        self.index = load_index_from_storage(storage_context, embed_model=Settings.embed_model)
        
        print(f"✅ 索引已从 {self.persist_path} 加载")
        return self.index
    
    def get_index(self) -> VectorStoreIndex:
        """获取当前索引（如果不存在则返回None）"""
        return self.index
