"""
索引管理器
负责索引的构建、保存、加载，支持 Milvus / memory 等多种存储后端
"""
import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core import StorageContext, load_index_from_storage
from src.config import INDEX_DIR
from src.embedding.embedder import Embedder
from src.storage.vector_store import VectorStoreManager


class IndexManager:
    """索引管理器"""

    def __init__(self, persist_path: str = None, store_type: str = 'memory',
                 milvus_uri: str = None, collection_name: str = None):
        """
        Args:
            persist_path: 索引持久化路径
            store_type: 存储类型 ('memory' / 'milvus')
            milvus_uri: Milvus 连接 URI（仅 milvus 类型需要）
            collection_name: Milvus 集合名
        """
        self.persist_path = persist_path or INDEX_DIR / "default"
        self.store_type = store_type
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name
        self.embedder = Embedder()
        self.index = None
        self.vector_store_manager = VectorStoreManager(
            store_type=store_type,
            persist_path=str(self.persist_path),
            milvus_uri=milvus_uri,
            collection_name=collection_name
        )

    def build_index(self, documents, force_rebuild: bool = False) -> VectorStoreIndex:
        """
        构建索引

        Args:
            documents: 文档列表
            force_rebuild: 是否强制重建

        Returns:
            VectorStoreIndex 实例
        """
        # 检查是否存在已保存的索引
        if not force_rebuild and self._index_exists():
            print(f"发现已存在的索引: {self.persist_path}")
            print("使用 load_index() 方法加载现有索引")
            return self.load_index()

        print("正在构建新索引...")

        # 设置全局 Embed 模型
        Settings.embed_model = self.embedder.get_model()

        # 通过 VectorStoreManager 创建索引
        self.index = self.vector_store_manager.create_index(
            documents,
            embed_model=Settings.embed_model
        )

        # 保存索引
        self.save_index()

        print("[OK] 索引构建完成")
        return self.index

    def _index_exists(self) -> bool:
        """检查索引是否存在"""
        if self.store_type == 'milvus':
            # Milvus 的集合是否存在
            milvus_path = self.milvus_uri or str(self.persist_path / "milvus.db")
            return os.path.exists(milvus_path)
        else:
            return os.path.exists(str(self.persist_path))

    def save_index(self):
        """保存索引到后端"""
        self.vector_store_manager.persist()

    def load_index(self) -> VectorStoreIndex:
        """加载索引"""
        Settings.embed_model = self.embedder.get_model()

        self.index = VectorStoreManager.load_index(
            persist_path=str(self.persist_path),
            embed_model=Settings.embed_model,
            store_type=self.store_type,
            milvus_uri=self.milvus_uri,
            collection_name=self.collection_name
        )

        return self.index

    def get_index(self) -> VectorStoreIndex:
        """获取当前索引"""
        return self.index
