"""
向量存储管理
支持内存存储、Milvus 本地存储等多种方案
"""
import os
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from typing import Optional


class VectorStoreManager:
    """向量存储管理器"""

    STORE_TYPES = {
        'memory': '内存存储',
        'milvus': 'Milvus 本地',
        'qdrant': 'Qdrant 本地',
        'dashvector': '阿里云 DashVector'
    }

    def __init__(self, store_type: str = 'memory', persist_path: Optional[str] = None,
                 milvus_uri: str = None, collection_name: str = None, dim: int = None):
        """
        Args:
            store_type: 存储类型 ('memory' / 'milvus' / ...)
            persist_path: 持久化路径（用于 memory 存储）
            milvus_uri: Milvus 连接 URI（默认本地文件）
            collection_name: Milvus 集合名
            dim: Embedding 向量维度
        """
        self.store_type = store_type
        self.persist_path = persist_path
        self.milvus_uri = milvus_uri
        self.collection_name = collection_name or "rag_default"
        self.dim = dim or 1024  # text-embedding-v3 默认 1024 维
        self.index = None
        self.vector_store = None

    def _create_vector_store(self):
        """创建底层向量存储"""
        if self.store_type == 'memory':
            return None  # 内存存储不需要 vector_store

        elif self.store_type == 'milvus':
            # Milvus Lite: 用本地文件作为后端
            uri = self.milvus_uri or (
                os.path.join(self.persist_path, "milvus.db")
                if self.persist_path else "./milvus.db"
            )

            # 确保目录存在
            milvus_dir = os.path.dirname(uri)
            if milvus_dir and not os.path.exists(milvus_dir):
                os.makedirs(milvus_dir, exist_ok=True)

            self.vector_store = MilvusVectorStore(
                uri=uri,
                collection_name=self.collection_name,
                dim=self.dim,
                overwrite=False
            )
            print(f"  Milvus URI: {uri}")
            print(f"  Collection: {self.collection_name}")
            return self.vector_store

        else:
            print(f"[WARN] {self.store_type} 存储暂未实现，使用内存存储")
            return None

    def create_index(self, documents, embed_model) -> VectorStoreIndex:
        """
        创建向量索引

        Args:
            documents: 文档列表
            embed_model: Embedding 模型

        Returns:
            VectorStoreIndex 实例
        """
        print(f"正在使用 {self.STORE_TYPES.get(self.store_type, self.store_type)} 创建索引...")

        vector_store = self._create_vector_store()

        if vector_store is not None:
            # 使用外部向量存储
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model
            )
        else:
            # 内存存储
            self.index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embed_model
            )

        print("[OK] 索引创建完成")
        return self.index

    def persist(self, persist_path: Optional[str] = None):
        """
        保存索引到本地（仅对 memory 存储有效，Milvus 自动持久化）
        """
        path = persist_path or self.persist_path
        if self.store_type == 'memory' and path and self.index:
            self.index.storage_context.persist(str(path))
            print(f"[OK] 索引已保存到: {path}")
        elif self.store_type == 'milvus':
            print(f"[OK] Milvus 数据已自动保存到 {self.milvus_uri or 'milvus.db'}")

    @classmethod
    def load_index(cls, persist_path: str, embed_model,
                   store_type: str = 'memory',
                   milvus_uri: str = None,
                   collection_name: str = "rag_default",
                   dim: int = 1024) -> VectorStoreIndex:
        """
        从存储加载索引

        Args:
            persist_path: 保存路径（memory 存储用）
            embed_model: Embedding 模型
            store_type: 存储类型
            milvus_uri: Milvus URI
            collection_name: Milvus 集合名
            dim: 向量维度

        Returns:
            VectorStoreIndex 实例
        """
        manager = cls(
            store_type=store_type,
            persist_path=persist_path,
            milvus_uri=milvus_uri,
            collection_name=collection_name,
            dim=dim
        )

        if store_type == 'milvus':
            vector_store = manager._create_vector_store()
            # Milvus 需要将集合加载到内存
            if vector_store and hasattr(vector_store, 'client'):
                vector_store.client.load_collection(collection_name)
            manager.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model
            )
        else:
            # memory 存储
            storage_context = StorageContext.from_defaults(persist_dir=str(persist_path))
            from llama_index.core import load_index_from_storage
            manager.index = load_index_from_storage(storage_context, embed_model=embed_model)

        print(f"[OK] 索引已从 {store_type} 存储加载")
        return manager.index

    def get_index(self) -> Optional[VectorStoreIndex]:
        """获取当前索引"""
        return self.index

    def drop_collection(self):
        """删除 Milvus 集合（仅 milvus 类型有效）"""
        if self.store_type == 'milvus' and self.vector_store:
            self.vector_store.client.drop_collection(self.collection_name)
            print(f"[OK] Milvus 集合 {self.collection_name} 已删除")
