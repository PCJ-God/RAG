"""索引管理：构建 / 加载 / 判断是否存在。

支持两种后端（由 .env 的 VECTOR_STORE 决定）：
- local      ：本地 JSON 文件（默认，零依赖）
- dashvector ：阿里云向量检索服务（经 LlamaIndex DashVectorStore 接入 VectorStoreIndex）
"""
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import BaseNode

from src.config import INDEX_DIR, Config, settings
from src.embedder import CompatibleEmbedding
from src.indexing.dashvector_store import PatchedDashVectorStore, get_collection


def create_embed_model():
    """创建 text-embedding-v3 模型（兼容模式），并注册为全局默认。"""
    embed_model = CompatibleEmbedding()
    Settings.embed_model = embed_model
    return embed_model


class Indexer:
    """索引管理器。"""

    def __init__(self, config: Optional[Config] = None, domain: str = "rag"):
        self.config = config or settings
        self.domain = domain
        self.store_type = self.config.vector_store
        self.embed_model = create_embed_model()
        # DashVector 原始 collection（供标签检索使用；本地模式为 None）
        self.collection = None
        # LlamaIndex VectorStoreIndex（本地与 DashVector 模式统一）
        self.index = None

    @property
    def persist_dir(self) -> Path:
        return INDEX_DIR / self.domain

    @property
    def collection_name(self) -> str:
        return self.config.dashvector_collection or f"{self.domain}_knowledge"

    # ---------- DashVector ----------
    def _get_collection(self, create: bool = False, drop_first: bool = False):
        return get_collection(
            self.config,
            self.collection_name,
            self.config.dashvector_dim,
            fields_schema={"tag": str},
            create=create,
            drop_first=drop_first,
        )

    # ---------- 通用接口 ----------
    def exists(self) -> bool:
        if self.store_type == "dashvector":
            try:
                return self._get_collection(create=False) is not None
            except ValueError:
                return False
        return (self.persist_dir / "docstore.json").exists()

    def build(self, nodes: List[BaseNode], rebuild: bool = False):
        if self.store_type == "dashvector":
            self._build_dashvector(nodes, rebuild)
            return

        # 本地 JSON 模式
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        index = VectorStoreIndex(nodes, embed_model=self.embed_model)
        index.storage_context.persist(persist_dir=str(self.persist_dir))
        self.index = index

    def load(self):
        if self.store_type == "dashvector":
            collection = self._get_collection(create=False)
            if collection is None:
                raise FileNotFoundError(
                    f"DashVector collection 不存在: {self.collection_name}"
                )
            store = PatchedDashVectorStore(collection=collection)
            index = VectorStoreIndex.from_vector_store(
                store, embed_model=self.embed_model
            )
            self.collection = collection
            self.index = index
            return index

        storage_context = StorageContext.from_defaults(
            persist_dir=str(self.persist_dir)
        )
        index = load_index_from_storage(
            storage_context, embed_model=self.embed_model
        )
        self.index = index
        return index

    # ---------- DashVector 建库（LlamaIndex DashVectorStore）----------
    def _build_dashvector(self, nodes: List[BaseNode], rebuild: bool) -> None:
        collection = self._get_collection(create=True, drop_first=rebuild)
        store = PatchedDashVectorStore(collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=store)
        index = VectorStoreIndex(
            nodes, storage_context=storage_context, embed_model=self.embed_model
        )
        self.collection = collection
        self.index = index
        print(
            f"[indexer] 已通过 VectorStoreIndex 写入 "
            f"DashVector collection「{self.collection_name}」"
        )
