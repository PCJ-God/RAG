"""索引管理：构建 / 加载 / 判断是否存在。

支持两种后端（由 .env 的 VECTOR_STORE 决定）：
- local      ：本地 JSON 文件（默认，零依赖）
- dashvector ：阿里云向量检索服务（直接用官方 SDK，向量 + 文本 + 标签都存云端）
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
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
)

from src.config import INDEX_DIR, Config, settings
from src.embedder import EMBED_BATCH_SIZE, embed_texts


def create_embed_model():
    """创建 text-embedding-v3 模型，并注册为全局默认。"""
    embed_model = DashScopeEmbedding(
        model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3,
        api_key=settings.api_key,
    )
    # 修复 llama-index-embeddings-dashscope 0.2.x 的已知 bug：
    # 它在 super().__init__() 之前给私有属性赋值，pydantic v2 会在
    # super().__init__() 里重置它们，导致运行时拿不到 _api_key/_text_type。
    # 因此构造完成后手动补一次。
    embed_model._api_key = settings.api_key
    embed_model._text_type = "document"
    Settings.embed_model = embed_model
    return embed_model


class Indexer:
    """索引管理器。"""

    def __init__(self, config: Optional[Config] = None, domain: str = "rag"):
        self.config = config or settings
        self.domain = domain
        self.store_type = self.config.vector_store
        self.embed_model = create_embed_model()
        # DashVector 原始 collection（供检索直接使用；本地模式为 None）
        self.collection = None

    @property
    def persist_dir(self) -> Path:
        return INDEX_DIR / self.domain

    @property
    def collection_name(self) -> str:
        return self.config.dashvector_collection or f"{self.domain}_knowledge"

    # ---------- DashVector ----------
    def _get_client(self):
        if not self.config.dashvector_api_key or not self.config.dashvector_endpoint:
            raise ValueError(
                "使用 DashVector 需在 .env 配置 DASHSVECTOR_API_KEY 和 DASHSVECTOR_ENDPOINT"
            )
        import dashvector

        return dashvector.Client(
            api_key=self.config.dashvector_api_key,
            endpoint=self.config.dashvector_endpoint,
        )

    def _get_collection(self, create: bool = False, drop_first: bool = False):
        import time

        client = self._get_client()

        if drop_first:
            # 重建：先删后建。DashVector 删除是异步的，直接 create 会因
            # 旧集合仍存在而失败，因此这里删除后重试 create 直到成功。
            try:
                client.delete(self.collection_name)
                print(f"[indexer] 已删除旧 collection: {self.collection_name}")
            except Exception as e:
                print(f"[indexer] 删除旧 collection 失败（可能本就不存在）: {e}")

            last = None
            for _ in range(15):
                ret = client.create(
                    name=self.collection_name,
                    dimension=self.config.dashvector_dim,
                    fields_schema={"tag": str},
                )
                if ret:
                    break
                last = (
                    f"code={getattr(ret, 'code', '?')}, "
                    f"message={getattr(ret, 'message', '?')}"
                )
                time.sleep(2)
            else:
                raise RuntimeError(f"重建 collection 失败: {last}")

            collection = client.get(self.collection_name)
            if collection is None:
                raise RuntimeError(f"重建后无法获取 collection: {self.collection_name}")
            print(
                f"[indexer] 已创建 collection: {self.collection_name}"
                f"（dimension={self.config.dashvector_dim}）"
            )
            return collection

        # 非重建：先尝试获取已存在的 collection
        try:
            collection = client.get(self.collection_name)
        except Exception as e:
            print(f"[indexer] 获取 collection 失败: {e}")
            collection = None

        if collection is None and create:
            # 单向量集合：dimension=1024，预定义 tag 字段用于标签过滤
            ret = client.create(
                name=self.collection_name,
                dimension=self.config.dashvector_dim,
                fields_schema={"tag": str},
            )
            if not ret:
                raise RuntimeError(f"创建 collection 失败: {self.collection_name}")
            collection = client.get(self.collection_name)
            if collection is None:
                raise RuntimeError(
                    f"创建后仍无法获取 collection: {self.collection_name}"
                )
            print(
                f"[indexer] 已创建 collection: {self.collection_name}"
                f"（dimension={self.config.dashvector_dim}）"
            )

        return collection

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

    def load(self):
        if self.store_type == "dashvector":
            collection = self._get_collection(create=False)
            if collection is None:
                raise FileNotFoundError(
                    f"DashVector collection 不存在: {self.collection_name}"
                )
            self.collection = collection
            return collection

        storage_context = StorageContext.from_defaults(
            persist_dir=str(self.persist_dir)
        )
        return load_index_from_storage(
            storage_context, embed_model=self.embed_model
        )

    # ---------- DashVector 建库（直接使用官方 SDK）----------
    def _build_dashvector(self, nodes: List[BaseNode], rebuild: bool) -> None:
        from dashvector import Doc

        collection = self._get_collection(create=True, drop_first=rebuild)
        self.collection = collection

        texts = [node.get_content() for node in nodes]
        print(f"[indexer] 正在向量化 {len(texts)} 个文档块...")

        # 分批向量化，避免单次请求过大
        embeddings = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            embeddings.extend(embed_texts(self.config.api_key, batch))
            print(
                f"[indexer] 向量化进度: "
                f"{min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)}"
            )

        docs = []
        for node, embedding in zip(nodes, embeddings):
            meta = node.metadata
            docs.append(
                Doc(
                    id=node.node_id,
                    vector=embedding,
                    fields={
                        "text": node.get_content(),
                        "tag": str(meta.get("tag") or ""),
                        "file_name": str(meta.get("file_name") or ""),
                        "header_path": str(meta.get("header_path") or ""),
                    },
                )
            )

        print(
            f"[indexer] 向量维度: {len(embeddings[0])}"
            f"（collection 要求 {self.config.dashvector_dim}）"
        )

        # 分批插入，避免单次请求过大，并输出具体失败原因
        insert_batch = 100
        for i in range(0, len(docs), insert_batch):
            chunk = docs[i : i + insert_batch]
            resp = collection.insert(chunk)
            if not resp:
                print(
                    f"[indexer] 插入失败: code={getattr(resp, 'code', '?')}, "
                    f"message={getattr(resp, 'message', '?')}"
                )
                for r in getattr(resp, "output", None) or []:
                    if not getattr(r, "success", True):
                        print(
                            f"[indexer]   失败条目 id={getattr(r, 'id', '?')}, "
                            f"code={getattr(r, 'code', '?')}, "
                            f"message={getattr(r, 'message', '?')}"
                        )
                raise RuntimeError("插入数据失败，详见上方日志")
            print(
                f"[indexer] 已插入 {min(i + insert_batch, len(docs))}/{len(docs)} 条"
            )

        try:
            stats = collection.stats()
            print(f"[indexer] collection「{self.collection_name}」统计: {stats}")
        except Exception as e:
            print(f"[indexer] 获取统计信息失败: {e}")
