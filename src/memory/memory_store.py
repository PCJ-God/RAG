"""多轮对话记忆：双层记忆（会话记忆 + 全局记忆 / 用户画像）。

存储统一走 DashVector 的 chat_memory collection（LlamaIndex 基座）：
- 会话记忆：memory_type="session"，按 session_id 隔离，保当前话题连贯；
- 全局记忆：memory_type="global"，跨会话，记录用户长期偏好/身份。

每条记录是一个 LlamaIndex TextNode，元数据扁平存成可过滤字段：
user_id / session_id / memory_type / turn_id / timestamp / role。
"""
import time
import uuid
from typing import List, Optional

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node

from src.config import Config, settings
from src.embedder import CompatibleEmbedding
from src.indexing.dashvector_store import (
    PatchedDashVectorStore,
    get_collection,
)


class MemoryStore:
    """双层记忆存储。"""

    def __init__(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        config: Optional[Config] = None,
    ):
        self.config = config or settings
        self.user_id = user_id or self.config.user_id
        self.session_id = session_id or uuid.uuid4().hex
        self._embed_model = CompatibleEmbedding()

        collection = get_collection(
            self.config,
            self.config.memory_collection,
            self.config.dashvector_dim,
            fields_schema={
                "user_id": str,
                "session_id": str,
                "memory_type": str,
            },
            create=True,
        )
        self._collection = collection
        self._store = PatchedDashVectorStore(collection=collection)

    # ---------- 写入 ----------
    def add_turn(self, question: str, answer: str, turn_id: int) -> None:
        """写入一条会话记忆（一轮合并一条）。"""
        text = f"用户：{question}\n助手：{answer}"
        node = TextNode(
            text=text,
            metadata={
                "user_id": self.user_id,
                "session_id": self.session_id,
                "memory_type": "session",
                "turn_id": turn_id,
                "timestamp": int(time.time()),
                "role": "assistant",
            },
        )
        node.embedding = self._embed_model.get_text_embedding(text)
        self._store.add([node])

    def add_global_fact(self, fact: str) -> None:
        """写入一条全局记忆（用户长期事实）。"""
        fact = fact.strip()
        if not fact:
            return
        node = TextNode(
            text=fact,
            metadata={
                "user_id": self.user_id,
                "session_id": "global",
                "memory_type": "global",
                "turn_id": 0,
                "timestamp": int(time.time()),
                "role": "system",
            },
        )
        node.embedding = self._embed_model.get_text_embedding(fact)
        self._store.add([node])

    # ---------- 检索 ----------
    def retrieve_session(self, query: str, top_k: Optional[int] = None) -> List[TextNode]:
        """检索当前会话相关的历史轮次（语义 + session_id 前置过滤）。"""
        top_k = top_k or self.config.memory_top_k
        return self._query(
            query=query,
            top_k=top_k,
            filter_str=(
                f"session_id='{self.session_id}' and memory_type='session'"
            ),
        )

    def retrieve_global(self, query: str, top_k: int = 10) -> List[TextNode]:
        """检索用户画像（跨会话，按 user_id + memory_type=global）。"""
        return self._query(
            query=query,
            top_k=top_k,
            filter_str=f"user_id='{self.user_id}' and memory_type='global'",
        )

    # ---------- 工具 ----------
    def _query(self, query: str, top_k: int, filter_str: str) -> List[TextNode]:
        q_embed = self._embed_model.get_query_embedding(query)
        try:
            resp = self._collection.query(
                vector=q_embed,
                topk=top_k,
                filter=filter_str,
            )
        except Exception as e:
            print(f"[memory] 记忆检索失败: {e}")
            return []
        if not resp:
            return []

        nodes: List[TextNode] = []
        for doc in resp:
            try:
                nodes.append(metadata_dict_to_node(getattr(doc, "fields", None) or {}))
            except Exception as e:
                print(f"[memory] 记忆记录解析失败: {e}")
        return nodes
