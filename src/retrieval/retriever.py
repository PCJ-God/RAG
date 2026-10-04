"""融合式检索：语义 + 标签 两路召回，合并去重后交给重排。

语义路：VectorStoreIndex.as_retriever（本地与 DashVector 统一走 LlamaIndex）。
标签路：直接对 DashVector collection 做 tag 字段过滤（纯条件查询，无向量）。
"""
from typing import List, Optional

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.utils import metadata_dict_to_node


class FusionRetriever:
    """语义 + 标签 两路召回。"""

    def __init__(
        self,
        top_k: int = 20,
        index=None,          # LlamaIndex VectorStoreIndex（本地 / DashVector）
        collection=None,     # DashVector 原始 collection（仅标签路需要）
    ):
        self.top_k = top_k
        self._index = index
        self._collection = collection
        self._vector_retriever = (
            index.as_retriever(similarity_top_k=top_k) if index is not None else None
        )

    def retrieve(
        self, query: str, tag: Optional[str] = None
    ) -> List[NodeWithScore]:
        nodes = self._semantic_search(query)
        if tag and self._collection is not None:
            nodes = self._merge(nodes, self._retrieve_by_tag(tag))
        return nodes

    # ---------- 语义路 ----------
    def _semantic_search(self, query: str) -> List[NodeWithScore]:
        if self._vector_retriever is None:
            return []
        try:
            return self._vector_retriever.retrieve(query)
        except Exception as e:
            print(f"[retriever] 语义检索失败: {e}")
            return []

    # ---------- 标签路 ----------
    def _retrieve_by_tag(self, tag: str) -> List[NodeWithScore]:
        try:
            # 不传 vector：仅按 tag 字段做条件过滤，即纯标签召回
            resp = self._collection.query(filter=f"tag='{tag}'", topk=self.top_k * 2)
        except Exception as e:
            print(f"[retriever] 标签召回失败: {e}")
            return []

        if not resp:
            return []

        results: List[NodeWithScore] = []
        for doc in resp:
            try:
                node = metadata_dict_to_node(getattr(doc, "fields", None) or {})
                results.append(NodeWithScore(node=node, score=0.0))
            except Exception as e:
                print(f"[retriever] 标签结果解析失败: {e}")
        return results

    # ---------- 工具 ----------
    @staticmethod
    def _merge(
        first: List[NodeWithScore], second: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        seen = {}
        order = []
        for item in first + second:
            node_id = item.node.node_id
            if node_id in seen:
                if item.score > seen[node_id].score:
                    seen[node_id] = item
            else:
                seen[node_id] = item
                order.append(node_id)
        return [seen[node_id] for node_id in order]
