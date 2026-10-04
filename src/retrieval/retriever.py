"""融合式检索：语义检索 + 标签检索 两路召回，合并去重后交给重排。

语义路：向量相似度召回（DashVector 直接 query / 本地 LlamaIndex retriever）。
标签路：按 tag 字段过滤召回（DashVector filter，只看主题不看语义）。
两路结果只做去重合并，不在这里排序——排序交给 Reranker（rerank 替代 RRF）。
"""
from typing import List, Optional

from llama_index.core.schema import NodeWithScore, TextNode

from src.embedder import embed_query


class FusionRetriever:
    """语义 + 标签 两路召回。"""

    def __init__(
        self,
        top_k: int = 20,
        index=None,          # 本地模式：LlamaIndex 索引
        collection=None,     # DashVector 模式：原始 collection
        api_key=None,        # DashVector 模式：用于把问题向量化
    ):
        self.top_k = top_k
        self._index = index
        self._collection = collection
        self._api_key = api_key
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
        if self._vector_retriever is not None:
            return self._vector_retriever.retrieve(query)

        q_emb = embed_query(self._api_key, query)
        try:
            resp = self._collection.query(vector=q_emb, topk=self.top_k)
        except Exception as e:
            print(f"[retriever] 语义检索失败: {e}")
            return []

        if not resp:
            return []

        results: List[NodeWithScore] = []
        for doc in resp:
            node = self._doc_to_node(doc)
            if node is not None:
                results.append(
                    NodeWithScore(node=node, score=getattr(doc, "score", 0.0))
                )
        return results

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
            node = self._doc_to_node(doc)
            if node is not None:
                results.append(NodeWithScore(node=node, score=0.0))
        return results

    # ---------- 工具 ----------
    @staticmethod
    def _doc_to_node(doc) -> Optional[TextNode]:
        fields = getattr(doc, "fields", None) or {}
        text = fields.get("text", "")
        if not text:
            return None
        meta = {k: v for k, v in fields.items() if k != "text"}
        return TextNode(id_=doc.id, text=text, metadata=meta)

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
