"""端到端 RAG 主链路编排：摄入 -> 索引 -> 检索 -> 重排 -> 生成。

检索阶段为融合式：语义检索 + 标签检索 两路召回，合并去重后交给重排。
"""
from typing import Optional

from llama_index.core import Settings

from src.config import KNOWLEDGE_DIR, Config, check_api_key, settings
from src.document.loader import MarkdownLoader
from src.document.splitter import QualitySplitter
from src.generation.generator import Answer, AnswerGenerator
from src.generation.llm import LLMClient
from src.indexing.indexer import Indexer, create_embed_model
from src.retrieval.retriever import FusionRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.tag_extractor import TagExtractor


class RAGPipeline:
    """基础 RAG 主链路。"""

    def __init__(self, config: Optional[Config] = None, domain: str = "rag"):
        self.config = config or settings
        self.domain = domain
        check_api_key()

        create_embed_model()
        self.llm_client = LLMClient(self.config)
        Settings.llm = self.llm_client.get_llm()

        self.indexer = Indexer(self.config, domain=domain)
        self.generator = AnswerGenerator(self.llm_client)
        self.tag_extractor = TagExtractor(self.llm_client, self.config)
        self.retriever: Optional[FusionRetriever] = None
        self.reranker: Optional[Reranker] = None

    # ---------- 构建 ----------
    def build(self, domain: Optional[str] = None, rebuild: bool = False) -> int:
        """加载文档、分块、打标签、建索引，返回文档块数量。"""
        domain = domain or self.domain
        docs_dir = KNOWLEDGE_DIR / domain

        documents = MarkdownLoader(str(docs_dir)).load()
        splitter = QualitySplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        nodes = splitter.get_nodes(documents)

        self.indexer = Indexer(self.config, domain=domain)
        self.indexer.build(nodes, rebuild=rebuild)
        return len(nodes)

    # ---------- 加载 ----------
    def load(self, domain: Optional[str] = None) -> "RAGPipeline":
        """加载索引并组装检索器与重排器。"""
        domain = domain or self.domain
        self.indexer = Indexer(self.config, domain=domain)
        if not self.indexer.exists():
            raise FileNotFoundError(
                f"索引不存在（domain={domain}）。请先运行: python scripts/build_index.py"
            )

        if self.indexer.store_type == "dashvector":
            collection = self.indexer.load()
            self.retriever = FusionRetriever(
                top_k=self.config.top_k,
                collection=collection,
                api_key=self.config.api_key,
            )
        else:
            index = self.indexer.load()
            self.retriever = FusionRetriever(top_k=self.config.top_k, index=index)

        self.reranker = Reranker(self.config, top_n=self.config.rerank_top_n)
        return self

    # ---------- 查询 ----------
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_rerank: bool = True,
        use_tag: bool = True,
    ) -> Answer:
        """端到端问答。"""
        if self.retriever is None:
            self.load(self.domain)
        assert self.retriever is not None

        if top_k:
            self.reranker = Reranker(self.config, top_n=top_k)

        tag = None
        if use_tag:
            tag = self.tag_extractor.extract(question)
            print(f"[tag] 问题标签: {tag}")

        nodes = self.retriever.retrieve(question, tag=tag)
        if use_rerank and self.reranker is not None:
            nodes = self.reranker.rerank(nodes, question)

        return self.generator.generate(question, nodes)
