"""端到端 RAG 主链路编排：摄入 -> 索引 -> 检索 -> 重排 -> 生成。

检索阶段为融合式：语义检索 + 标签检索 两路召回，合并去重后交给重排。
"""
from typing import List, Optional

from llama_index.core import Settings

from src.config import KNOWLEDGE_DIR, Config, check_api_key, settings
from src.document.loader import MarkdownLoader
from src.document.splitter import QualitySplitter
from src.generation.generator import Answer, AnswerGenerator, Source
from src.generation.llm import LLMClient
from src.indexing.indexer import Indexer, create_embed_model
from src.retrieval.decomposer import QueryDecomposer
from src.retrieval.retriever import FusionRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.rewriter import QueryRewriter
from src.retrieval.tag_extractor import TagExtractor


class RAGPipeline:
    """基础 RAG 主链路。"""

    def __init__(
        self,
        config: Optional[Config] = None,
        domain: str = "rag",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ):
        self.config = config or settings
        self.domain = domain
        # 多轮会话：session_id 为 None 表示不启用记忆（如评测）
        self.session_id = session_id
        self.user_id = user_id
        check_api_key()

        create_embed_model()
        self.llm_client = LLMClient(self.config)
        Settings.llm = self.llm_client.get_llm()

        self.indexer = Indexer(self.config, domain=domain)
        self.generator = AnswerGenerator(self.llm_client)
        self.tag_extractor = TagExtractor(self.llm_client, self.config)
        self.rewriter = QueryRewriter(self.llm_client)
        self.decomposer = QueryDecomposer(self.llm_client)
        self.retriever: Optional[FusionRetriever] = None
        self.reranker: Optional[Reranker] = None

        # 多轮记忆（懒加载）
        self.memory = None
        self.global_extractor = None
        self._recent_turns = []  # [(turn_id, question, answer), ...]
        self._turn_id = 0

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

        index = self.indexer.load()
        self.retriever = FusionRetriever(
            top_k=self.config.top_k,
            index=index,
            collection=self.indexer.collection,
        )

        self.reranker = Reranker(self.config, top_n=self.config.rerank_top_n)
        return self

    # ---------- 查询 ----------
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        use_rerank: bool = True,
        use_tag: bool = True,
        use_memory: bool = True,
        use_rewrite: bool = True,
        use_decompose: bool = True,
    ) -> Answer:
        """端到端问答：改写 → 拆解 → 检索 → 重排 → 生成 → 记忆。"""
        if self.retriever is None:
            self.load(self.domain)
        assert self.retriever is not None

        if top_k:
            self.reranker = Reranker(self.config, top_n=top_k)

        memory_on = (
            use_memory
            and self.session_id is not None
            and self.config.vector_store == "dashvector"
        )
        history_str = ""
        profile_str = ""
        if memory_on:
            self._init_memory()
            profile_str = self._load_profile(question)
            history_str = self._load_history(question)

        # 1) 模糊/指代问题改写（结合历史）
        query_for_search = question
        if use_rewrite:
            query_for_search = self.rewriter.rewrite(question, history=history_str)
            print(f"[rewrite] {query_for_search}")

        # 2) 复杂问题拆解
        sub_questions = [query_for_search]
        if use_decompose:
            sub_questions = self.decomposer.decompose(query_for_search)

        # 3) 检索生成
        if len(sub_questions) <= 1:
            answer = self._single_pass(
                sub_questions[0],
                question,
                profile=profile_str,
                history=history_str,
                use_rerank=use_rerank,
                use_tag=use_tag,
            )
        else:
            answer = self._multi_pass(
                question, sub_questions, use_rerank=use_rerank, use_tag=use_tag
            )

        if memory_on:
            self._record_turn(question, answer.answer)

        return answer

    # ---------- 检索生成 ----------
    def _single_pass(
        self,
        query: str,
        answer_question: str,
        profile: str = "",
        history: str = "",
        use_rerank: bool = True,
        use_tag: bool = True,
    ) -> Answer:
        """单次检索 + 重排 + 生成。query 用于召回，answer_question 用于生成。"""
        tag = None
        if use_tag:
            tag = self.tag_extractor.extract(query)
            print(f"[tag] 问题标签: {tag}")

        nodes = self.retriever.retrieve(query, tag=tag)
        if use_rerank and self.reranker is not None:
            nodes = self.reranker.rerank(nodes, query)

        return self.generator.generate(
            answer_question, nodes, history=history, profile=profile
        )

    def _multi_pass(
        self,
        question: str,
        sub_questions: List[str],
        use_rerank: bool = True,
        use_tag: bool = True,
    ) -> Answer:
        """逐个回答子问题，再整合成最终回答。"""
        qa_pairs = []
        sources = []
        for sub in sub_questions:
            print(f"[decompose] 子问题: {sub}")
            ans = self._single_pass(sub, sub, use_rerank=use_rerank, use_tag=use_tag)
            qa_pairs.append((sub, ans.answer))
            sources.extend(ans.sources)

        merged = self.decomposer.merge(question, qa_pairs)
        merged_sources = [
            Source(
                index=i,
                text=s.text,
                score=s.score,
                header_path=s.header_path,
                file_name=s.file_name,
            )
            for i, s in enumerate(sources, start=1)
        ]
        return Answer(answer=merged, sources=merged_sources)

    # ---------- 记忆 ----------
    def _init_memory(self) -> None:
        if self.memory is None:
            from src.memory import GlobalMemoryExtractor, MemoryStore

            self.memory = MemoryStore(
                session_id=self.session_id,
                user_id=self.user_id,
                config=self.config,
            )
            self.global_extractor = GlobalMemoryExtractor(
                self.llm_client, self.config
            )

    def _load_profile(self, question: str) -> str:
        """检索用户画像并拼成文本。"""
        facts, seen = [], set()
        for node in self.memory.retrieve_global(question):
            text = node.get_content().strip()
            if text and text not in seen:
                seen.add(text)
                facts.append(text)
        return "\n".join(f"- {f}" for f in facts)

    def _load_history(self, question: str) -> str:
        """最近 N 轮纯文本 + 语义检索的更早历史（按 turn_id 去重）。"""
        window = self._recent_turns[-self.config.memory_window :]
        recent_ids = {tid for tid, _, _ in window}

        lines = []
        for node in self.memory.retrieve_session(question):
            if node.metadata.get("turn_id") in recent_ids:
                continue
            text = node.get_content().strip()
            if text:
                lines.append(text)

        lines += [f"用户：{q}\n助手：{a}" for _, q, a in window]
        return "\n\n".join(lines)

    def _record_turn(self, question: str, answer: str) -> None:
        """写入会话记忆并提取全局事实。"""
        turn_id = self._turn_id
        self._turn_id += 1

        self.memory.add_turn(question, answer, turn_id)
        self._recent_turns.append((turn_id, question, answer))
        self._recent_turns = self._recent_turns[-self.config.memory_window :]

        try:
            for fact in self.global_extractor.extract(question, answer):
                self.memory.add_global_fact(fact)
        except Exception as e:
            print(f"[memory] 全局记忆提取失败: {e}")
