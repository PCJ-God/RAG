"""
FastAPI Web 服务
提供 /query, /chat, /retrieve, /health 端点
"""
import uuid
import asyncio
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import sys
from pathlib import Path

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import check_api_key, DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_TOP_K
from src.document.loader import DocumentLoader
from src.document.splitter import SplitterFactory
from src.storage.index_manager import IndexManager
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient
from src.generation.answer_generator import AnswerGenerator
from src.chat.multi_turn_chat import CondenseQuestionChatEngine
from src.api.schemas import (
    QueryRequest, QueryResponse,
    ChatRequest, ChatResponse,
    RetrieverRequest, RetrieverResponse,
    HealthResponse, SessionInfo, ContextNode
)


# ============ 全局状态 ============

class RAGState:
    """RAG 服务全局状态"""

    def __init__(self):
        self.query_engine = None
        self.retriever: Optional[Retriever] = None
        self.llm_client: Optional[LLMClient] = None
        self.answer_generator: Optional[AnswerGenerator] = None
        self.index_loaded = False
        # 多轮对话会话存储
        self.sessions: Dict[str, CondenseQuestionChatEngine] = {}

    def initialize(self, use_rerank: bool = False, use_hyde: bool = False):
        """初始化 RAG 系统"""
        check_api_key()

        loader = DocumentLoader(str(DOCS_DIR))
        documents = loader.load()

        splitter = SplitterFactory.create_sentence_splitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        nodes = splitter.get_nodes_from_documents(documents)

        index_manager = IndexManager()
        try:
            index = index_manager.load_index()
        except FileNotFoundError:
            index = index_manager.build_index(nodes)

        self.llm_client = LLMClient()
        self.retriever = Retriever(index, similarity_top_k=SIMILARITY_TOP_K)

        if use_rerank:
            self.retriever.enable_rerank(llm_client=self.llm_client)
        if use_hyde:
            self.retriever.enable_hyde()

        self.retriever.create_query_engine(llm=self.llm_client.get_llm())
        self.query_engine = self.retriever.query_engine

        self.answer_generator = AnswerGenerator(llm_client=self.llm_client)
        self.index_loaded = True

    def get_retriever(self, use_rerank: bool = False, use_hyde: bool = False,
                      top_k: int = None) -> Retriever:
        """获取（或临时创建）一个带指定策略的 retriever"""
        if top_k is None:
            top_k = SIMILARITY_TOP_K

        # 如果请求的策略和全局一致，直接用
        if (use_rerank == self.retriever.use_rerank and
                use_hyde == self.retriever.use_hyde and
                top_k == SIMILARITY_TOP_K):
            return self.retriever

        # 否则临时创建一个
        retriever = Retriever(self.retriever.index, similarity_top_k=top_k)
        if use_rerank:
            retriever.enable_rerank(llm_client=self.llm_client)
        if use_hyde:
            retriever.enable_hyde()
        retriever.create_query_engine(llm=self.llm_client.get_llm())
        return retriever

    def get_or_create_session(self, session_id: Optional[str] = None,
                              use_rerank: bool = False,
                              use_hyde: bool = False) -> tuple:
        """获取或创建多轮对话会话"""
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        if session_id not in self.sessions:
            # 获取带策略的 retriever
            retriever = self.get_retriever(use_rerank, use_hyde)

            chat_engine = CondenseQuestionChatEngine(
                query_engine=retriever.query_engine,
                llm=self.llm_client.get_llm(),
                verbose=False
            )
            self.sessions[session_id] = chat_engine

        return session_id, self.sessions[session_id]


state = RAGState()


# ============ 应用生命周期 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化 RAG 系统"""
    state.initialize()
    print("✅ RAG API 服务启动完成")
    yield
    # 关闭时清理
    state.sessions.clear()
    print("👋 RAG API 服务已关闭")


# ============ FastAPI 应用 ============

app = FastAPI(
    title="RAG 智能问答 API",
    description="提供单轮问答、多轮对话、检索等端点",
    version="1.0.0",
    lifespan=lifespan
)


# ============ 端点 ============

@app.get("/health", response_model=HealthResponse)
async def health():
    """健康检查"""
    strategy = []
    if state.retriever:
        if state.retriever.use_rerank:
            strategy.append("rerank")
        if state.retriever.use_hyde:
            strategy.append("hyde")
    return HealthResponse(
        status="ok",
        index_loaded=state.index_loaded,
        strategy="+".join(strategy) if strategy else "baseline"
    )


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    单轮问答
    - 检索相关文档
    - 用 AnswerGenerator 生成回答
    """
    try:
        retriever = state.get_retriever(
            use_rerank=req.use_rerank,
            use_hyde=req.use_hyde,
            top_k=req.top_k
        )

        nodes = retriever.retrieve(req.question)
        contexts = [node.text for node in nodes]

        answer = state.answer_generator.generate(req.question, contexts)

        strategy_parts = []
        if req.use_rerank:
            strategy_parts.append("rerank")
        if req.use_hyde:
            strategy_parts.append("hyde")
        strategy = "+".join(strategy_parts) if strategy_parts else "baseline"

        return QueryResponse(
            question=req.question,
            answer=answer,
            contexts=[
                ContextNode(text=n.text, score=n.score or 0.0)
                for n in nodes
            ],
            strategy=strategy
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    """
    流式单轮问答（SSE）
    """

    async def event_generator():
        try:
            retriever = state.get_retriever(
                use_rerank=req.use_rerank,
                use_hyde=req.use_hyde,
                top_k=req.top_k
            )

            nodes = retriever.retrieve(req.question)
            contexts = [node.text for node in nodes]

            # 发送上下文信息
            context_json = '{"contexts": [' + ", ".join(
                f'{{"text": "{n.text[:100]}", "score": {n.score or 0.0}}}'
                for n in nodes
            ) + ']}'
            yield f"data: {context_json}\n\n"

            # 流式生成回答
            for chunk in state.answer_generator.generate_stream(req.question, contexts):
                yield f"data: {chunk}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    多轮对话
    - 不传 session_id 则自动创建
    - 后续请求带上 session_id 保持上下文
    """
    try:
        session_id, chat_engine = state.get_or_create_session(
            session_id=req.session_id,
            use_rerank=req.use_rerank,
            use_hyde=req.use_hyde
        )

        reply = chat_engine.chat(req.message)

        # 获取参考文档
        if hasattr(chat_engine.chat_engine, 'retrieve'):
            nodes = chat_engine.chat_engine.retrieve(req.message)
        else:
            nodes = []

        return ChatResponse(
            session_id=session_id,
            message=req.message,
            reply=reply,
            contexts=[
                ContextNode(text=n.text, score=n.score or 0.0)
                for n in nodes
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式多轮对话（SSE）
    """

    async def event_generator():
        try:
            session_id, chat_engine = state.get_or_create_session(
                session_id=req.session_id,
                use_rerank=req.use_rerank,
                use_hyde=req.use_hyde
            )

            # 发送 session_id
            yield f'data: {{"session_id": "{session_id}"}}\n\n'

            # 流式输出回复
            response = chat_engine.chat_engine.stream_chat(req.message)
            for token in response.response_gen:
                yield f"data: {token}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/sessions", response_model=dict)
async def list_sessions():
    """列出所有活跃会话"""
    sessions_info = {}
    for sid, engine in state.sessions.items():
        history = engine.get_history()
        sessions_info[sid] = {
            "message_count": len(history) // 2,
            "last_message": history[-1].content if history else None
        }
    return {"sessions": sessions_info}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    if session_id in state.sessions:
        del state.sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail=f"Session {session_id} not found")


@app.post("/retrieve", response_model=RetrieverResponse)
async def retrieve(req: RetrieverRequest):
    """
    纯检索（不生成回答）
    """
    try:
        retriever = state.get_retriever(
            use_rerank=req.use_rerank,
            use_hyde=req.use_hyde,
            top_k=req.top_k
        )

        nodes = retriever.retrieve(req.question)

        return RetrieverResponse(
            question=req.question,
            contexts=[
                ContextNode(text=n.text, score=n.score or 0.0)
                for n in nodes
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """启动入口"""
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
