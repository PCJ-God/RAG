"""
API 请求/响应模型
"""
from pydantic import BaseModel, Field
from typing import Optional, List


# ============ 请求模型 ============

class QueryRequest(BaseModel):
    """单轮问答请求"""
    question: str = Field(..., description="用户问题", min_length=1)
    use_rerank: bool = Field(False, description="是否启用 Rerank")
    use_hyde: bool = Field(False, description="是否启用 HyDE")
    top_k: int = Field(5, ge=1, le=20, description="检索 top_k")


class ChatRequest(BaseModel):
    """多轮对话请求"""
    message: str = Field(..., description="用户消息", min_length=1)
    session_id: Optional[str] = Field(None, description="会话 ID（不传则自动创建）")
    use_rerank: bool = Field(False, description="是否启用 Rerank")
    use_hyde: bool = Field(False, description="是否启用 HyDE")


class RetrieverRequest(BaseModel):
    """纯检索请求"""
    question: str = Field(..., description="检索问题", min_length=1)
    top_k: int = Field(5, ge=1, le=20, description="检索 top_k")
    use_rerank: bool = Field(False, description="是否启用 Rerank")
    use_hyde: bool = Field(False, description="是否启用 HyDE")


# ============ 响应模型 ============

class ContextNode(BaseModel):
    """检索到的文档片段"""
    text: str
    score: float


class QueryResponse(BaseModel):
    """单轮问答响应"""
    question: str
    answer: str
    contexts: List[ContextNode]
    strategy: str  # "baseline" / "rerank" / "hyde" / "rerank+hyde"


class ChatResponse(BaseModel):
    """多轮对话响应"""
    session_id: str
    message: str
    reply: str
    contexts: List[ContextNode]


class RetrieverResponse(BaseModel):
    """纯检索响应"""
    question: str
    contexts: List[ContextNode]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "ok"
    index_loaded: bool
    strategy: str = "baseline"


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    message_count: int
