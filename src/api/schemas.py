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
    use_query_expand: bool = Field(False, description="是否启用 LLM 问题扩展")
    use_multistep: bool = Field(False, description="是否启用多步骤查询分解")
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
    use_query_expand: bool = Field(False, description="是否启用 LLM 问题扩展")
    use_multistep: bool = Field(False, description="是否启用多步骤查询分解")


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
    strategy: str = "baseline"  # "baseline"/"rerank"/"hyde"/"query_expand"/"multistep"/组合


class SessionInfo(BaseModel):
    """会话信息"""
    session_id: str
    message_count: int


class PromptCoachRequest(BaseModel):
    """提示词教练请求"""
    prompt: str = Field(..., description="待优化的提示词", min_length=1)
    mode: str = Field("fixed", description="模式: 'fixed'=固定迭代, 'auto'=AI裁判驱动")
    iterations: int = Field(1, ge=1, le=10, description="固定迭代次数（mode=fixed时使用）")
    max_iterations: int = Field(5, ge=1, le=10, description="AI裁判模式最大迭代次数（mode=auto时使用）")


class PromptCoachResponse(BaseModel):
    """提示词教练响应"""
    original_prompt: str
    analysis: str
    optimized_prompt: str
    iterations: int
    mode: str = "fixed"
    judge_verdict: Optional[str] = None  # "通过" / "不通过"（仅 mode=auto 时有值）
