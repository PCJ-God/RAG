"""集中配置：API Key、模型、RAG 参数、路径与知识点标签。

优先级：.env 文件 > 代码默认值。
"""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# override=True：让 .env 的值为准，覆盖 shell 里可能残留的旧环境变量
# （否则旧的 DASHSCOPE_API_KEY 会遮蔽 .env 里的新值，导致 InvalidApiKey）
load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = PROJECT_ROOT / "knowledge_base"

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class Config:
    """RAG 应用配置。"""

    # API
    api_key: str = ""
    base_url: str = BASE_URL

    # 模型
    llm_model: str = "qwen-plus"
    embedding_model: str = "text-embedding-v3"
    rerank_model: str = "qwen3-rerank"

    # 分块
    chunk_size: int = 512
    chunk_overlap: int = 50
    # 语义切分：相邻句向量相似度的断裂点百分位阈值（0-100，越大切得越少）
    semantic_breakpoint_threshold: int = 95

    # 检索：先粗排召回 top_k 个候选，再精排到 rerank_top_n 个
    top_k: int = 20
    rerank_top_n: int = 3

    # 生成（问答助手追求准确，温度建议 0~0.3）
    temperature: float = 0.1
    max_tokens: int = 1024

    # 向量库：local（本地 JSON，默认）/ dashvector（阿里云）
    vector_store: str = "local"
    dashvector_api_key: str = ""
    dashvector_endpoint: str = ""
    dashvector_collection: str = ""  # 留空则用 "<domain>_knowledge"
    dashvector_dim: int = 1024       # text-embedding-v3 维度，建库后不可改

    # 多轮记忆（chat_memory collection，仅 dashvector 模式启用）
    memory_collection: str = "chat_memory"
    memory_top_k: int = 4            # 语义检索的历史轮次数量
    memory_window: int = 3           # 最近 N 轮以纯文本形式直接带入
    user_id: str = "default_user"    # 单用户多会话；未来多用户可换

    # 追问澄清：意图不明确时与用户交互补全问题（仅多轮会话启用）
    clarify_max_rounds: int = 3

    # AI 安全护栏（阿里云内容安全 / 绿网，未配置 AccessKey 时自动禁用）
    aliyun_access_key_id: str = ""
    aliyun_access_key_secret: str = ""
    content_security_region: str = "cn-shanghai"
    content_security_endpoint: str = "green-cip.cn-shanghai.aliyuncs.com"
    guard_query_service: str = "llm_query_moderation"      # 输入护栏（检测提问）
    guard_response_service: str = "llm_response_moderation"  # 输出护栏（检测回答）
    guard_block_levels: str = "high,medium"  # 命中这些风险等级即拦截

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            base_url=os.getenv("BASE_URL", BASE_URL),
            llm_model=os.getenv("LLM_MODEL", "qwen-plus"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-v3"),
            rerank_model=os.getenv("RERANK_MODEL", "qwen3-rerank"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            semantic_breakpoint_threshold=int(
                os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD", "95")
            ),
            top_k=int(os.getenv("TOP_K", "20")),
            rerank_top_n=int(os.getenv("RERANK_TOP_N", "3")),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
            vector_store=os.getenv("VECTOR_STORE", "local"),
            dashvector_api_key=os.getenv("DASHSVECTOR_API_KEY", ""),
            dashvector_endpoint=os.getenv("DASHSVECTOR_ENDPOINT", ""),
            dashvector_collection=os.getenv("DASHSVECTOR_COLLECTION", ""),
            dashvector_dim=int(os.getenv("DASHSVECTOR_DIM", "1024")),
            memory_collection=os.getenv("MEMORY_COLLECTION", "chat_memory"),
            memory_top_k=int(os.getenv("MEMORY_TOP_K", "4")),
            memory_window=int(os.getenv("MEMORY_WINDOW", "3")),
            user_id=os.getenv("USER_ID", "default_user"),
            clarify_max_rounds=int(os.getenv("CLARIFY_MAX_ROUNDS", "3")),
            aliyun_access_key_id=os.getenv("ALIBABA_CLOUD_ACCESS_KEY_ID", ""),
            aliyun_access_key_secret=os.getenv("ALIBABA_CLOUD_ACCESS_KEY_SECRET", ""),
            content_security_region=os.getenv("CONTENT_SECURITY_REGION", "cn-shanghai"),
            content_security_endpoint=os.getenv(
                "CONTENT_SECURITY_ENDPOINT", "green-cip.cn-shanghai.aliyuncs.com"
            ),
            guard_query_service=os.getenv(
                "CONTENT_SECURITY_QUERY_SERVICE", "llm_query_moderation"
            ),
            guard_response_service=os.getenv(
                "CONTENT_SECURITY_RESPONSE_SERVICE", "llm_response_moderation"
            ),
            guard_block_levels=os.getenv("CONTENT_SECURITY_BLOCK_LEVELS", "high,medium"),
        )


settings = Config.from_env()


# ==================== 知识点标签 ====================
# 建索引时按文件名给 chunk 打 tag；提问时也从同一集合里抽取，保证两端标签一致。
TAG_MAP = {
    "2_1_用大模型构建新人答疑机器人.md": "大模型基础",
    "2_2_扩展答疑机器人的知识范围.md": "RAG基础",
    "2_3_优化提示词改善答疑机器人回答质量.md": "提示词",
    "2_4_自动化评测答疑机器人的表现.md": "评测",
    "2_5_优化RAG应用提升问答准确度.md": "RAG优化",
}

TAG_LIST = sorted(set(TAG_MAP.values()))


def get_tag(file_name: str) -> str:
    """根据文档文件名返回知识点标签。"""
    return TAG_MAP.get(file_name, "其他")


def check_api_key() -> None:
    """确保 API Key 已配置。"""
    if not settings.api_key or settings.api_key == "sk-your-api-key-here":
        raise ValueError(
            "请先在 .env 文件中配置 DASHSCOPE_API_KEY\n"
            "获取地址: https://bailian.console.aliyun.com/?apiKey=1"
        )
