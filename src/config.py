"""
配置管理模块
集中管理API Key、模型参数、RAG配置等
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
INDEX_DIR = PROJECT_ROOT / "knowledge_base"


# ==================== API配置 ====================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


# ==================== 模型配置 ====================
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")
RERANK_MODEL = os.getenv("RERANK_MODEL", "qwen3-rerank")


# ==================== RAG配置 ====================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "5"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))


# ==================== 生成配置 ====================
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.8"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))


def check_api_key():
    """检查API Key是否配置"""
    if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY == "sk-your-api-key-here":
        raise ValueError(
            "请先在.env文件中配置DASHSCOPE_API_KEY\n"
            "获取地址: https://bailian.console.aliyun.com/?apiKey=1"
        )
    return True
