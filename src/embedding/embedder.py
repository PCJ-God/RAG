"""
向量化模块
"""
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels


class Embedder:
    """向量化器"""
    
    def __init__(self, model_name: str = "text-embedding-v3", 
                 embed_batch_size: int = 6, 
                 embed_input_length: int = 8192):
        """
        Args:
            model_name: Embedding模型名称
            embed_batch_size: 批处理大小
            embed_input_length: 最大输入长度
        """
        self.model_name = model_name
        self.embed_batch_size = embed_batch_size
        self.embed_input_length = embed_input_length
        
        # 映射到LlamaIndex的模型枚举
        model_map = {
            'text-embedding-v2': DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
            'text-embedding-v3': DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3
        }
        
        dashscope_model = model_map.get(model_name, DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3)
        
        self.embed_model = DashScopeEmbedding(
            model_name=dashscope_model,
            embed_batch_size=embed_batch_size,
            embed_input_length=embed_input_length
        )
    
    def get_model(self):
        """获取Embedding模型实例"""
        return self.embed_model
