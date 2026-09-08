"""
答案生成器
"""
from llama_index.core import PromptTemplate
from src.config import LLM_MODEL
from src.generation.llm_client import LLMClient
from src.generation.prompt_templates import PromptTemplates


class AnswerGenerator:
    """答案生成器"""
    
    def __init__(self, llm_client: LLMClient = None, prompt_template: str = None):
        """
        Args:
            llm_client: LLM客户端
            prompt_template: Prompt模板
        """
        self.llm_client = llm_client or LLMClient()
        self.prompt_template = prompt_template or PromptTemplates.get_template('rag')
    
    def generate(self, question: str, contexts: list) -> str:
        """
        生成答案
        
        Args:
            question: 用户问题
            contexts: 检索到的上下文
            
        Returns:
            生成的答案
        """
        context_str = "\n\n".join(contexts)
        
        prompt = self.prompt_template.format(
            context_str=context_str,
            query_str=question
        )
        
        return self.llm_client.invoke(prompt)
    
    def generate_stream(self, question: str, contexts: list):
        """
        流式生成答案
        
        Args:
            question: 用户问题
            contexts: 检索到的上下文
            
        Yields:
            流式回复内容
        """
        context_str = "\n\n".join(contexts)
        
        prompt = self.prompt_template.format(
            context_str=context_str,
            query_str=question
        )
        
        for chunk in self.llm_client.stream_response(prompt):
            yield chunk
    
    def update_template(self, new_template: str):
        """更新Prompt模板"""
        self.prompt_template = new_template
