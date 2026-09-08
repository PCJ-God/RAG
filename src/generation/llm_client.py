"""
LLM客户端
封装大语言模型的调用
"""
import os
from openai import OpenAI
from llama_index.llms.openai_like import OpenAILike
from src.config import (
    DASHSCOPE_API_KEY, BASE_URL, LLM_MODEL,
    TEMPERATURE, TOP_P, MAX_TOKENS
)


class LLMClient:
    """LLM客户端"""
    
    def __init__(self, model: str = None, temperature: float = None, 
                 top_p: float = None, max_tokens: int = None):
        """
        Args:
            model: 模型名称
            temperature: 温度参数
            top_p: top_p参数
            max_tokens: 最大token数
        """
        self.model = model or LLM_MODEL
        self.temperature = temperature or TEMPERATURE
        self.top_p = top_p or TOP_P
        self.max_tokens = max_tokens or MAX_TOKENS
        
        # 创建OpenAI客户端
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=BASE_URL
        )
        
        # 创建LlamaIndex兼容的LLM
        self.llm = OpenAILike(
            model=self.model,
            api_base=BASE_URL,
            api_key=DASHSCOPE_API_KEY,
            is_chat_model=True,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
    
    def get_llm(self):
        """获取LlamaIndex LLM实例"""
        return self.llm
    
    def invoke(self, prompt: str, system_prompt: str = None) -> str:
        """
        调用LLM
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            
        Returns:
            LLM回复
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
    
    def stream_response(self, prompt: str, system_prompt: str = None):
        """
        流式调用LLM
        
        Args:
            prompt: 用户输入
            system_prompt: 系统提示词
            
        Yields:
            流式回复内容
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
