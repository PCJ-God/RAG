"""LLM 客户端：OpenAI 兼容协议调用通义千问，同时提供 LlamaIndex 实例。"""
from typing import Optional

from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI

from src.config import Config, settings


class LLMClient:
    """LLM 客户端。"""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or settings
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        self.llm = OpenAILike(
            model=self.config.llm_model,
            api_base=self.config.base_url,
            api_key=self.config.api_key,
            is_chat_model=True,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

    def get_llm(self):
        """返回 LlamaIndex 兼容的 LLM 实例。"""
        return self.llm

    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """调用 LLM 并返回文本回复。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content
