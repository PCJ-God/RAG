"""
标签提取器
从文本中提取结构化标签用于过滤
"""
import os
import json
from openai import OpenAI
from src.config import DASHSCOPE_API_KEY, BASE_URL


class TagExtractor:
    """标签提取器"""
    
    SYSTEM_PROMPT = """你是一个标签提取专家。请从文本中提取结构化信息，并按要求输出标签。
---
【支持的标签类型】
- 人名
- 部门名称
- 职位名称
- 技术领域
- 产品名称
---
【输出要求】
1. 请用 JSON 格式输出，如：[{"key": "部门名称", "value": "教研部"}]
2. 如果某类标签未识别到，则不输出该类
---
待分析文本如下：
"""
    
    def __init__(self, model: str = "qwen-turbo"):
        """
        Args:
            model: 用于提取的模型
        """
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url=BASE_URL
        )
        self.model = model
    
    def extract_tags(self, text: str) -> list:
        """
        从文本中提取标签
        
        Args:
            text: 输入文本
            
        Returns:
            标签列表
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.SYSTEM_PROMPT},
                    {'role': 'user', 'content': text}
                ],
                response_format={"type": "json_object"}
            )
            
            result = completion.choices[0].message.content
            tags = json.loads(result)
            
            return tags
        except Exception as e:
            print(f"⚠️ 标签提取失败: {e}")
            return []
