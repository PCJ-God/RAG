"""
文档解析器
使用DashScope文档智能解析PDF等复杂格式
"""
import os
import json
import logging
from llama_index.readers.dashscope.utils import ResultType
from llama_index.readers.dashscope.base import DashScopeParse
from typing import List


class DocumentParser:
    """文档解析器 - 使用DashScope文档智能"""
    
    def __init__(self, category_id: str = None):
        """
        Args:
            category_id: DashScope解析分类ID
        """
        self.category_id = category_id
        # 设置静默日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)
    
    def parse_pdf(self, file_path: str) -> str:
        """解析PDF文件为Markdown格式"""
        print(f"正在解析PDF: {file_path}")
        
        parse = self._create_parser()
        documents = parse.load_data(file_path=file_path)
        
        markdown_content = ""
        for doc in documents:
            doc_json = json.loads(json.loads(doc.text))
            for item in doc_json.get("layouts", []):
                if item.get("text") in item.get("markdownContent", ""):
                    markdown_content += item["markdownContent"]
        
        print(f"✅ PDF解析完成，生成Markdown")
        return markdown_content
    
    def polish_markdown(self, md_content: str, llm_client=None) -> str:
        """
        润色Markdown，修正格式问题
        可选：使用LLM修正格式
        
        Args:
            md_content: 原始Markdown内容
            llm_client: LLM客户端（可选）
        """
        if llm_client is None:
            return md_content
        
        # 使用LLM润色
        prompt = f"""
        下面这段文本是由PDF转为Markdown的，格式和内容可能存在一些问题，需要你帮我优化下：
        1、目录层级，如果目录层级顺序不对请以markdown形式补全或修改；
        2、内容错误，如果存在上下文不一致的情况，请你修改下；
        3、如果有表格，注意上下行不一致的情况；
        4、输出文本整体应该与输入没有较大差异，不要自己制造内容，我是需要对原文进行润色；
        5、输出格式要求：markdown文本，你的所有回答都应该放在一个markdown文件里面。
        
        特别注意：只输出转换后的 markdown 内容本身，不输出任何其他信息。
        
        需要处理的内容是：
        {md_content}
        """
        
        response = llm_client.invoke(prompt)
        print(f"✅ Markdown润色完成")
        return response
    
    def _create_parser(self):
        """创建DashScope解析器"""
        return DashScopeParse(
            result_type=ResultType.DASHSCOPE_DOCMIND,
            category_id=self.category_id
        )
