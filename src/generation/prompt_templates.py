"""
Prompt模板管理
"""


class PromptTemplates:
    """Prompt模板集合"""
    
    # 基础RAG模板
    RAG_DEFAULT = """你是公司的客服小蜜，你需要简明扼要的回答用户的问题
【注意事项】：
1. 依据上下文信息来回答用户问题。
2. 你只需要回答用户的问题，不要输出其他信息
以下是参考信息。
---------------------
{context_str}
---------------------
问题：{query_str}
回答："""
    
    # 结构化输出模板
    STRUCTURED_OUTPUT = """
【任务要求】
你将看到一句话或一段话。你需要审查这段话中有没有错别字。如果出现了错别字，你要指出错误，并给出解释。
"的" 和 "地" 混淆不算错别字，没有错误
---
【输出要求】
请你只输出json格式，不要输出代码段
其中，label只能取0或1，0代表有错误，1代表没有错误
reason是错误的原因
correct是修正后的文档内容
---
【用户输入】
以下是用户输入，请审阅：
{input_text}
"""
    
    # 问题改写模板
    QUERY_REWRITE = """\
系统角色设定:
你是一个专业的问题改写助手。你的任务是将用户的原始问题扩充为一个更完整、更全面的问题。

规则：
1. 将可能的歧义、相关概念和上下文信息整合到一个完整的问题中
2. 使用括号对歧义概念进行补充说明
3. 添加关键的限定词和修饰语
4. 确保改写后的问题清晰且语义完整

原始问题:
{query}

请生成一个综合的改写问题。
"""
    
    # 标签提取模板
    TAG_EXTRACTION = """你是一个标签提取专家。请从文本中提取结构化信息，并按要求输出标签。
---
【支持的标签类型】
- 人名
- 部门名称
- 职位名称
- 技术领域
- 产品名称
---
【输出要求】
1. 请用 JSON 格式输出
2. 如果某类标签未识别到，则不输出该类
---
待分析文本如下：
{text}
"""
    
    @classmethod
    def get_template(cls, template_name: str) -> str:
        """
        获取模板
        
        Args:
            template_name: 模板名称
            
        Returns:
            模板字符串
        """
        templates = {
            'rag': cls.RAG_DEFAULT,
            'structured': cls.STRUCTURED_OUTPUT,
            'rewrite': cls.QUERY_REWRITE,
            'tag': cls.TAG_EXTRACTION
        }
        
        return templates.get(template_name, cls.RAG_DEFAULT)
