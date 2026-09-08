"""
多轮对话引擎
使用CondenseQuestionChatEngine实现上下文感知对话
"""
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.chat_engine import CondenseQuestionChatEngine


class CondenseQuestionChatEngine:
    """多轮对话引擎"""
    
    DEFAULT_PROMPT = """
    给定一段对话历史（人类与助手之间）和人类的后续问题，
    请将该问题改写为一个独立的问题，包含对话中所有相关的上下文信息。

    <对话历史>
    {chat_history}

    <后续问题>
    {question}

    <改写后的独立问题>
"""
    
    def __init__(self, query_engine, llm, chat_history: list = None, 
                 prompt: str = None, verbose: bool = False):
        """
        Args:
            query_engine: 查询引擎
            llm: 大语言模型
            chat_history: 历史对话
            prompt: 问题改写Prompt
            verbose: 是否详细输出
        """
        self.query_engine = query_engine
        self.llm = llm
        self.chat_history = chat_history or []
        self.prompt = prompt or self.DEFAULT_PROMPT
        self.verbose = verbose
        
        # 创建对话引擎
        self.chat_engine = self._create_chat_engine()
    
    def _create_chat_engine(self):
        """创建CondenseQuestionChatEngine"""
        custom_prompt = PromptTemplate(self.prompt)
        
        return CondenseQuestionChatEngine.from_defaults(
            query_engine=self.query_engine,
            condense_question_prompt=custom_prompt,
            chat_history=self.chat_history,
            llm=self.llm,
            verbose=self.verbose
        )
    
    def chat(self, message: str):
        """
        发送消息并获取回复
        
        Args:
            message: 用户消息
            
        Returns:
            助手回复
        """
        response = self.chat_engine.stream_chat(message)
        
        # 收集完整回复
        full_response = ""
        for token in response.response_gen:
            full_response += token
            if self.verbose:
                print(token, end="")
        
        # 更新对话历史
        self.chat_history.append(
            ChatMessage(role=MessageRole.USER, content=message)
        )
        self.chat_history.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=full_response)
        )
        
        return full_response
    
    def get_history(self):
        """获取对话历史"""
        return self.chat_history
    
    def reset_history(self):
        """重置对话历史"""
        self.chat_history = []
        self.chat_engine = self._create_chat_engine()
