"""
命令行交互界面
"""
import sys
from pathlib import Path

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import check_api_key
from src.document.loader import DocumentLoader
from src.document.splitter import SplitterFactory
from src.storage.index_manager import IndexManager
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient
from src.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_TOP_K


class CLIApp:
    """命令行应用"""
    
    def __init__(self, docs_dir=None, rebuild_index=False):
        """
        Args:
            docs_dir: 文档目录
            rebuild_index: 是否重建索引
        """
        check_api_key()
        
        self.docs_dir = docs_dir or DOCS_DIR
        self.rebuild_index = rebuild_index
        self.query_engine = None
    
    def initialize(self):
        """初始化应用"""
        print("=" * 60)
        print("🤖 智能问答系统初始化中...")
        print("=" * 60)
        
        # 加载文档
        loader = DocumentLoader(str(self.docs_dir))
        documents = loader.load()
        
        # 分块
        splitter = SplitterFactory.create_sentence_splitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        nodes = splitter.get_nodes_from_documents(documents)
        
        # 构建/加载索引
        index_manager = IndexManager()
        if self.rebuild_index:
            index = index_manager.build_index(nodes, force_rebuild=True)
        else:
            try:
                index = index_manager.load_index()
            except FileNotFoundError:
                print("未找到现有索引，正在构建新索引...")
                index = index_manager.build_index(nodes)
        
        # 创建查询引擎
        llm_client = LLMClient()
        self.query_engine = index.as_query_engine(
            streaming=True,
            llm=llm_client.get_llm(),
            similarity_top_k=SIMILARITY_TOP_K
        )
        
        print("\n✅ 系统初始化完成！\n")
    
    def run(self, question=None):
        """
        运行问答
        
        Args:
            question: 如果提供则直接回答该问题，否则进入交互模式
        """
        if self.query_engine is None:
            self.initialize()
        
        if question:
            # 直接回答单个问题
            self._answer_question(question)
        else:
            # 进入交互模式
            self._interactive_mode()
    
    def _interactive_mode(self):
        """交互模式"""
        print("=" * 60)
        print("💬 进入交互模式")
        print("输入问题后按回车提问，输入 'quit' 或 'exit' 退出")
        print("=" * 60)
        
        while True:
            question = input("\n👤 请输入你的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not question:
                continue
            
            self._answer_question(question)
    
    def _answer_question(self, question: str):
        """回答问题"""
        print(f"\n🤔 问题: {question}")
        print("🤖 AI正在思考...", end="")
        
        response = self.query_engine.query(question)
        
        print("\n" + "-" * 60)
        print("🤖 回答:")
        print("-" * 60)
        
        # 流式输出
        if hasattr(response, 'print_response_stream') and callable(response.print_response_stream):
            response.print_response_stream()
        else:
            print(str(response))
        
        # 显示参考文档
        print("\n" + "-" * 60)
        print("📚 参考文档:")
        print("-" * 60)
        for i, node in enumerate(response.source_nodes, 1):
            print(f"\n文档片段 {i} (相关度: {node.score:.4f}):")
            print(f"  {node.text[:100]}...")
        
        print("\n" + "=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="智能问答系统CLI")
    parser.add_argument("--question", "-q", type=str, help="直接回答的问题")
    parser.add_argument("--rebuild", action="store_true", help="重建索引")
    parser.add_argument("--docs-dir", type=str, help="文档目录路径")
    
    args = parser.parse_args()
    
    app = CLIApp(docs_dir=args.docs_dir, rebuild_index=args.rebuild)
    app.run(question=args.question)


if __name__ == "__main__":
    main()
