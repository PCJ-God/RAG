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
from src.generation.answer_generator import AnswerGenerator
from src.chat.multi_turn_chat import CondenseQuestionChatEngine
from src.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SIMILARITY_TOP_K


class CLIApp:
    """命令行应用"""

    def __init__(self, docs_dir=None, rebuild_index=False,
                 use_rerank=False, use_hyde=False, use_multi_turn=False,
                 use_query_expand=False, use_multistep=False):
        """
        Args:
            docs_dir: 文档目录
            rebuild_index: 是否重建索引
            use_rerank: 是否启用 Rerank
            use_hyde: 是否启用 HyDE
            use_multi_turn: 是否启用多轮对话模式
            use_query_expand: 是否启用 LLM 问题扩展
            use_multistep: 是否启用多步骤查询分解
        """
        check_api_key()

        self.docs_dir = docs_dir or DOCS_DIR
        self.rebuild_index = rebuild_index
        self.use_rerank = use_rerank
        self.use_hyde = use_hyde
        self.use_multi_turn = use_multi_turn
        self.use_query_expand = use_query_expand
        self.use_multistep = use_multistep
        self.query_engine = None
        self.chat_engine = None
        self.answer_generator = None

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

        # 创建检索器
        llm_client = LLMClient()
        self.retriever = Retriever(
            index,
            similarity_top_k=SIMILARITY_TOP_K
        )

        # 启用策略
        if self.use_rerank:
            self.retriever.enable_rerank(llm_client=llm_client)
            print("🔀 Rerank 已启用")
        if self.use_hyde:
            self.retriever.enable_hyde()
            print("🧪 HyDE 已启用")
        if self.use_query_expand:
            self.retriever.enable_query_expand()
            print("🔄 问题扩展已启用")
        if self.use_multistep:
            self.retriever.enable_multistep()
            print("🔗 多步骤查询已启用")

        # 创建查询引擎
        self.retriever.create_query_engine(llm=llm_client.get_llm())
        self.query_engine = self.retriever.query_engine

        # 多轮对话模式
        if self.use_multi_turn:
            self.chat_engine = CondenseQuestionChatEngine(
                query_engine=self.query_engine,
                llm=llm_client.get_llm(),
                verbose=False
            )
            print("💬 多轮对话模式已启用")

        # 答案生成器
        self.answer_generator = AnswerGenerator(llm_client=llm_client)

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
        if self.use_multi_turn:
            print("（多轮对话模式：后续问题将结合历史上下文）")
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

        # 多轮对话模式：使用 CondenseQuestionChatEngine
        if self.use_multi_turn and self.chat_engine:
            print("🤖 AI正在思考...")
            full_response = self.chat_engine.chat(question)

            print("\n" + "-" * 60)
            print("🤖 回答:")
            print("-" * 60)
            print(full_response)

            # 显示参考文档
            print("\n" + "-" * 60)
            print("📚 参考文档:")
            print("-" * 60)
            if hasattr(self.chat_engine.chat_engine, 'retrieve'):
                nodes = self.chat_engine.chat_engine.retrieve(question)
                for i, node in enumerate(nodes, 1):
                    print(f"\n文档片段 {i} (相关度: {node.score:.4f}):")
                    print(f"  {node.text[:100]}...")
        else:
            # 单轮模式：用 AnswerGenerator
            print("🤖 AI正在思考...", end="")

            # 先检索上下文
            nodes = self.retriever.retrieve(question)
            contexts = [node.text for node in nodes]

            # 用 AnswerGenerator 生成回答
            response = self.answer_generator.generate(question, contexts)

            print("\n" + "-" * 60)
            print("🤖 回答:")
            print("-" * 60)
            print(response)

            # 显示参考文档
            print("\n" + "-" * 60)
            print("📚 参考文档:")
            print("-" * 60)
            for i, node in enumerate(nodes, 1):
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
    parser.add_argument("--rerank", action="store_true", help="启用 Rerank 重排序")
    parser.add_argument("--hyde", action="store_true", help="启用 HyDE 假设文档嵌入")
    parser.add_argument("--multi-turn", action="store_true", help="启用多轮对话模式")
    parser.add_argument("--query-expand", action="store_true", help="启用 LLM 问题扩展")
    parser.add_argument("--multistep", action="store_true", help="启用多步骤查询分解")

    args = parser.parse_args()

    app = CLIApp(
        docs_dir=args.docs_dir,
        rebuild_index=args.rebuild,
        use_rerank=args.rerank,
        use_hyde=args.hyde,
        use_multi_turn=args.multi_turn,
        use_query_expand=args.query_expand,
        use_multistep=args.multistep
    )
    app.run(question=args.question)


if __name__ == "__main__":
    main()
