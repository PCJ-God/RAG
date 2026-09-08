#!/usr/bin/env python
"""
构建索引脚本
"""
import sys
from pathlib import Path

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import check_api_key, DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.document.loader import DocumentLoader
from src.document.splitter import SplitterFactory
from src.storage.index_manager import IndexManager


def main():
    """主函数"""
    check_api_key()
    
    print("=" * 60)
    print("📚 开始构建索引...")
    print("=" * 60)
    
    # 加载文档
    loader = DocumentLoader(str(DOCS_DIR))
    documents = loader.load()
    
    # 选择分块策略
    print("\n选择分块策略:")
    print("1. Sentence (默认)")
    print("2. Sentence Window")
    print("3. Semantic")
    print("4. Markdown")
    print("5. Token")
    
    choice = input("\n请选择 (1-5, 默认1): ").strip()
    
    if choice == '2':
        splitter = SplitterFactory.create_window()
    elif choice == '3':
        print("⚠️ Semantic分块需要embed_model，请使用默认分块")
        splitter = SplitterFactory.create_sentence(CHUNK_SIZE, CHUNK_OVERLAP)
    elif choice == '4':
        splitter = SplitterFactory.create_markdown()
    elif choice == '5':
        splitter = SplitterFactory.create_token(100, 20)
    else:
        splitter = SplitterFactory.create_sentence(CHUNK_SIZE, CHUNK_OVERLAP)
    
    # 分块
    print("\n正在分块...")
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"✅ 分块完成，共 {len(nodes)} 个文档块")
    
    # 构建索引
    index_manager = IndexManager()
    index = index_manager.build_index(nodes, force_rebuild=True)
    
    print("\n" + "=" * 60)
    print("✅ 索引构建完成！")
    print(f"索引保存路径: {index_manager.persist_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
