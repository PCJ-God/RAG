# 项目结构说明

## 目录结构

```
RAG/
├── README.md                    # 项目主文档（项目介绍、架构图、快速开始）
├── requirements.txt             # Python依赖包列表
├── .env.example                 # 环境变量配置模板
├── .gitignore                   # Git忽略文件配置
├── LICENSE                      # MIT开源许可证
│
├── data/                        # 数据目录
│   ├── docs/                    # 原始知识库文档（PDF/MD/DOCX/TXT）
│   │   └── README.md            # 文档使用说明（占位符）
│   └── qa_pairs.json            # 测试问答对（用于自动化评测）
│
├── src/                         # 源代码目录
│   ├── __init__.py              # 模块初始化
│   ├── config.py                # 配置管理（API Key、模型参数、RAG配置）
│   │
│   ├── document/                # 文档处理模块
│   │   ├── __init__.py
│   │   ├── loader.py            # 文档加载器（支持多格式加载）
│   │   ├── parser.py            # 文档解析器（DashScope文档智能）
│   │   └── splitter.py          # 分块策略工厂（5种分块方法）
│   │
│   ├── embedding/               # 向量化模块
│   │   ├── __init__.py
│   │   └── embedder.py          # Embedding封装（text-embedding-v2/v3）
│   │
│   ├── storage/                 # 存储模块
│   │   ├── __init__.py
│   │   ├── vector_store.py      # 向量存储管理（内存/本地/云服务）
│   │   └── index_manager.py     # 索引管理器（构建/保存/加载）
│   │
│   ├── retrieval/               # 检索模块
│   │   ├── __init__.py
│   │   ├── retriever.py         # 基础检索器
│   │   ├── query_rewriter.py    # 查询改写（多步查询/HyDE）
│   │   ├── tag_extractor.py     # 标签提取（人名/部门/职位）
│   │   └── reranker.py          # 重排序模块
│   │
│   ├── generation/              # 生成模块
│   │   ├── __init__.py
│   │   ├── llm_client.py        # LLM客户端（OpenAI SDK兼容）
│   │   ├── prompt_templates.py  # Prompt模板集合
│   │   └── answer_generator.py  # 答案生成器
│   │
│   ├── chat/                    # 对话模块
│   │   ├── __init__.py
│   │   └── multi_turn_chat.py   # 多轮对话引擎（CondenseQuestion）
│   │
│   ├── evaluation/              # 评测模块
│   │   ├── __init__.py
│   │   └── ragas_eval.py        # Ragas自动化评测
│   │
│   └── app/                     # 应用入口
│       ├── __init__.py
│       └── cli.py               # 命令行交互界面
│
├── scripts/                     # 运行脚本
│   ├── build_index.py           # 构建索引脚本
│   ├── run_qa.py                # 运行问答脚本
│   └── run_eval.py              # 运行评测脚本
│
├── knowledge_base/              # （自动生成）索引存储目录
│
└── tests/                       # （可选）单元测试目录

```

## 核心模块说明

### 1. document（文档处理）
- **loader.py**: 使用LlamaIndex的SimpleDirectoryReader加载文档
- **parser.py**: 使用DashScope文档智能解析PDF等复杂格式
- **splitter.py**: 提供5种分块策略
  - Sentence: 默认句子分块
  - Sentence Window: 句子窗口（保留上下文）
  - Semantic: 语义分块
  - Markdown: Markdown标题分块
  - Token: Token数量分块

### 2. embedding（向量化）
- **embedder.py**: 封装DashScope Embedding模型
  - 支持text-embedding-v2/v3
  - 可配置批处理和最大输入长度

### 3. storage（存储）
- **vector_store.py**: 向量存储管理
  - 内存存储（开发测试）
  - 本地持久化
  - 云服务（Milvus/Qdrant/DashVector）
- **index_manager.py**: 索引生命周期管理
  - 构建/保存/加载索引
  - 避免重复构建

### 4. retrieval（检索）
- **retriever.py**: 基础检索器
  - 向量相似度检索
  - 可配置top_k和相似度阈值
- **query_rewriter.py**: 查询优化
  - Query Rewriting: LLM改写
  - MultiStepQuery: 多步查询分解
  - HyDE: 假设文档嵌入
- **tag_extractor.py**: 标签提取
  - 提取人名/部门/职位等结构化标签
  - 用于后续标签过滤+向量混合检索
- **reranker.py**: 重排序
  - 先粗排top-20，再精排top-3

### 5. generation（生成）
- **llm_client.py**: LLM客户端
  - 支持qwen-turbo/plus/max/Qwen3
  - 支持流式/非流式调用
  - 可配置temperature/top_p/seed等参数
- **prompt_templates.py**: Prompt模板
  - RAG默认模板
  - 结构化输出模板
  - 问题改写模板
  - 标签提取模板
- **answer_generator.py**: 答案生成
  - 基于检索结果生成答案
  - 支持流式输出

### 6. chat（对话）
- **multi_turn_chat.py**: 多轮对话引擎
  - CondenseQuestionChatEngine
  - 问题改写+历史对话管理
  - 支持对话历史持久化

### 7. evaluation（评测）
- **ragas_eval.py**: Ragas评测
  - Answer Correctness（答案正确性）
  - Context Recall（上下文召回率）
  - Context Precision（上下文精确度）
  - 支持单个/批量评测
  - 支持分块策略对比

### 8. app（应用）
- **cli.py**: 命令行交互界面
  - 初始化（加载文档→分块→建索引）
  - 交互模式（循环问答）
  - 单次模式（直接回答单个问题）
  - 显示参考文档和相关度

## 数据流

```
用户提问
    ↓
CLI/Web界面
    ↓
查询优化（可选）
    ↓ Query Rewrite / MultiStep / HyDE
    ↓
检索模块
    ↓ 向量相似度检索 → top_k个文档块
    ↓ （可选）标签过滤 + Rerank精排
    ↓
答案生成
    ↓ 将问题+上下文输入LLM
    ↓ 流式/非流式输出答案
    ↓
返回给用户
```

## 快速上手

1. **配置环境**: `cp .env.example .env` 并填入API Key
2. **安装依赖**: `pip install -r requirements.txt`
3. **准备文档**: 将PDF/MD/DOCX放入 `data/docs/`
4. **构建索引**: `python scripts/build_index.py`
5. **开始问答**: `python scripts/run_qa.py`
6. **运行评测**: `python scripts/run_eval.py`
