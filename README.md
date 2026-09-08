# RAG 智能问答系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 项目简介

本项目是一个基于 **RAG（检索增强生成）** 架构的企业级智能问答系统，源自阿里云大模型 ACP 认证课程的实践与扩展。面向教育内容开发公司场景，帮助新员工通过自然语言提问快速获取公司内部知识。

项目从课程 Notebook 教学代码演进为**生产级 Python 包**，具备模块化架构、Web API 服务、自动化评测和多种检索优化策略。

## ✨ 核心特性

### 数据处理
- **多格式文档解析**：PDF（DashScope 文档智能）/ Markdown / DOCX / TXT
- **5 种分块策略**：Sentence / SentenceWindow / Semantic / Markdown / Token，支持 Ragas 对比评测

### 检索优化
- **LLM 问题扩展**：将模糊/歧义问题改写为完整、无歧义的综合问题
- **多步骤查询分解**：将复杂问题自动分解为多个子查询依次执行
- **HyDE 假设文档嵌入**：生成假设答案文档来增强向量检索
- **Rerank 重排序**：LLM 精排，先粗排 top-20 再精排 top-3
- **标签过滤**：自动提取人名/部门/职位等结构化标签用于过滤

### 对话与生成
- **多轮对话**：CondenseQuestionChatEngine，自动理解上下文指代
- **答案生成器**：支持流式输出和自定义 Prompt 模板
- **Meta Prompting 提示词教练**：AI 裁判驱动，自动分析并优化提示词质量

### 工程化
- **Web API 服务**：FastAPI + 8 个 REST 端点 + SSE 流式输出
- **CLI 交互界面**：支持多种策略开关
- **自动化评测**：集成 Ragas 框架，一键对比不同策略效果
- **Milvus 向量存储**：嵌入式本地持久化，无需 Docker

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户交互层                                │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  CLI 交互界面 │  │  Web API 服务 │                             │
│  │ (run_qa.py)  │  │ (FastAPI)    │                             │
│  └──────┬───────┘  └──────┬───────┘                             │
└─────────┼────────────────┼──────────────────────────────────────┘
          │                │
┌─────────┼────────────────┼──────────────────────────────────────┐
│         ▼       应用逻辑层        ▼                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  多轮对话引擎  │  │  查询优化器   │  │  答案生成器   │           │
│  │ (CondenseQ)  │  │ (扩展/分解/  │  │ (LLM+Prompt) │           │
│  │              │  │  HyDE)       │  │              │           │
│  └──────────────┘  └──────┬───────┘  └──────┬───────┘           │
│                           │                 │                    │
│                    ┌──────┴───────┐         │                    │
│                    │ Meta Prompting        │                    │
│                    │ (Prompt Coach) │◄──────┘                    │
│                    │ · 分析/优化提示词        │                    │
│                    │ · AI 裁判自动判定        │                    │
│                    └──────────────┘                              │
└─────────┼────────────────┼──────────────────────────────────────┘
          │                │
┌─────────┼────────────────┼──────────────────────────────────────┐
│         ▼       检索与存储层      ▼                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  向量检索     │  │  Rerank 模块  │  │  标签过滤器   │           │
│  │ (Embedding)  │  │ (LLMRerank)  │  │ (Tag Filter) │           │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘           │
│         │                │                                       │
│  ┌──────┴────────────────┴──────────────────────────┐            │
│  │            向量索引与持久化                        │            │
│  │    内存 (开发) / Milvus (本地) / DashVector (云)   │            │
│  └───────────────────────────────────────────────────┘            │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────┼─────────────────────────────────────────┐
│                        ▼       数据处理层                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  文档加载器   │  │  文档解析器   │  │  分块策略模块  │            │
│  │ (PDF/MD/DOCX)│  │(DashScope AI)│  │ (5 种方法)    │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└───────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
git clone <your-repo-url>
cd RAG

conda create -n rag_learn python=3.10
conda activate rag_learn
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 DashScope API Key
```

### 3. 构建索引

```bash
python scripts/build_index.py
```

### 4. 运行问答

```bash
# CLI 交互模式
python scripts/run_qa.py

# 直接提问
python scripts/run_qa.py --question "张伟是哪个部门的"

# 启用检索策略
python scripts/run_qa.py --rerank          # Rerank 重排序
python scripts/run_qa.py --hyde            # HyDE 假设文档
python scripts/run_qa.py --query-expand    # LLM 问题扩展
python scripts/run_qa.py --multistep       # 多步骤查询分解
python scripts/run_qa.py --multi-turn      # 多轮对话模式
```

### 5. 启动 Web API

```bash
# 开发模式（auto-reload）
PYTHONIOENCODING=utf-8 python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload

# 生产模式
PYTHONIOENCODING=utf-8 python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 6. 运行评测

```bash
python scripts/run_eval.py              # 默认策略
python scripts/run_eval.py --all        # 对比 4 种策略
python scripts/run_eval.py --rerank     # 仅 Rerank
python scripts/run_eval.py --hyde       # 仅 HyDE
```

## 📡 API 端点

| 端点 | 方法 | 功能 |
|------|------|------|
| `GET /health` | GET | 健康检查 |
| `POST /query` | POST | 单轮问答 |
| `POST /query/stream` | POST (SSE) | 流式问答 |
| `POST /chat` | POST | 多轮对话 |
| `POST /chat/stream` | POST (SSE) | 流式多轮对话 |
| `POST /retrieve` | POST | 纯检索 |
| `POST /prompt-coach` | POST | 提示词教练 |
| `GET /sessions` | GET | 列出会话 |
| `DELETE /sessions/{id}` | DELETE | 删除会话 |

### API 示例

```bash
# 单轮问答
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "张伟是哪个部门的？", "use_query_expand": true}'

# 多轮对话
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'

# 提示词教练（AI 裁判驱动）
curl -X POST http://localhost:8000/prompt-coach \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你是一个客服", "mode": "auto", "max_iterations": 5}'
```

## 📁 项目结构

```
RAG/
├── README.md                    # 项目说明
├── requirements.txt             # Python 依赖
├── .env.example                 # 环境变量模板
├── PROJECT_STRUCTURE.md         # 详细模块说明
├── QUICKSTART.md                # 5 分钟快速开始
│
├── data/                        # 数据
│   ├── docs/                    # 原始文档 (PDF/MD/DOCX)
│   └── qa_pairs.json            # 测试问答对
│
├── src/                         # 核心代码 (9 个模块)
│   ├── config.py                # 配置管理 (API Key/模型参数/RAG 配置)
│   ├── document/                # 文档处理
│   │   ├── loader.py            #   文档加载 (SimpleDirectoryReader)
│   │   ├── parser.py            #   PDF 解析 (DashScopeParse)
│   │   └── splitter.py          #   分块策略 (5 种方法)
│   ├── embedding/               # 向量化
│   │   └── embedder.py          #   Embedding 模型封装
│   ├── storage/                 # 索引存储
│   │   ├── vector_store.py      #   向量存储管理器 (Memory/Milvus)
│   │   └── index_manager.py     #   索引生命周期管理
│   ├── retrieval/               # 检索模块
│   │   ├── retriever.py         #   基础检索器 (支持多种策略)
│   │   ├── reranker.py          #   Rerank 后处理器
│   │   ├── query_rewriter.py    #   查询改写 (Rewrite/HyDE/MultiStep)
│   │   └── tag_extractor.py     #   标签提取 (人名/部门/职位)
│   ├── generation/              # 答案生成
│   │   ├── llm_client.py        #   LLM 客户端 (OpenAI 兼容)
│   │   ├── answer_generator.py  #   答案生成器
│   │   ├── prompt_templates.py  #   Prompt 模板管理
│   │   └── prompt_coach.py      #   Meta Prompting 提示词教练
│   ├── chat/                    # 多轮对话
│   │   └── multi_turn_chat.py   #   CondenseQuestionChatEngine
│   ├── evaluation/              # 自动化评测
│   │   └── ragas_eval.py        #   Ragas 评测 (Answer Correctness/Context Recall/Precision)
│   └── app/                     # 应用入口
│       ├── cli.py               #   CLI 交互界面
│       └── api/                 #   Web API 服务
│           ├── app.py           #     FastAPI 应用 + 所有端点
│           └── schemas.py       #     Pydantic 请求/响应模型
│
├── scripts/                     # 运行脚本
│   ├── build_index.py           # 构建索引
│   ├── run_qa.py                # 命令行问答
│   ├── run_eval.py              # 自动化评测 (支持策略对比)
│   └── meta_prompting.py        # 提示词教练 CLI
│
└── knowledge_base/              # 持久化索引存储 (自动生成)
```

## 🧪 核心功能演示

### 基础 RAG 问答

```python
from src.retrieval.retriever import Retriever
from src.generation.answer_generator import AnswerGenerator

retriever = Retriever()
generator = AnswerGenerator()

question = "我们公司项目管理应该用什么工具？"
contexts = retriever.retrieve(question, top_k=3)
answer = generator.generate(question, contexts)
```

### 多轮对话

```python
from src.chat.multi_turn_chat import CondenseQuestionChatEngine

chat_engine = CondenseQuestionChatEngine()
chat_engine.chat("张伟是哪个部门的？")     # 第一轮
chat_engine.chat("他的联系方式是什么？")   # 自动理解"他"指代张伟
```

### Meta Prompting 提示词教练

```python
from src.generation.prompt_coach import PromptCoach

coach = PromptCoach()

# 固定迭代模式
result = coach.coach("你是一个客服，回答问题要简短")

# AI 裁判驱动模式（自动判断何时停止）
result = coach.coach_auto("你是一个客服，回答问题要简短")
print(result['judge_verdict'])   # "通过" / "不通过"
print(result['iterations'])      # 实际迭代次数
```

### 检索策略对比评测

```bash
python scripts/run_eval.py --all
```

输出对比表：

```
策略对比结果
------------------------------------------------------------
策略                 | answer_correctness | context_recall | context_precision
baseline             |           0.7115   |       0.5000   |          1.0000
rerank               |           0.6580   |       0.0000   |          1.0000
hyde                 |           0.6616   |       1.0000   |          1.0000
rerank+hyde          |           0.6568   |       0.5000   |          0.0000
```

## 📊 评测指标

| 指标 | 说明 | 优化方向 |
|------|------|----------|
| **Answer Correctness** | 答案正确性（语义相似度 + 事实准确度） | 优化 Prompt、升级 LLM |
| **Context Recall** | 有多少相关文档被召回 | 升级 Embedding、改进分块 |
| **Context Precision** | 召回文档中相关条目的排名 | Rerank、Query 改写 |

## 🛠️ 技术栈

| 类别 | 技术选型 |
|------|----------|
| **编程语言** | Python 3.10+ |
| **LLM 框架** | LlamaIndex 0.11.0 |
| **大模型** | Qwen-Max / Qwen-Plus / Qwen3-Thinking |
| **Embedding** | DashScope text-embedding-v3 |
| **向量存储** | 内存 (开发) / Milvus (本地) / DashVector (云) |
| **Rerank** | LLMRerank (LLM 基于) |
| **评测框架** | Ragas |
| **Web 框架** | FastAPI + Uvicorn |

## 📚 学习资源

本项目基于 [阿里云大模型 ACP 认证课程](https://edu.aliyun.com/course/3130200) 开发：

- [C2_构造问答系统](https://edu.aliyun.com/course/3130200/) — RAG 原理与实践
- [C3_构建 Agent 系统](https://edu.aliyun.com/course/3130200/) — Agent 设计与开发
- [C4_交付上线](https://edu.aliyun.com/course/3130200/) — 模型优化与上线

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交 Pull Request

## 📄 许可证

MIT 许可证，详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [LlamaIndex](https://www.llamaindex.ai/) — RAG 开发框架
- [Ragas](https://docs.ragas.io/) — RAG 评测框架
- [阿里云百炼](https://bailian.console.aliyun.com/) — 大模型服务平台
