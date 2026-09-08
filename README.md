# 基于RAG的企业智能问答系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 项目简介

本项目是一个基于**RAG（检索增强生成）**架构的企业级智能问答系统，面向教育内容开发公司场景，帮助新员工通过自然语言提问快速获取公司内部知识（规章制度、写作规范、工具使用指南等），替代传统规则匹配型问答系统，显著降低人工答疑成本。

### 核心特性

- ✅ **多格式文档解析**: 支持PDF(DashScope文档智能)/Markdown/DOCX
- ✅ **多种分块策略**: Sentence/SentenceWindow/Semantic/Markdown/Token，支持对比评测
- ✅ **智能检索优化**: Query Rewriting(LLM问题扩展)、MultiStepQuery分解、HyDE假设文档检索
- ✅ **重排序增强**: 先粗排top-20，再用Rerank模型精排top-3
- ✅ **标签过滤**: 从文档/问题中提取结构化标签(人名/部门/职位)+向量混合检索
- ✅ **多轮对话**: CondenseQuestionChatEngine，支持上下文感知对话
- ✅ **Meta Prompting**: 让大模型成为提示词教练，自动分析并优化提示词质量
- ✅ **自动化评测**: 集成Ragas框架，量化评估Answer Correctness/Context Recall/Context Precision
- ✅ **流式输出**: 提升用户体验，减少等待时间
- ✅ **模块化架构**: 分块策略/Embedding模型/Rerank模型/LLM均可通过配置切换

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户交互层                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  CLI交互界面  │  │  Web API服务  │  │  Jupyter教程  │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼────────────────┼────────────────┼────────────────────┘
          │                │                │
┌─────────┼────────────────┼────────────────┼────────────────────┐
│         ▼       应用逻辑层        ▼                ▼            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  多轮对话引擎  │  │  查询优化器   │  │  答案生成器   │           │
│  │ (CondenseQ)  │  │ (Rewrite/HyDE)│  │ (LLM+Prompt) │           │
│  └──────────────┘  └──────────────┘  └──────┬───────┘           │
│                       ┌──────────────┐       │                    │
│                       │ Meta Prompting│      │                    │
│                       │ (Prompt Coach)│◄─────┘                    │
│                       └──────────────┘                            │
└─────────┼────────────────┼────────────────┼────────────────────┘
          │                │                │
┌─────────┼────────────────┼────────────────┼────────────────────┐
│         ▼       检索与存储层      ▼                ▼            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  向量检索     │  │  重排序模块   │  │  标签过滤器   │           │
│  │ (Embedding)  │  │  (Rerank)    │  │ (Tag Filter) │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                │                │                     │
│  ┌──────┴────────────────┴────────────────┴───────┐             │
│  │              向量索引与持久化                    │             │
│  │  (内存/本地Milvus/云服务DashVector)             │             │
│  └────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
          │
┌─────────┼──────────────────────────────────────────────────────┐
│         ▼       数据处理层                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  文档加载器   │  │  文档解析器   │  │  分块策略模块  │           │
│  │ (PDF/MD/DOCX)│  │(DashScope AI)│  │(5种分块方法)  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd RAG

# 创建虚拟环境
conda create -n rag_learn python=3.10
conda activate rag_learn

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置API Key

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑.env文件，填入你的DashScope API Key
```

`.env`文件内容：
```
DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 构建索引

```bash
python scripts/build_index.py
```

### 4. 运行问答

```bash
# 命令行交互模式
python scripts/run_qa.py

# 或直接提问
python scripts/run_qa.py --question "张伟是哪个部门的"
```

### 5. 运行评测

```bash
python scripts/run_eval.py
```

## 📁 项目结构

```
RAG/
├── README.md                    # 项目说明
├── requirements.txt             # Python依赖
├── .env.example                 # 环境变量模板
│
├── data/                        # 示例数据
│   ├── docs/                    # 原始文档(PDF/MD/DOCX)
│   └── qa_pairs.json            # 测试问答对
│
├── src/                         # 核心代码
│   ├── config.py                # 配置管理
│   ├── document/                # 文档处理
│   ├── embedding/               # 向量化
│   ├── storage/                 # 索引存储
│   ├── retrieval/               # 检索模块
│   ├── generation/              # 答案生成
│   ├── chat/                    # 多轮对话
│   ├── evaluation/              # 自动化评测
│   └── app/                     # 应用入口
│
├── tests/                       # 单元测试
└── scripts/                     # 运行脚本
```

## 🧪 核心功能演示

### 基础RAG问答

```python
from src.retrieval.retriever import Retriever
from src.generation.answer_generator import AnswerGenerator

retriever = Retriever()
generator = AnswerGenerator()

question = "我们公司项目管理应该用什么工具？"
contexts = retriever.retrieve(question, top_k=3)
answer = generator.generate(question, contexts)
print(answer)
```

### 多轮对话

```python
from src.chat.multi_turn_chat import CondenseQuestionChatEngine

chat_engine = CondenseQuestionChatEngine()

# 第一轮
response1 = chat_engine.chat("张伟是哪个部门的？")
print(response1)

# 第二轮（自动理解"他"指代张伟）
response2 = chat_engine.chat("他的联系方式是什么？")
print(response2)
```

### Meta Prompting 提示词教练

参考 ACP 教程 2_3 节 4.7 "让大模型帮你打造专属 AI 裁判"。

```python
from src.generation.prompt_coach import PromptCoach

coach = PromptCoach()

# 模式 1: 固定迭代（人为指定次数）
result = coach.coach("你是一个客服，回答问题要简短")

# 模式 2: AI 裁判驱动（自动判断何时停止）
result = coach.coach_auto("你是一个客服，回答问题要简短")
print(result['judge_verdict'])   # "通过" / "不通过"
print(result['iterations'])      # 实际迭代次数
print(result['optimized_prompt']) # 优化后的 Prompt
```

**CLI 交互模式：**
```bash
# 固定迭代
python scripts/meta_prompting.py --prompt "你是一个客服" --iterations 3

# AI 裁判驱动（自动停止）
python scripts/meta_prompting.py --prompt "你是一个客服" --auto --max-iterations 5
```

**API 调用：**
```bash
# 固定迭代
curl -X POST http://localhost:8000/prompt-coach \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你是一个客服", "mode": "fixed", "iterations": 2}'

# AI 裁判驱动
curl -X POST http://localhost:8000/prompt-coach \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你是一个客服", "mode": "auto", "max_iterations": 5}'
```

### 分块策略对比

```python
from src.evaluation.ragas_eval import compare_splitter_strategies

# 对比不同分块策略的Ragas指标
compare_splitter_strategies(
    documents=documents,
    question="张伟是哪个部门的？",
    ground_truth="张伟在教研部、课程开发部和IT部..."
)
```

## 📊 评测指标说明

本项目使用[Ragas](https://docs.ragas.io/)框架进行自动化评测，主要指标包括：

| 指标 | 说明 | 优化方向 |
|------|------|----------|
| **Answer Correctness** | 答案正确性（语义相似度+事实准确度） | 优化Prompt、升级LLM |
| **Context Recall** | 有多少相关文档被召回 | 升级Embedding模型、改进分块 |
| **Context Precision** | 召回文档中相关条目的排名 | 添加Rerank、Query改写 |

## 🛠️ 技术栈

| 类别 | 技术选型 |
|------|----------|
| **编程语言** | Python 3.10+ |
| **LLM框架** | OpenAI SDK(兼容模式) / LlamaIndex |
| **大模型** | Qwen-Max / Qwen-Plus / Qwen3-Thinking |
| **Embedding** | DashScope text-embedding-v3 |
| **向量存储** | 内存(开发) / Milvus-Qdrant(生产) / DashVector(云) |
| **Rerank** | qwen3-rerank / gte-rerank |
| **评测框架** | Ragas |
| **Web框架** | FastAPI (可选) |

## 📚 学习资源

本项目基于[阿里云大模型ACP认证课程](https://edu.aliyun.com/course/3130200)开发，如果你想深入学习，可以参考：

- [C2_构造问答系统](https://edu.aliyun.com/course/3130200/?spm=5176.40615594.J_ZPkBgKtTYq68YQ8A0zB1K.4.f671559ciKx6w2) - RAG原理与实践
- [C3_构建Agent系统](https://edu.aliyun.com/course/3130200/?spm=5176.40615594.J_ZPkBgKtTYq68YQ8A0zB1K.4.f671559ciKx6w2) - Agent设计与开发
- [C4_交付上线](https://edu.aliyun.com/course/3130200/?spm=5176.40615594.J_ZPkBgKtTYq68YQ8A0zB1K.4.f671559ciKx6w2) - 模型优化与上线

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 提交Pull Request

## 📄 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。

## 🙏 致谢

- [LlamaIndex](https://www.llamaindex.ai/) - RAG开发框架
- [Ragas](https://docs.ragas.io/) - RAG评测框架
- [阿里云百炼](https://bailian.console.aliyun.com/) - 大模型服务平台
