# RAG 知识答疑助手

一个基于 **RAG（检索增强生成）** 的学习助手：对某个 RAG 知识点不清楚时直接提问，它会从课程知识库中检索相关内容，给出**带来源引用**的回答，帮助你学习而非闲聊。

## 项目定位

- 回答的是「RAG 知识点不清晰」这类学习问题，不做通用聊天。
- 知识库按主题分域管理，当前只有 `rag` 域，后续可自行添加 `agent` 等域，主流程无需改动。
- **质量优先于功能数量**：只保留一条高质量基础链路（结构感知分块 → 融合式检索 → 重排序 → 带引用生成），不堆砌策略。

## 当前已实现的内容

整条链路分 5 个阶段，每阶段对应一个 `src/` 子包：

| 阶段 | 模块 | 已实现内容 |
| --- | --- | --- |
| 文档摄入 | `document/` | Markdown 加载；结构感知分块（按标题层级切块、标题路径写入元数据、超长小节按句子二次切分、清洗 Notebook 导出的 HTML 噪声）；按文件名打知识点标签 |
| 索引 | `indexing/` | `build / load / exists` 统一接口；本地 JSON 与 DashVector 两种后端；DashVector 直连官方 SDK（建 1024 维 collection + tag 字段、分批插入、异步删除后重建） |
| 检索 | `retrieval/` | 融合式检索：语义 + 标签两路召回、合并去重；提问时先由 LLM 抽取知识点标签再按 `tag` 字段过滤；DashScope 原生 rerank 精排到 top-3，失败自动降级 LLM 精排 |
| 生成 | `generation/` | LLM 走 DashScope OpenAI 兼容模式；强制只依据检索上下文回答、防幻觉；引用编号带文档名（如 `[1]（2_2_扩展答疑机器人的知识范围.md）`） |
| 评测 | `evaluation/` | Ragas 评测，指标：`answer_correctness` / `faithfulness` / `context_recall` / `context_precision` |

## 系统架构

```
document/ ──> indexing/ ──> retrieval/ ──> generation/ ──> evaluation/
(加载+分块)   (构建/加载)   (融合检索+重排)  (LLM+提示词)    (Ragas)
                └────────── pipeline.py 端到端编排 ──────────┘
```

一次提问的完整过程：

1. 抽出问题对应的知识点标签（`TagExtractor`）。
2. 语义检索 + 标签检索两路召回（`FusionRetriever`），合并去重。
3. 重排序精排到 top-3（`Reranker`）。
4. 把 top-3 文档块连同文件名、章节编号后拼进提示词，生成带引用的回答（`AnswerGenerator`）。

## 快速开始

1. 配置环境：`cp .env.example .env`，填入 `DASHSCOPE_API_KEY`（百炼平台获取）。
2. 安装依赖：`pip install -r requirements.txt`。
3. 构建索引：`python scripts/build_index.py`。
4. 提问：`python scripts/ask.py "RAG 建立索引包括哪几步？"`。
5. 评测：`python scripts/eval.py`。

## 向量库

默认用本地 JSON（`VECTOR_STORE=local`），零依赖、开箱即用。

要用阿里云 DashVector，在 `.env` 中设置：

```ini
VECTOR_STORE=dashvector
DASHSVECTOR_API_KEY=你的 DashVector API-KEY
DASHSVECTOR_ENDPOINT=你的 DashVector endpoint
```

> 注意：DashVector 的 API-KEY 与百炼的 `DASHSCOPE_API_KEY` 是两套独立的，需在 DashVector 控制台单独获取。标签检索依赖 DashVector 的字段过滤能力，本地模式下自动跳过。

## 配置说明（.env）

```ini
# 模型
DASHSCOPE_API_KEY=        # 百炼 API Key（调用 LLM / Embedding / Rerank）
LLM_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v3
RERANK_MODEL=qwen3-rerank

# 分块
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# 检索：先粗排召回 TOP_K，再精排到 RERANK_TOP_N
TOP_K=20
RERANK_TOP_N=3

# 生成
TEMPERATURE=0.1           # 问答助手追求准确，建议 0~0.3
MAX_TOKENS=1024

# 向量库
VECTOR_STORE=local        # local / dashvector
DASHSVECTOR_API_KEY=
DASHSVECTOR_ENDPOINT=
DASHSVECTOR_COLLECTION=   # 留空则自动用 "<domain>_knowledge"
DASHSVECTOR_DIM=1024      # text-embedding-v3 固定 1024，建库后不可改
```

知识点标签集合定义在 `src/config.py` 的 `TAG_MAP` 中，当前映射了课程 5 篇文档对应的标签；新文档若不在映射里，会归入「其他」。

## 关键技术说明（踩坑记录）

这些是调试过程中固化为代码实现的事实，后续改动时注意：

1. **`sk-ws-` 开头的百炼 Key 只支持 OpenAI 兼容模式**：原生 `TextEmbedding` 接口会返回 `InvalidApiKey`。因此 LLM 与向量化都走 `/compatible-mode/v1`（`src/embedder.py`、`src/generation/llm.py`）；而 rerank 的原生接口接受该 Key，所以重排走 DashScope 原生。
2. **兼容模式向量化单次最多 10 条文本**：见 `src/embedder.py` 的 `EMBED_BATCH_SIZE=10`。
3. **DashVector 三个注意点**：维度 1024（建库后不可改）；`delete` 是异步的，`--rebuild` 采用「删除后重试 create」；公网访问需在控制台配置 IP 白名单，否则插入时报 `-2939 Cluster whiteList validate fail`。
4. **`.env` 为权威来源**：`src/config.py` 使用 `load_dotenv(override=True)`，避免 shell 里残留的旧环境变量遮蔽 `.env`。
5. **本地模式 embedding 的已知 bug 修复**：`create_embed_model()` 中对 `llama-index-embeddings-dashscope` 0.2.x 的 pydantic 私有属性 bug 做了手动补丁。

## 命令速查

```bash
# 构建索引
python scripts/build_index.py                     # 构建默认域 rag
python scripts/build_index.py --domain agent      # 构建其他知识域
python scripts/build_index.py --rebuild           # 强制重建

# 提问
python scripts/ask.py "RAG 建立索引包括哪几步？"
python scripts/ask.py "..." --show-sources        # 展示来源清单（文件名/章节/分数/原文）
python scripts/ask.py "..." --no-rerank           # 关闭重排序
python scripts/ask.py "..." --no-tag              # 关闭标签检索
python scripts/ask.py "..." --top-k 5             # 调整最终用于生成的块数
python scripts/ask.py                            # 进入交互模式

# 评测
python scripts/eval.py
python scripts/eval.py --qa-file data/eval/qa_pairs.json
```

## 目录结构

```
├── data/
│   ├── knowledge/          # 知识库（按主题分域）
│   │   └── rag/            # 当前域：RAG 课程笔记
│   └── eval/               # 评测问答对
├── src/
│   ├── config.py           # 集中配置（.env 覆盖）+ 知识点标签
│   ├── embedder.py         # 向量化（兼容模式，批量上限 10）
│   ├── document/           # 文档加载 + 结构感知分块
│   ├── indexing/           # 索引构建 / 加载
│   ├── retrieval/          # 融合检索 + 标签抽取 + 重排序
│   ├── generation/         # LLM 客户端 + 提示词 + 生成器
│   ├── evaluation/         # Ragas 评测
│   └── pipeline.py         # 端到端编排
├── scripts/                # CLI 入口
│   ├── build_index.py
│   ├── ask.py
│   └── eval.py
├── requirements.txt
├── .env.example
└── knowledge_base/         # 本地索引持久化（gitignore）
```

## 扩展新知识域

以 `agent` 为例，两步即可：

1. 把文档放入 `data/knowledge/agent/`，并在 `src/config.py` 的 `TAG_MAP` 中补充该域文档对应的标签。
2. `python scripts/build_index.py --domain agent`，再 `python scripts/ask.py "..." --domain agent`。

主流程各阶段无需改动。
