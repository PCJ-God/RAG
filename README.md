# RAG 知识答疑助手

一个基于 **RAG（检索增强生成）** 的学习助手：对某个 RAG 知识点不清楚时直接提问，它会从课程知识库中检索相关内容，给出**带来源引用**的回答。

当前已支持：多轮对话（双层记忆 + 问题改写 + 复杂问题拆解）、AI 安全护栏（阿里云内容安全）。

## 项目全景

这是一条完整的 RAG 学习链路，基于 **LlamaIndex 基座**搭建，覆盖 7 个环节：

```
文档摄入 → 索引 → 检索 → 生成 → 记忆 → 安全护栏 → 评测
 document   indexing  retrieval  generation  memory  safety  evaluation
```

每个环节对应一个 `src/` 子包，由 `pipeline.py` 统一编排。下面先给一张「知识地图」，标出每个环节踩到了哪些知识点，再逐模块展开解读。

## 知识地图总览

| 环节 | 模块 | 一句话干什么 | 涉及的知识点 |
| --- | --- | --- | --- |
| 文档摄入 | `document/` | 加载 Markdown、结构感知分块、打标签 | 文本分块 Chunking、元数据 Metadata、标签体系 |
| 索引 | `indexing/` | 向量化并写入本地 / DashVector | Embedding、向量检索 ANN、向量库抽象 |
| 检索 | `retrieval/` | 改写 → 拆解 → 标签 → 融合召回 → 重排 | 查询改写/扩展、指代消解、问题分解、多跳推理、混合检索、两阶段检索 Rerank |
| 生成 | `generation/` | 拼上下文 → 调 LLM → 带引用回答 | RAG 生成、上下文学习 ICL、提示词工程、防幻觉 Grounding |
| 记忆 | `memory/` | 双层记忆读写 | 长短期记忆、滑动窗口、记忆语义召回、事实抽取 |
| 安全护栏 | `safety/` | 输入/输出双向内容审核 | 内容安全、输入/输出护栏、fail-open/fail-closed |
| 评测 | `evaluation/` | Ragas 跑 QA 对 | RAG 评测指标：忠实度/正确性/召回率/精确率 |
| 编排 | `pipeline.py` | 串起以上所有环节 | RAG 完整链路、可开关的 Pipeline 设计 |

---

## 模块解读

### 1. 文档摄入 `document/`

**过程（略）**：加载 Markdown → 结构感知分块 → 清洗噪声 → 按文件名打标签。

**涉及的知识点**：

- **文本分块（Chunking）**：检索和生成都以"块"为单位，块切得好不好直接决定召回质量。本项目不是固定字数硬切，而是**结构感知分块**——按标题层级切，保证每块语义完整；超长小节再按句子二次切；`chunk_overlap` 保留相邻块的重叠，防止语义在边界处被切断。
- **元数据（Metadata）**：每块把「章节路径 header_path」「文件名」写入元数据，供生成阶段的引用标注与溯源使用——这是 RAG 里"答案可解释、可追溯"的数据基础。
- **标签体系（Tag）**：按文件名映射知识点标签（`TAG_MAP`），让"文档组织方式"与"提问分类方式"两端对齐，标签才能作为检索的过滤条件。

### 2. 索引 `indexing/`

**过程（略）**：调用 Embedding 把每块转成向量，写入本地 JSON 或 DashVector；`build / load / exists` 统一接口。

**涉及的知识点**：

- **向量化（Embedding）**：用 `text-embedding-v3` 把文本映射成 1024 维稠密向量，使"语义相近 → 向量相近"，这是语义检索的数学前提。
- **近似最近邻（ANN）检索**：向量库里不可能逐条精确比对，`VectorStoreIndex` / DashVector 底层用 HNSW 等 ANN 索引，用精度换速度，快速召回 top-k。
- **向量库抽象**：本地 JSON（零依赖、开发友好）与云向量库（可扩展、支持字段过滤）共用同一套 LlamaIndex 接口，切换后端不改主流程。

### 3. 检索 `retrieval/`

**过程（略）**：问题改写 → 复杂问题拆解 → 标签抽取 → 语义+标签两路召回 → 重排。

**涉及的知识点**：

- **查询改写 / 查询扩展（Query Rewriting / Expansion）**——`rewriter.py`：用户口语化的模糊问题（如 "RAG"）或含指代的问题（如 "那第二步呢"）直接拿去检索效果很差，先用 LLM 扩充成完整、明确的检索问题。
- **指代消解（Coreference Resolution）**——`rewriter.py`：把 "它""第二步" 这类指代结合对话历史还原成具体对象，这是多轮对话检索的关键难点。
- **问题分解（Question Decomposition）与多跳推理（Multi-hop）**——`decomposer.py`：复杂问题（"建立索引和检索生成分别有哪些步骤？"）拆成多个子问题，每个子问题独立检索，最后 LLM 整合。本质是把"多跳问题"降成"单跳问题"逐个击破。
- **意图识别 / 主题分类**——`tag_extractor.py`：用 LLM 把问题映射到有限标签集合（闭集分类），结果作为召回前的过滤条件，属于"先分类、后检索"的思路。
- **混合检索（Hybrid Retrieval）**——`retriever.py`：**语义路**（稠密向量相似度召回）+ **标签路**（按 `tag` 字段条件过滤）两路召回再合并去重。向量检索擅长"语义相近"，字段过滤擅长"精确限定"，两者互补。
- **两阶段检索 / 重排序（Retrieve-then-Rerank）**——`reranker.py`：先用轻量模型粗排召回较多候选，再用更强的精排模型（`qwen3-rerank`，交叉编码器式打分）筛到 top-n。粗排管"召回率"，精排管"精确率"。

### 4. 生成 `generation/`

**过程（略）**：把 top-n 文档块拼成【参考资料】→ 填入提示词 → 调 LLM → 返回带引用的回答。

**涉及的知识点**：

- **RAG 的生成侧**——`generator.py`：把检索到的上下文作为输入喂给 LLM，属于**上下文学习（In-Context Learning）**——模型不靠训练记忆，而是"边看边答"。
- **提示词工程（Prompt Engineering）**——`prompts.py`：角色设定 + 约束（只依据参考资料）+ 引用格式要求，是控制 RAG 输出质量的主要手段。
- **防幻觉 / 接地（Grounding）**——`prompts.py`：强制"只用资料回答、资料不足就明说"，并用引用编号让答案可溯源，是对抗 LLM 幻觉的核心机制。
- **采样参数**——`llm.py`：问答场景用低 `temperature`（0.1）降低随机性、`max_tokens` 限制长度；走 OpenAI 兼容协议（`/compatible-mode/v1`）。
- **画像注入**——`generator.py`：用户画像作为 system prompt 注入，与正文隔离，属于"角色/约束层与内容层分离"的提示词结构。

### 5. 记忆 `memory/`

**过程（略）**：提问时读记忆（画像 + 历史），生成后写回（会话记忆 + 全局事实）。

**涉及的知识点**：

- **长短期记忆（Long/Short-term Memory）**：会话记忆（当前 session 话题，短期）+ 全局画像（跨会话偏好，长期），是 Agent 记忆的经典分层。
- **滑动窗口（Sliding Window）**：最近 3 轮历史纯文本直给，更早历史按语义召回——用"窗口 + 检索"缓解长上下文的成本与噪声。
- **记忆检索**：把历史对话当向量库，按语义相似度召回相关轮次，而非简单全塞，属于"记忆即检索"的思想。
- **事实抽取（Fact Extraction）**：每轮后由 LLM 判断并抽取长期事实写入画像，类似"事实验证 + 画像沉淀"。
- **多租户字段过滤**：`user_id / session_id` 前置过滤隔离，是"记忆私有化"的实现基础。

### 6. 安全护栏 `safety/`

**过程（略）**：输入检测 → 正常链路 → 输出检测，命中风险替换为固定话术。

**涉及的知识点**：

- **内容安全（Content Moderation）**：接入阿里云绿网文本审核 PLUS，`llm_query_moderation`（检测提问）+ `llm_response_moderation`（检测回答）。
- **输入护栏 + 输出护栏（Input/Output Guardrail）**：双向防护，分别卡在链路最前和最后，属于 LLM 应用安全的标准位置。
- **风险分级处置**：按 `RiskLevel`（high/medium/low/none）分级，命中 `high,medium` 即拦截。
- **fail-open vs fail-closed**：护栏服务异常时"放行并告警"（fail-open），在"可用性"与"安全性"之间做取舍——本项目选择优先保证主链路可用。

### 7. 评测 `evaluation/`

**过程（略）**：用 Ragas 对 QA 对评测。

**涉及的知识点**：

- **RAG 评测四指标**：`faithfulness`（回答是否忠实于上下文）、`answer_correctness`（答案是否正确）、`context_recall`（该召回的是否召回全）、`context_precision`（召回的是否相关）。前两者评"生成"，后两者评"检索"。

### 8. 编排 `pipeline.py`

**过程（略）**：`query()` 串起 8 步——输入护栏 → 读记忆 → 改写 → 拆解 → 标签+召回 → 重排 → 生成 → 输出护栏 → 写回记忆。

**涉及的知识点**：

- **RAG 完整链路**：把散件串成一条可运行的问答流水线。
- **可开关的 Pipeline**：每个环节都能单独关闭（`--no-*`），方便做消融实验（Ablation）对比各环节贡献。

---

## 快速开始

1. 配置环境：`cp .env.example .env`，填入 `DASHSCOPE_API_KEY`（百炼平台获取）。
2. 安装依赖：`pip install -r requirements.txt`。
3. 构建索引：`python scripts/build_index.py`。
4. 提问：`python scripts/ask.py "RAG 建立索引包括哪几步？"`。
5. 评测：`python scripts/eval.py`。

## 向量库

默认本地 JSON（`VECTOR_STORE=local`），零依赖。要用 DashVector，在 `.env` 里配 `VECTOR_STORE=dashvector` + `DASHSVECTOR_API_KEY/ENDPOINT`。标签检索与多轮记忆依赖 DashVector 字段过滤，本地模式自动跳过。

## 配置说明（.env）

```ini
# 模型
DASHSCOPE_API_KEY=        # 百炼 API Key（LLM / Embedding / Rerank）
LLM_MODEL=qwen-plus
EMBEDDING_MODEL=text-embedding-v3
RERANK_MODEL=qwen3-rerank

# 分块
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# 检索：粗排 TOP_K，精排到 RERANK_TOP_N
TOP_K=20
RERANK_TOP_N=3

# 生成
TEMPERATURE=0.1
MAX_TOKENS=1024

# 向量库
VECTOR_STORE=local        # local / dashvector
DASHSVECTOR_API_KEY=
DASHSVECTOR_ENDPOINT=
DASHSVECTOR_COLLECTION=
DASHSVECTOR_DIM=1024

# 多轮记忆（仅 dashvector 模式）
MEMORY_COLLECTION=chat_memory
MEMORY_TOP_K=4
MEMORY_WINDOW=3
USER_ID=default_user

# AI 安全护栏（未配置 AccessKey 自动禁用）
ALIBABA_CLOUD_ACCESS_KEY_ID=
ALIBABA_CLOUD_ACCESS_KEY_SECRET=
CONTENT_SECURITY_REGION=cn-shanghai
CONTENT_SECURITY_ENDPOINT=green-cip.cn-shanghai.aliyuncs.com
CONTENT_SECURITY_QUERY_SERVICE=llm_query_moderation
CONTENT_SECURITY_RESPONSE_SERVICE=llm_response_moderation
CONTENT_SECURITY_BLOCK_LEVELS=high,medium
```

## 命令速查

```bash
# 构建索引
python scripts/build_index.py                 # 默认域 rag
python scripts/build_index.py --domain agent
python scripts/build_index.py --rebuild

# 提问
python scripts/ask.py "..."                   # 单次提问
python scripts/ask.py "..." --show-sources    # 展示来源（文件/章节/分数/原文）
python scripts/ask.py "..." --top-k 5         # 最终用于生成的块数
python scripts/ask.py "..." --no-rerank       # 关闭重排
python scripts/ask.py "..." --no-tag          # 关闭标签检索
python scripts/ask.py "..." --no-rewrite      # 关闭改写
python scripts/ask.py "..." --no-decompose    # 关闭拆解
python scripts/ask.py "..." --no-memory       # 关闭记忆
python scripts/ask.py "..." --no-guard        # 关闭安全护栏
python scripts/ask.py                         # 交互模式（自动开新会话）
python scripts/ask.py --session-id <id>       # 恢复会话
python scripts/ask.py --domain agent "..."    # 指定知识域

# 评测
python scripts/eval.py
```

## 目录结构

```
├── data/knowledge/rag/    # 知识库（按主题分域）
├── src/
│   ├── config.py          # 集中配置 + 知识点标签
│   ├── embedder.py        # Embedding（LlamaIndex BaseEmbedding）
│   ├── document/          # 加载 + 结构感知分块
│   ├── indexing/          # 索引构建 / 加载
│   ├── retrieval/         # 改写 / 拆解 / 标签 / 融合检索 / 重排
│   ├── generation/        # LLM 客户端 + 提示词 + 生成器
│   ├── memory/            # 双层记忆存储 + 画像抽取
│   ├── safety/            # AI 安全护栏
│   ├── evaluation/        # Ragas 评测
│   └── pipeline.py        # 端到端编排
├── scripts/               # CLI 入口（build_index / ask / eval）
├── requirements.txt
├── .env.example
└── knowledge_base/        # 本地索引持久化（gitignore）
```

## 关键技术说明（踩坑记录）

1. **`sk-ws-` 开头的百炼 Key 只支持 OpenAI 兼容模式**：原生接口报 `InvalidApiKey`，所以 LLM 与向量化都走 `/compatible-mode/v1`；rerank 原生接口接受该 Key，走 DashScope 原生。
2. **兼容模式向量化单次最多 10 条**：`src/embedder.py` 的 `EMBED_BATCH_SIZE=10`。
3. **DashVector `client.get()` 对不存在的 collection 也返回句柄**（内部 `code=-2021`），要用 `client.list()` 判断；且创建是异步的，建完立刻写入会报错，需轮询 `stats()` 直到可用。
4. **`llama-index-vector-stores-dashvector` 0.2.0 有两个 bug**：未实现抽象属性 `client`；漏初始化 `_support_sparse_vector/_encoder`，用 `PatchedDashVectorStore` 子类修复。
5. **DashVector 其他**：维度 1024 建库后不可改；`delete` 异步，`--rebuild` 需"删后重试 create"；公网访问需配 IP 白名单。
6. **`.env` 权威来源**：`load_dotenv(override=True)`，避免 shell 旧变量遮蔽。
7. **本地模式 embedding 的 pydantic 私有属性 bug**：`create_embed_model()` 里手动补丁。
8. **通义千问上下文缓存**：隐式缓存默认自动生效（前缀 ≥1024 token 才激活，命中按 20% 计费），无需代码；显式缓存需 `cache_control` 标记（命中 10%、建缓存 125%）。本项目稳定前缀不足 1024，暂不启用。

## 扩展新知识域

1. 文档放入 `data/knowledge/agent/`，在 `src/config.py` 的 `TAG_MAP` 补标签。
2. `python scripts/build_index.py --domain agent`，再 `python scripts/ask.py "..." --domain agent`。

主流程各环节无需改动。
