# RAG 知识答疑助手

一个基于 **RAG（检索增强生成）** 的学习助手：对某个 RAG 知识点不清楚时直接提问，它会从课程知识库中检索相关内容，给出**带来源引用**的回答，帮助你学习而非闲聊。

当前已支持**多轮对话**：会话记忆保持话题连贯、全局记忆沉淀用户长期偏好、模糊问题自动改写、复杂问题自动拆解。

## 项目定位

- 回答的是「RAG 知识点不清晰」这类学习问题，不做通用聊天。
- 知识库按主题分域管理，当前只有 `rag` 域，后续可自行添加 `agent` 等域，主流程无需改动。
- **质量优先于功能数量**：一条高质量主链路，各环节可独立开关、可独立替换。

## 核心能力总览

| 阶段 | 模块 | 已实现内容 |
| --- | --- | --- |
| 文档摄入 | `document/` | Markdown 加载；结构感知分块（按标题层级切块、标题路径写入元数据、超长小节按句子二次切分、清洗 Notebook 导出的 HTML 噪声）；按文件名打知识点标签 |
| 索引 | `indexing/` | `build / load / exists` 统一接口；本地 JSON 与 DashVector 两种后端；DashVector 经 LlamaIndex `DashVectorStore` 接入 `VectorStoreIndex`（见踩坑记录里的官方包补丁） |
| 检索 | `retrieval/` | 问题改写 + 复杂问题拆解 + 标签抽取 + 融合检索（语义 + 标签两路）+ 重排序，详见下文 |
| 生成 | `generation/` | LLM 走 DashScope OpenAI 兼容模式；只依据检索上下文回答、防幻觉；引用带文档名；用户画像注入 system prompt，详见下文 |
| 记忆 | `memory/` | 双层记忆（会话记忆 + 全局画像），存 DashVector 的 `chat_memory` collection，单用户多会话 |
| 评测 | `evaluation/` | Ragas 评测，指标：`answer_correctness` / `faithfulness` / `context_recall` / `context_precision` |

## 系统架构

```
document/ ──> indexing/ ──> retrieval/ ──> generation/
(加载+分块)   (构建/加载)   (改写/拆解/      (LLM+提示词)
                           融合检索+重排)
      └──────────── memory/（双层记忆，DashVector）────────────┘
      └────────────── pipeline.py 端到端编排 ──────────────────┘
```

一次提问的完整过程（`RAGPipeline.query`）：

1. **读记忆**（多轮会话）：检索用户画像（全局记忆）＋ 最近 3 轮纯文本 ＋ 语义召回更早历史。
2. **问题改写**：模糊 / 含指代的问题由 LLM 扩写为完整检索问题。
3. **问题拆解**：复杂问题由 LLM 拆成 ≤4 个子问题。
4. **标签抽取 + 融合召回 + 重排**：语义路 + 标签路两路召回，去重后精排。
5. **生成**：用户画像注入 system prompt，对话历史与参考资料进主提示词，生成带引用回答。
6. **写回记忆**：本轮问答写入会话记忆，并由 LLM 抽取长期事实写入全局画像。

---

# 检索（retrieval）

检索链路位于 `src/retrieval/`，按执行顺序由 5 个组件构成：

```
原始问题 ─> QueryRewriter(改写) ─> QueryDecomposer(拆解) ─> TagExtractor(标签)
                │                        │                      │
                │                        └─ 子问题逐个走右侧 ──┤
                └────────────────────────────────────────────┤
                        FusionRetriever(语义+标签两路召回) <──┘
                                │
                        Reranker(精排 top_n)
                                │
                            交给 generation/
```

## 1. 问题改写 `rewriter.py`

`QueryRewriter` 用 LLM 把两类"不好检索"的问题改写成完整、明确的检索问题：

- **模糊问题**：如用户只输入 `"RAG"`，改写为「RAG（检索增强生成）的技术原理、典型架构、主流应用场景及常见挑战是什么？」。
- **指代问题**：如多轮里的「那第二步是什么？」，结合【对话历史】改写为「RAG 建立索引的第二步是什么？」。

要点：

- 改写 prompt 约束**控制在 1~2 句话、不超过 50 字**，避免过度扩写稀释检索向量。
- 改写结果**只用于召回**，最终生成仍针对用户的原始问题，保证回答忠实于用户意图。

## 2. 复杂问题拆解 `decomposer.py`

`QueryDecomposer` 判断问题是否复杂：

- 简单问题 → 原样返回（单路检索）。
- 复杂问题 → LLM 拆成 **2~4 个子问题**（代码硬上限 4，宁少勿多）。

每个子问题独立走一遍「标签抽取 → 融合召回 → 重排 → 生成」，最后由 LLM 把各子答案**整合成一份连贯的最终回答**（`merge`）。

> 说明：这里没有直接用 LlamaIndex 的 `MultiStepQueryEngine`，因为它只能包一个 `query_engine`，若包 `index.as_query_engine()` 会绕过本项目自己的融合检索、重排与记忆注入。因此拆解用 LLM 产出子问题列表，子问题仍走自己的检索链路，保留全部能力。

## 3. 标签抽取 `tag_extractor.py`

`TagExtractor` 用 LLM 从问题中抽取**一个**知识点标签，候选集合来自 `src/config.py` 的 `TAG_LIST`（由 `TAG_MAP` 按文档文件名映射而来，两端一致）。

- 抽取结果不在候选集合内 → 返回 `None`，标签路自动跳过。
- 标签用于下面融合检索的**标签路**前置过滤。

## 4. 融合召回 `retriever.py`

`FusionRetriever` 做**两路召回、合并去重**：

| 路 | 实现 | 说明 |
| --- | --- | --- |
| 语义路 | `VectorStoreIndex.as_retriever(similarity_top_k=top_k)` | 本地 JSON 与 DashVector **统一走 LlamaIndex** 的向量索引，语义相似度召回 top-k |
| 标签路 | DashVector collection 直接 `query(filter="tag='...'", topk=top_k*2)` | **纯条件过滤、无向量**，只召回同一知识点标签下的块；仅 DashVector 模式启用，本地模式自动跳过 |

合并规则：两路结果按 `node_id` 去重，同一片段保留分数更高的一条，保持各自原始顺序。

## 5. 重排序 `reranker.py`

`Reranker` 把粗排结果精排到 `top_n`：

- 首选 **DashScope 专用 rerank 端点**（`qwen3-rerank`），相关度打分更准。
- 未安装对应包时**自动降级**为 LLM 打分精排（`LLMRerank`），保证可运行。

最终 top_n 个文档块交给生成模块。

---

# 生成（generation）

生成链路位于 `src/generation/`，由 3 个文件构成。

## 1. LLM 客户端 `llm.py`

`LLMClient` 提供两个入口：

- `invoke(prompt, system_prompt)`：直接走 OpenAI 兼容协议的 `chat.completions`，返回文本，供改写 / 拆解 / 标签抽取 / 全局事实抽取 / 答案生成等所有提示词类调用使用。
- `get_llm()`：返回 LlamaIndex 的 `OpenAILike` 实例，挂到 `Settings.llm`，供需要 LlamaIndex LLM 接口的组件使用。

关键点：

- 走 **DashScope OpenAI 兼容模式**（`/compatible-mode/v1`），因为 `sk-ws-` 开头的百炼 Key 只支持该协议（见踩坑记录）。
- 统一使用 `temperature`（默认 0.1，问答追求准确）与 `max_tokens`（默认 1024）配置。

## 2. 生成器 `generator.py`

`AnswerGenerator.generate(question, nodes, history, profile)` 完成"拼提示词 → 调 LLM → 组装结果"：

1. **上下文拼装**：把重排后的每个文档块带上标签，形如：

   ```
   [1]（2_2_扩展答疑机器人的知识范围.md · 1.1 建立索引）
   <块原文>
   ```

2. **提示词填充**：主提示词 `RAG_ANSWER_TEMPLATE` 含【对话历史】【参考资料】【问题】三段。
3. **画像注入**：若有用户画像，作为 **system prompt** 注入（`invoke(prompt, system_prompt=...)`）。
4. **返回** `Answer(answer, sources)`，其中 `sources` 保存每个来源的编号、原文、分数、文件名、章节路径，供 `--show-sources` 展示。

## 3. 提示词 `prompts.py`

`RAG_ANSWER_TEMPLATE` 的核心约束（防幻觉是重点）：

1. 只依据【参考资料】回答，不使用资料之外的知识臆测。
2. 回答简明、准确、结构化，先给直接结论再补充解释。
3. 资料不足以回答时，明确说明"参考资料中未找到相关内容"。
4. 答案末尾列出实际引用的资料编号与文档信息，例如：`参考来源：[1]（2_2_扩展答疑机器人的知识范围.md）、[3]（2_3_优化提示词改善答疑机器人回答质量.md）`。

---

# 多轮对话记忆（memory）

双层记忆位于 `src/memory/`，统一存 DashVector 的 `chat_memory` collection（1024 维）。

| 层 | 范围 | 用途 | 检索方式 |
| --- | --- | --- | --- |
| 会话记忆 | 单个 `session_id` | 保持当前话题连贯 | `session_id` 前置过滤 + 语义 top-k |
| 全局画像 | 跨会话，按 `user_id` | 记住身份 / 偏好 / 长期目标 | `user_id` + `memory_type="global"`，每轮注入 system prompt |

每条记录字段：`user_id / session_id / memory_type / turn_id / timestamp / role`，正文存在节点里。

- `memory_store.py`：`MemoryStore`，提供 `add_turn`（一轮合并一条：`用户：q\n助手：a`）、`add_global_fact`、`retrieve_session`、`retrieve_global`。
- `extractor.py`：`GlobalMemoryExtractor`，每轮对话结束后由 LLM 判断是否有长期事实并逐条抽取写入全局画像。

**滑动窗口策略**：最近 3 轮以纯文本直接带进 prompt（`memory_window=3`），更早历史按语义检索召回（`memory_top_k=4`），按 `turn_id` 去重。摘要生成、记忆衰减、定期清理留待后续版本。

**会话管理**：`scripts/ask.py` 交互模式自动生成新会话（UUID），`--session-id <id>` 恢复旧会话；单次命令行提问默认不开会话，避免写脏记忆。

---

# 快速开始

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

> 注意：DashVector 的 API-KEY 与百炼的 `DASHSCOPE_API_KEY` 是两套独立的，需在 DashVector 控制台单独获取。标签检索与多轮记忆依赖 DashVector 的字段过滤能力，本地模式下自动跳过。

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

# 多轮记忆（仅 dashvector 模式）
MEMORY_COLLECTION=chat_memory
MEMORY_TOP_K=4            # 语义检索的历史轮次数量
MEMORY_WINDOW=3           # 最近 N 轮以纯文本直接带入
USER_ID=default_user      # 单用户多会话；未来多用户可换
```

知识点标签集合定义在 `src/config.py` 的 `TAG_MAP` 中，当前映射了课程 5 篇文档对应的标签；新文档若不在映射里，会归入「其他」。

## 命令速查

```bash
# 构建索引
python scripts/build_index.py                     # 构建默认域 rag
python scripts/build_index.py --domain agent      # 构建其他知识域
python scripts/build_index.py --rebuild           # 强制重建

# 提问
python scripts/ask.py "RAG 建立索引包括哪几步？"
python scripts/ask.py "..." --show-sources        # 展示来源清单（文件名/章节/分数/原文）
python scripts/ask.py "..." --top-k 5             # 调整最终用于生成的块数
python scripts/ask.py "..." --no-rerank           # 关闭重排序
python scripts/ask.py "..." --no-tag              # 关闭标签检索
python scripts/ask.py "..." --no-rewrite          # 关闭问题改写
python scripts/ask.py "..." --no-decompose        # 关闭问题拆解
python scripts/ask.py "..." --no-memory           # 关闭多轮记忆
python scripts/ask.py                            # 进入交互模式（自动开新会话）
python scripts/ask.py --session-id <id>          # 恢复指定会话
python scripts/ask.py --domain agent "..."       # 指定知识域

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
│   ├── embedder.py         # 向量化（LlamaIndex BaseEmbedding，兼容模式，批量 10）
│   ├── document/
│   │   ├── loader.py       # Markdown 加载
│   │   └── splitter.py     # 结构感知分块
│   ├── indexing/
│   │   ├── indexer.py      # 索引构建 / 加载（VectorStoreIndex）
│   │   └── dashvector_store.py  # DashVectorStore 补丁 + 建/取 collection 工具
│   ├── retrieval/
│   │   ├── rewriter.py     # 问题改写
│   │   ├── decomposer.py   # 复杂问题拆解 + 子答案整合
│   │   ├── tag_extractor.py# 知识点标签抽取
│   │   ├── retriever.py    # 融合检索（语义 + 标签）
│   │   └── reranker.py     # 重排序
│   ├── generation/
│   │   ├── llm.py          # LLM 客户端（OpenAI 兼容 + LlamaIndex）
│   │   ├── generator.py    # 答案生成器
│   │   └── prompts.py      # 提示词模板
│   ├── memory/
│   │   ├── memory_store.py # 双层记忆存储
│   │   └── extractor.py    # 全局画像（长期事实）抽取
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

## 关键技术说明（踩坑记录）

这些是调试过程中固化为代码实现的事实，后续改动时注意：

1. **`sk-ws-` 开头的百炼 Key 只支持 OpenAI 兼容模式**：原生 `TextEmbedding` 接口会返回 `InvalidApiKey`。因此 LLM 与向量化都走 `/compatible-mode/v1`（`src/embedder.py`、`src/generation/llm.py`）；而 rerank 的原生接口接受该 Key，所以重排走 DashScope 原生。
2. **兼容模式向量化单次最多 10 条文本**：见 `src/embedder.py` 的 `EMBED_BATCH_SIZE=10`。
3. **DashVector 的 `client.get()` 对不存在的 collection 也返回句柄**（内部带 `code=-2021`），不能靠 `get() is None` 判断是否存在；正确做法是查 `client.list()`。且**创建是异步的**，建完立刻写入会报 `Not found collection`，需轮询 `stats()` 直到可用（见 `src/indexing/dashvector_store.py`）。
4. **`llama-index-vector-stores-dashvector` 0.2.0 有两个 bug**：未实现基类抽象属性 `client`；`support_sparse_vector=False` 时漏初始化 `_support_sparse_vector/_encoder`。用 `PatchedDashVectorStore` 子类修复。
5. **DashVector 其他注意点**：维度 1024（建库后不可改）；`delete` 是异步的，`--rebuild` 需「删除后重试 create」；公网访问需在控制台配置 IP 白名单，否则插入报 `-2939 Cluster whiteList validate fail`。
6. **`.env` 为权威来源**：`src/config.py` 使用 `load_dotenv(override=True)`，避免 shell 里残留的旧环境变量遮蔽 `.env`。
7. **本地模式 embedding 的已知 bug 修复**：`create_embed_model()` 中对 `llama-index-embeddings-dashscope` 0.2.x 的 pydantic 私有属性 bug 做了手动补丁。

## 扩展新知识域

以 `agent` 为例，两步即可：

1. 把文档放入 `data/knowledge/agent/`，并在 `src/config.py` 的 `TAG_MAP` 中补充该域文档对应的标签。
2. `python scripts/build_index.py --domain agent`，再 `python scripts/ask.py "..." --domain agent`。

主流程各阶段无需改动。
