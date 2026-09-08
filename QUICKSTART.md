# 🚀 快速启动指南

## 5分钟快速上手

### 第一步：配置API Key

1. 复制环境变量模板：
   ```bash
   cp .env.example .env
   ```

2. 编辑 `.env` 文件，填入你的DashScope API Key：
   ```
   DASHSCOPE_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
   ```

   > 获取API Key: https://bailian.console.aliyun.com/?apiKey=1

### 第二步：安装依赖

```bash
pip install -r requirements.txt
```

### 第三步：准备文档

将你的公司文档（PDF/Markdown/DOCX/TXT）放入 `data/docs/` 目录。

示例：
```bash
# 复制你的文档到data/docs/
cp /path/to/your/company_rules.pdf data/docs/
cp /path/to/your/employee_handbook.md data/docs/
```

### 第四步：构建索引

```bash
python scripts/build_index.py
```

首次运行会自动：
- 加载 `data/docs/` 下的所有文档
- 使用默认的句子分块策略
- 构建向量索引并保存到 `knowledge_base/default/`

### 第五步：开始问答

**方式一：交互模式**
```bash
python scripts/run_qa.py
```

进入交互界面，可以连续提问：
```
👤 请输入你的问题: 我们公司用什么项目管理工具？
🤖 回答: ...
👤 请输入你的问题: 张伟是哪个部门的？
🤖 回答: ...
👤 请输入你的问题: quit
👋 再见！
```

**方式二：单次提问**
```bash
python scripts/run_qa.py --question "张伟是哪个部门的"
```

### 第六步（可选）：运行评测

```bash
python scripts/run_eval.py
```

会自动使用 `data/qa_pairs.json` 中的测试问答对进行评测。

---

## 进阶使用

### 切换分块策略

在构建索引时，可以选择不同的分块策略：

```bash
python scripts/build_index.py
# 程序会提示你选择：
# 1. Sentence (默认)
# 2. Sentence Window
# 3. Semantic
# 4. Markdown
# 5. Token
```

### 重建索引

如果你想重新构建索引（例如添加了新文档）：

```bash
python scripts/run_qa.py --rebuild
```

### 自定义文档目录

```bash
python scripts/run_qa.py --docs-dir /path/to/your/docs
```

### 代码示例：在Python中使用

```python
from src.config import check_api_key
from src.document.loader import DocumentLoader
from src.storage.index_manager import IndexManager
from src.retrieval.retriever import Retriever
from src.generation.llm_client import LLMClient

# 初始化
check_api_key()

# 加载文档
loader = DocumentLoader("data/docs")
documents = loader.load()

# 构建索引
index_manager = IndexManager()
index = index_manager.build_index(documents)

# 创建检索器
retriever = Retriever(index, similarity_top_k=5)

# 创建LLM客户端
llm_client = LLMClient()

# 提问
question = "张伟是哪个部门的？"
response = retriever.query(question)
print(response)
```

---

## 常见问题

### Q1: 如何获取API Key？

访问 https://bailian.console.aliyun.com/?apiKey=1 ，登录后创建一个API Key。

### Q2: 索引构建失败？

- 检查 `.env` 文件中的API Key是否正确配置
- 确认 `data/docs/` 目录下有文档（PDF/MD/DOCX/TXT格式）
- 检查网络连接是否正常

### Q3: 回答不准确？

- 尝试增加 `SIMILARITY_TOP_K` 值（在 `.env` 中配置）
- 尝试不同的分块策略（推荐先试Sentence Window）
- 升级Embedding模型为 `text-embedding-v3`

### Q4: 如何添加新文档？

1. 将文档放入 `data/docs/` 目录
2. 运行 `python scripts/build_index.py --rebuild` 重建索引

### Q5: 如何自定义Prompt？

编辑 `src/generation/prompt_templates.py` 文件，修改对应的模板字符串。

---

## 下一步

- 阅读 [README.md](README.md) 了解项目完整架构
- 阅读 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) 了解每个模块的详细说明
- 查看 `notebooks/` 目录中的Jupyter教程（如果有）
- 查看阿里云大模型ACP课程获取更深入的知识

---

## 技术支持

如果遇到问题，可以：
1. 查看项目的Issues是否有类似问题的解决方案
2. 提交新的Issue描述你的问题
3. 参考[阿里云百炼文档](https://help.aliyun.com/zh/model-studio/)
