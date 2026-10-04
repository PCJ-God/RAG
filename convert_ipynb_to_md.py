import os
import nbformat
from nbconvert import MarkdownExporter

src_dir = r"E:\github项目经历\大模型ACP认证\aliyun_acp_learning\大模型ACP认证教程\C2_构造问答系统"
out_dir = r"E:\github项目经历\大模型应用开发\RAG\rag_data"

files = [
    "2_1_用大模型构建新人答疑机器人.ipynb",
    "2_2_扩展答疑机器人的知识范围.ipynb",
    "2_3_优化提示词改善答疑机器人回答质量.ipynb",
    "2_4_自动化评测答疑机器人的表现.ipynb",
    "2_5_优化RAG应用提升问答准确度.ipynb",
]

os.makedirs(out_dir, exist_ok=True)
exporter = MarkdownExporter()

for f in files:
    src = os.path.join(src_dir, f)
    if not os.path.exists(src):
        print("MISSING:", src)
        continue
    nb = nbformat.read(src, as_version=4)
    body, resources = exporter.from_notebook_node(nb)
    out_name = os.path.splitext(f)[0] + ".md"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(body)
    # 导出 markdown 单元格里引用的图片附件
    for img_name, img_data in resources.get("outputs", {}).items():
        img_path = os.path.join(out_dir, img_name)
        with open(img_path, "wb") as fh:
            fh.write(img_data)
    print("OK:", out_path)

print("DONE")
