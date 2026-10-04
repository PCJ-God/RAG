"""DashVector 的 LlamaIndex 集成与共享工具。

供索引模块（知识库）与记忆模块（chat_memory）共同复用：
- PatchedDashVectorStore：官方 llama-index-vector-stores-dashvector 0.2.0 的补丁子类
- get_client / get_collection：DashVector 客户端与 collection 管理
"""
import time
from typing import Any, Dict, Optional

from llama_index.vector_stores.dashvector import DashVectorStore

from src.config import Config, settings


class PatchedDashVectorStore(DashVectorStore):
    """补丁：DashVectorStore 0.2.0 与 llama-index-core 0.11 的两处不兼容。

    1. 未实现基类抽象属性 client；
    2. support_sparse_vector=False 时漏初始化 _support_sparse_vector/_encoder
       私有属性，导致 add() 访问时报 AttributeError。
    """

    @property
    def client(self):
        return self._collection

    def __init__(
        self, collection=None, support_sparse_vector: bool = False, encoder=None
    ):
        super().__init__(
            collection=collection,
            support_sparse_vector=support_sparse_vector,
            encoder=encoder,
        )
        self._support_sparse_vector = support_sparse_vector
        self._encoder = encoder


def get_client(config: Optional[Config] = None):
    """创建 DashVector 客户端。"""
    config = config or settings
    if not config.dashvector_api_key or not config.dashvector_endpoint:
        raise ValueError(
            "使用 DashVector 需在 .env 配置 DASHSVECTOR_API_KEY 和 DASHSVECTOR_ENDPOINT"
        )
    import dashvector

    return dashvector.Client(
        api_key=config.dashvector_api_key,
        endpoint=config.dashvector_endpoint,
    )


def _wait_until_ready(collection, name: str, timeout: int = 60) -> None:
    """DashVector collection 创建是异步的，轮询直到真正可用。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            collection.stats()
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"collection 创建后长时间不可用: {name}")


def _list_names(client) -> set:
    """返回当前已存在的 collection 名称集合。"""
    try:
        resp = client.list()
    except Exception as e:
        print(f"[dashvector] 获取 collection 列表失败: {e}")
        return set()
    return set(getattr(resp, "output", None) or [])


def get_collection(
    config: Optional[Config],
    name: str,
    dimension: int,
    fields_schema: Dict[str, Any],
    create: bool = False,
    drop_first: bool = False,
):
    """获取（或创建）一个 DashVector collection。

    注意：client.get() 对不存在的 collection 也会返回句柄（内部携带错误码），
    因此用 client.list() 判断是否真正存在。
    """
    client = get_client(config)

    if drop_first:
        try:
            client.delete(name)
            print(f"[dashvector] 已删除旧 collection: {name}")
        except Exception as e:
            print(f"[dashvector] 删除旧 collection 失败（可能本就不存在）: {e}")
        # 删除是异步的：等 name 从列表中消失
        for _ in range(15):
            if name not in _list_names(client):
                break
            time.sleep(2)

    if name in _list_names(client):
        collection = client.get(name)
        if collection is None:
            raise RuntimeError(f"collection 存在但无法获取: {name}")
        return collection

    if not create:
        return None

    ret = client.create(
        name=name,
        dimension=dimension,
        fields_schema=fields_schema,
    )
    if not ret:
        raise RuntimeError(f"创建 collection 失败: {name}")

    # 创建是异步的：等 name 出现在列表中
    for _ in range(15):
        if name in _list_names(client):
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"创建 collection 超时未生效: {name}")

    collection = client.get(name)
    if collection is None:
        raise RuntimeError(f"创建后仍无法获取 collection: {name}")
    _wait_until_ready(collection, name)
    print(f"[dashvector] 已创建 collection: {name}（dimension={dimension}）")
    return collection
