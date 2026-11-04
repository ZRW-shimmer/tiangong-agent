"""项目统一的 Embedding 模型工厂。

默认使用本地 BGE 模型（bge-large-zh-v1.5，1024 维），
也可以通过 `EMBEDDING_BACKEND=dashscope` 切回阿里云 DashScope。
"""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

from langchain_core.embeddings import Embeddings
from loguru import logger

from src.core.config import get_settings

# bge 系列模型查询侧建议添加的指令前缀
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LocalBGEEmbeddings(Embeddings):
    """基于 sentence-transformers 的本地 BGE Embedding。"""

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        path = Path(model_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(
                f"本地 Embedding 模型不存在: {path}\n"
                "请先下载 BAAI/bge-large-zh-v1.5 到该目录。"
            )

        logger.info("加载本地 Embedding 模型: {} (device={})", path, device)
        self._model = SentenceTransformer(str(path), device=device)
        self._batch_size = batch_size
        # sentence-transformers 的 encode 不是线程安全的，查询并发时串行执行
        self._lock = threading.Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            vectors = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([BGE_QUERY_INSTRUCTION + text])[0]


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """返回全局单例 Embedding 模型。"""
    settings = get_settings()
    backend = (settings.EMBEDDING_BACKEND or "local").lower()

    if backend == "dashscope":
        from langchain_community.embeddings import DashScopeEmbeddings

        logger.info("使用 DashScope Embedding: {}", settings.EMBEDDING_MODEL)
        return DashScopeEmbeddings(
            model=settings.EMBEDDING_MODEL,
            dashscope_api_key=settings.DASHSCOPE_API_KEY,
        )

    if backend != "local":
        raise ValueError(
            f"不支持的 EMBEDDING_BACKEND: {backend}，可选 local / dashscope"
        )

    return LocalBGEEmbeddings(
        model_path=settings.LOCAL_EMBEDDING_MODEL,
        device=settings.LOCAL_EMBEDDING_DEVICE,
        batch_size=settings.LOCAL_EMBEDDING_BATCH_SIZE,
    )
