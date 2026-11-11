"""可追踪的 RAG 包装。

用 @instrument 标记检索/生成/入口三个 span，TruLens 基于 OpenTelemetry
自动记录输入、输出与耗时，并把检索返回值作为 context 供 RAG Triad 指标使用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pymilvus import MilvusClient
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from src.agents.knowledge.doc_rag import search_docs_raw
from src.agents.knowledge.prompts import DOC_QA_PROMPT

SpanType = SpanAttributes.SpanType


@dataclass
class EvalRAGConfig:
    """一次 RAG 实验的配置（只切换检索策略，不改动已入库数据）。"""

    name: str
    top_k: int = 20
    rerank_top_k: int = 5
    use_rewrite: bool = False
    use_hyde: bool = False
    use_rerank: bool = True
    doc_type: str | None = None
    role: str = "patient"
    collection_name: str = "knowledge_docs"

    def describe(self) -> str:
        return (
            f"collection={self.collection_name}, top_k={self.top_k}, "
            f"rerank_top_k={self.rerank_top_k}, "
            f"rewrite={self.use_rewrite}, hyde={self.use_hyde}, "
            f"rerank={self.use_rerank}, doc_type={self.doc_type or '-'}"
        )


class TrackedRAG:
    """把现有 doc_rag 检索链路包装成 TruLens 可追踪应用。"""

    def __init__(
        self,
        llm: BaseChatModel,
        embedding_model: Embeddings,
        milvus_client: MilvusClient,
        config: EvalRAGConfig,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.milvus_client = milvus_client
        self.config = config

        # 评估脚本用于计算 Recall@K / MRR / 延迟
        self.last_hits: list[dict] = []
        self.last_retrieval_query: str = ""
        self.last_retrieval_ms: float = 0.0
        self.last_total_ms: float = 0.0

    @instrument(
        span_type=SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    async def retrieve(self, query: str) -> list[str]:
        """检索步骤：返回带来源前缀的上下文列表。"""
        retrieval_query = query
        if self.config.use_rewrite:
            from src.agents.knowledge.query_rewriter import rewrite_query

            rewritten = await rewrite_query(query, self.llm, self.config.role)
            queries = rewritten.get("queries") or []
            if queries:
                retrieval_query = queries[0]

        self.last_retrieval_query = retrieval_query
        started = time.perf_counter()
        hits = await search_docs_raw(
            question=retrieval_query,
            milvus_client=self.milvus_client,
            embedding_model=self.embedding_model,
            top_k=self.config.top_k,
            rerank_top_k=self.config.rerank_top_k,
            doc_type=self.config.doc_type,
            llm=self.llm,
            use_hyde=self.config.use_hyde,
            use_rerank=self.config.use_rerank,
            collection_name=self.config.collection_name,
        )
        self.last_retrieval_ms = (time.perf_counter() - started) * 1000
        self.last_hits = hits

        return [
            f"[{hit.get('doc_name', '未知来源')}, 第{hit.get('page_number', '?')}页]\n{hit['text']}"
            for hit in hits
        ]

    @instrument(span_type=SpanType.GENERATION)
    async def generate(self, query: str, contexts: list[str]) -> str:
        """生成步骤：只基于检索到的上下文回答。"""
        if not contexts:
            return "当前知识库中未找到与您问题相关的文档内容。"

        context = "\n\n---\n\n".join(
            f"片段{i} {ctx}" for i, ctx in enumerate(contexts, 1)
        )
        prompt = DOC_QA_PROMPT.format(
            question=query, context=context, role=self.config.role
        )
        response = await self.llm.ainvoke(prompt)
        return response.content

    # 入口方法交给 TruApp(main_method=...) 自动标记为 record root，
    # 2.x 中手动设置 SpanType.RECORD_ROOT 会被降级为 UNKNOWN。
    async def query(self, query: str) -> str:
        """完整 RAG 流程入口。"""
        started = time.perf_counter()
        contexts = await self.retrieve(query)
        answer = await self.generate(query, contexts)
        self.last_total_ms = (time.perf_counter() - started) * 1000
        return answer
