import asyncio

from langchain_core.messages import SystemMessage
from loguru import logger
from pymilvus import MilvusClient
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.agents.knowledge.prompts import DOC_QA_PROMPT


COLLECTION_NAME="knowledge_docs"

async def search_docs_raw(
        question:str,
        milvus_client: MilvusClient,
        embedding_model:Embeddings,
        top_k:int=20,
        rerank_top_k:int=5,
        doc_type:str | None=None,
        llm:BaseChatModel |None=None,
        use_hyde:bool=False,
        use_rerank:bool=True,
        collection_name:str=COLLECTION_NAME,
) ->list[dict]:
    """
    文档向量检索，返回原始检索结果列表。
    支持 HyDE 增强检索和 Reranker 精排，可通过 collection_name 指定评估集合。
    """
    if use_hyde and llm is not None:
        from src.agents.knowledge.hyde import generate_hyde_embedding
        query_vec=await generate_hyde_embedding(question,llm,embedding_model)
    else:
        query_vec=await embedding_model.aembed_query(question)

    search_params = {"metric_type":"COSINE","params":{"nprobe":16}}
    search_kwargs = {
        "collection_name": collection_name,
        "data": [query_vec],
        "limit": top_k,
        "output_fields": ["id", "doc_name", "doc_type", "page_number", "chunk_index", "text"],
        "search_params": search_params,
    }
    if doc_type:
        search_kwargs["filter"] = f'doc_type == "{doc_type}"'

    try:
        # pymilvus 的 MilvusClient.search 是同步方法，不能直接 await
        results = await asyncio.to_thread(milvus_client.search, **search_kwargs)
    except Exception as e:
        logger.warning(f"文档检索失败: {e}")
        return []
    if not results or not results[0]:
        return []
    hits = [
        {**hit["entity"], "score": hit.get("distance", 0.0)}
        for hit in results[0]
    ]

    if not use_rerank:
        return hits[:rerank_top_k]

    from src.agents.knowledge.reranker import rerank_docs
    reranked = await rerank_docs(question, hits, top_k=rerank_top_k)
    return reranked

def format_doc_context(hits:list[dict]) -> str:
    """
    将检索结果格式化为 LLM 可读的上下文字符串。
       统一格式: 片段1 [文档名, 第X页]: 内容... 。
    """
    if not hits:
        return ""
    parts = []
    for i, hit in enumerate(hits, 1):
        source = f"[{hit['doc_name']}, 第{hit.get('page_number', '?')}页]"
        parts.append(f"片段{i} {source}:\n{hit['text']}")
    return "\n\n---\n\n".join(parts)

async def generate_doc_answer(
        question:str,
        hits:list[dict],
        llm:BaseChatModel,
        role:str="patient",
) -> str:
    """基于检索结果生成文档问答回答。检索与生成分离，便于评估复用。"""
    if not hits:
        return "当前知识库中未找到与您问题相关的文档内容。"
    context = format_doc_context(hits)
    prompt = DOC_QA_PROMPT.format(question=question, context=context, role=role)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content


async def search_docs(
        question:str,
        embedding_model:Embeddings,
        milvus_client: MilvusClient,
        llm:BaseChatModel,
        top_k:int=20,
        rerank_top_k:int=5,
        doc_type:str | None=None,
        role:str="patient",
        use_hyde:bool=True,
        use_rerank:bool=True,
        collection_name:str=COLLECTION_NAME,
) -> str:
    """文档 RAG 检索 + Reranker 精排 + HyDE 增强 + 生成回答。"""
    hits=await search_docs_raw(
        question=question,
        milvus_client=milvus_client,
        embedding_model=embedding_model,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        doc_type=doc_type,
        llm=llm,
        use_hyde=use_hyde,
        use_rerank=use_rerank,
        collection_name=collection_name,
    )
    return await generate_doc_answer(question, hits, llm, role=role)
