from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.embeddings import Embeddings
from loguru import logger
from pymilvus import DataType, MilvusClient

from src.agents.knowledge.doc_rag import COLLECTION_NAME
from src.agents.knowledge.mineru_client import MinerUResult, parse_document

EMBEDDING_DIM = 1024
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBED_BATCH_SIZE = 10


def ensure_knowledge_collection(milvus_client: MilvusClient) -> None:
    """确保 knowledge_docs collection 存在并已 load。"""
    if milvus_client.has_collection(COLLECTION_NAME):
        milvus_client.load_collection(COLLECTION_NAME)
        return

    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("id", DataType.VARCHAR, max_length=256, is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
    schema.add_field("doc_name", DataType.VARCHAR, max_length=256)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=50)
    schema.add_field("category", DataType.VARCHAR, max_length=100)
    schema.add_field("page_number", DataType.INT64)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        params={"nlist": 128},
    )

    milvus_client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )
    milvus_client.load_collection(COLLECTION_NAME)
    logger.info(f"Collection '{COLLECTION_NAME}' 创建成功")


def _block_text(block: dict[str, Any]) -> str:
    """把 MinerU content_list 的单个 block 转成可入库文本。"""
    kind = block.get("type")

    if kind == "table":
        parts = [
            *(block.get("table_caption") or []),
            block.get("table_body") or "",
            *(block.get("table_footnote") or []),
        ]
        return "\n".join(str(p) for p in parts if p).strip()

    if kind == "image":
        return " ".join(str(c) for c in (block.get("img_caption") or [])).strip()

    return str(block.get("text") or "").strip()


def _content_list_to_pages(
    content_list: list[dict[str, Any]],
) -> list[tuple[int, str]]:
    """按 page_idx 聚合 content_list，返回 (1 基页码, 页面文本)。"""
    pages: dict[int, list[tuple[str, str]]] = {}
    heading = ""

    for block in content_list:
        if not isinstance(block, dict):
            continue

        text = _block_text(block)
        if not text:
            continue

        if block.get("type") == "title":
            heading = text

        page_number = int(block.get("page_idx", 0)) + 1
        pages.setdefault(page_number, []).append((heading, text))

    result: list[tuple[int, str]] = []
    for page_number in sorted(pages):
        parts = []
        for heading_text, block_text in pages[page_number]:
            parts.append(
                f"【{heading_text}】\n{block_text}"
                if heading_text and block_text != heading_text
                else block_text
            )
        result.append((page_number, "\n\n".join(parts)))
    return result


def _chunks_from_content_list(
    content_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按页切块，保留 page_number，供引用使用。"""
    chunks: list[dict[str, Any]] = []
    chunk_index = 0

    for page_number, page_text in _content_list_to_pages(content_list):
        for piece in _split_markdown(page_text):
            piece = piece.strip()
            if len(piece) < 20:
                continue
            chunks.append(
                {
                    "text": piece,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

    return chunks


def _split_markdown(
    md_text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """对 Markdown 按段落分块，超长段落再做滑动窗口切分。"""
    paragraphs = md_text.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current = current + "\n\n" + para if current else para
            continue

        if current:
            chunks.append(current)

        if len(para) > chunk_size:
            step = max(1, chunk_size - overlap)
            for i in range(0, len(para), step):
                chunks.append(para[i : i + chunk_size])
            current = ""
        else:
            current = para

    if current:
        chunks.append(current)

    return chunks


async def _parse_with_mineru(
    file_path: str,
    file_name: str,
    is_ocr: bool | None = None,
) -> MinerUResult | None:
    """尝试用 MinerU 解析文档，失败则返回 None。"""
    try:
        result = await parse_document(file_path, file_name, is_ocr=is_ocr)
        if result.markdown.strip() or result.content_list:
            logger.info(
                f"MinerU 解析成功: {file_name} "
                f"({len(result.markdown)} chars, {result.page_count} pages)"
            )
            return result
    except Exception as e:
        logger.warning(f"MinerU 解析失败，回退到 LlamaIndex: {e}")
    return None


async def _parse_with_llamaindex(file_path: str) -> list:
    """用 LlamaIndex 解析文档（兜底方案）。"""
    from llama_index.core import SimpleDirectoryReader

    reader = SimpleDirectoryReader(input_files=[file_path])
    return reader.load_data()


async def ingest_file(
    file_path: str,
    doc_name: str,
    doc_type: str,
    category: str,
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    is_ocr: bool | None = None,
) -> int:
    """
    导入单个文档到 Milvus。
    优先使用 MinerU 解析（高精度、带页码），失败回退到 LlamaIndex。
    """
    ensure_knowledge_collection(milvus_client)
    doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:16]

    # 优先 MinerU，失败回退 LlamaIndex
    parsed = await _parse_with_mineru(file_path, doc_name, is_ocr=is_ocr)
    pages: list[int] = []
    texts: list[str] = []

    if parsed:
        chunk_items = _chunks_from_content_list(parsed.content_list)
        if not chunk_items and parsed.markdown.strip():
            chunk_items = [
                {"text": text, "page_number": 0, "chunk_index": i}
                for i, text in enumerate(_split_markdown(parsed.markdown))
            ]

        if chunk_items:
            texts = [item["text"] for item in chunk_items]
            pages = [item["page_number"] for item in chunk_items]
        else:
            logger.warning(
                f"文档 '{doc_name}' 的 MinerU 结果没有可入库文本，回退到 LlamaIndex"
            )
            parsed = None

    if not parsed:
        from llama_index.core.node_parser import SentenceSplitter

        documents = await _parse_with_llamaindex(file_path)
        splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        nodes = splitter.get_nodes_from_documents(documents)
        if not nodes:
            logger.warning(f"文档 '{doc_name}' 解析后无内容")
            return 0

        texts = [node.get_content() for node in nodes]
        for node in nodes:
            page = node.metadata.get("page_label", 0)
            try:
                pages.append(int(page))
            except (ValueError, TypeError):
                pages.append(0)

    if not texts:
        return 0

    all_data: list[dict[str, Any]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch_texts = texts[i : i + EMBED_BATCH_SIZE]
        embeddings = await embedding_model.aembed_documents(batch_texts)

        for j, (text_content, emb) in enumerate(zip(batch_texts, embeddings)):
            chunk_idx = i + j
            all_data.append(
                {
                    "id": f"{doc_id}_{chunk_idx}",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "doc_type": doc_type,
                    "category": category,
                    "page_number": pages[chunk_idx] if chunk_idx < len(pages) else 0,
                    "chunk_index": chunk_idx,
                    "text": text_content[:65000],
                    "embedding": emb,
                }
            )

    # 解析和向量化都成功后再替换旧数据，避免解析失败时丢数据
    milvus_client.delete(
        collection_name=COLLECTION_NAME,
        filter=f'doc_id == "{doc_id}"',
    )
    milvus_client.insert(collection_name=COLLECTION_NAME, data=all_data)
    milvus_client.flush(COLLECTION_NAME)
    logger.info(f"文档 '{doc_name}' 导入完成，共 {len(all_data)} 个分块")
    return len(all_data)


async def ingest_drug_instructions(
    embedding_model: Embeddings,
    milvus_client: MilvusClient,
    db_session,
) -> int:
    """从 PostgreSQL drug_details 表导入药品说明书到 Milvus。"""
    from sqlalchemy import select

    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    from src.modules.medical.model import Drug, DrugDetail

    ensure_knowledge_collection(milvus_client)

    result = await db_session.execute(
        select(Drug.name, DrugDetail.full_instruction)
        .join(DrugDetail, Drug.id == DrugDetail.drug_id)
        .where(DrugDetail.full_instruction.isnot(None))
    )
    rows = result.all()

    if not rows:
        logger.warning("drug_details 表中无说明书数据")
        return 0

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    total = 0

    for drug_name, instruction in rows:
        if not instruction or not instruction.strip():
            continue

        doc_id = hashlib.md5(f"drug_{drug_name}".encode()).hexdigest()[:16]

        doc = Document(text=instruction, metadata={"drug_name": drug_name})
        nodes = splitter.get_nodes_from_documents([doc])
        texts = [n.get_content() for n in nodes]

        if not texts:
            continue

        embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            embeddings.extend(await embedding_model.aembed_documents(batch))

        data = []
        for idx, (text_content, emb) in enumerate(zip(texts, embeddings)):
            data.append(
                {
                    "id": f"{doc_id}_{idx}",
                    "doc_id": doc_id,
                    "doc_name": f"{drug_name}说明书",
                    "doc_type": "drug_instruction",
                    "category": "药剂科",
                    "page_number": 0,
                    "chunk_index": idx,
                    "text": text_content[:65000],
                    "embedding": emb,
                }
            )

        milvus_client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'doc_id == "{doc_id}"',
        )
        milvus_client.insert(collection_name=COLLECTION_NAME, data=data)
        total += len(data)

    milvus_client.flush(COLLECTION_NAME)
    logger.info(f"药品说明书导入完成，共 {total} 个分块")
    return total