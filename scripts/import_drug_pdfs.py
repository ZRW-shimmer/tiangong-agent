"""导入中文药品说明书 PDF 到 Milvus。

数据源：
    SHPDRG/drug-instructions-sample-20
    - 20 份中文药品说明书 PDF
    - Apache-2.0
    - 仅用于文档解析、检索和知识库测试，不构成用药建议

用法：
    # 先测试 2 份
    python scripts/import_drug_pdfs.py --limit 2 --reset

    # 导入全部 20 份
    python scripts/import_drug_pdfs.py --reset
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

from loguru import logger
from pypdf import PdfReader
from pymilvus import MilvusClient
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.knowledge.doc_ingestion import ensure_knowledge_collection
from src.agents.knowledge.doc_rag import COLLECTION_NAME
from src.core.config import get_settings
from src.infra.embedding import get_embedding_model

settings = get_settings()

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "drug_instructions_sample20"
DEFAULT_PDF_DIR = BASE_DIR / "data" / "pdfs"
DEFAULT_MANIFEST = BASE_DIR / "metadata" / "drug_instructions_manifest.jsonl"

DOC_TYPE = "drug_instruction"
EMBED_BATCH_SIZE = 10
WRITE_BATCH_SIZE = 200
MAX_CHUNK_CHARS = 800
CHUNK_OVERLAP = 100


def _normalize_pdf_text(text: str) -> str:
    """清理 pypdf 抽取的换行和重复标题。

    PDF 说明书常见问题：
    - 章节标题重复出现两次
    - 数字、单位被拆成单独一行
    - 每行都有硬换行
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    # 去掉连续重复行
    deduped: list[str] = []
    for line in lines:
        if not deduped or line != deduped[-1]:
            deduped.append(line)

    # 按【章节】合并为自然段
    sections: list[str] = []
    current = ""
    for line in deduped:
        if line.startswith("【"):
            if current:
                sections.append(current)
            current = line
        else:
            current = f"{current} {line}".strip() if current else line
    if current:
        sections.append(current)

    return "\n\n".join(sections).strip()


def _load_manifest(path: Path) -> dict[str, dict]:
    manifest: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            filename = row.get("pdf_file_name") or Path(row.get("pdf_path", "")).name
            if filename:
                manifest[filename] = row
    return manifest


def _split_text(text: str) -> list[str]:
    """按段落/标点切分，单块不超过 MAX_CHUNK_CHARS。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)
    boundaries = ("\n", "。", "！", "？", "；", ".", "!", "?", ";")

    while start < length:
        end = min(start + MAX_CHUNK_CHARS, length)
        if end < length:
            min_cut = start + int(MAX_CHUNK_CHARS * 0.6)
            cut = max(text.rfind(ch, min_cut, end) for ch in boundaries)
            if cut >= min_cut:
                end = cut + 1

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break
        next_start = end - CHUNK_OVERLAP
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


async def _embed_texts(
    embedding_model,
    texts: list[str],
    max_retries: int = 6,
) -> list[list[float]]:
    """DashScope 网络异常时指数退避重试。"""
    for attempt in range(max_retries):
        try:
            return await embedding_model.aembed_documents(texts)
        except (RequestsConnectionError, RequestsTimeout) as exc:
            if attempt >= max_retries - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(
                f"[WARN] Embedding 网络异常，{wait}s 后重试 "
                f"({attempt + 1}/{max_retries}): {exc}"
            )
            await asyncio.sleep(wait)
    return []


async def _flush(
    milvus_client: MilvusClient,
    embedding_model,
    records: list[dict],
    doc_ids: list[str],
) -> int:
    """删除旧切片、向量化并写入 Milvus。"""
    if not records:
        return 0

    if doc_ids:
        expr = f"doc_id in {json.dumps(doc_ids, ensure_ascii=False)}"
        try:
            milvus_client.delete(collection_name=COLLECTION_NAME, filter=expr)
        except Exception as exc:
            logger.warning(f"删除旧药品切片失败: {exc}")

    for start in range(0, len(records), EMBED_BATCH_SIZE):
        batch = records[start: start + EMBED_BATCH_SIZE]
        vectors = await _embed_texts(
            embedding_model,
            [record["text"] for record in batch],
        )
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedding 返回数量不一致: {len(vectors)} != {len(batch)}"
            )
        for record, vector in zip(batch, vectors):
            record["embedding"] = vector

    for start in range(0, len(records), 500):
        milvus_client.upsert(
            collection_name=COLLECTION_NAME,
            data=records[start: start + 500],
        )
    return len(records)


def _records_for_pdf(
    pdf_path: Path,
    metadata: dict,
) -> tuple[str, str, list[dict]]:
    drug_name = metadata.get("drug_name") or pdf_path.stem
    approval_number = metadata.get("approval_number") or ""
    manufacturer = "、".join(metadata.get("manufacturer") or [])
    dosage_form = metadata.get("dosage_form") or ""

    doc_id = "zhpdf_" + hashlib.sha1(
        f"{approval_number}|{pdf_path.name}".encode("utf-8")
    ).hexdigest()[:16]

    header = [
        "数据来源：SHPDRG/drug-instructions-sample-20（Apache-2.0，测试数据）",
        f"药品名称：{drug_name}",
    ]
    if approval_number:
        header.append(f"批准文号：{approval_number}")
    if dosage_form:
        header.append(f"剂型：{dosage_form}")
    if manufacturer:
        header.append(f"生产企业：{manufacturer}")

    reader = PdfReader(str(pdf_path))
    records: list[dict] = []
    global_index = 0

    for page_number, page in enumerate(reader.pages, 1):
        page_text = _normalize_pdf_text(page.extract_text() or "")
        if len(page_text) < 20:
            continue

        body = "\n".join(header) + f"\n\n【第{page_number}页】\n" + page_text
        for chunk in _split_text(body):
            records.append(
                {
                    "id": f"{doc_id}_{global_index}",
                    "doc_id": doc_id,
                    "doc_name": drug_name[:200],
                    "doc_type": DOC_TYPE,
                    "category": "药品说明书样本20份 (Apache-2.0)",
                    "page_number": page_number,
                    "chunk_index": global_index,
                    "text": chunk[:65000],
                }
            )
            global_index += 1

    return doc_id, drug_name, records


async def main() -> None:
    parser = argparse.ArgumentParser(description="导入中文药品说明书 PDF 到 Milvus")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少份 PDF")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="导入前删除 doc_type=drug_instruction 的旧文档",
    )
    args = parser.parse_args()

    if not args.pdf_dir.exists():
        raise SystemExit(f"PDF 目录不存在: {args.pdf_dir}")
    if not args.manifest.exists():
        raise SystemExit(f"manifest 不存在: {args.manifest}")

    pdf_files = sorted(args.pdf_dir.glob("*.pdf"))
    if args.limit and args.limit > 0:
        pdf_files = pdf_files[: args.limit]
    if not pdf_files:
        raise SystemExit("没有找到 PDF 文件")

    manifest = _load_manifest(args.manifest)

    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    ensure_knowledge_collection(milvus_client)

    if args.reset:
        print(f"[INFO] 清理旧药品说明书: doc_type={DOC_TYPE}")
        milvus_client.delete(
            collection_name=COLLECTION_NAME,
            filter=f'doc_type == "{DOC_TYPE}"',
        )

    embedding_model = get_embedding_model()

    total_docs = 0
    total_chunks = 0
    records_buffer: list[dict] = []
    doc_ids_buffer: list[str] = []

    for index, pdf_path in enumerate(pdf_files, 1):
        metadata = manifest.get(pdf_path.name, {})
        doc_id, drug_name, records = _records_for_pdf(pdf_path, metadata)
        if not records:
            print(f"[WARN] 没有抽取到文本，跳过: {pdf_path.name}")
            continue

        total_docs += 1
        records_buffer.extend(records)
        doc_ids_buffer.append(doc_id)
        print(
            f"[{index}/{len(pdf_files)}] {drug_name}: "
            f"{len(records)} 个切片"
        )

        if len(records_buffer) >= WRITE_BATCH_SIZE:
            total_chunks += await _flush(
                milvus_client,
                embedding_model,
                records_buffer,
                doc_ids_buffer,
            )
            records_buffer = []
            doc_ids_buffer = []

    if records_buffer:
        total_chunks += await _flush(
            milvus_client,
            embedding_model,
            records_buffer,
            doc_ids_buffer,
        )

    if total_chunks:
        milvus_client.flush(COLLECTION_NAME)

    print(
        f"[DONE] 导入完成：说明书 {total_docs} 份，切片 {total_chunks} 条"
    )
    print(
        "[INFO] 数据来源：SHPDRG/drug-instructions-sample-20 "
        "(Apache-2.0，仅用于测试)"
    )


if __name__ == "__main__":
    asyncio.run(main())
