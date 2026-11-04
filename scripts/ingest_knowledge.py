"""把本地文件或 MinIO 对象解析后写入 knowledge_docs。

用法：
    python -m scripts.ingest_knowledge --file data/raw/guide.pdf --doc-type guideline
    python -m scripts.ingest_knowledge --object guidelines/guide.pdf --doc-type guideline
    python -m scripts.ingest_knowledge --object guidelines/scan.pdf --ocr
"""
from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from pymilvus import MilvusClient

from src.agents.knowledge.doc_ingestion import ingest_file
from src.core.config import get_settings
from src.infra.embedding import get_embedding_model
from src.infra.minio_client import download_file


def _build_embedding_model():
    return get_embedding_model()


async def main() -> None:
    parser = argparse.ArgumentParser(description="MinerU 解析并入库知识文档")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="本地文件路径")
    source.add_argument("--object", help="MinIO 对象名，如 guidelines/xxx.pdf")
    parser.add_argument(
        "--doc-type",
        default="guideline",
        help="文档类型：guideline / drug_instruction / sop / literature",
    )
    parser.add_argument("--category", default="通用", help="文档分类，如 心内科/药剂科")
    parser.add_argument("--name", help="文档显示名，默认取文件名")
    parser.add_argument("--ocr", action="store_true", help="扫描件强制 OCR")
    args = parser.parse_args()

    settings = get_settings()
    tmp_path: Path | None = None

    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise SystemExit(f"文件不存在: {path}")
        doc_name = args.name or path.name
    else:
        data = download_file(args.object)
        doc_name = args.name or Path(args.object).name
        suffix = Path(doc_name).suffix or ".pdf"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(data)
        tmp.close()
        path = Path(tmp.name)
        tmp_path = path

    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )

    try:
        count = await ingest_file(
            file_path=str(path),
            doc_name=doc_name,
            doc_type=args.doc_type,
            category=args.category,
            embedding_model=_build_embedding_model(),
            milvus_client=milvus_client,
            is_ocr=True if args.ocr else None,
        )
        print(f"入库完成: doc_name={doc_name}, chunks={count}")
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())