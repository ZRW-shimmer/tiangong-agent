"""
检查药品说明书入库质量。

用法：
  # 只看入库统计和样本文本
  python scripts/check_drug_ingestion.py

  # 额外做一次检索测试
  python scripts/check_drug_ingestion.py --question "富马酸比索洛尔片的用法用量是什么？"

  # 指定返回条数
  python scripts/check_drug_ingestion.py --question "..." --top-k 5
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DOC_TYPE = "drug_instruction"
COLLECTION_NAME = "knowledge_docs"


def _preview(text: str, limit: int = 300) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def check_rows(milvus_client) -> list[dict]:
    rows = milvus_client.query(
        collection_name=COLLECTION_NAME,
        filter=f'doc_type == "{DOC_TYPE}"',
        output_fields=["id", "doc_name", "doc_id", "chunk_index", "text"],
        limit=10000,
        consistency_level="Strong",
    )
    return rows or []


def print_stats(rows: list[dict]) -> None:
    print("=" * 70)
    print("药品说明书入库统计")
    print("=" * 70)

    if not rows:
        print(f"[WARN] 没有找到 doc_type={DOC_TYPE} 的数据")
        print("       请先运行: python scripts/import_drug_pdfs.py --limit 2 --reset")
        return

    print(f"总分块数: {len(rows)}")

    by_doc = Counter(row["doc_name"] for row in rows)
    print(f"文档数: {len(by_doc)}")
    for name, count in sorted(by_doc.items()):
        print(f"  - {name}: {count} 块")

    lengths = [len(row.get("text") or "") for row in rows]
    print("\n分块长度:")
    print(f"  min={min(lengths)}, median={statistics.median(lengths):.0f}, max={max(lengths)}")

    garbled = sum(
        1
        for row in rows
        if "�" in (row.get("text") or "")
        or "□□" in (row.get("text") or "")
    )
    print(f"疑似乱码块: {garbled}/{len(rows)}")

    print("\n样本文本:")
    for row in rows[:2]:
        print("-" * 70)
        print(f"[{row['doc_name']}] chunk_index={row['chunk_index']} len={len(row.get('text') or '')}")
        print(_preview(row.get("text") or ""))


def retrieval_test(milvus_client, question: str, top_k: int) -> None:
    from src.infra.embedding import get_embedding_model

    print("\n" + "=" * 70)
    print(f"检索测试: {question}")
    print("=" * 70)

    embedding_model = get_embedding_model()
    query_vec = embedding_model.embed_query(question)

    results = milvus_client.search(
        collection_name=COLLECTION_NAME,
        data=[query_vec],
        limit=top_k,
        output_fields=["doc_name", "chunk_index", "text"],
        filter=f'doc_type == "{DOC_TYPE}"',
        search_params={"metric_type": "COSINE", "params": {"nprobe": 16}},
    )

    hits = results[0] if results else []
    if not hits:
        print("[WARN] 没有检索到结果")
        return

    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        print("-" * 70)
        print(
            f"Top{i} score={hit.get('distance', 0):.4f} "
            f"doc={entity.get('doc_name')} chunk={entity.get('chunk_index')}"
        )
        print(_preview(entity.get("text") or ""))


def main():
    parser = argparse.ArgumentParser(description="检查药品说明书入库质量")
    parser.add_argument("--question", default="", help="可选：做一次检索测试")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回条数")
    parser.add_argument("--skip-stats", action="store_true", help="只做检索测试，不打印入库统计")
    args = parser.parse_args()

    from pymilvus import MilvusClient

    from src.core.config import get_settings

    settings = get_settings()
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )

    rows = check_rows(milvus_client)
    if not args.skip_stats:
        print_stats(rows)

    if args.question:
        retrieval_test(milvus_client, args.question, args.top_k)


if __name__ == "__main__":
    main()
