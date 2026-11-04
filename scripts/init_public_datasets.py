"""
公共医学数据集初始化脚本。
从 HuggingFace 下载公共医学数据集，清洗后导入 Milvus 知识库。

用法：
    # 导入全部数据集
    python scripts/init_public_datasets.py

    # 只导入指定数据集
    python scripts/init_public_datasets.py --dataset cmirb
    python scripts/init_public_datasets.py --dataset dialogue
    python scripts/init_public_datasets.py --dataset medqa

依赖：
    pip install datasets
"""

import asyncio
import argparse
import hashlib
import json
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pymilvus import MilvusClient
from loguru import logger
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from src.core.config import get_settings
from src.infra.embedding import get_embedding_model
from src.infra.milvus_client import get_milvus_client_alias
from src.agents.knowledge.doc_ingestion import ensure_knowledge_collection
from src.agents.knowledge.doc_rag import COLLECTION_NAME

settings = get_settings()
# text-embedding-v3 单次请求最多 10 条文本
BATCH_SIZE = 10

# 优先使用本地 CMIRB 文件，避免离线/无 VPN 时访问 HuggingFace
CMIRB_LOCAL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "CMIRB", "corpus.jsonl"
)


def _get_deps(reset: bool = False):
    get_milvus_client_alias()
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    embedding_model = get_embedding_model()
    ensure_knowledge_collection(milvus_client)
    if reset:
        # 只清理文献数据，保留 drug_instruction 药品说明书
        print('[INFO] 清理旧文献数据: doc_type=literature')
        milvus_client.delete(
            collection_name=COLLECTION_NAME,
            filter='doc_type == "literature"',
        )
    return milvus_client, embedding_model


def _chunk_already_imported(
    milvus_client: MilvusClient,
    doc_id: str,
    expected: int,
) -> bool:
    """检查这一批数据是否已经完整导入，支持中断后续跑。"""
    try:
        rows = milvus_client.query(
            collection_name=COLLECTION_NAME,
            filter=f'doc_id == "{doc_id}"',
            output_fields=["id"],
            limit=expected + 1,
            consistency_level="Strong",
        )
        return len(rows) >= expected
    except Exception:
        return False


async def _embed_batch(
    embedding_model,
    batch: list[str],
    max_retries: int = 6,
) -> list[list[float]]:
    """带重试的 Embedding 调用。

    DashScope 的 langchain 封装只重试 HTTPError，不会重试网络层的
    SSLError / ConnectionError。批量导入时偶发 SSL EOF，这里做指数退避。
    """
    for attempt in range(max_retries):
        try:
            return await embedding_model.aembed_documents(batch)
        except (RequestsConnectionError, RequestsTimeout) as exc:
            if attempt >= max_retries - 1:
                raise
            wait = min(2 ** attempt, 30)
            print(
                f"[WARN] Embedding 网络异常，{wait}s 后重试 "
                f"({attempt + 1}/{max_retries}): {exc}"
            )
            await asyncio.sleep(wait)


async def _insert_texts(
    milvus_client: MilvusClient,
    embedding_model,
    texts: list[str],
    doc_id: str,
    doc_name: str,
    doc_type: str,
    category: str,
) -> int:
    """批量向量化并写入 Milvus。"""
    milvus_client.delete(
        collection_name=COLLECTION_NAME,
        filter=f'doc_id == "{doc_id}"',
    )

    all_data = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        embeddings = await _embed_batch(embedding_model, batch)
        for j, (text_content, emb) in enumerate(zip(batch, embeddings)):
            idx = i + j
            all_data.append({
                "id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_type": doc_type,
                "category": category,
                "page_number": 0,
                "chunk_index": idx,
                "text": text_content[:65000],
                "embedding": emb,
            })

    if all_data:
        for i in range(0, len(all_data), 1000):
            milvus_client.insert(
                collection_name=COLLECTION_NAME,
                data=all_data[i:i + 1000],
            )
    return len(all_data)


# ═══════════════════════════════════════════════════════════════════════
# 数据集 1：CMIRB/MedicalRetrieval — 10万条医学检索语料
# ═══════════════════════════════════════════════════════════════════════

def _iter_cmirb_rows():
    """优先读本地 JSONL；本地不存在时才访问 HuggingFace。"""
    if os.path.exists(CMIRB_LOCAL_PATH):
        print(f"[INFO] 使用本地 CMIRB 文件: {CMIRB_LOCAL_PATH}")
        with open(CMIRB_LOCAL_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    from datasets import load_dataset

    print("[INFO] 本地文件不存在，尝试从 HuggingFace 下载...")
    ds = load_dataset("CMIRB/MedicalRetrieval", "corpus", split="corpus")
    yield from ds


async def download_cmirb(milvus_client, embedding_model, limit: int = 0):
    """
    导入 CMIRB/MedicalRetrieval 数据集。
    包含 ~101K 条中文医学文本，适合作为文档 RAG 的基础语料。
    """
    texts = []
    for row in _iter_cmirb_rows():
        text = row.get("text", "") or row.get("content", "")
        if text and len(text.strip()) > 20:
            texts.append(text.strip()[:2000])

    if limit and limit < len(texts):
        random.seed(42)
        texts = random.sample(texts, limit)
        print(f"[INFO] 按 --limit 随机抽取 {len(texts)} 条文本")

    print(f"[INFO] CMIRB 共 {len(texts)} 条有效文本，开始向量化...")

    # 每 500 条写一次，方便观察进度，也避免中断后丢失整批结果
    chunk_size = 500
    total = 0
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        doc_id = hashlib.md5(f"cmirb_{i}".encode()).hexdigest()[:16]

        if _chunk_already_imported(milvus_client, doc_id, len(chunk)):
            total += len(chunk)
            print(f"  [跳过] 已完成 {min(i + chunk_size, len(texts))}/{len(texts)}")
            continue

        count = await _insert_texts(
            milvus_client, embedding_model, chunk,
            doc_id=doc_id,
            doc_name=f"CMIRB医学检索语料_{i // chunk_size + 1}",
            doc_type="literature",
            category="医学文献",
        )
        total += count
        print(f"  [进度] {min(i + chunk_size, len(texts))}/{len(texts)}")

    print(f"[OK] CMIRB 导入完成，共 {total} 条")
    return total


# ═══════════════════════════════════════════════════════════════════════
# 数据集 2：Chinese-medical-dialogue-data — 6科室医患对话
# ═══════════════════════════════════════════════════════════════════════

async def download_med_dialogue(milvus_client, embedding_model, limit: int = 0):
    """
    下载中文医患对话数据集并导入 Milvus。
    覆盖：男科、内科、妇产科、肿瘤科、儿科、外科。
    每条记录格式化为"问：xxx 答：xxx"，作为知识文档。
    """
    from datasets import load_dataset

    print("[INFO] 下载 Chinese-medical-dialogue-data 数据集...")
    ds = load_dataset("BillGPT/Chinese-medical-dialogue-data")

    texts = []
    for split_name in ds:
        for row in ds[split_name]:
            q = row.get("ask", "") or row.get("question", "") or ""
            a = row.get("answer", "") or ""
            if q and a and len(q) > 5 and len(a) > 10:
                text = f"问：{q.strip()}\n答：{a.strip()}"
                texts.append(text[:2000])

    if limit and limit < len(texts):
        random.seed(42)
        texts = random.sample(texts, limit)
        print(f"[INFO] 按 --limit 随机抽取 {len(texts)} 条记录")

    print(f"[INFO] 医患对话共 {len(texts)} 条有效记录，开始向量化...")

    chunk_size = 500
    total = 0
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        doc_id = hashlib.md5(f"med_dialogue_{i}".encode()).hexdigest()[:16]

        if _chunk_already_imported(milvus_client, doc_id, len(chunk)):
            total += len(chunk)
            print(f"  [跳过] 已完成 {min(i + chunk_size, len(texts))}/{len(texts)}")
            continue

        count = await _insert_texts(
            milvus_client, embedding_model, chunk,
            doc_id=doc_id,
            doc_name=f"中文医患对话_{i // chunk_size + 1}",
            doc_type="literature",
            category="医患对话",
        )
        total += count
        print(f"  [进度] {min(i + chunk_size, len(texts))}/{len(texts)}")

    print(f"[OK] 医患对话导入完成，共 {total} 条")
    return total


# ═══════════════════════════════════════════════════════════════════════
# 数据集 3：MedQA — 执业医师考试题（评测用，存到本地文件）
# ═══════════════════════════════════════════════════════════════════════

async def download_medqa():
    """
    下载 MedQA 中文执业医师考试题，保存到 data/eval/ 目录。
    用于评测知识 Agent 的准确率，不导入 Milvus。
    """
    from datasets import load_dataset
    import json

    print("[INFO] 下载 MedQA 数据集...")
    ds = load_dataset("bigbio/med_qa", "med_qa_zh_4options_bigbio_qa")

    eval_dir = os.path.join(os.path.dirname(__file__), "..", "data", "eval")
    os.makedirs(eval_dir, exist_ok=True)

    records = []
    for split_name in ds:
        for row in ds[split_name]:
            question = row.get("question", "")
            choices = row.get("choices", [])
            answer = row.get("answer", [])
            if question:
                records.append({
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "split": split_name,
                })

    output_path = os.path.join(eval_dir, "medqa_zh.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[OK] MedQA 保存完成：{output_path}，共 {len(records)} 道题")
    return len(records)


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="公共医学数据集初始化")
    parser.add_argument(
        "--dataset",
        choices=["cmirb", "dialogue", "medqa", "all"],
        default="all",
        help="要导入的数据集（默认 all）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只随机抽取 N 条，0 表示全部导入",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="导入前删除 knowledge_docs collection（更换 Embedding 模型时必须）",
    )
    args = parser.parse_args()

    milvus_client, embedding_model = _get_deps(reset=args.reset)

    if args.dataset in ("cmirb", "all"):
        await download_cmirb(milvus_client, embedding_model, limit=args.limit)

    if args.dataset in ("dialogue", "all"):
        await download_med_dialogue(
            milvus_client, embedding_model, limit=args.limit
        )

    if args.dataset in ("medqa", "all"):
        await download_medqa()

    print("\n[DONE] 数据集初始化完成！")


if __name__ == "__main__":
    asyncio.run(main())