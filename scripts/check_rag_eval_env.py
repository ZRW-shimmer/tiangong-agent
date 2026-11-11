"""
RAG 评估环境自检。

检查项：
  1. Python / TruLens / LiteLLM 依赖
  2. Judge LLM 密钥
  3. trulens_eval 数据库和表
  4. Milvus knowledge_docs 知识库
  5. data/eval/rag_eval_dataset.json 评估数据集

用法：
  python scripts/check_rag_eval_env.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# TruLens 的依赖提示不影响使用，先关掉
logging.getLogger("trulens.core.utils.imports").setLevel(logging.ERROR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DATASET = os.path.join(PROJECT_ROOT, "data", "eval", "rag_eval_dataset.json")

OK = "[OK]"
WARN = "[WARN]"
FAIL = "[FAIL]"


def check_python() -> bool:
    print(f"\n== Python ==")
    print(f"  {sys.version.splitlines()[0]}")
    print(f"  executable: {sys.executable}")
    return sys.version_info >= (3, 10)


def check_packages() -> bool:
    print(f"\n== 评估依赖 ==")
    ok = True
    packages = [
        ("trulens.core", "trulens-core"),
        ("trulens.apps.app", "trulens-core / TruApp"),
        ("trulens.providers.litellm", "trulens-providers-litellm"),
        ("trulens.dashboard", "trulens-dashboard"),
    ]
    for module, label in packages:
        try:
            __import__(module)
            print(f"  {OK} {label}")
        except Exception as exc:
            ok = False
            print(f"  {FAIL} {label}: {exc}")
    return ok


def check_keys() -> bool:
    from src.core.config import get_settings

    settings = get_settings()
    print(f"\n== Judge LLM ==")
    print(f"  model: {settings.EVAL_JUDGE_MODEL}")
    if settings.DEEPSEEK_API_KEY:
        print(f"  {OK} DEEPSEEK_API_KEY 已配置")
        return True
    print(f"  {FAIL} DEEPSEEK_API_KEY 未配置，无法计算 RAG Triad")
    return False


def check_trulens_db() -> bool:
    from sqlalchemy import create_engine, text

    from src.core.config import get_settings

    settings = get_settings()
    print(f"\n== TruLens 评估库 ==")
    print(f"  database: {settings.TRULENS_DB_NAME}")

    try:
        engine = create_engine(settings.TRULENS_DATABASE_URL)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "select tablename from pg_tables "
                    "where schemaname='public' order by tablename"
                )
            ).fetchall()
        tables = [r[0] for r in rows]
        if not tables:
            print(f"  {WARN} 数据库存在，但还没有 TruLens 表")
            print("        首次运行评估脚本时会自动建表")
        else:
            print(f"  {OK} 已初始化 {len(tables)} 张表")
        return True
    except Exception as exc:
        print(f"  {FAIL} 无法连接: {exc}")
        print(
            "        请先创建数据库：\n"
            f'        docker exec tiangong-postgres psql -U {settings.DB_USER} '
            f'-d postgres -c "CREATE DATABASE {settings.TRULENS_DB_NAME};"'
        )
        return False


def check_milvus() -> bool:
    from pymilvus import MilvusClient

    from src.core.config import get_settings

    settings = get_settings()
    print(f"\n== Milvus 知识库 ==")
    try:
        client = MilvusClient(
            uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
        )
        collections = client.list_collections()
        print(f"  collections: {collections}")
        if "knowledge_docs" not in collections:
            print(f"  {FAIL} knowledge_docs 不存在")
            print("        请先运行: python scripts/init_public_datasets.py --dataset cmirb")
            return False

        stats = client.get_collection_stats("knowledge_docs")
        count = int(stats.get("row_count", 0))
        if count > 0:
            print(f"  {OK} knowledge_docs 共 {count} 条")
            return True
        print(f"  {WARN} knowledge_docs 为空")
        return False
    except Exception as exc:
        print(f"  {FAIL} Milvus 连接失败: {exc}")
        return False


def check_dataset() -> bool:
    print(f"\n== 评估数据集 ==")
    print(f"  path: {EVAL_DATASET}")
    if not os.path.exists(EVAL_DATASET):
        print(f"  {FAIL} 数据集不存在")
        print("        请运行: python scripts/prepare_eval_dataset.py")
        return False

    with open(EVAL_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not dataset:
        print(f"  {FAIL} 数据集为空")
        return False

    sources = Counter(item.get("source", "unknown") for item in dataset)
    print(f"  {OK} 共 {len(dataset)} 条")
    print(f"  来源分布: {dict(sources)}")

    sample = dataset[0]
    missing = [k for k in ("question", "reference_answer", "source") if k not in sample]
    if missing:
        print(f"  {WARN} 字段缺失: {missing}")
        return False
    return True


def main():
    print("=" * 60)
    print("RAG 评估环境自检")
    print("=" * 60)

    results = {
        "Python": check_python(),
        "依赖": check_packages(),
        "Judge LLM": check_keys(),
        "TruLens DB": check_trulens_db(),
        "Milvus": check_milvus(),
        "评估数据集": check_dataset(),
    }

    print("\n" + "=" * 60)
    print("检查结果")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {OK if passed else FAIL} {name}")

    if all(results.values()):
        print("\n初始化完成，可以开始第一步优化：")
        print("  python scripts/run_rag_experiments.py --limit 10 --only baseline,v1_rerank --reset")
        print("  python scripts/run_rag_experiments.py --dashboard")
    else:
        print("\n请先修复上面的 FAIL 项，再运行评估。")


if __name__ == "__main__":
    main()
