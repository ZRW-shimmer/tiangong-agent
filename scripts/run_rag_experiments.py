"""
RAG 实验矩阵：每组记录为不同 app_version，在 TruLens Dashboard 中对比。

当前知识库已按固定 512 字符分块入库，因此这里先做“不重建索引”的检索策略实验：
  baseline    向量检索 Top20 -> 直接取 Top5
  v1_rerank   向量检索 Top20 -> Reranker 精排 Top5
  v2_hyde     查询改写 + HyDE + Reranker
  v3_rewrite  查询改写 + HyDE + Reranker（v2 基础上加 Query 改写）
  v4_topk40   查询改写 + HyDE + Reranker，粗排扩到 Top40

分块策略（语义分块/父子分块）和混合检索需要重建索引，放在后续阶段。

用法：
  python scripts/run_rag_experiments.py --limit 10
  python scripts/run_rag_experiments.py --only baseline
  python scripts/run_rag_experiments.py --only baseline,v1_rerank
  python scripts/run_rag_experiments.py --dashboard
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 提前导入 TruLens 应用类，避免后台评估线程并发 import 死锁
from trulens.apps.app import TruApp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "data", "eval")
RUNS_DIR = os.path.join(EVAL_DIR, "runs")

from src.agents.knowledge.evaluation.feedbacks import build_rag_triad_metrics
from src.agents.knowledge.evaluation.tracked_rag import EvalRAGConfig, TrackedRAG
from src.agents.knowledge.evaluation.trulens_config import (
    get_llm_provider,
    get_trulens_session,
    launch_dashboard,
)
from src.core.config import get_llm, get_settings
from src.infra.embedding import get_embedding_model

settings = get_settings()


EXPERIMENTS: dict[str, EvalRAGConfig] = {
    "baseline": EvalRAGConfig(
        name="baseline",
        top_k=20,
        rerank_top_k=5,
        use_rewrite=False,
        use_hyde=False,
        use_rerank=False,
    ),
    "v1_rerank": EvalRAGConfig(
        name="v1_rerank",
        top_k=20,
        rerank_top_k=5,
        use_rewrite=False,
        use_hyde=False,
        use_rerank=True,
    ),
    "v1_topk10": EvalRAGConfig(
        name="v1_topk10",
        top_k=10,
        rerank_top_k=5,
        use_rewrite=False,
        use_hyde=False,
        use_rerank=True,
    ),
    "v1_topk40": EvalRAGConfig(
        name="v1_topk40",
        top_k=40,
        rerank_top_k=5,
        use_rewrite=False,
        use_hyde=False,
        use_rerank=True,
    ),
    # 药品说明书专项（只检索 doc_type=drug_instruction）
    "drug_baseline": EvalRAGConfig(
        name="drug_baseline",
        top_k=20,
        rerank_top_k=5,
        use_rewrite=False,
        use_hyde=False,
        use_rerank=False,
        doc_type="drug_instruction",
    ),
    "drug_rerank": EvalRAGConfig(
        name="drug_rerank",
        top_k=20,
        rerank_top_k=5,
        use_rewrite=False,
        use_hyde=False,
        use_rerank=True,
        doc_type="drug_instruction",
    ),
    "drug_top1": EvalRAGConfig(
        name="drug_top1",
        top_k=20,
        rerank_top_k=1,
        use_rewrite=False,
        use_hyde=False,
        use_rerank=False,
        doc_type="drug_instruction",
    ),
    "drug_top3": EvalRAGConfig(
        name="drug_top3",
        top_k=20,
        rerank_top_k=3,
        use_rewrite=False,
        use_hyde=False,
        use_rerank=False,
        doc_type="drug_instruction",
    ),
    "v2_hyde": EvalRAGConfig(
        name="v2_hyde",
        top_k=20,
        rerank_top_k=5,
        use_rewrite=False,
        use_hyde=True,
        use_rerank=True,
    ),
    "v3a_rewrite_only": EvalRAGConfig(
        name="v3a_rewrite_only",
        top_k=20,
        rerank_top_k=5,
        use_rewrite=True,
        use_hyde=False,
        use_rerank=True,
    ),
    "v3_rewrite": EvalRAGConfig(
        name="v3_rewrite",
        top_k=20,
        rerank_top_k=5,
        use_rewrite=True,
        use_hyde=True,
        use_rerank=True,
    ),
    "v4_topk40": EvalRAGConfig(
        name="v4_topk40",
        top_k=40,
        rerank_top_k=5,
        use_rewrite=True,
        use_hyde=True,
        use_rerank=True,
    ),
}


def _build_app(config: EvalRAGConfig) -> TrackedRAG:
    from pymilvus import MilvusClient

    milvus = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    return TrackedRAG(
        llm=get_llm(temperature=0.1),
        embedding_model=get_embedding_model(),
        milvus_client=milvus,
        config=config,
    )


def _finalize_evaluation(tru_app, session) -> None:
    """确保 trace 落库，并把后台评估切到主线程同步完成。"""
    try:
        session.force_flush()
    except Exception:
        pass

    try:
        tru_app.stop_evaluator()
    except Exception:
        pass

    for call in (
        lambda: tru_app.compute_feedbacks(raise_error_on_no_feedbacks_computed=False),
        lambda: tru_app.compute_feedbacks(),
    ):
        try:
            call()
            break
        except TypeError:
            continue
        except AttributeError:
            break
        except Exception as exc:
            print(f"  [WARN] compute_feedbacks 失败: {exc}")
            break

    try:
        session.force_flush()
    except Exception:
        pass


def _hit_stats(
    hits: list[dict],
    expected_chunk_id: str | None,
    expected_doc_name: str | None = None,
    expected_section: str | None = None,
) -> tuple[int | None, int | None]:
    """返回 (是否命中, 命中排名)。

    优先用 expected_chunk_id 精确匹配；
    药品说明书没有固定 chunk id，则用 doc_name + section 标题匹配。
    """
    if expected_chunk_id:
        ids = [str(h.get("id", "")) for h in hits]
        if expected_chunk_id not in ids:
            return 0, None
        return 1, ids.index(expected_chunk_id) + 1

    if expected_doc_name and expected_section:
        for rank, hit in enumerate(hits, 1):
            text = str(hit.get("text", ""))
            if (
                hit.get("doc_name") == expected_doc_name
                and expected_section in text
            ):
                return 1, rank
        return 0, None

    return None, None


async def run_single_experiment(
    version: str,
    config: EvalRAGConfig,
    dataset: list[dict],
    limit: int,
) -> dict:
    session = get_trulens_session()
    provider = get_llm_provider()
    metrics = build_rag_triad_metrics(provider)

    app = _build_app(config)
    tru_app = TruApp(
        app,
        main_method=app.query,
        app_name="tiangong_medical_rag",
        app_version=version,
        feedbacks=metrics,
    )

    items = dataset[:limit] if limit > 0 else dataset
    print(f"\n{'=' * 70}")
    print(f"实验: {version} | {config.describe()}")
    print(f"样本数: {len(items)}")
    print(f"{'=' * 70}")

    details: list[dict] = []
    hit_values: list[int] = []
    reciprocal_ranks: list[float] = []
    latencies: list[float] = []
    retrieval_latencies: list[float] = []

    with tru_app as _recording:
        for i, item in enumerate(items, 1):
            question = item["question"]
            print(f"  [{i}/{len(items)}] {question[:50]}...", end="", flush=True)
            started = time.perf_counter()
            answer = ""
            error = None
            try:
                answer = await app.query(question)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(f" 失败: {error}")
            else:
                print(f" {app.last_total_ms:.0f}ms")

            hit, rank = _hit_stats(
                app.last_hits,
                item.get("expected_chunk_id"),
                item.get("doc_name"),
                item.get("expected_section"),
            )
            if hit is not None:
                hit_values.append(hit)
                if rank:
                    reciprocal_ranks.append(1.0 / rank)

            latencies.append((time.perf_counter() - started) * 1000)
            retrieval_latencies.append(app.last_retrieval_ms)
            details.append(
                {
                    "id": item.get("id"),
                    "question": question,
                    "source": item.get("source"),
                    "expected_chunk_id": item.get("expected_chunk_id"),
                    "hit": hit,
                    "rank": rank,
                    "retrieval_query": app.last_retrieval_query,
                    "retrieval_ms": round(app.last_retrieval_ms, 1),
                    "total_ms": round(app.last_total_ms, 1),
                    "answer_preview": answer[:300],
                    "error": error,
                }
            )

    _finalize_evaluation(tru_app, session)

    summary = {
        "version": version,
        "config": asdict(config),
        "n": len(items),
        "errors": sum(1 for d in details if d["error"]),
        "hit_rate": round(sum(hit_values) / len(hit_values), 4) if hit_values else None,
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)
        if reciprocal_ranks
        else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "avg_retrieval_ms": round(
            sum(retrieval_latencies) / len(retrieval_latencies), 1
        )
        if retrieval_latencies
        else None,
    }

    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(os.path.join(RUNS_DIR, f"{version}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(
        os.path.join(RUNS_DIR, f"{version}_details.jsonl"), "w", encoding="utf-8"
    ) as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"  >>> 汇总: {json.dumps(summary, ensure_ascii=False)}")
    try:
        leaderboard = session.get_leaderboard()
        print(f"\n{leaderboard}")

        full = leaderboard.reset_index()
        full.to_csv(
            os.path.join(RUNS_DIR, f"{version}_leaderboard.csv"), index=False
        )

        # 只取当前版本那一行，避免把其他版本的指标写进 summary
        if "app_version" in full.columns:
            current = full[full["app_version"] == version]
        else:
            current = full.iloc[[0]]
        summary["leaderboard"] = current.to_dict(orient="records")

        # 补写一次，让 summary JSON 里也包含三项指标
        with open(
            os.path.join(RUNS_DIR, f"{version}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        print(f"  [WARN] 读取 Leaderboard 失败: {exc}")

    return summary


async def main():
    parser = argparse.ArgumentParser(description="RAG 实验矩阵")
    parser.add_argument(
        "--dataset",
        default=os.path.join(EVAL_DIR, "rag_eval_dataset.json"),
        help="评估数据集路径",
    )
    parser.add_argument("--limit", type=int, default=10, help="每个实验使用的样本数，0=全部")
    parser.add_argument("--only", default="", help="只运行指定实验，逗号分隔")
    parser.add_argument("--dashboard", action="store_true", help="启动 TruLens Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Dashboard 端口")
    parser.add_argument("--list", action="store_true", help="列出所有实验配置")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="运行前清空 trulens_eval 中已有评估数据（重新对比时使用）",
    )
    args = parser.parse_args()

    if args.list:
        for name, cfg in EXPERIMENTS.items():
            print(f"{name:12s} {cfg.describe()}")
        return

    if args.dashboard:
        print(f"启动 TruLens Dashboard: http://localhost:{args.port}")
        launch_dashboard(port=args.port)
        return

    if not os.path.exists(args.dataset):
        raise SystemExit(
            f"评估数据集不存在: {args.dataset}\n"
            "请先运行: python scripts/prepare_eval_dataset.py"
        )

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if args.reset:
        print("清空 trulens_eval 中已有评估数据...")
        get_trulens_session().reset_database()

    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"未知实验: {unknown}，可选: {list(EXPERIMENTS)}")

    summaries = []
    for version in names:
        summaries.append(
            await run_single_experiment(
                version, EXPERIMENTS[version], dataset, args.limit
            )
        )

    print(f"\n{'=' * 70}")
    print("全部实验完成")
    print(f"结果目录: {RUNS_DIR}")
    for s in summaries:
        metrics = s.get("leaderboard")
        print(
            f"  {s['version']:12s} n={s['n']:3d} "
            f"hit_rate={s['hit_rate']} mrr={s['mrr']} "
            f"latency={s['avg_latency_ms']}ms"
        )
        if metrics:
            print(f"    {metrics}")
    print("\n启动 Dashboard 查看对比:")
    print("  python scripts/run_rag_experiments.py --dashboard")


if __name__ == "__main__":
    asyncio.run(main())
