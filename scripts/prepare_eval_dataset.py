"""
生成 RAG 评估数据集。

数据来源：
  1. 手动编写的高质量问题（通用医学场景）
  2. MedQA 中文考试题（如果已运行 scripts/init_public_datasets.py --dataset medqa）
  3. medical.json 疾病知识（自动生成症状/病因/用药等问题）
  4. CMIRB 语料（从知识库原文反向生成问题，并记录 expected_chunk_id，可算召回率）

输出：
  data/eval/rag_eval_dataset.json

用法：
  # 默认：10 条手动 + 20 条 CMIRB + 20 条 medical.json
  python scripts/prepare_eval_dataset.py

  # 快速冒烟：只用 10 条手动题
  python scripts/prepare_eval_dataset.py --cmirb 0 --medical 0

  # 指定每条来源的数量
  python scripts/prepare_eval_dataset.py --manual 10 --cmirb 30 --medical 30 --medqa 20
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "data", "eval")
MEDICAL_JSON = os.path.join(PROJECT_ROOT, "data", "raw", "medical.json")
CMIRB_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "CMIRB", "corpus.jsonl")

CMIRB_BATCH_SIZE = 500
CMIRB_MAX_TEXT_LEN = 2000


def load_manual_questions() -> list[dict]:
    """手动编写的高质量评估问题（贴近真实用户场景）。"""
    return [
        {
            "question": "高血压患者能吃布洛芬吗？",
            "reference_answer": "高血压患者应慎用布洛芬，因为 NSAIDs 类药物可能升高血压、影响降压药效果。",
            "source": "manual",
            "category": "drug_interaction",
        },
        {
            "question": "糖尿病人可以吃西瓜吗？",
            "reference_answer": "糖尿病患者可以少量食用西瓜，但西瓜 GI 值较高（约 72），建议每次不超过 200g，并计入全天碳水。",
            "source": "manual",
            "category": "diet",
        },
        {
            "question": "感冒和流感有什么区别？",
            "reference_answer": "普通感冒症状较轻，以鼻部症状为主；流感起病急、全身症状重（高热、肌肉酸痛），由流感病毒引起。",
            "source": "manual",
            "category": "differential_diagnosis",
        },
        {
            "question": "孕妇发烧了能吃什么药？",
            "reference_answer": "孕妇退热首选对乙酰氨基酚，禁用布洛芬和阿司匹林；体温超过 38.5°C 或持续不退应及时就医。",
            "source": "manual",
            "category": "drug_safety",
        },
        {
            "question": "头孢和酒能一起吗？",
            "reference_answer": "不能。头孢类抗生素与酒精同服可能引起双硫仑样反应，出现面部潮红、心悸，严重时可致休克。",
            "source": "manual",
            "category": "drug_interaction",
        },
        {
            "question": "小孩反复咳嗽一个月了怎么办？",
            "reference_answer": "儿童慢性咳嗽（>4 周）需排除咳嗽变异性哮喘、上气道咳嗽综合征、感染后咳嗽等，建议儿科就诊并做肺功能检查。",
            "source": "manual",
            "category": "clinical",
        },
        {
            "question": "阿莫西林和头孢有什么区别？",
            "reference_answer": "阿莫西林属于青霉素类，头孢属于头孢菌素类；两者都是 β-内酰胺类抗生素，但抗菌谱不同，青霉素过敏者用头孢需谨慎。",
            "source": "manual",
            "category": "drug_knowledge",
        },
        {
            "question": "胃疼应该做什么检查？",
            "reference_answer": "胃痛建议做胃镜（金标准）、幽门螺杆菌 C13/C14 呼气试验，必要时做腹部超声排除胆胰疾病。",
            "source": "manual",
            "category": "check",
        },
        {
            "question": "血压高到多少需要吃药？",
            "reference_answer": "一般收缩压 ≥140mmHg 或舒张压 ≥90mmHg，且生活方式干预 3 个月无效时需要药物治疗；合并糖尿病或肾病者标准更低。",
            "source": "manual",
            "category": "clinical_decision",
        },
        {
            "question": "甲状腺结节需要手术吗？",
            "reference_answer": "多数良性结节无需手术，定期随访即可。TI-RADS 4 类以上、细针穿刺提示恶性、结节 >4cm 且有压迫症状时考虑手术。",
            "source": "manual",
            "category": "clinical_decision",
        },
    ]


def load_medqa_questions(max_count: int) -> list[dict]:
    """从 MedQA 提取开放式问题（去掉选项）。"""
    medqa_path = os.path.join(EVAL_DIR, "medqa_zh.json")
    if not os.path.exists(medqa_path):
        print(
            f"[WARN] {medqa_path} 不存在，跳过 MedQA。"
            "如需下载：python scripts/init_public_datasets.py --dataset medqa"
        )
        return []

    with open(medqa_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    test_items = [r for r in raw if r.get("split") == "test"] or raw
    random.shuffle(test_items)

    questions: list[dict] = []
    for item in test_items[:max_count]:
        q = item.get("question", "").strip()
        if not q:
            continue

        # 去掉选择题尾巴，转成开放式问题
        for suffix in ["，最可能的诊断是", "，应首先考虑", "，最佳治疗方案是", "，最可能的原因是"]:
            if suffix in q:
                q = q.split(suffix)[0] + suffix + "什么？"
                break

        answer_text = ""
        choices = item.get("choices") or []
        answers = item.get("answer") or []
        if choices and answers:
            for choice in choices:
                if any(choice.startswith(label) for label in answers):
                    answer_text = choice[2:].strip() if len(choice) > 2 else choice
                    break

        questions.append(
            {
                "question": q,
                "reference_answer": answer_text,
                "source": "medqa",
                "category": "clinical",
            }
        )

    return questions


def generate_from_medical_json(max_count: int) -> list[dict]:
    """从 medical.json 自动生成评估问题。"""
    if max_count <= 0:
        return []
    if not os.path.exists(MEDICAL_JSON):
        print(f"[WARN] {MEDICAL_JSON} 不存在，跳过 medical.json")
        return []

    diseases: list[dict] = []
    with open(MEDICAL_JSON, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                diseases.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    random.shuffle(diseases)

    templates = [
        ("symptom", "{name}有哪些症状？", lambda d: "、".join(d.get("symptom", [])[:5])),
        ("cause", "{name}的病因是什么？", lambda d: d.get("cause", "")),
        ("prevent", "{name}如何预防？", lambda d: d.get("prevent", "")),
        (
            "drug",
            "{name}常用什么药物治疗？",
            lambda d: "、".join(
                (d.get("recommand_drug") or [])[:5]
                or [x.get("drug_name", "") for x in (d.get("drug_detail") or [])[:5]]
            ),
        ),
        ("check", "{name}需要做哪些检查？", lambda d: "、".join(d.get("check", [])[:5])),
        (
            "department",
            "{name}应该挂什么科？",
            lambda d: "、".join(d.get("cure_department", [])[:3]),
        ),
        ("cure", "{name}能治好吗？", lambda d: d.get("cured_prob", "")),
    ]

    questions: list[dict] = []
    for disease in diseases:
        if len(questions) >= max_count:
            break

        name = (disease.get("name") or "").strip()
        if len(name) < 2:
            continue

        tpl_type, tpl_question, answer_fn = random.choice(templates)
        try:
            ref_answer = answer_fn(disease)
        except Exception:
            continue
        if isinstance(ref_answer, list):
            ref_answer = "、".join(str(x) for x in ref_answer if x)
        ref_answer = (ref_answer or "").strip()
        if len(ref_answer) < 3:
            continue

        questions.append(
            {
                "question": tpl_question.format(name=name),
                "reference_answer": ref_answer[:500],
                "source": "medical_json",
                "category": tpl_type,
            }
        )

    return questions


def _iter_valid_cmirb_rows():
    """按入库脚本同样的规则遍历 CMIRB，返回 (有效文本序号, 文本)。"""
    valid_index = 0
    with open(CMIRB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = row.get("text", "") or row.get("content", "")
            if text and len(text.strip()) > 20:
                yield valid_index, text.strip()[:CMIRB_MAX_TEXT_LEN]
                valid_index += 1


def _cmirb_chunk_id(valid_index: int) -> str:
    """还原 init_public_datasets.py 写入 Milvus 时的 chunk id。"""
    batch_start = (valid_index // CMIRB_BATCH_SIZE) * CMIRB_BATCH_SIZE
    doc_id = hashlib.md5(f"cmirb_{batch_start}".encode()).hexdigest()[:16]
    return f"{doc_id}_{valid_index - batch_start}"


QUESTION_FROM_CONTEXT_PROMPT = """你是医学检索评估数据集生成器。
请根据下面这段医学资料，写一个普通患者可能会问的自然问题。

要求：
1. 问题必须能由这段资料回答
2. 问题必须自包含：包含关键症状、疾病、人群或场景，不能出现“这种情况”“我这样”“这个病”等指代
3. 用患者的口语表达，不要直接抄资料里的句子
4. 只输出问题本身，不要解释、不要加引号

医学资料：
{text}

问题："""


async def generate_from_cmirb(max_count: int, concurrency: int = 4) -> list[dict]:
    """从 CMIRB 原文反向生成问题，并记录 expected_chunk_id 用于召回率评估。"""
    if max_count <= 0:
        return []
    if not os.path.exists(CMIRB_PATH):
        print(f"[WARN] {CMIRB_PATH} 不存在，跳过 CMIRB")
        return []

    all_indices = [idx for idx, _ in _iter_valid_cmirb_rows()]
    if not all_indices:
        return []

    random.shuffle(all_indices)
    selected = set(all_indices[:max_count])

    rows: dict[int, str] = {}
    for idx, text in _iter_valid_cmirb_rows():
        if idx in selected:
            rows[idx] = text

    from langchain_core.messages import HumanMessage

    from src.core.config import get_llm

    llm = get_llm(temperature=0.3)
    semaphore = asyncio.Semaphore(concurrency)

    async def _make_one(idx: int, text: str) -> dict | None:
        async with semaphore:
            try:
                response = await llm.ainvoke(
                    [HumanMessage(content=QUESTION_FROM_CONTEXT_PROMPT.format(text=text[:1200]))]
                )
                question = response.content.strip().strip('"').strip("“”")
            except Exception as exc:
                print(f"[WARN] CMIRB 问题生成失败 (idx={idx}): {exc}")
                return None

        if not question:
            return None
        return {
            "question": question,
            "reference_answer": text,
            "source": "cmirb",
            "category": "retrieval_grounded",
            "expected_chunk_id": _cmirb_chunk_id(idx),
        }

    tasks = [_make_one(idx, text) for idx, text in rows.items()]
    results = await asyncio.gather(*tasks)
    questions = [r for r in results if r]
    print(f"[INFO] CMIRB 反向生成问题: {len(questions)}/{len(rows)}")
    return questions


async def main():
    parser = argparse.ArgumentParser(description="生成 RAG 评估数据集")
    parser.add_argument("--out", default=os.path.join(EVAL_DIR, "rag_eval_dataset.json"))
    parser.add_argument("--manual", type=int, default=10, help="手动题数量")
    parser.add_argument("--medqa", type=int, default=0, help="MedQA 题目数量")
    parser.add_argument("--medical", type=int, default=20, help="medical.json 生成题数量")
    parser.add_argument("--cmirb", type=int, default=20, help="CMIRB 反向生成题数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=4, help="CMIRB 问题生成并发数")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    manual = load_manual_questions()[: args.manual] if args.manual > 0 else []
    medqa = load_medqa_questions(args.medqa)
    medical = generate_from_medical_json(args.medical)
    cmirb = await generate_from_cmirb(args.cmirb, args.concurrency)

    dataset = manual + medqa + medical + cmirb

    # 去重（按问题文本）
    seen: set[str] = set()
    unique: list[dict] = []
    for item in dataset:
        key = item["question"].strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item)

    # 按来源交错排列，保证 --limit N 时各来源都有覆盖
    by_source: dict[str, list[dict]] = {}
    for item in unique:
        by_source.setdefault(item["source"], []).append(item)
    for group in by_source.values():
        random.shuffle(group)

    source_order = ["manual", "medqa", "medical_json", "cmirb"]
    interleaved: list[dict] = []
    while any(by_source.get(s) for s in source_order):
        for source in source_order:
            group = by_source.get(source) or []
            if group:
                interleaved.append(group.pop(0))
    unique = interleaved

    # 写稳定 id，方便 Dashboard 对照
    for i, item in enumerate(unique, 1):
        item.setdefault("id", f"q{i:04d}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(unique, f, ensure_ascii=False, indent=2)

    sources: dict[str, int] = {}
    categories: dict[str, int] = {}
    for item in unique:
        sources[item["source"]] = sources.get(item["source"], 0) + 1
        categories[item["category"]] = categories.get(item["category"], 0) + 1

    print(f"\n[OK] 评估数据集: {args.out}")
    print(f"     总量: {len(unique)}")
    print(f"     来源: {sources}")
    print(f"     类别: {categories}")
    print("\n[示例]")
    for item in unique[:3]:
        print(f"  Q: {item['question'][:60]}")
        print(f"  A: {item['reference_answer'][:60]}")


if __name__ == "__main__":
    asyncio.run(main())
