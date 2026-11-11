"""RAG Triad 评估指标（TruLens Metric API）。

三大核心指标：
  1. 答案相关性 (Question -> Answer)
  2. 上下文相关性 (Question -> Context)
  3. 有据性 (Context -> Answer)
"""

from __future__ import annotations

import numpy as np

# 提前导入，避免评估器后台线程并发 import 造成死锁
from trulens.core import Metric, Selector
from trulens.core.metric.metric import GroundednessConfigs


def build_rag_triad_metrics(provider) -> list:
    """根据 Judge Provider 构建 RAG Triad 指标列表。"""

    # 1. 答案相关性：最终回答是否切题
    m_answer_relevance = Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="答案相关性",
        selectors={
            "prompt": Selector.select_record_input(),
            "response": Selector.select_record_output(),
        },
    )

    # 2. 上下文相关性：每个检索片段与问题的相关程度
    m_context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name="上下文相关性",
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=np.mean,
    )

    # 3. 有据性：回答中的每个声明是否能被检索内容支撑
    # 使用 consider_answerability 版本，同时传入 question，可识别“问题本身无法回答”的情况
    groundedness_impl = getattr(
        provider,
        "groundedness_measure_with_cot_reasons_consider_answerability",
        provider.groundedness_measure_with_cot_reasons,
    )
    m_groundedness = Metric(
        implementation=groundedness_impl,
        name="有据性",
        selectors={
            "source": Selector.select_context(collect_list=True),
            "statement": Selector.select_record_output(),
            "question": Selector.select_record_input(),
        },
        # 中文回答不使用 NLTK 英文分句，交给 Judge LLM 分句
        groundedness_configs=GroundednessConfigs(
            use_sent_tokenize=False,
            filter_trivial_statements=True,
        ),
    )

    return [m_answer_relevance, m_context_relevance, m_groundedness]
