# src/agents/workers/knowledge_agent.py

from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from pymilvus import MilvusClient

from src.core.config import get_settings
from src.infra.embedding import get_embedding_model
from src.infra.neo4j_client import get_neo4j_driver
from src.infra.milvus_client import get_milvus_client_alias
from src.agents.knowledge.tools import KnowledgeDeps, build_knowledge_tools

settings = get_settings()

KNOWLEDGE_SYSTEM_PROMPT = """你是天宫医疗的知识问答助手，面向医生、药师和企业内部员工提供专业知识服务。

## 你的能力

你拥有五个检索工具，根据问题类型选择最合适的工具：

1. search_knowledge_docs — 文档检索
   适用：查询临床指南、药品说明书、医院制度SOP、医学文献
   示例："高血压诊疗指南推荐的一线用药"、"阿莫西林的禁忌症"

2. search_knowledge_graph — 知识图谱检索
   适用：查询疾病与症状/药物/科室/检查/食物的关系，多跳推理
   示例："糖尿病的常用药有哪些"、"头痛可能是什么疾病需要做什么检查"

3. search_knowledge_sql — 运营数据查询
   适用：统计数据、数量、排名、趋势、库存
   示例："上个月各科室问诊量排名"、"库存不足100的OTC药品"

4. search_knowledge_multi — 多通道融合检索（复杂问题首选）
   适用：需要同时参考文档和知识图谱才能回答的复杂问题
   示例："糖尿病合并肾功能不全的降糖方案"、"高血压患者能用哪些感冒药"

5. review_prescription_tool — 处方审核
   适用：校验处方安全性（剂量、配伍禁忌、过敏冲突、重复用药）
   示例："帮我审核这张处方：阿莫西林0.5g tid + 甲硝唑0.4g bid"

## 工具选择策略

- 简单的文档查询 → search_knowledge_docs
- 药品说明书、用法用量、禁忌、不良反应、药物相互作用 → search_knowledge_docs，并传 doc_type="drug_instruction"
- 医学文献、研究资料、通用医学知识 → search_knowledge_docs，并传 doc_type="literature"
- 实体关系查询 → search_knowledge_graph
- 统计数据查询 → search_knowledge_sql
- 涉及"合并症+用药"、"禁忌+推荐"等复杂问题 → search_knowledge_multi
- 处方审核、用药安全校验 → review_prescription_tool
- 不确定文档类型时 doc_type 留空，系统会同时检索文献和药品说明书

## 工作原则

1. 回答必须基于工具返回的检索结果，不要编造信息
2. 回答中必须标注信息来源
3. 如果所有工具都未找到相关信息，明确告知用户
4. 涉及用药安全的问题，必须提醒"请遵医嘱"
5. 语言专业但易懂"""


def _build_deps(db_session=None, user_id="anonymous", role="patient") -> KnowledgeDeps:
    llm = ChatDeepSeek(
        model=settings.CHAT_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.3,
    )
    embedding_model = get_embedding_model()
    neo4j_driver = get_neo4j_driver()
    get_milvus_client_alias()
    milvus_client = MilvusClient(
        uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
    )
    return KnowledgeDeps(
        llm=llm,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        neo4j_driver=neo4j_driver,
        db_session=db_session,
        user_id=user_id,
        role=role,
    )


def create_knowledge_agent(db_session=None, user_id="anonymous", role="patient"):
    deps = _build_deps(db_session, user_id, role)
    tools = build_knowledge_tools(deps)
    llm = ChatDeepSeek(
        model=settings.CHAT_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        temperature=0.3,
    )
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
        name="knowledge_agent",
    )


_knowledge_agent = None

def get_knowledge_agent(
    db_session=None,
    user_id="anonymous",
    role="patient",
):
    """默认请求复用单例；带 db/user/role 的请求按需构建，避免串会话。"""
    global _knowledge_agent
    if db_session is None and user_id == "anonymous" and role == "patient":
        if _knowledge_agent is None:
            _knowledge_agent = create_knowledge_agent()
        return _knowledge_agent
    return create_knowledge_agent(db_session, user_id, role)