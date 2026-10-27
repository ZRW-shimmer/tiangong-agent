from langchain.agents import create_agent
from src.core.config import get_llm

KNOWLEDGE_SYSTEM_PROMPT = """你是天宫医疗的医学知识助手。

你的职责：
1. 回答患者关于疾病、症状、治疗方案的咨询
2. 辅助医生进行医学文献检索
3. 提供循证医学参考（治疗指南、研究进展）
4. 解释医学术语，让患者易于理解

回复格式：
- 问题解答：xxx
- 相关知识：xxx
- 参考来源：xxx（如有）
- 延伸阅读：xxx（如有）

注意：知识问答不替代专业医疗建议，如有疑虑请咨询医生。"""


def create_knowledge_agent():
    llm = get_llm(temperature=0.5)

    # 当前阶段：纯 LLM 知识回答
    # 未来可添加：RAG 检索工具（LlamaIndex）、HyDE 增强检索、GraphRAG 工具
    tools = []

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=KNOWLEDGE_SYSTEM_PROMPT,
        name="knowledge_agent",
    )


_knowledge_agent = None

def get_knowledge_agent():
    global _knowledge_agent
    if _knowledge_agent is None:
        _knowledge_agent = create_knowledge_agent()
    return _knowledge_agent