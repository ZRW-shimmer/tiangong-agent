from __future__ import annotations
import asyncio
import re
import json
from loguru import logger
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from src.agents.knowledge.prompts import NL2SQL_PROMPT, SQL_QA_PROMPT

MAX_SQL_RETRIES = 2
SQL_TIMEOUT_SECONDS = 10

FORBIDDEN_PATTERNS = [
    re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE)\b", re.IGNORECASE),
    re.compile(r"\bpatients\b[^;]*\b(phone|id_card)\b", re.IGNORECASE),
]


def _validate_sql(sql: str) -> tuple[bool, str]:
    stripped = sql.strip().rstrip(";")
    if not stripped.upper().startswith("SELECT"):
        return False, "只允许 SELECT 查询"
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(stripped):
            return False, "查询包含禁止的操作或字段"
    if "LIMIT" not in stripped.upper():
        stripped += " LIMIT 100"
    return True, stripped


async def _generate_sql(
    question: str, llm: BaseChatModel, error_hint: str = "",
) -> str:
    extra = ""
    if error_hint:
        extra = f"\n\n上一次生成的 SQL 执行报错：{error_hint}\n请修正后重新生成。"
    prompt = NL2SQL_PROMPT.format(question=question) + extra
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    sql = response.content.strip()
    if "```" in sql:
        sql = sql.split("```")[1].lstrip("sql").strip()
    return sql

async def search_sql_raw(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
)-> tuple[str | None, list[dict] | str]:
    """NL2SQL：LLM 生成 SQL → 安全校验 → 执行（带重试）。

    返回 (最终执行的 SQL, 查询结果)。成功时 SQL 有值、第二个元素是数据列表；
    安全拦截 / 超时 / 执行失败时 SQL 为 None，第二个元素是给用户的错误提示字符串。
    """
    error_hint = ""
    for attempt in range(MAX_SQL_RETRIES + 1):
        raw_sql = await _generate_sql(question, llm, error_hint)
        logger.info(f"NL2SQL SQL (attempt {attempt + 1}): {raw_sql}")

        valid, validated = _validate_sql(raw_sql)
        if not valid:
            logger.warning(f"SQL 安全校验失败: {validated}")
            return None, f"查询被安全策略拦截：{validated}。请换一种方式提问。"

        try:
            query_result = await asyncio.wait_for(
                db.execute(text(validated)),
                timeout=SQL_TIMEOUT_SECONDS,
            )
            rows = query_result.mappings().all()
            data = [dict(row) for row in rows[:100]]
            return validated, data
        except asyncio.TimeoutError:
            logger.warning(f"SQL 执行超时 ({SQL_TIMEOUT_SECONDS}s): {validated}")
            return None, f"查询执行超时（{SQL_TIMEOUT_SECONDS}秒），请简化查询条件后重试。"
        except Exception as e:
            error_hint = str(e)
            logger.warning(f"SQL 执行失败 (attempt {attempt + 1}): {e}")
            if attempt == MAX_SQL_RETRIES:
                return None, "数据查询执行失败，请尝试换一种方式提问。"

    # 正常情况下循环里一定会 return，这行只是给类型检查/意外情况兜底
    return None, "数据查询执行失败，请尝试换一种方式提问。"

async def search_sql(
    question: str,
    llm: BaseChatModel,
    db: AsyncSession,
) -> str:
    sql, data = await search_sql_raw(question, llm, db)
    if sql is None:
        # 安全拦截 / 超时 / 执行失败时，data 是给用户的错误提示字符串，直接返回
        if isinstance(data, str):
            return data
        return "数据查询执行失败，请尝试换一种方式提问。"

    result_str = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    # 增强生成回答
    prompt = SQL_QA_PROMPT.format(question=question, sql=sql, result=result_str)
    response = await llm.ainvoke([SystemMessage(content=prompt)])
    return response.content