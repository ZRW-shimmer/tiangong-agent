"""TruLens 会话与 Judge LLM 配置。

评估结果写入独立的 PostgreSQL 数据库 `trulens_eval`，与业务库隔离。
Judge LLM 默认使用 DeepSeek（LiteLLM 的 deepseek/deepseek-chat）。
"""

from __future__ import annotations

import os
from functools import lru_cache

from loguru import logger

from src.core.config import get_settings

# LiteLLM 默认会在线拉取价格表，国内网络容易超时；使用本地备份即可
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# TruLens 会提示 jsonschema 不在它自己的依赖清单里，项目已安装，忽略即可
import logging  # noqa: E402

logging.getLogger("trulens.core.utils.imports").setLevel(logging.ERROR)

# 提前导入，避免评估器后台线程与主线程同时 import 造成 module lock 死锁
from trulens.core import TruSession  # noqa: E402
from trulens.providers.litellm import LiteLLM  # noqa: E402

import litellm  # noqa: E402

litellm.suppress_debug_info = True
# Judge 请求卡住时尽快失败，避免整个实验看起来“假死”
litellm.request_timeout = 30
litellm.num_retries = 0

settings = get_settings()


@lru_cache(maxsize=1)
def get_trulens_session():
    """初始化 TruLens 会话（单例）。"""
    try:
        session = TruSession(database_url=settings.TRULENS_DATABASE_URL)
    except Exception as exc:  # pragma: no cover - 依赖外部数据库
        raise RuntimeError(
            "无法初始化 TruLens 评估库。请先创建数据库：\n"
            f'  docker exec tiangong-postgres psql -U {settings.DB_USER} -d postgres '
            f'-c "CREATE DATABASE {settings.TRULENS_DB_NAME};"\n'
            "然后重新运行评估脚本。"
        ) from exc

    logger.info("TruLens 评估库: {}", settings.TRULENS_DB_NAME)
    return session


class DeepSeekLiteLLM(LiteLLM):
    """兼容 DeepSeek 官方 API 的 LiteLLM Provider。

    TruLens 默认使用 OpenAI 的 json_schema structured output，
    DeepSeek 官方 API 当前只支持 json_object，因此这里丢弃 response_format，
    让 TruLens 从普通文本中解析 JSON 分数。
    """

    def _create_chat_completion(
        self,
        prompt=None,
        messages=None,
        response_format=None,
        **kwargs,
    ):
        return super()._create_chat_completion(
            prompt=prompt,
            messages=messages,
            response_format=None,
            **kwargs,
        )


def get_llm_provider():
    """评估用 LLM Provider（LiteLLM 对接 DeepSeek）。"""
    if not settings.DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置，无法调用 Judge LLM")

    # 通过环境变量传递密钥，避免被 TruLens 序列化进 completion_args
    os.environ.setdefault("DEEPSEEK_API_KEY", settings.DEEPSEEK_API_KEY)

    return DeepSeekLiteLLM(
        model_engine=settings.EVAL_JUDGE_MODEL,
        completion_kwargs={},
    )


def launch_dashboard(session=None, port: int = 8501):
    """启动 TruLens Streamlit Dashboard。"""
    import sys
    from pathlib import Path

    from trulens.dashboard import run_dashboard

    # run_dashboard 通过 subprocess 调用 `streamlit`，需要把当前环境 Scripts 放进 PATH
    exe_dir = Path(sys.executable).parent
    scripts_dir = exe_dir / ("Scripts" if os.name == "nt" else "bin")
    os.environ["PATH"] = str(scripts_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    s = session or get_trulens_session()
    try:
        return run_dashboard(session=s, port=port)
    except TypeError:
        # 某些 TruLens 版本的 run_dashboard 不接受 port，改用环境变量
        os.environ["STREAMLIT_SERVER_PORT"] = str(port)
        return run_dashboard(session=s)
