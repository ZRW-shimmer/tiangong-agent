# 读取环境变量
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "tiangong-agent"
    APP_ENV: str = "dev"
    APP_DEBUG: bool = True

    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "medical"
    DB_PASSWORD: str = "medical123"
    DB_NAME: str = "medical_db"

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_DB: int = 0

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "knowledge-docs"
    MINIO_SECURE: bool = False

    # Milvus
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530

    # Neo4j
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "medical123"

    # 模型
    DASHSCOPE_API_KEY: str = ""
       # 聊天模型
    BASE_URL_CHAT: str = ""
    DEEPSEEK_API_KEY: str = ""
    CHAT_MODEL: str = "deepseek-chat"
    EMBEDDING_MODEL: str = "text-embedding-v3"
    # Embedding 后端：local=本地 BGE，dashscope=阿里云
    EMBEDDING_BACKEND: str = "dashscope"
    LOCAL_EMBEDDING_MODEL: str = "models/bge-large-zh-v1.5"
    LOCAL_EMBEDDING_DEVICE: str = "cpu"
    LOCAL_EMBEDDING_BATCH_SIZE: int = 16
    VL_MODEL: str = "qwen-vl"

    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: str = "logs"

    # RAG 评估（TruLens）
    TRULENS_DB_NAME: str = "trulens_eval"
    # LiteLLM 模型名：deepseek/deepseek-chat 表示走 DeepSeek 官方 API
    EVAL_JUDGE_MODEL: str = "deepseek/deepseek-chat"


    # MinerU 云端解析（mineru.net 精准解析 API v4）
    MINERU_API_BASE: str = "https://mineru.net"
    MINERU_API_TOKEN: str = ""
    MINERU_MODEL_VERSION: str = "vlm"   # vlm / pipeline / MinerU-HTML
    MINERU_LANGUAGE: str = "ch"
    MINERU_POLL_INTERVAL: int = 5       # 轮询间隔，秒
    MINERU_POLL_TIMEOUT: int = 900      # 总等待时间，秒
    MINERU_REQUEST_TIMEOUT: int = 120   # 单次 HTTP 超时，秒

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def TRULENS_DATABASE_URL(self) -> str:
        """TruLens 评估库连接串（与业务库隔离）。"""
        from sqlalchemy.engine import URL

        return URL.create(
            drivername="postgresql",
            username=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            database=self.TRULENS_DB_NAME,
        ).render_as_string(hide_password=False)

    # 指定环境变量文件
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

# 保存到内存缓存中。以后直接获取。
@lru_cache  # lru 把对象实例保存到内存中。这是一种单例的实现
def get_settings() -> Settings:
    return Settings()


def get_llm(temperature: float = 0.1):
    from langchain_deepseek import ChatDeepSeek
    settings = get_settings()
    return ChatDeepSeek(
        model=settings.CHAT_MODEL,  # "deepseek-chat"
        api_key=settings.DEEPSEEK_API_KEY,  # 从 .env 读
        temperature=temperature,
    )

