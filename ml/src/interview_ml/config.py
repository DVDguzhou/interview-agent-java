"""运行配置：统一从环境变量读取，变量名与后端 .env 保持一致。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    dashscope_api_key: str = field(default_factory=lambda: _env("DASHSCOPE_API_KEY"))
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen-plus"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "text-embedding-v3"))
    embedding_dim: int = field(default_factory=lambda: int(_env("EMBEDDING_DIM", "1024")))

    pg_dsn: str = field(
        default_factory=lambda: _env("PG_DSN", "postgresql+psycopg://interview:interview@localhost:5432/interview_agent")
    )
    milvus_uri: str = field(default_factory=lambda: _env("MILVUS_URI", "http://localhost:19530"))
    milvus_collection: str = field(default_factory=lambda: _env("MILVUS_COLLECTION", "interview_questions"))

    # 评分模型（QLoRA 微调产物）
    scorer_base_model: str = field(default_factory=lambda: _env("SCORER_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    scorer_adapter_path: str = field(default_factory=lambda: _env("SCORER_ADAPTER_PATH"))
    scorer_load_in_4bit: bool = field(default_factory=lambda: _env("SCORER_LOAD_IN_4BIT", "true").lower() == "true")


def get_settings() -> Settings:
    return Settings()
