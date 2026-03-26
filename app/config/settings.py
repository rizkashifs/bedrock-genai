"""
Centralised application settings loaded from environment variables / .env file.
All modules import from here — never hardcode values elsewhere.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ────────────────────────────────────────────────────────
    app_name: str = "Bedrock GenAI RAG"
    environment: str = Field(default="local", description="local | staging | production")
    debug: bool = False

    # ── Storage paths ──────────────────────────────────────────────────────
    base_dir: Path = Path(__file__).resolve().parents[2]
    uploads_dir: Path = base_dir / "app" / "uploads"
    logs_dir: Path = base_dir / "logs"

    # ── AWS / Bedrock ──────────────────────────────────────────────────────
    aws_access_key_id: Optional[str] = Field(default=None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(default="us-east-2", env="AWS_REGION")

    # Chat / generation model
    bedrock_model_id: str = Field(
        default="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        env="MODEL_ID",
        description="Bedrock model ID for Claude invocations",
    )

    # Titan embeddings model (used for RAG vector store)
    titan_embed_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        description="Bedrock Titan Embeddings model ID",
    )

    # ── Model capability limits ────────────────────────────────────────────
    # Derived from model type; used by ingestion and CSV truncation logic
    @property
    def model_type(self) -> str:
        return "sonnet" if "sonnet" in self.bedrock_model_id else "haiku"

    @property
    def max_tokens(self) -> int:
        return 200_000 if self.model_type == "sonnet" else 4_096

    @property
    def max_csv_rows(self) -> Optional[int]:
        """None means no limit (Sonnet). Haiku capped at 300."""
        return None if self.model_type == "sonnet" else 300

    # ── S3 / DynamoDB (optional persistence) ──────────────────────────────
    s3_bucket: Optional[str] = Field(default=None, env="S3_BUCKET")
    dynamo_table: Optional[str] = Field(default=None, env="DYNAMO_TABLE")

    # ── Retrieval ──────────────────────────────────────────────────────────
    max_chunks_per_query: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    s = Settings()  # type: ignore[call-arg]
    s.uploads_dir.mkdir(parents=True, exist_ok=True)
    s.logs_dir.mkdir(parents=True, exist_ok=True)
    return s


# Module-level singleton — import this everywhere
settings = get_settings()
