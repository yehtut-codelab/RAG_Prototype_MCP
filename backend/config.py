from pydantic_settings import BaseSettings
from functools import cached_property
from pathlib import Path

_ENV_FILE = Path(__file__).parent.parent / ".env"


class Settings(BaseSettings):
    postgres_url: str = "postgresql://postgres:Password123@localhost:5432/postgres"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "rag_documents"
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    mcp_enabled: bool = True

    @cached_property
    def postgres_sqlalchemy_url(self) -> str:
        url = self.postgres_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        return url

    model_config = {"env_file": _ENV_FILE, "env_file_encoding": "utf-8"}


settings = Settings()
