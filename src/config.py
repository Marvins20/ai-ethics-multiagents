from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_ethics"
    # Sync URL used by Alembic migrations (psycopg3 driver)
    database_url_sync: str = "postgresql+psycopg://postgres:postgres@localhost:5432/ai_ethics"

    # Redis / ARQ
    redis_url: str = "redis://localhost:6379"

    # JWT
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 1 day

    # Google Gemini
    google_api_key: str = ""

    # Anthropic Claude
    claude_api_key: str = ""

    # LangSmith (optional tracing)
    langchain_tracing_v2: bool = False
    langchain_project: str = "ai-ethics-multiagents"
    langsmith_api_key: str = ""

    # Embedding model
    embedding_model_name: str = "models/gemini-embedding-001"

    # Static API key required from the frontend on every request.
    # Leave empty to disable the check (local dev).
    frontend_api_key: str = ""


settings = Settings()
