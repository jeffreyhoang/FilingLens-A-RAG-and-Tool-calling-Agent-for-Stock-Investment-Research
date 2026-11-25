from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    # Users running the app need only database_url (the read-only login) and
    # openai_api_key. The rest are for the data owner.
    database_url: str
    openai_api_key: str | None = None
    supabase_url: str | None = None  # not used by the app's code
    supabase_key: str | None = None
    anthropic_api_key: str | None = None
    sec_user_agent: str | None = None  # "Name email", required by SEC EDGAR


@lru_cache
def get_settings() -> Settings:
    return Settings()
