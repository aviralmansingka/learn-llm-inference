from dataclasses import dataclass
from functools import cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_name: str = "Qwen/Qwen3-0.6B"

    model_config: SettingsConfigDict = SettingsConfigDict(yaml_file="config.yaml", frozen=True)


@cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
