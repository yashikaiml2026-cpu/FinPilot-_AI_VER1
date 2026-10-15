from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=False)


def read_env_value(key: str, default: str = "") -> str:
    value = os.getenv(key, default)
    return value.strip() if isinstance(value, str) else default


def sync_env_aliases() -> None:
    provider = read_env_value("MODEL_PROVIDER") or read_env_value("LLM_PROVIDER") or read_env_value("Gemini_PROVIDER") or "google"
    model = read_env_value("MODEL_NAME") or read_env_value("Gemini_MODEL") or read_env_value("GOOGLE_MODEL") or ""
    api_key = read_env_value("API_KEY") or read_env_value("Gemini_API_KEY") or read_env_value("GOOGLE_API_KEY") or ""

    os.environ["MODEL_PROVIDER"] = provider
    os.environ["MODEL_NAME"] = model
    os.environ["API_KEY"] = api_key

    if provider.lower() == "google" and not model:
        os.environ["MODEL_NAME"] = read_env_value("Gemini_MODEL")
    if provider.lower() == "google" and not api_key:
        os.environ["API_KEY"] = read_env_value("Gemini_API_KEY")


sync_env_aliases()


def validate_runtime_configuration() -> None:
    missing = []
    for key in ("MODEL_PROVIDER", "MODEL_NAME", "API_KEY"):
        if not read_env_value(key):
            missing.append(key)
    if missing:
        labels = ", ".join(missing)
        raise RuntimeError(
            "FinPilot configuration is incomplete. Please add the required values in the project .env file: "
            f"{labels}."
        )


class Settings:
    project_root = BASE_DIR
    data_dir = BASE_DIR / "data"
    knowledge_base_dir = BASE_DIR / "knowledge_base"
    db_path = BASE_DIR / "data" / "finpilot.db"
    chroma_path = BASE_DIR / "data" / "chroma"
    model_provider = read_env_value("MODEL_PROVIDER")
    model_name = read_env_value("MODEL_NAME")
    api_key = read_env_value("API_KEY")


settings = Settings()
