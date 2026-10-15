from __future__ import annotations

from typing import Any

from src.config.settings import settings, validate_runtime_configuration


try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover
    ChatGoogleGenerativeAI = None

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:  # pragma: no cover
    ChatAnthropic = None


def build_llm_client() -> Any:
    validate_runtime_configuration()
    provider = (settings.model_provider or "").lower()
    api_key = settings.api_key
    model_name = settings.model_name

    if provider == "google":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("langchain-google-genai is not installed. Install the required provider package.")
        return ChatGoogleGenerativeAI(model=model_name, api_key=api_key)

    if provider == "openai":
        if ChatOpenAI is None:
            raise RuntimeError("langchain-openai is not installed. Install the required provider package.")
        return ChatOpenAI(model=model_name, api_key=api_key)

    if provider == "anthropic":
        if ChatAnthropic is None:
            raise RuntimeError("langchain-anthropic is not installed. Install the required provider package.")
        return ChatAnthropic(model=model_name, api_key=api_key)

    raise RuntimeError(f"Unsupported MODEL_PROVIDER '{settings.model_provider}'. Supported values: google, openai, anthropic.")
