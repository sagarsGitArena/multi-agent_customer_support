import os
from dotenv import load_dotenv
from functools import lru_cache
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_MODEL = "gpt-4o"
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")


DEFAULT_TEMPERATURE = 0


def _get_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _get_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


@lru_cache
def get_llm(model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
    resolved_model = model or os.getenv("MODEL_NAME", DEFAULT_MODEL)
    resolved_temperature = temperature if temperature is not None else float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE))
    return ChatOpenAI(model=resolved_model, temperature=resolved_temperature, base_url=API_BASE_URL)


PORT = _get_int("PORT", "7860")
TEMPERATURE = _get_float("TEMPERATURE", "0.0")
