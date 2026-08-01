import os
from dotenv import load_dotenv
from functools import lru_cache
from langchain_openai import ChatOpenAI



DEFAULT_MODEL = "gpt-4o"
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL)

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")


DEFAULT_TEMPERATURE = 0




def _get_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _get_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))

# @lru_cache
# def get_llm() -> ChatOpenAI:
#     """Reads MODEL and TEMPERATURE from the environment, falling back
#     to defaults if unset."""
#     model = os.getenv("MODEL", DEFAULT_MODEL)
#     temperature = float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE))
#     return ChatOpenAI(model=model, temperature=temperature)
    
@lru_cache
def get_llm(model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
    resolved_model = model or os.getenv("MODEL", DEFAULT_MODEL)
    resolved_temperature = temperature if temperature is not None else float(os.getenv("TEMPERATURE", DEFAULT_TEMPERATURE))
    return ChatOpenAI(model=resolved_model, temperature=resolved_temperature)





load_dotenv()
PORT = _get_int("PORT", "8000")
TEMPERATURE = _get_float("TEMPERATURE", "0.0")