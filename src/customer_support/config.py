import os
from dotenv import load_dotenv

load_dotenv()


def _get_float(name: str, default: str) -> float:
    return float(os.getenv(name, default))


def _get_int(name: str, default: str) -> int:
    return int(os.getenv(name, default))


API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE = _get_float("TEMPERATURE", "0.0")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
PORT = _get_int("PORT", "8000")