import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    load_dotenv(ROOT_DIR / ".env")


def get_api_key(explicit_api_key: str = "") -> str:
    _load_env()
    api_key = (explicit_api_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not configured. Set it in Comfyui_gpt_2.0/.env or pass api_key directly."
        )
    return api_key
