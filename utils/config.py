import os
from dotenv import load_dotenv

load_dotenv()


def _get(key, default=""):
    """Read from env vars first, then Streamlit secrets (for cloud deployment)."""
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


class Config:
    # groq (primary LLM provider)
    GROQ_API_KEY = _get("GROQ_API_KEY")
    GROQ_MODEL = _get("GROQ_MODEL", "openai/gpt-oss-120b")

    # ollama fallback
    OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    OLLAMA_MODEL = _get("OLLAMA_MODEL", "llama3.2")

    # storage
    CHROMA_PERSIST_DIR = _get("CHROMA_PERSIST_DIR", "./data/chroma_db")
    MEMORY_PERSIST_DIR = _get("MEMORY_PERSIST_DIR", "./data/memory")

    # retrieval
    MAX_RETRIEVAL_K = int(_get("MAX_RETRIEVAL_K", "5"))
    CONFIDENCE_THRESHOLD = float(_get("CONFIDENCE_THRESHOLD", "0.7"))

    @classmethod
    def validate(cls):
        problems = []
        if not cls.GROQ_API_KEY:
            problems.append("GROQ_API_KEY not set")
        return problems


# module-level aliases so other files can do: from utils.config import CHROMA_PERSIST_DIR
GROQ_API_KEY = Config.GROQ_API_KEY
GROQ_MODEL = Config.GROQ_MODEL
OLLAMA_BASE_URL = Config.OLLAMA_BASE_URL
OLLAMA_MODEL = Config.OLLAMA_MODEL
CHROMA_PERSIST_DIR = Config.CHROMA_PERSIST_DIR
MEMORY_PERSIST_DIR = Config.MEMORY_PERSIST_DIR
MAX_RETRIEVAL_K = Config.MAX_RETRIEVAL_K
CONFIDENCE_THRESHOLD = Config.CONFIDENCE_THRESHOLD