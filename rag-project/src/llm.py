"""
llm.py
------
LLM factory.

Industry pattern: environment config loaded once at module import,
never inline.  Temperature and token budget are explicit constants so
they're easy to tune without hunting through business logic.
"""

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME  = "llama-3.3-70b-versatile"
TEMPERATURE = 0.2     # lower = more factual, less creative
MAX_TOKENS  = 2048    # raised from 1024 — complex synthesis needs room


def get_llm() -> ChatGroq:
    """Return a configured Groq LLM instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. "
            "Add it to your .env file:  GROQ_API_KEY=gsk_..."
        )

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        groq_api_key=api_key,
    )
    print(f"✅ Groq LLM loaded: {MODEL_NAME}")
    return llm