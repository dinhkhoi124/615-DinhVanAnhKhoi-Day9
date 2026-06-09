"""Shared LLM factory for all agents.

Uses OpenRouter as an OpenAI-compatible API, so any provider's model
can be selected via the OPENROUTER_MODEL env var.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at OpenAI or OpenRouter."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    
    if openai_api_key and not openai_api_key.startswith("your_key_here") and openai_api_key != "xxx":
        model = os.getenv("OPENROUTER_MODEL", "gpt-4o")
        if "/" in model or ":" in model:
            model = "gpt-4o"
        return ChatOpenAI(
            model=model,
            openai_api_key=openai_api_key,
            temperature=0.3,
        )
    else:
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4-5"),
            openai_api_key=openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.3,
        )