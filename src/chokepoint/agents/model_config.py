"""
ToolShield-Bench: Model Configuration

Provides a unified interface to instantiate LLM backends across providers
(Anthropic, OpenAI, Google, Ollama) using LangChain's ChatModel abstraction.
"""

import os

from langchain_core.language_models.chat_models import BaseChatModel

# ─── Supported Model Registry ───────────────────────────────────────────────
# Maps human-friendly labels to (provider, api_identifier) tuples.

MODEL_REGISTRY = {
    # ── Tier 1: Frontier Models (Cloud API) ──────────────────────────────
    "claude-opus-5":      ("anthropic", "claude-opus-5"),
    "claude-fable-5":     ("anthropic", "claude-fable-5"),
    "claude-sonnet-5":    ("anthropic", "claude-sonnet-5"),
    "gpt-5.6-sol":        ("openai",    "gpt-5.6-sol"),
    "gpt-5.6-terra":      ("openai",    "gpt-5.6-terra"),
    "gemini-3.7-flash":   ("google",    "gemini-3.7-flash"),

    # ── Tier 2: Local SLMs (Ollama) ──────────────────────────────────────
    "gemma4:12b":          ("ollama",   "gemma4:12b"),
    "qwen2.5-coder:7b":   ("ollama",   "qwen2.5-coder:7b"),
    "llama3.1:8b":         ("ollama",   "llama3.1:8b"),
}


#: Models that reject an explicit ``temperature`` argument (reasoning-tier
#: models on both OpenAI and Anthropic). Matched as substrings of the API id.
_TEMPERATURE_REJECTING = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "gpt-5.6",
)


def _rejects_temperature(api_id: str) -> bool:
    return any(marker in api_id for marker in _TEMPERATURE_REJECTING)


def get_llm(
    model_id: str,
    temperature: float = 0.0,
    base_url: str | None = None,
) -> BaseChatModel:
    """
    Instantiate a LangChain ChatModel for the given model identifier.

    Args:
        model_id: A key from MODEL_REGISTRY or a raw provider:model string.
        temperature: Sampling temperature (default 0 for deterministic eval).
        base_url: Override base URL (useful for custom Ollama hosts).

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If the model_id is not recognized and cannot be parsed.
    """
    # Resolve from registry
    if model_id in MODEL_REGISTRY:
        provider, api_id = MODEL_REGISTRY[model_id]
    elif ":" in model_id and model_id.split(":")[0] in ("anthropic", "openai", "google", "ollama"):
        # Allow raw "provider:model" format, e.g. "ollama:phi4-mini"
        provider, api_id = model_id.split(":", 1)
    else:
        # Assume it's an Ollama model tag if not recognized
        provider, api_id = "ollama", model_id

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs: dict = {"model": api_id, "max_tokens": 4096}
        # Newer Anthropic models reject an explicit temperature. Only send it to
        # models that still accept it; the evaluation is deterministic-by-intent
        # either way (these models default to greedy-ish decoding).
        if not _rejects_temperature(api_id):
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs)

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {"model": api_id}
        if _rejects_temperature(api_id):
            # Reasoning models reject temperature and take reasoning_effort as a
            # first-class argument rather than in model_kwargs. These models use
            # "none" to disable extended reasoning for low-latency evaluation.
            kwargs["reasoning_effort"] = "none"
        else:
            kwargs["temperature"] = temperature
        return ChatOpenAI(**kwargs)

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=api_id,
            temperature=temperature,
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        # A client-side request timeout so a hung local generation aborts the
        # HTTP call itself. The evaluation runner's thread-pool timeout cannot
        # kill a blocked thread, so without this a single stuck Ollama call
        # stalls the whole sweep.
        return ChatOllama(
            model=api_id,
            temperature=temperature,
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            client_kwargs={"timeout": 90},
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}' for model '{model_id}'. "
            f"Supported providers: anthropic, openai, google, ollama"
        )


def list_available_models() -> dict:
    """Return the full model registry for display purposes."""
    result = {"frontier": {}, "local": {}}
    for model_id, (provider, api_id) in MODEL_REGISTRY.items():
        tier = "local" if provider == "ollama" else "frontier"
        result[tier][model_id] = {"provider": provider, "api_id": api_id}
    return result
