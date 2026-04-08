"""
Factory for creating LLM clients.

Routes model traffic to the appropriate provider:
- OpenAI models (gpt-*, o1*, o3*) → OpenAI API directly
- All other models (Claude, Llama, etc.) → OpenRouter

Model names without a provider prefix are automatically normalised
(e.g. "gpt-4o-mini" → "openai/gpt-4o-mini", "claude-3.5-sonnet" → "anthropic/claude-3.5-sonnet").
"""

from typing import Tuple
from src.llm.providers.openai_client import OpenAIClient
from src.llm.providers.openrouter_client import OpenRouterClient
from src.llm.providers.parallel_openrouter_async import AsyncOpenRouterParallelClient


def is_openai_model(model: str) -> bool:
    """
    Check if a model should use OpenAI API directly.

    Args:
        model: Model identifier (e.g., "gpt-4o-mini", "openai/gpt-5.2", "claude-3.5-sonnet")

    Returns:
        True if model should route to OpenAI, False if it should use OpenRouter

    Examples:
        >>> is_openai_model("gpt-4o-mini")
        True
        >>> is_openai_model("openai/gpt-5.2")
        True
        >>> is_openai_model("o1-mini")
        True
        >>> is_openai_model("claude-3.5-sonnet")
        False
        >>> is_openai_model("meta-llama/llama-3.3-70b")
        False
    """
    # Remove provider prefix if present
    base_model = model.split("/")[-1].lower()

    # Check if it's a GPT or o1/o3 model
    return (base_model.startswith("gpt-") or
            base_model.startswith("o1") or
            base_model.startswith("o3"))


def normalize_model_name(model: str) -> str:
    """
    Normalise a model name for API routing.

    If the name already contains a provider prefix (e.g. "openai/gpt-4o"),
    it is returned unchanged.  Otherwise a prefix is inferred:
      - gpt-*, o1*, o3*  → openai/<model>
      - claude-*         → anthropic/<model>
      - everything else  → returned as-is (OpenRouter will route it)

    Examples:
        >>> normalize_model_name("gpt-4o-mini")
        "openai/gpt-4o-mini"
        >>> normalize_model_name("openai/gpt-4o")
        "openai/gpt-4o"
        >>> normalize_model_name("claude-3.5-sonnet")
        "anthropic/claude-3.5-sonnet"
    """
    if "/" in model:
        return model  # already has provider prefix

    lower = model.lower()
    if lower.startswith("gpt-") or lower.startswith("o1") or lower.startswith("o3"):
        return f"openai/{model}"
    if "claude" in lower:
        return f"anthropic/{model}"
    return model


def create_llm_clients(
    model: str,
    openrouter_api_key: str = None,
    openai_api_key: str = None,
    temperature: float = 1.0,
    max_workers: int = 20,
) -> Tuple[object, object]:
    """
    Create LLM client and matching parallel client, routing to appropriate provider.

    Routes to:
    - OpenAI API for GPT models (gpt-*, o1*, o3*)
    - OpenRouter API for all other models (Claude, Llama, etc.)

    Args:
        model:              Model identifier (normalised automatically).
        openrouter_api_key: OpenRouter API key (required for non-OpenAI models).
        openai_api_key:     OpenAI API key (required for OpenAI models).
        temperature:        Sampling temperature.
        max_workers:        Max concurrent requests for the parallel client.

    Returns:
        (llm_client, parallel_client)
    """
    normalised = normalize_model_name(model)

    # Determine if this is an OpenAI model
    if is_openai_model(model):
        # Route to OpenAI API
        if not openai_api_key:
            raise ValueError(
                f"Model '{model}' requires OPENAI_API_KEY but it's not set. "
                "Please add OPENAI_API_KEY to your .env file."
            )

        # Extract base model name (remove openai/ prefix if present)
        base_model = normalised.split("/")[-1]

        print(f"Using OpenAI API with model: {base_model}")
        llm_client = OpenAIClient(
            api_key=openai_api_key,
            model=base_model,
            temperature=temperature,
            timeout=120,
        )
        # Note: Parallel client not used for OpenAI in current implementation
        # Will be handled by queued client in trial_runner.py
        parallel_client = None
        return llm_client, parallel_client

    else:
        # Route to OpenRouter API
        if not openrouter_api_key:
            raise ValueError(
                f"Model '{model}' requires OPENROUTER_API_KEY but it's not set. "
                "Please add OPENROUTER_API_KEY to your .env file."
            )

        is_reasoning = (
            "qwen" in normalised.lower()
            or "deepseek-r1" in normalised.lower()
            or "/r1" in normalised.lower()
        )
        timeout = 300 if is_reasoning else 120

        print(f"Using OpenRouter with model: {normalised}")
        llm_client = OpenRouterClient(
            api_key=openrouter_api_key,
            model=normalised,
            temperature=temperature,
            timeout=timeout,
        )
        parallel_client = AsyncOpenRouterParallelClient(
            api_key=openrouter_api_key,
            max_concurrent=max_workers,
            timeout=timeout,
        )
        return llm_client, parallel_client
