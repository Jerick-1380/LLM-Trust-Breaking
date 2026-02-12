"""
Factory for creating the appropriate LLM client based on model name.

Automatically routes:
- GPT models (gpt-*) → OpenAI API
- O-series models (o1*, o3*) → OpenAI API
- Claude models (claude-*, anthropic/*) → OpenRouter API
- Everything else → OpenRouter API
"""

from typing import Tuple
from src.llm.providers.openai_client import OpenAIClient
from src.llm.providers.openrouter_client import OpenRouterClient
from src.llm.providers.parallel_openai import ParallelLLMClient
from src.llm.providers.parallel_openrouter_async import AsyncOpenRouterParallelClient


def should_use_openai(model: str) -> bool:
    """
    Determine if a model should use OpenAI API (True) or OpenRouter (False).

    Args:
        model: Model identifier (e.g., "gpt-4o", "anthropic/claude-3.5-sonnet")

    Returns:
        True if should use OpenAI API, False if should use OpenRouter

    Examples:
        >>> should_use_openai("gpt-4o")
        True
        >>> should_use_openai("gpt-4o-mini")
        True
        >>> should_use_openai("o1-preview")
        True
        >>> should_use_openai("anthropic/claude-3.5-sonnet")
        False
        >>> should_use_openai("claude-3.5-sonnet")
        False
        >>> should_use_openai("openai/gpt-4o")
        False  # OpenRouter prefix, use OpenRouter
    """
    model_lower = model.lower()

    # If model has openrouter prefix (provider/model), use OpenRouter
    if '/' in model:
        return False

    # GPT models use OpenAI
    if model_lower.startswith('gpt-'):
        return True

    # O-series models (o1, o3) use OpenAI
    if model_lower.startswith('o1') or model_lower.startswith('o3'):
        return True

    # Everything else (Claude, Gemini, Llama, etc.) uses OpenRouter
    return False


def normalize_model_name(model: str, use_openai: bool) -> str:
    """
    Normalize model name for the appropriate API.

    Args:
        model: Model identifier
        use_openai: Whether using OpenAI API

    Returns:
        Normalized model name

    Examples:
        >>> normalize_model_name("gpt-4o", True)
        "gpt-4o"
        >>> normalize_model_name("anthropic/claude-3.5-sonnet", False)
        "anthropic/claude-3.5-sonnet"
        >>> normalize_model_name("claude-3.5-sonnet", False)
        "anthropic/claude-3.5-sonnet"
    """
    # If using OpenRouter and model doesn't have provider prefix, add it
    if not use_openai and '/' not in model:
        model_lower = model.lower()

        # Claude models need anthropic/ prefix
        if 'claude' in model_lower:
            return f"anthropic/{model}"

        # GPT models need openai/ prefix (if someone wants to use them via OpenRouter)
        if model_lower.startswith('gpt-') or model_lower.startswith('o1') or model_lower.startswith('o3'):
            return f"openai/{model}"

        # For other models, return as-is and let OpenRouter handle it
        return model

    return model


def create_llm_clients(
    model: str,
    openai_api_key: str,
    openrouter_api_key: str,
    temperature: float = 1.0,
    max_workers: int = 20
) -> Tuple[object, object, bool]:
    """
    Create appropriate LLM clients based on model name.

    Automatically determines whether to use OpenAI or OpenRouter based on model name.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-3.5-sonnet")
        openai_api_key: OpenAI API key
        openrouter_api_key: OpenRouter API key
        temperature: Temperature for generation
        max_workers: Max parallel workers for batch requests

    Returns:
        Tuple of (llm_client, parallel_client, used_openai: bool)

    Examples:
        >>> client, parallel, is_openai = create_llm_clients("gpt-4o", key1, key2)
        # Uses OpenAI API

        >>> client, parallel, is_openai = create_llm_clients("claude-3.5-sonnet", key1, key2)
        # Uses OpenRouter API with anthropic/claude-3.5-sonnet
    """
    use_openai = should_use_openai(model)
    normalized_model = normalize_model_name(model, use_openai)

    if use_openai:
        print(f"🔵 Using OpenAI API with model: {normalized_model}")
        llm_client = OpenAIClient(
            api_key=openai_api_key,
            model=normalized_model,
            temperature=temperature
        )
        parallel_client = ParallelLLMClient(
            api_key=openai_api_key,
            max_workers=max_workers
        )
    else:
        print(f"🟣 Using OpenRouter API with model: {normalized_model}")
        # Use longer timeout for reasoning models (Qwen, DeepSeek R1, etc.)
        is_reasoning_model = ('qwen' in normalized_model.lower() or
                             'deepseek-r1' in normalized_model.lower() or
                             'r1' in normalized_model.lower())
        timeout = 300 if is_reasoning_model else 120

        llm_client = OpenRouterClient(
            api_key=openrouter_api_key,
            model=normalized_model,
            temperature=temperature,
            timeout=timeout
        )
        parallel_client = AsyncOpenRouterParallelClient(
            api_key=openrouter_api_key,
            max_concurrent=max_workers,  # Use max_concurrent instead of max_workers
            timeout=timeout
        )

    return llm_client, parallel_client, use_openai
