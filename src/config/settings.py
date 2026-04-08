"""Configuration settings loaded from environment variables."""
from __future__ import annotations
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API CONFIGURATION - Loaded from .env file
# ============================================================================

# Model / temperature defaults (used when no explicit model is passed)
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
TEMPERATURE = float(os.getenv('TEMPERATURE', '1.0'))

# OpenAI Configuration (for GPT models: gpt-*, o1*, o3*)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# OpenRouter Configuration (for non-OpenAI models: Claude, Llama, etc.)
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')

# Reasoning Model Configuration (for o1/o3/gpt-5 models)
# Valid values: "low", "medium", "high"
# Higher effort = more reasoning tokens = potentially better quality but slower/more expensive
REASONING_EFFORT = os.getenv('REASONING_EFFORT', 'low')

# ============================================================================
# GAME CONFIGURATION
# ============================================================================

NUM_AGENTS = int(os.getenv('NUM_AGENTS', '5'))

# ============================================================================
# VALIDATION
# ============================================================================

if not OPENAI_API_KEY and not OPENROUTER_API_KEY:
    raise ValueError(
        "Neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set. "
        "Please create a .env file with at least one API key (see .env.example)"
    )
