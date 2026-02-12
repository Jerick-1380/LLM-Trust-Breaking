"""Configuration settings loaded from environment variables."""
from __future__ import annotations
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# API CONFIGURATION - Loaded from .env file
# ============================================================================

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
TEMPERATURE = float(os.getenv('TEMPERATURE', '1.0'))

# OpenRouter Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')

# API Routing (AUTOMATIC - no need to manually change)
# The system automatically routes models to the correct API:
# - GPT models (gpt-*, o1*, o3*) → OpenAI API
# - Claude models (claude-*, anthropic/*) → OpenRouter API
# - Other models (Gemini, Llama, etc.) → OpenRouter API

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

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not set. "
        "Please create a .env file with your API keys (see .env.example)"
    )

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not set. "
        "Please create a .env file with your API keys (see .env.example)"
    )
