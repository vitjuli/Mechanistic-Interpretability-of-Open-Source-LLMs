# Behavior prompts module
from .behavior_prompts import (
    BEHAVIOR_PROMPTS,
    FACTUAL_RECALL_PROMPTS,
    REASONING_PROMPTS,
    CODE_GENERATION_PROMPTS,
    MULTILINGUAL_PROMPTS,
    get_prompts,
    get_all_prompts,
)

__all__ = [
    "BEHAVIOR_PROMPTS",
    "FACTUAL_RECALL_PROMPTS",
    "REASONING_PROMPTS",
    "CODE_GENERATION_PROMPTS",
    "MULTILINGUAL_PROMPTS",
    "get_prompts",
    "get_all_prompts",
]
