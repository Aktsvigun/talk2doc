from typing import Any

from .constants import DEFAULT_LLM_KWARGS


def validate_llm_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    for kwarg, default_value in DEFAULT_LLM_KWARGS.items():
        if kwarg not in kwargs:
            kwargs[kwarg] = default_value
    return kwargs
