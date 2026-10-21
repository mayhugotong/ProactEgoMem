from __future__ import annotations
from typing import Callable, Dict, Type
from proact.models.llm.base import LLM, LLMConfig
from proact.models.llm.providers import anthropic, gemini, openai, openrouter, transformers_vl, vertex, vllm_local
from proact.core.registry import MODEL_REGISTRY


def register_llm(provider_name: str):
    def deco(cls: Type[LLM]):
        MODEL_REGISTRY[provider_name] = cls
        return cls
    return deco

def create_llm(**cfg_kwargs) -> LLM:
    cls = MODEL_REGISTRY.get(cfg_kwargs["provider"])
    if not cls:
        raise ValueError(f"Unknown LLM provider '{cfg_kwargs['provider']}'. Registered: {list(MODEL_REGISTRY)}")
    return cls(LLMConfig(**cfg_kwargs))
