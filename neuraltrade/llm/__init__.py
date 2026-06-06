"""Multi-model LLM routing system with intelligent provider selection."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"


class TaskType(str, Enum):
    QUICK = "quick"           # Fast, cheap: classification, extraction
    STANDARD = "standard"     # Balanced: analysis, reasoning
    COMPLEX = "complex"       # Best model: final decisions, risk assessment


@dataclass
class ModelInfo:
    """Information about a specific model."""
    provider: Provider
    model_id: str
    display_name: str
    max_tokens: int = 128000
    supports_tools: bool = True
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    speed: str = "medium"  # fast, medium, slow
    quality: int = 3  # 1-5 scale


# ── Model Catalog ──
MODEL_CATALOG: dict[str, ModelInfo] = {
    # OpenAI
    "openai/gpt-4o": ModelInfo(Provider.OPENAI, "gpt-4o", "GPT-4o", 128000, True, 2.50, 10.00, "medium", 5),
    "openai/gpt-4o-mini": ModelInfo(Provider.OPENAI, "gpt-4o-mini", "GPT-4o Mini", 128000, True, 0.15, 0.60, "fast", 4),
    "openai/gpt-4-turbo": ModelInfo(Provider.OPENAI, "gpt-4-turbo", "GPT-4 Turbo", 128000, True, 10.00, 30.00, "medium", 5),
    "openai/o1": ModelInfo(Provider.OPENAI, "o1", "o1", 200000, True, 15.00, 60.00, "slow", 5),
    "openai/o3-mini": ModelInfo(Provider.OPENAI, "o3-mini", "o3-mini", 200000, True, 1.10, 4.40, "medium", 5),

    # Anthropic
    "anthropic/claude-sonnet-4": ModelInfo(Provider.ANTHROPIC, "claude-sonnet-4-20250514", "Claude Sonnet 4", 200000, True, 3.00, 15.00, "medium", 5),
    "anthropic/claude-haiku-3.5": ModelInfo(Provider.ANTHROPIC, "claude-3-5-haiku-20241022", "Claude Haiku 3.5", 200000, True, 0.80, 4.00, "fast", 4),
    "anthropic/claude-opus-4": ModelInfo(Provider.ANTHROPIC, "claude-opus-4-20250514", "Claude Opus 4", 200000, True, 15.00, 75.00, "slow", 5),

    # Google
    "google/gemini-2.0-flash": ModelInfo(Provider.GOOGLE, "gemini-2.0-flash", "Gemini 2.0 Flash", 1000000, True, 0.10, 0.40, "fast", 4),
    "google/gemini-2.0-flash-lite": ModelInfo(Provider.GOOGLE, "gemini-2.0-flash-lite", "Gemini 2.0 Flash Lite", 1000000, True, 0.075, 0.30, "fast", 3),
    "google/gemini-1.5-pro": ModelInfo(Provider.GOOGLE, "gemini-1.5-pro", "Gemini 1.5 Pro", 2000000, True, 1.25, 5.00, "medium", 5),

    # OpenRouter (access many models through one API)
    "openrouter/auto": ModelInfo(Provider.OPENROUTER, "auto", "OpenRouter Auto", 128000, True, 0.0, 0.0, "medium", 4),
    "openrouter/anthropic/claude-sonnet-4": ModelInfo(Provider.OPENROUTER, "anthropic/claude-sonnet-4", "Claude Sonnet 4 (OR)", 200000, True, 3.00, 15.00, "medium", 5),
    "openrouter/openai/gpt-4o": ModelInfo(Provider.OPENROUTER, "openai/gpt-4o", "GPT-4o (OR)", 128000, True, 2.50, 10.00, "medium", 5),

    # Groq (ultra-fast inference)
    "groq/llama-3.3-70b": ModelInfo(Provider.GROQ, "llama-3.3-70b-versatile", "Llama 3.3 70B", 128000, True, 0.59, 0.79, "fast", 4),
    "groq/llama-3.1-8b": ModelInfo(Provider.GROQ, "llama-3.1-8b-instant", "Llama 3.1 8B", 128000, True, 0.05, 0.08, "fast", 3),

    # DeepSeek
    "deepseek/deepseek-chat": ModelInfo(Provider.DEEPSEEK, "deepseek-chat", "DeepSeek V3", 128000, True, 0.27, 1.10, "fast", 4),
    "deepseek/deepseek-reasoner": ModelInfo(Provider.DEEPSEEK, "deepseek-reasoner", "DeepSeek R1", 128000, True, 0.55, 2.19, "medium", 5),
}


def get_model_info(model_key: str) -> ModelInfo | None:
    """Get model info from catalog."""
    return MODEL_CATALOG.get(model_key)


def list_available_models(has_keys: dict[str, bool] | None = None) -> list[ModelInfo]:
    """List all available models, optionally filtered by available API keys."""
    models = list(MODEL_CATALOG.values())
    if has_keys:
        models = [m for m in models if has_keys.get(m.provider.value, False)]
    return models


def recommend_model(task: TaskType, has_keys: dict[str, bool] | None = None) -> str:
    """Recommend the best model for a given task type."""
    available = list_available_models(has_keys)
    if not available:
        return "openai/gpt-4o-mini"

    if task == TaskType.QUICK:
        # Fastest, cheapest
        candidates = [m for m in available if m.speed == "fast"]
        if candidates:
            return f"{candidates[0].provider.value}/{candidates[0].model_id}"
    elif task == TaskType.COMPLEX:
        # Highest quality
        candidates = sorted(available, key=lambda m: m.quality, reverse=True)
        if candidates:
            return f"{candidates[0].provider.value}/{candidates[0].model_id}"

    # STANDARD: balanced
    candidates = [m for m in available if m.quality >= 4]
    if candidates:
        return f"{candidates[0].provider.value}/{candidates[0].model_id}"

    return f"{available[0].provider.value}/{available[0].model_id}"


class LLMRouter:
    """Intelligent LLM router that selects the best model for each task."""

    def __init__(self, config):
        self.config = config
        self._clients: dict[str, Any] = {}

    def _get_client(self, provider: str):
        """Get or create LLM client for a provider."""
        if provider in self._clients:
            return self._clients[provider]

        if provider == "openai":
            from .providers.openai_client import OpenAIClient
            client = OpenAIClient(self.config.openai_key)
        elif provider == "anthropic":
            from .providers.anthropic_client import AnthropicClient
            client = AnthropicClient(self.config.anthropic_key)
        elif provider == "google":
            from .providers.google_client import GoogleClient
            client = GoogleClient(self.config.google_key)
        elif provider == "openrouter":
            from .providers.openrouter_client import OpenRouterClient
            client = OpenRouterClient(self.config.openrouter_key)
        elif provider == "groq":
            from .providers.groq_client import GroqClient
            client = GroqClient(self.config.groq_key)
        elif provider == "deepseek":
            from .providers.deepseek_client import DeepSeekClient
            client = DeepSeekClient(self.config.deepseek_key)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self._clients[provider] = client
        return client

    def route(self, model_key: str | None = None, task: TaskType = TaskType.STANDARD) -> tuple:
        """Route to the appropriate model and client."""
        if not model_key:
            has_keys = {
                "openai": bool(self.config.openai_key),
                "anthropic": bool(self.config.anthropic_key),
                "google": bool(self.config.google_key),
                "openrouter": bool(self.config.openrouter_key),
                "groq": bool(self.config.groq_key),
                "deepseek": bool(self.config.deepseek_key),
            }
            model_key = recommend_model(task, has_keys)

        parts = model_key.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid model key format: {model_key}. Use 'provider/model'")

        provider, model_id = parts
        client = self._get_client(provider)
        return client, model_id

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        task: TaskType = TaskType.STANDARD,
        **kwargs,
    ) -> str:
        """Send a chat completion request with automatic routing."""
        client, model_id = self.route(model, task)
        return await client.chat(messages, model_id, **kwargs)
