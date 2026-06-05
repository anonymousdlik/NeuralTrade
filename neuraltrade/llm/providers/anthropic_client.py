"""Anthropic LLM client."""

from __future__ import annotations

from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client."""

    def __init__(self, api_key: str):
        super().__init__(api_key)
        try:
            from anthropic import AsyncAnthropic
            self.client = AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    async def chat(self, messages: list[dict], model_id: str, temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> str:
        # Convert messages format for Anthropic
        system_msg = ""
        chat_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                chat_messages.append(m)

        response = await self.client.messages.create(
            model=model_id, max_tokens=max_tokens, temperature=temperature,
            system=system_msg, messages=chat_messages, **kwargs
        )
        return response.content[0].text if response.content else ""

    def supports_tools(self, model_id: str) -> bool:
        return True
