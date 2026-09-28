"""Groq client using the OpenAI SDK Responses API."""
from __future__ import annotations
from openai import OpenAI, OpenAIError

BASE_URL = "https://api.groq.com/openai/v1"


class LLMUnavailable(RuntimeError):
    pass


class GroqClient:
    def __init__(self, api_key: str, model: str, timeout: float = 30, retries: int = 3):
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=BASE_URL,
            timeout=timeout,
            max_retries=max(0, retries - 1),
        )

    def complete_json(self, system: str, user: str) -> str:
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=system,
                input=user,
                text={"format": {"type": "json_object"}},
                temperature=0,
            )
            return response.output_text
        except OpenAIError as exc:
            raise LLMUnavailable(f"Groq request failed: {exc}") from exc
