from __future__ import annotations
from typing import Protocol


class LLMClient(Protocol):
    model: str

    def complete_json(self, system: str, user: str) -> str:
        """Return the raw JSON string produced by the model."""
        ...
