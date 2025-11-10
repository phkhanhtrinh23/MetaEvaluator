"""Configuration helpers for the shift descriptor pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ModelSpec:
    """Describes a Hugging Face model to be used for embedding extraction."""

    model_id: str
    alias: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False

    @property
    def name(self) -> str:
        if self.alias:
            return self.alias
        return self.model_id.split("/")[-1]


def default_model_specs() -> List[ModelSpec]:
    """Return a curated list of lightweight, readily accessible models."""

    return [
        ModelSpec(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            alias="tinyllama-1.1b-chat",
        ),
        ModelSpec(
            model_id="HuggingFaceH4/zephyr-3b-beta",
            alias="zephyr-3b-beta",
        ),
        ModelSpec(
            model_id="microsoft/Phi-3-mini-4k-instruct",
            alias="phi-3-mini-4k",
            trust_remote_code=True,
        ),
    ]
