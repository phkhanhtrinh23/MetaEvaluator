"""Embedding extraction and caching for FusionSQL."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from shift_descriptor.config import ModelSpec
from shift_descriptor.embeddings import EmbeddingConfig, EmbeddingExtractor, get_device


def _build_extractor(model: ModelSpec, batch_size: int, max_length: int, lora_r: int | None, device: torch.device | None) -> EmbeddingExtractor:
    cfg = EmbeddingConfig(
        model_id=model.model_id,
        alias=model.alias or model.name,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=model.trust_remote_code,
        revision=model.revision,
        lora_r=lora_r,
    )
    return EmbeddingExtractor(cfg, device=device)


@dataclass
class EmbeddingCacheConfig:
    output_dir: Path
    batch_size: int = 4
    max_length: int = 512
    lora_r: int | None = 8
    device: str | None = None
    max_points_per_split: int | None = None
    subsample_seed: int = 13


class EmbeddingCache:
    """Lazy loader for per-model, per-split embeddings with on-disk caching."""

    def __init__(self, config: EmbeddingCacheConfig):
        self.cfg = config
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device(config.device)
        self._extractors: Dict[str, EmbeddingExtractor] = {}

    def _get_extractor(self, model: ModelSpec) -> EmbeddingExtractor:
        key = model.alias or model.name
        if key not in self._extractors:
            self._extractors[key] = _build_extractor(
                model,
                batch_size=self.cfg.batch_size,
                max_length=self.cfg.max_length,
                lora_r=self.cfg.lora_r,
                device=self.device,
            )
        return self._extractors[key]

    def _embedding_path(self, model: ModelSpec, split_name: str) -> Path:
        alias = model.alias or model.name
        sanitized = alias.replace("/", "-")
        return self.cfg.output_dir / f"{sanitized}_{split_name}_embeddings.npz"

    def _subsample(self, embeddings: np.ndarray) -> np.ndarray:
        if not self.cfg.max_points_per_split or embeddings.shape[0] <= self.cfg.max_points_per_split:
            return embeddings
        rng = np.random.default_rng(self.cfg.subsample_seed)
        idx = rng.choice(embeddings.shape[0], size=self.cfg.max_points_per_split, replace=False)
        return embeddings[idx]

    def load_or_compute(self, model: ModelSpec, split_name: str, texts: Sequence[str]) -> np.ndarray:
        path = self._embedding_path(model, split_name)
        if path.exists():
            data = np.load(path)
            return data["embeddings"]

        extractor = self._get_extractor(model)
        embeddings = extractor.encode(texts)
        embeddings = self._subsample(embeddings)
        np.savez_compressed(path, embeddings=embeddings)
        gc.collect()
        return embeddings

    def clear_extractors(self) -> None:
        self._extractors.clear()
        gc.collect()
