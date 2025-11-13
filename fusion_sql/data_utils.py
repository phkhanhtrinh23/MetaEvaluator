"""Data loading and splitting helpers for FusionSQL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from shift_descriptor.datasets import load_prompt_texts, load_text_field
from shift_descriptor.prompts import default_template_path


SplitMap = Mapping[str, Sequence[int]]


@dataclass(frozen=True)
class PromptSplit:
    """Holds prompted texts and the indices they originated from."""

    name: str
    prompts: List[str]
    indices: Sequence[int]
    samples: List[dict]


def load_prompted_texts(
    json_path: str | Path,
    *,
    template_path: str | Path | None = None,
    context_fields: Iterable[str] | None = ("evidence", "matched_contents", "text"),
    use_plain_text: bool = False,
    text_field: str = "text",
) -> List[str]:
    """Load Text-to-SQL prompts, optionally applying the repo's template."""

    if use_plain_text:
        return load_text_field(json_path, field=text_field)
    return load_prompt_texts(
        json_path,
        template_path=template_path or default_template_path(),
        context_fields=context_fields,
    )


def _compute_split_indices(num_items: int, ratios: Sequence[float], seed: int) -> Dict[str, np.ndarray]:
    if len(ratios) != 3:
        raise ValueError("Expected three ratios for meta_train/meta_val/meta_test.")
    if not np.isclose(sum(ratios), 1.0, atol=1e-6):
        raise ValueError(f"Split ratios should sum to 1.0, got {ratios}.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_items)

    boundaries = np.cumsum(ratios) * num_items
    boundaries = boundaries.astype(int)
    meta_train_idx = perm[: boundaries[0]]
    meta_val_idx = perm[boundaries[0] : boundaries[1]]
    meta_test_idx = perm[boundaries[1] :]

    return {
        "meta_train": meta_train_idx,
        "meta_val": meta_val_idx,
        "meta_test": meta_test_idx,
    }


def build_prompt_splits(
    prompts: Sequence[str],
    samples: Sequence[dict],
    *,
    ratios: Sequence[float] = (0.6, 0.2, 0.2),
    seed: int = 13,
) -> Dict[str, PromptSplit]:
    """Split prompted texts into meta-train/meta-val/meta-test subsets."""

    if len(prompts) != len(samples):
        raise ValueError("Prompts and samples must have matching lengths.")
    indices = _compute_split_indices(len(prompts), ratios, seed)
    split_map: Dict[str, PromptSplit] = {}
    for split_name, split_idx in indices.items():
        split_prompts = [prompts[i] for i in split_idx]
        split_samples = [samples[i] for i in split_idx]
        split_map[split_name] = PromptSplit(name=split_name, prompts=split_prompts, indices=split_idx, samples=split_samples)
    return split_map


def load_raw_dataset(json_path: str | Path) -> List[dict]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {json_path}.")
    return [dict(item) for item in payload]


def maybe_cap_examples(prompts: List[str], samples: List[dict], cap: int) -> tuple[List[str], List[dict]]:
    if cap and cap < len(prompts):
        return prompts[:cap], samples[:cap]
    return prompts, samples
