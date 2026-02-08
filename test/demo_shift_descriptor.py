"""Small, commented demo that extracts a few embeddings and prints shift descriptors.

Usage:
    python -m shift_descriptor.demo_shift_descriptor
"""

from __future__ import annotations

import argparse
from typing import List, Sequence

import numpy as np

from .config import ModelSpec, default_model_specs
from .datasets import load_text_field
from .embeddings import EmbeddingConfig, EmbeddingExtractor, get_device
from .metrics import compute_stats, frechet_distance, mahalanobis_distance, sliced_wasserstein_distance


def _parse_args() -> argparse.Namespace:
    """CLI kept small on purpose so it runs quickly during exploration."""

    default_model = default_model_specs()[0]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-path",
        default="data/sft_spider_train_text2sql.json",
        help="JSON file used for the training split.",
    )
    parser.add_argument(
        "--dev-path",
        default="data/sft_spider_dev_text2sql.json",
        help="JSON file used for the dev split.",
    )
    parser.add_argument("--text-field", default="text", help="Field to embed from the JSON examples.")
    parser.add_argument("--train-samples", type=int, default=500, help="How many training examples to embed.")
    parser.add_argument("--dev-samples", type=int, default=800, help="How many dev examples to embed.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the embedding extractor.")
    parser.add_argument("--max-length", type=int, default=512, help="Max tokens per example.")
    parser.add_argument("--num-projections", type=int, default=128, help="Directions used by sliced Wasserstein.")
    parser.add_argument("--seed", type=int, default=13, help="RNG seed for subsampling examples.")
    parser.add_argument("--device", default=None, help="Torch device override (cpu, cuda, mps).")
    parser.add_argument("--model-id", default=default_model.model_id, help="HF model id for embeddings.")
    parser.add_argument(
        "--model-alias",
        default=default_model.alias or default_model.name,
        help="Friendly alias used when printing results.",
    )
    parser.add_argument("--revision", default=None, help="Optional HF revision/commit for the model.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for HF loads.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (<=0 disables).")
    return parser.parse_args()


def _subsample(texts: Sequence[str], limit: int, seed: int) -> List[str]:
    """Return up to `limit` texts without replacement (deterministic)."""

    if limit <= 0 or len(texts) <= limit:
        return list(texts)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=limit, replace=False)
    return [texts[i] for i in idx]


def _build_extractor(spec: ModelSpec, args: argparse.Namespace) -> EmbeddingExtractor:
    """Wrap EmbeddingConfig creation so the demo logic stays tidy."""

    cfg = EmbeddingConfig(
        model_id=args.model_id,
        alias=args.model_alias or spec.alias or spec.name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code or spec.trust_remote_code,
        revision=args.revision or spec.revision,
        lora_r=args.lora_r if args.lora_r > 0 else None,
    )
    return EmbeddingExtractor(cfg, device=get_device(args.device))


def _compute_descriptor(train_emb: np.ndarray, dev_emb: np.ndarray, *, num_projections: int, seed: int) -> dict:
    """Assemble the descriptor using existing metric helpers."""
    print("Computing shift descriptor metrics ...")
    print("Train embeddings shape:", train_emb.shape)
    print("Dev embeddings shape:  ", dev_emb.shape)
    stats_train = compute_stats(train_emb)
    stats_dev = compute_stats(dev_emb)
    frechet = frechet_distance(stats_train, stats_dev)
    mahala = mahalanobis_distance(stats_train, stats_dev)
    swd = sliced_wasserstein_distance(train_emb, dev_emb, num_projections=num_projections, seed=seed)
    return {
        "frechet_distance": frechet["frechet_distance"],
        "frechet_mean_shift": frechet["frechet_mean_shift"],
        "mahalanobis_distance": mahala,
        "swd_mean": swd["swd_mean"],
        "swd_std": swd["swd_std"],
        "swd_max": swd["swd_max"],
    }


def main() -> None:
    """Procedure:

    1) Load the training/dev JSON files and pull the requested text field.
    2) Deterministically subsample to keep the demo light.
    3) Build one EmbeddingExtractor and embed both splits.
    4) Run the shift descriptor metrics and print the numeric features.
    """

    args = _parse_args()
    model_spec = ModelSpec(args.model_id, alias=args.model_alias, revision=args.revision, trust_remote_code=args.trust_remote_code)

    print(f"Loading train/dev texts from {args.train_path} and {args.dev_path} ...")
    train_texts = load_text_field(args.train_path, field=args.text_field)
    dev_texts = load_text_field(args.dev_path, field=args.text_field)

    train_texts = _subsample(train_texts, args.train_samples, args.seed)
    dev_texts = _subsample(dev_texts, args.dev_samples, args.seed + 1)
    print(f"Using {len(train_texts)} train examples and {len(dev_texts)} dev examples.")

    extractor = _build_extractor(model_spec, args)
    try:
        print(f"Extracting embeddings with {args.model_alias} on device={extractor.device} ...")
        emb_train = extractor.encode(train_texts)
        emb_dev = extractor.encode(dev_texts)
    finally:
        extractor.shutdown()

    descriptor = _compute_descriptor(emb_train, emb_dev, num_projections=args.num_projections, seed=args.seed)

    print("\nShift descriptor (train -> dev):")
    for key, value in descriptor.items():
        print(f"  {key:24s}: {value:.6f}")


if __name__ == "__main__":
    main()
