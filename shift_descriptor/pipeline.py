"""Command-line pipeline for computing shift descriptors using lightweight LLMs."""

from __future__ import annotations

import argparse
import json
from itertools import count
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import ModelSpec, default_model_specs
from .datasets import load_prompt_texts, load_text_field
from .embeddings import EmbeddingConfig, EmbeddingExtractor, get_device
from .prompts import default_template_path
from .metrics import (
    build_descriptor_matrix,
    compute_pairwise_similarities,
    compute_stats,
    frechet_distance,
    mahalanobis_distance,
    pca_project,
    sliced_wasserstein_distance,
    sanitize_embeddings,
)
from .visualize import plot_multi_model_scatter, plot_overlay_scatter, plot_pca


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="data/sft_spider_train_text2sql.json", help="Training JSON file.")
    parser.add_argument("--test-path", default="data/sft_spider_dev_text2sql.json", help="Testing JSON file.")
    parser.add_argument("--text-field", default="text", help="Primary field to embed when --use-plain-text is set.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per model.")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum token length.")
    parser.add_argument("--output-dir", default="outputs", help="Directory where artifacts will be saved.")
    parser.add_argument(
        "--model-ids",
        nargs="*",
        default=[
            "meta-llama/Llama-3.2-3B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "XGenerationLab/XiYanSQL-QwenCoder-3B-2502",
            "stabilityai/stablelm-2-1_6b",
            "allenai/OLMo-1B-hf",
            "internlm/internlm2-1_8b",
            # "state-spaces/mamba-2.8b-slimpj",
            # "RWKV/RWKV7-Goose-World3-1.5B-HF",
            "google/recurrentgemma-2b-it",
            # "openbmb/MiniCPM-2B-sft-bf16",
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            # "bigscience/bloomz-3b"
        ],
        help="Optional list of model ids. When omitted, the curated defaults are used.",
    )
    parser.add_argument("--num-projections", type=int, default=128, help="Number of directions for sliced Wasserstein.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed for projections.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g., cuda or cpu.")
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Path to the Text-to-SQL prompt template. Defaults to prompts/text2sql_prompt.tmpl.",
    )
    parser.add_argument(
        "--context-fields",
        nargs="*",
        default=("evidence", "matched_contents", "text"),
        help="Auxiliary fields to append under 'Additional context' in the prompt.",
    )
    parser.add_argument(
        "--use-plain-text",
        action="store_true",
        help="Disable prompt templating and embed the raw field referenced by --text-field instead.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="Rank for on-the-fly LoRA adapters (set <=0 to disable).",
    )
    parser.add_argument(
        "--skip-embedding-plots",
        action="store_true",
        help="Disable embedding scatter plot that overlays all models.",
    )
    parser.add_argument(
        "--scatter-max-points",
        type=int,
        default=2000,
        help="Maximum number of samples per split when drawing embedding scatter plots.",
    )
    parser.add_argument(
        "--scatter-seed",
        type=int,
        default=7,
        help="Random seed for subsampling points in embedding scatter plots.",
    )
    parser.add_argument(
        "--metric-max-points",
        type=int,
        default=4096,
        help="Optional cap on number of embeddings per split when computing shift metrics (helps reduce RAM usage).",
    )
    return parser.parse_args()


REMOTE_REQUIRED_MODELS = {
    "state-spaces/mamba-2.8b-slimpj",
    "RWKV/RWKV7-Goose-World3-1.5B-HF",
    "openbmb/MiniCPM-2B-sft-bf16",
    "internlm/internlm2-1_8b",
}


def _parse_model_entry(raw: str) -> ModelSpec:
    alias = None
    entry = raw
    if "=" in raw:
        alias, entry = raw.split("=", 1)
    trust_remote = False
    if entry.endswith(":remote"):
        entry = entry[: -len(":remote")]
        trust_remote = True
    entry = entry.strip()
    trust_remote = trust_remote or entry in REMOTE_REQUIRED_MODELS
    return ModelSpec(model_id=entry, alias=alias, trust_remote_code=trust_remote)


def resolve_models(args: argparse.Namespace) -> List[ModelSpec]:
    if args.model_ids:
        return [_parse_model_entry(raw) for raw in args.model_ids]
    return default_model_specs()


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = output_dir / "shift_descriptor_matrix.npy"
    summary_path = output_dir / "shift_descriptor_summary.json"
    pca_plot_path = output_dir / "shift_descriptor_pca.png"
    similarity_path = output_dir / "pairwise_similarity.json"

    previous_summary: Dict[str, Dict] = {}
    previous_metrics: Dict[str, Dict[str, float]] = {}
    previous_embeddings: Dict[str, Dict[str, str]] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            previous_summary = json.load(f)
        previous_metrics = previous_summary.get("metrics", {})
        previous_embeddings = previous_summary.get("embeddings", {})

    rng = np.random.default_rng(args.scatter_seed)
    metric_rng = np.random.default_rng(args.seed)
    metric_rng = np.random.default_rng(args.seed)

    def sample_for_scatter(arr: np.ndarray) -> np.ndarray:
        if args.scatter_max_points is None or arr.shape[0] <= args.scatter_max_points:
            return arr.copy()
        idx = rng.choice(arr.shape[0], size=args.scatter_max_points, replace=False)
        return arr[idx]

    def downsample_for_metrics(arr: np.ndarray) -> np.ndarray:
        if args.metric_max_points and arr.shape[0] > args.metric_max_points:
            idx = metric_rng.choice(arr.shape[0], size=args.metric_max_points, replace=False)
            return arr[idx]
        return arr

    print("Loading datasets...")
    if args.use_plain_text:
        train_texts = load_text_field(args.train_path, field=args.text_field)
        test_texts = load_text_field(args.test_path, field=args.text_field)
        template_used = None
    else:
        template_path = args.prompt_template or default_template_path()
        train_texts = load_prompt_texts(
            args.train_path,
            template_path=template_path,
            context_fields=args.context_fields,
        )
        test_texts = load_prompt_texts(
            args.test_path,
            template_path=template_path,
            context_fields=args.context_fields,
        )
        template_used = str(template_path)
    # train_texts = train_texts[:200]
    # test_texts = test_texts[:100]
    print(f"Train samples: {len(train_texts)}, Test samples: {len(test_texts)}")

    specs = resolve_models(args)
    device = get_device(args.device)
    print(f"Using device: {device}")

    metric_records: Dict[str, Dict[str, float]] = {}
    embedding_paths: Dict[str, Dict[str, str]] = {}
    scatter_data: Dict[str, Dict[str, np.ndarray]] = {}

    failures: Dict[str, str] = {}

    for spec in specs:
        print(f"\nProcessing model: {spec.model_id}")
        embed_cfg = EmbeddingConfig(
            model_id=spec.model_id,
            alias=spec.name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            trust_remote_code=spec.trust_remote_code,
            revision=spec.revision,
            lora_r=args.lora_r if args.lora_r and args.lora_r > 0 else None,
        )

        default_train_path = output_dir / f"{spec.name}_train_embeddings.npz"
        default_test_path = output_dir / f"{spec.name}_test_embeddings.npz"
        cached_embed = previous_embeddings.get(spec.name, {})
        train_emb_path = Path(cached_embed.get("train_embeddings", default_train_path))
        test_emb_path = Path(cached_embed.get("test_embeddings", default_test_path))
        cached_metric = previous_metrics.get(spec.name)

        use_cached_descriptor = cached_metric is not None and train_emb_path.exists() and test_emb_path.exists()

        try:
            if use_cached_descriptor:
                print(f"  Using cached metrics and embeddings for {spec.name}")
                if not args.skip_embedding_plots:
                    scatter_data.setdefault(spec.name, {})
                    train_embeddings = sanitize_embeddings(np.load(train_emb_path)["embeddings"]).astype(
                        np.float32, copy=False
                    )
                    test_embeddings = sanitize_embeddings(np.load(test_emb_path)["embeddings"]).astype(
                        np.float32, copy=False
                    )
                    scatter_data[spec.name]["train"] = sample_for_scatter(train_embeddings)
                    scatter_data[spec.name]["test"] = sample_for_scatter(test_embeddings)
                    del train_embeddings, test_embeddings

                metric_records[spec.name] = cached_metric
                embedding_paths[spec.name] = {
                    "train_embeddings": str(train_emb_path),
                    "test_embeddings": str(test_emb_path),
                }
                continue

            extractor = None
            missing_embeddings = (not train_emb_path.exists()) or (not test_emb_path.exists())
            if missing_embeddings:
                extractor = EmbeddingExtractor(embed_cfg, device=device)

            if train_emb_path.exists():
                print(f"  Found cached embeddings: {train_emb_path.name}")
                train_embeddings = np.load(train_emb_path)["embeddings"]
            else:
                train_embeddings = extractor.encode(tqdm(train_texts, desc=f"{spec.name} train", leave=False))
                extractor.save_embeddings(train_embeddings, train_emb_path)
                clear_cuda_cache()

            if test_emb_path.exists():
                print(f"  Found cached embeddings: {test_emb_path.name}")
                test_embeddings = np.load(test_emb_path)["embeddings"]
            else:
                test_embeddings = extractor.encode(tqdm(test_texts, desc=f"{spec.name} test", leave=False))
                extractor.save_embeddings(test_embeddings, test_emb_path)
                clear_cuda_cache()

            if extractor is not None:
                del extractor
                clear_cuda_cache()

            train_embeddings = sanitize_embeddings(train_embeddings).astype(np.float32, copy=False)
            test_embeddings = sanitize_embeddings(test_embeddings).astype(np.float32, copy=False)

            if not args.skip_embedding_plots:
                scatter_data.setdefault(spec.name, {})
                scatter_data[spec.name]["train"] = sample_for_scatter(train_embeddings)
                scatter_data[spec.name]["test"] = sample_for_scatter(test_embeddings)

            train_metric = downsample_for_metrics(train_embeddings)
            test_metric = downsample_for_metrics(test_embeddings)

            stats_train = compute_stats(train_metric)
            stats_test = compute_stats(test_metric)

            frechet = frechet_distance(stats_train, stats_test)
            maha = mahalanobis_distance(stats_train, stats_test)
            sw = sliced_wasserstein_distance(
                train_metric, test_metric, num_projections=args.num_projections, seed=args.seed
            )

            metric_records[spec.name] = {
                "frechet_distance": frechet,
                "mahalanobis_distance": maha,
                "sliced_wasserstein_distance": sw,
            }
            embedding_paths[spec.name] = {
                "train_embeddings": str(train_emb_path),
                "test_embeddings": str(test_emb_path),
            }
            del train_metric, test_metric, train_embeddings, test_embeddings
            clear_cuda_cache()
        except Exception as exc:  # pragma: no cover - defensive
            failures[spec.name] = str(exc)
            print(f"  ! Failed on model {spec.model_id}: {exc}")
            print("Error: ", exc)
            for path in (train_emb_path, test_emb_path):
                if path.exists():
                    path.unlink()
            metric_records.pop(spec.name, None)
            embedding_paths.pop(spec.name, None)
            scatter_data.pop(spec.name, None)
            continue

    metric_order = ["frechet_distance", "mahalanobis_distance", "sliced_wasserstein_distance"]
    descriptor_matrix = None
    matrix_written = None
    pca_written = None
    similarity_written = None
    pairwise_sims = {}

    if metric_records:
        descriptor_matrix = build_descriptor_matrix(metric_records, metric_order)
        np.save(matrix_path, descriptor_matrix)
        matrix_written = matrix_path

        labels = list(metric_records.keys())
        if len(labels) >= 2:
            pca_points = pca_project(descriptor_matrix, n_components=2)
            plot_pca(pca_points, labels, pca_plot_path)
            pca_written = pca_plot_path
        else:
            print("Skipping PCA projection: need at least two models.")

        pairwise_sims = compute_pairwise_similarities(descriptor_matrix, labels)
        with similarity_path.open("w", encoding="utf-8") as f:
            json.dump({f"{a}__{b}": sim for (a, b), sim in pairwise_sims.items()}, f, indent=2)
        similarity_written = similarity_path

    overlay_plot_path = None
    per_model_plot_path = None
    if not args.skip_embedding_plots and scatter_data:
        overlay_path = output_dir / "all_models_embedding_scatter.png"
        plot_overlay_scatter(scatter_data, overlay_path)
        overlay_plot_path = str(overlay_path)

        multi_scatter_path = output_dir / "per_model_embedding_scatter.png"
        plot_multi_model_scatter(scatter_data, multi_scatter_path)
        per_model_plot_path = str(multi_scatter_path)

    summary = {
        "metrics": metric_records,
        "metric_order": metric_order,
        "descriptor_matrix_path": str(matrix_written) if matrix_written else None,
        "pca_plot_path": str(pca_written) if pca_written else None,
        "pairwise_similarity_path": str(similarity_written) if similarity_written else None,
        "embeddings": embedding_paths,
        "train_samples": len(train_texts),
        "test_samples": len(test_texts),
        "prompt_template": template_used,
        "global_embedding_scatter_path": overlay_plot_path,
        "per_model_embedding_scatter_path": per_model_plot_path,
        "failed_models": failures,
    }

    summary_path = output_dir / "shift_descriptor_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nShift descriptor computation completed.")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
