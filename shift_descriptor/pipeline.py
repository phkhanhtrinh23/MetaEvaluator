"""Command-line pipeline for computing shift descriptors using lightweight LLMs."""

from __future__ import annotations

import argparse
import json
from itertools import combinations, count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import ModelSpec, default_model_specs
from .datasets import load_prompt_texts, load_text_field
from .diagnostics import (
    build_normalization_view,
    compute_procrustes,
    match_dimensions,
    match_pair_samples,
    plot_cosine_histogram,
    plot_mean_shift_anisotropy,
    plot_pairwise_cosines,
    sample_cosine_hist,
    view_to_scatter_payload,
)
from .embeddings import EmbeddingConfig, EmbeddingExtractor, get_device
from .prompts import default_template_path
from .metrics import (
    build_descriptor_matrix,
    compute_pairwise_similarities,
    compute_stats,
    frechet_distance,
    linear_cka,
    mahalanobis_distance,
    pca_project,
    tail_drift,
    sliced_wasserstein_distance,
    sanitize_embeddings,
)
from .visualize import (
    plot_architecture_alignment,
    plot_cka_heatmap,
    plot_multi_model_scatter,
    plot_overlay_scatter,
    plot_pca,
)


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
            # Llama 3.2 family (Meta)
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B",

            # TinyLlama (Llama-compatible)
            "TinyLlama/TinyLlama_v1.1",
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",

            # QwenCoder (XiYanSQL is a finetune of Qwen/Qwen2.x Coder)
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            "Qwen/Qwen2.5-Coder-1.5B",
            "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "XGenerationLab/XiYanSQL-QwenCoder-3B-2502",

            # StableLM-2 (Stability AI)
            "stabilityai/stablelm-2-1_6b-chat",
            "stabilityai/stablelm-2-zephyr-1_6b",

            # OLMo (AI2)
            "allenai/OLMo-1B",
            "allenai/OLMo-1B-0724-hf",
            "allenai/OLMo-2-0425-1B-Instruct",

            # InternLM2 (Shanghai AI Lab)
            "internlm/internlm2-chat-1_8b",

            # # DeepSeek Coder family
            "deepseek-ai/deepseek-coder-1.3b-base",
            "deepseek-ai/deepseek-coder-6.7b-base",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
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
        default=0,
        help="Optional cap on number of embeddings per split when computing shift metrics (0 keeps all).",
    )
    parser.add_argument(
        "--cka-max-points",
        type=int,
        default=4096,
        help="Maximum number of normalized embeddings per model when computing representational similarity.",
    )
    parser.add_argument(
        "--diag-normalizations",
        nargs="*",
        default=("per_model", "global_l2", "global_zscore"),
        help="Normalization views to visualize (options: per_model, global_l2, global_zscore).",
    )
    parser.add_argument(
        "--cosine-hist-bins",
        type=int,
        default=40,
        help="Number of bins for cosine-distance histograms.",
    )
    parser.add_argument(
        "--cosine-hist-samples",
        type=int,
        default=2000,
        help="Max sampled pairs for cosine histograms (within/between).",
    )
    return parser.parse_args()


REMOTE_REQUIRED_MODELS = {
    "state-spaces/mamba-2.8b-slimpj",
    "RWKV/RWKV7-Goose-World3-1.5B-HF",
    "openbmb/MiniCPM-2B-sft-bf16",
    "internlm/internlm2-1_8b",
}

DEEPSEEK_67B_MODELS = {
    "deepseek-ai/deepseek-coder-6.7b-base",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
}

ARCH_FAMILY_EXPECTATIONS = {
    "transformer": "embeddings cluster in nearby subspace",
    "rwkv": "elongated manifold with temporal correlation",
    "recurrent": "elongated manifold with temporal correlation",
    "mamba": "smoother, lower-variance embedding spread",
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


def infer_arch_family(model_id: str) -> str:
    lower = model_id.lower()
    if "mamba" in lower or "state-spaces" in lower:
        return "mamba"
    if "rwkv" in lower:
        return "rwkv"
    if "recurrent" in lower or "gemma" in lower:
        return "recurrent"
    return "transformer"


def _flash_attention_available() -> bool:
    try:  # pragma: no cover - optional dependency
        import flash_attn  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _configure_specialized_loader(spec: ModelSpec, cfg: EmbeddingConfig) -> None:
    if spec.model_id not in DEEPSEEK_67B_MODELS:
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{spec.model_id} requires a CUDA device for 4-bit loading. Please choose a GPU-enabled environment."
        )
    try:  # pragma: no cover - optional dependency
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "bitsandbytes is required to load Deepseek 6.7B models in 4-bit mode. Install via `pip install bitsandbytes`."
        ) from exc

    cfg.dtype = torch.bfloat16
    cfg.tokenizer_kwargs.setdefault("use_fast", True)
    cfg.model_kwargs.setdefault(
        "quantization_config",
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    cfg.model_kwargs.setdefault("device_map", "auto")
    cfg.model_kwargs.setdefault("low_cpu_mem_usage", True)
    if _flash_attention_available():
        cfg.model_kwargs.setdefault("attn_implementation", "flash_attention_2")


def normalize_model_embeddings(train_arr: np.ndarray, test_arr: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    total_n = train_arr.shape[0] + test_arr.shape[0]
    total_sum = train_arr.sum(axis=0) + test_arr.sum(axis=0)
    total_sq = (train_arr ** 2).sum(axis=0) + (test_arr ** 2).sum(axis=0)
    mean = total_sum / total_n
    var = total_sq / total_n - mean**2
    var = np.maximum(var, eps)
    std = np.sqrt(var)
    return (train_arr - mean) / std, (test_arr - mean) / std


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
    previous_embeddings: Dict[str, Dict[str, str]] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            previous_summary = json.load(f)
        previous_embeddings = previous_summary.get("embeddings", {})

    rng = np.random.default_rng(args.scatter_seed)
    metric_rng = np.random.default_rng(args.seed)
    cka_rng = np.random.default_rng(args.seed + 73)

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

    def downsample_for_cka(arr: np.ndarray) -> np.ndarray:
        if args.cka_max_points and arr.shape[0] > args.cka_max_points:
            idx = cka_rng.choice(arr.shape[0], size=args.cka_max_points, replace=False)
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

    # Save first 3 samples for reference
    sample_output = {
        "train_samples": train_texts[:3],
        "test_samples": test_texts[:3],
    }
    sample_path = output_dir / "sample_inputs.json"
    with sample_path.open("w", encoding="utf-8") as f:
        json.dump(sample_output, f, indent=2)
    print(f"Sample inputs written to {sample_path}")

    specs = resolve_models(args)
    device = get_device(args.device)
    print(f"Using device: {device}")

    diag_dir = output_dir / "diagnostics"
    diag_dir_created = False

    def ensure_diag_dir() -> None:
        nonlocal diag_dir_created
        if not diag_dir_created:
            diag_dir.mkdir(parents=True, exist_ok=True)
            diag_dir_created = True

    metric_records: Dict[str, Dict[str, float]] = {}
    embedding_paths: Dict[str, Dict[str, str]] = {}
    scatter_data: Dict[str, Dict[str, np.ndarray]] = {}
    analysis_embeddings: Dict[str, Dict[str, Any]] = {}

    def register_analysis_embeddings(model_name: str, family: str, train_norm: np.ndarray, test_norm: np.ndarray) -> None:
        train_sample = downsample_for_cka(train_norm)
        test_sample = downsample_for_cka(test_norm)
        blocks = [arr for arr in (train_sample, test_sample) if arr.size > 0]
        if not blocks:
            return
        combined = np.vstack(blocks)
        analysis_embeddings[model_name] = {
            "combined": combined,
            "family": family,
        }

    def compute_shift_anisotropy(train_mat: np.ndarray, test_mat: np.ndarray) -> Tuple[float, float]:
        if train_mat.size == 0 or test_mat.size == 0:
            return 0.0, 1.0
        mean_train = train_mat.mean(axis=0)
        mean_test = test_mat.mean(axis=0)
        mean_shift = float(np.linalg.norm(mean_train - mean_test))
        cov_train = np.cov(train_mat, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov_train + np.eye(cov_train.shape[0]) * 1e-6)
        anisotropy = float(np.max(eigvals) / np.clip(np.min(eigvals), 1e-6, None))
        return mean_shift, anisotropy

    def summarize_architectural_alignment(cka_pairs: Dict[str, float], analysis_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        summary = {
            "within_family": {},
            "between_family": {},
            "expectations": ARCH_FAMILY_EXPECTATIONS,
        }

        def pair_key(a: str, b: str) -> str:
            return "__".join(sorted((a, b)))

        family_models: Dict[str, List[str]] = {}
        for model, info in analysis_map.items():
            family = info.get("family", "unknown")
            family_models.setdefault(family, []).append(model)

        for family, models in family_models.items():
            if len(models) < 2:
                continue
            vals = []
            for m1, m2 in combinations(models, 2):
                key = pair_key(m1, m2)
                if key in cka_pairs:
                    vals.append(cka_pairs[key])
            if vals:
                summary["within_family"][family] = {
                    "count": len(vals),
                    "mean_cka": float(np.mean(vals)),
                    "min_cka": float(np.min(vals)),
                    "max_cka": float(np.max(vals)),
                    "expectation": ARCH_FAMILY_EXPECTATIONS.get(family),
                }

        between_summary: Dict[str, Dict[str, float]] = {}
        families = sorted(family_models.keys())
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                fam_i, fam_j = families[i], families[j]
                vals = []
                for m_i in family_models[fam_i]:
                    for m_j in family_models[fam_j]:
                        key = pair_key(m_i, m_j)
                        if key in cka_pairs:
                            vals.append(cka_pairs[key])
                if vals:
                    between_summary[f"{fam_i}__{fam_j}"] = {
                        "count": len(vals),
                        "mean_cka": float(np.mean(vals)),
                        "min_cka": float(np.min(vals)),
                        "max_cka": float(np.max(vals)),
                    }
        summary["between_family"] = between_summary
        return summary


    failures: Dict[str, str] = {}

    for spec in specs:
        clear_cuda_cache()

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
        _configure_specialized_loader(spec, embed_cfg)

        default_train_path = output_dir / f"{spec.name}_train_embeddings.npz"
        default_test_path = output_dir / f"{spec.name}_test_embeddings.npz"
        cached_embed = previous_embeddings.get(spec.name, {})
        train_emb_path = Path(cached_embed.get("train_embeddings", default_train_path))
        test_emb_path = Path(cached_embed.get("test_embeddings", default_test_path))
        use_cached_descriptor = train_emb_path.exists() and test_emb_path.exists()

        extractor: EmbeddingExtractor | None = None
        try:
            try:
                if not (train_emb_path.exists() and test_emb_path.exists()):
                    extractor = EmbeddingExtractor(embed_cfg, device=device)
                elif use_cached_descriptor:
                    print(f"  Using cached embeddings for {spec.name}")

                if train_emb_path.exists():
                    train_embeddings = np.load(train_emb_path)["embeddings"]
                else:
                    train_embeddings = extractor.encode(tqdm(train_texts, desc=f"{spec.name} train", leave=False))
                    extractor.save_embeddings(train_embeddings, train_emb_path)
                    clear_cuda_cache()

                if test_emb_path.exists():
                    test_embeddings = np.load(test_emb_path)["embeddings"]
                else:
                    test_embeddings = extractor.encode(tqdm(test_texts, desc=f"{spec.name} test", leave=False))
                    extractor.save_embeddings(test_embeddings, test_emb_path)
                    clear_cuda_cache()
            finally:
                if extractor is not None:
                    extractor.shutdown()
                    extractor = None
                    clear_cuda_cache()

            train_embeddings = sanitize_embeddings(train_embeddings).astype(np.float32, copy=False)
            test_embeddings = sanitize_embeddings(test_embeddings).astype(np.float32, copy=False)

            train_norm, test_norm = normalize_model_embeddings(train_embeddings, test_embeddings)
            register_analysis_embeddings(spec.name, infer_arch_family(spec.model_id), train_norm, test_norm)

            if not args.skip_embedding_plots:
                scatter_data.setdefault(spec.name, {})
                scatter_data[spec.name]["train"] = sample_for_scatter(train_norm)
                scatter_data[spec.name]["test"] = sample_for_scatter(test_norm)

            train_metric = downsample_for_metrics(train_embeddings)
            test_metric = downsample_for_metrics(test_embeddings)

            stats_train = compute_stats(train_metric)
            stats_test = compute_stats(test_metric)

            frechet = frechet_distance(stats_train, stats_test)
            tail_metrics = tail_drift(stats_train, stats_test)
            maha = mahalanobis_distance(stats_train, stats_test)
            sw = sliced_wasserstein_distance(
                train_metric, test_metric, num_projections=args.num_projections, seed=args.seed
            )

            mean_shift, anisotropy = compute_shift_anisotropy(train_metric, test_metric)

            record = {}
            record.update(frechet)
            record.update(tail_metrics)
            record["mahalanobis_distance"] = maha
            record.update(sw)
            record["mean_shift_norm"] = mean_shift
            record["train_anisotropy"] = anisotropy

            metric_records[spec.name] = record
            embedding_paths[spec.name] = {
                "train_embeddings": str(train_emb_path),
                "test_embeddings": str(test_emb_path),
            }
            del train_metric, test_metric, train_embeddings, test_embeddings, train_norm, test_norm
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
            clear_cuda_cache()
            continue

    metric_order = [
        "frechet_distance",
        "frechet_mean_shift",
        "frechet_scale_mean",
        "frechet_scale_std",
        "tail_mean",
        "tail_std",
        "mahalanobis_distance",
        "swd_mean",
        "swd_std",
        "swd_max",
        "mean_shift_norm",
        "train_anisotropy",
    ]
    descriptor_matrix = None
    matrix_written = None
    pca_written = None
    similarity_written = None
    pairwise_sims = {}

    if metric_records:
        for record in metric_records.values():
            for key in metric_order:
                record.setdefault(key, 0.0)
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

    cka_path = None
    cka_heatmap_path = None
    architecture_path = None
    architecture_plot_path = None
    architecture_summary = {}
    if len(analysis_embeddings) >= 2:
        cka_pairs: Dict[str, float] = {}
        model_names = sorted(analysis_embeddings.keys())
        for a, b in combinations(model_names, 2):
            arr_a = analysis_embeddings[a]["combined"]
            arr_b = analysis_embeddings[b]["combined"]
            sample_a, sample_b = match_pair_samples(arr_a, arr_b, cka_rng)
            if sample_a is None:
                continue
            try:
                sample_a, sample_b = match_dimensions(sample_a, sample_b)
                cka_val = linear_cka(sample_a, sample_b)
            except ValueError:
                continue
            key = "__".join(sorted((a, b)))
            cka_pairs[key] = cka_val

        if cka_pairs:
            cka_path = output_dir / "cka_similarity.json"
            with cka_path.open("w", encoding="utf-8") as f:
                json.dump(cka_pairs, f, indent=2)

            labels = model_names
            size = len(labels)
            heatmap = np.ones((size, size), dtype=np.float32)
            for i in range(size):
                for j in range(size):
                    if i == j:
                        continue
                    pair_key = "__".join(sorted((labels[i], labels[j])))
                    heatmap[i, j] = cka_pairs.get(pair_key, 0.0)
            cka_heatmap_path = output_dir / "cka_heatmap.png"
            plot_cka_heatmap(heatmap, labels, cka_heatmap_path)

            architecture_summary = summarize_architectural_alignment(cka_pairs, analysis_embeddings)
            architecture_path = output_dir / "architecture_alignment.json"
            with architecture_path.open("w", encoding="utf-8") as f:
                json.dump(architecture_summary, f, indent=2)
            if architecture_summary.get("within_family") or architecture_summary.get("between_family"):
                architecture_plot_path = output_dir / "architecture_alignment.png"
                plot_architecture_alignment(architecture_summary, architecture_plot_path)

    diagnostics_summary: Dict[str, Dict[str, Any]] = {}
    mean_shift_plot_path = None
    if metric_records:
        ensure_diag_dir()
        mean_shift_plot_path = diag_dir / "mean_shift_anisotropy.png"
        plot_mean_shift_anisotropy(metric_records, mean_shift_plot_path)

    per_model_cosine_plot_path = None
    per_model_cosine_pairs_json = None

    if analysis_embeddings:
        for mode in args.diag_normalizations:
            try:
                view = build_normalization_view(analysis_embeddings, mode)
            except ValueError as exc:
                print(f"Skipping normalization mode {mode}: {exc}")
                continue
            if not view:
                continue
            ensure_diag_dir()
            scatter_payload = view_to_scatter_payload(view)
            overlay_norm_path = diag_dir / f"{mode}_overlay.png"
            plot_overlay_scatter(scatter_payload, overlay_norm_path)
            per_model_norm_path = diag_dir / f"{mode}_per_model.png"
            plot_multi_model_scatter(scatter_payload, per_model_norm_path)

            within_cos, between_cos, pair_cos = sample_cosine_hist(view, rng, args.cosine_hist_samples)
            cosine_hist_path = diag_dir / f"{mode}_cosine_hist.png"
            plot_cosine_histogram(within_cos, between_cos, mode, cosine_hist_path, args.cosine_hist_bins)

            cosine_pairs_path = diag_dir / f"{mode}_cosine_pairs.json"
            with cosine_pairs_path.open("w", encoding="utf-8") as f:
                json.dump(pair_cos, f, indent=2)

            # procrustes_scores = compute_procrustes(view, cka_rng)
            # procrustes_path = diag_dir / f"{mode}_procrustes.json"
            # with procrustes_path.open("w", encoding="utf-8") as f:
            #     json.dump(procrustes_scores, f, indent=2)

            # if mode == "per_model":
            #     per_model_cosine_plot_path = diag_dir / "per_model_cosine_hist.png"
            #     if not plot_pairwise_cosines(pair_cos, per_model_cosine_plot_path):
            #         per_model_cosine_plot_path = None
            #     per_model_cosine_pairs_json = cosine_pairs_path

            diagnostics_summary[mode] = {
                "overlay_path": str(overlay_norm_path),
                "per_model_path": str(per_model_norm_path),
                "cosine_hist_path": str(cosine_hist_path),
                "cosine_pairs_path": str(cosine_pairs_path),
                # "procrustes_path": str(procrustes_path),
                "within_cosine_pairs": int(len(within_cos)),
                "between_cosine_pairs": int(len(between_cos)),
            }

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
        "cka_similarity_path": str(cka_path) if cka_path else None,
        "cka_heatmap_path": str(cka_heatmap_path) if cka_heatmap_path else None,
        "architecture_alignment_path": str(architecture_path) if architecture_path else None,
        "architecture_alignment_plot_path": str(architecture_plot_path) if architecture_plot_path else None,
        "architecture_alignment_summary": architecture_summary,
        "mean_shift_anisotropy_plot_path": str(mean_shift_plot_path) if mean_shift_plot_path else None,
        "per_model_cosine_plot_path": str(per_model_cosine_plot_path) if per_model_cosine_plot_path else None,
        "per_model_cosine_pairs_path": str(per_model_cosine_pairs_json) if per_model_cosine_pairs_json else None,
        "diagnostics": diagnostics_summary,
        "failed_models": failures,
    }

    summary_path = output_dir / "shift_descriptor_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nShift descriptor computation completed.")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
