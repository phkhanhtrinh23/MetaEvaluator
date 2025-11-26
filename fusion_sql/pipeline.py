"""End-to-end pipeline to train and evaluate FusionSQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import gc
import numpy as np
import torch
from shift_descriptor.config import ModelSpec, default_model_specs
from tqdm.auto import tqdm

from .accuracy import ModelAccuracySource
from .data_utils import (
    PromptSplit,
    build_prompt_splits,
    load_prompted_texts,
    load_raw_dataset,
    maybe_cap_examples,
)
from .descriptors import DEFAULT_FEATURE_ORDER, compute_shift_descriptor
from .embedding_cache import EmbeddingCache, EmbeddingCacheConfig
from .evaluation import build_prediction_records
from .generation import GenerationSettings, SQLGenerator
from .meta_learning import FusionSQLMetaLearner, MetaLearningConfig, ShiftDescriptorTask
from .model import FusionSQL
from .optimal_transport import barycentric_mapping, sinkhorn_plan

def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _compute_descriptor_stats(tasks: Sequence[ShiftDescriptorTask]) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = []
    for task in tasks:
        tensors.append(task.support_descriptor)
        tensors.append(task.query_descriptor)
        if task.transfer_descriptor is not None:
            tensors.append(task.transfer_descriptor)
    all_desc = torch.cat(tensors, dim=0)
    mean = all_desc.mean(dim=0, keepdim=True)
    std = all_desc.std(dim=0, keepdim=True).clamp(min=1e-6)
    return mean, std


def _apply_descriptor_norm(tasks: Sequence[ShiftDescriptorTask], mean: torch.Tensor, std: torch.Tensor) -> None:
    for task in tasks:
        task.support_descriptor.sub_(mean).div_(std)
        task.query_descriptor.sub_(mean).div_(std)
        if task.transfer_descriptor is not None:
            task.transfer_descriptor.sub_(mean).div_(std)


def _parse_model_entry(raw: str) -> ModelSpec:
    alias = None
    entry = raw
    if "=" in raw:
        alias, entry = raw.split("=", 1)
    if entry.endswith(":remote"):
        entry = entry[: -len(":remote")]
    return ModelSpec(model_id=entry, alias=alias, trust_remote_code=True)


def parse_model_specs(raw_list: Sequence[str] | None) -> List[ModelSpec]:
    if raw_list:
        return [_parse_model_entry(entry) for entry in raw_list]
    return default_model_specs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default="data/sft_spider_train_text2sql.json", help="Path to training JSON.")
    parser.add_argument("--dev-path", default="data/sft_spider_dev_text2sql.json", help="Path to dev/test JSON.")
    parser.add_argument("--output-dir", default="outputs/fusionsql", help="Directory for FusionSQL artifacts.")
    parser.add_argument("--embedding-dir", default="outputs/fusionsql/embeddings", help="Embedding cache directory.")
    parser.add_argument("--model-ids", nargs="*", default=[
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

            # # DeepSeek Coder family
            "deepseek-ai/deepseek-coder-1.3b-base",
            "deepseek-ai/deepseek-coder-6.7b-base",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
        ], help="Optional HF model ids (alias=model_id or alias=model_id:remote).")
    parser.add_argument("--test-model-ids", 
                        nargs="*",
                        default=["meta-llama/Llama-3.2-3B-Instruct",
                                 "Qwen/Qwen3-0.6B",
                                 "Gensyn/Qwen2.5-0.5B-Instruct",
                                 "Qwen/Qwen2.5-0.5B-Instruct",
                                 "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                                 "Qwen/Qwen3-0.6B-Base",
                                 "deepseek-ai/deepseek-coder-1.3b-instruct",
                                 "Qwen/Qwen2-0.5B",
                                 "unsloth/Llama-3.2-1B-Instruct"
                                ], 
                        help="Extra model ids evaluated only after meta-training.")
    parser.add_argument("--ot-epsilon", type=float, default=0.1, help="Sinkhorn epsilon for OT mapping.")
    parser.add_argument("--ot-use-plan", type=bool, default=True, help="Use full Sinkhorn transport plan and barycentric label averaging.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per model during embedding extraction.")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length for embeddings.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank (<=0 disables).")
    parser.add_argument("--num-projections", type=int, default=128, help="Projection count for SWD.")
    parser.add_argument("--subsample-limit", type=int, default=0, help="Optional cap on embeddings per split before metrics.")
    parser.add_argument("--subsample-seed", type=int, default=13, help="Random seed for embedding subsampling.")
    parser.add_argument("--prompt-template", default=None, help="Path to prompt template (defaults to repo template).")
    parser.add_argument("--context-fields", nargs="*", default=("evidence", "matched_contents", "text"))
    parser.add_argument("--use-plain-text", action="store_true", help="Disable prompt templating.")
    parser.add_argument("--text-field", default="text", help="Field used when --use-plain-text is enabled.")
    parser.add_argument("--split-ratios", nargs=3, type=float, default=(0.6, 0.2, 0.2), help="Ratios for meta_train/meta_val/meta_test.")
    parser.add_argument("--split-seed", type=int, default=13, help="Seed for dataset split.")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional cap on training samples before splitting.")
    parser.add_argument("--max-dev-samples", type=int, default=0, help="Optional cap on dev samples.")
    parser.add_argument("--inner-lr", type=float, default=0.01, help="Inner-loop lr.")
    parser.add_argument("--outer-lr", type=float, default=1e-3, help="Outer-loop lr.")
    parser.add_argument("--inner-steps", type=int, default=5, help="Inner updates per task.")
    parser.add_argument("--epochs", type=int, default=500, help="Meta-training epochs.")
    parser.add_argument("--tasks-per-batch", type=int, default=4, help="Tasks per meta-batch.")
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument("--meta-val-key", default="meta_val", help="Accuracy key for meta validation split.")
    parser.add_argument("--meta-test-key", default="meta_test", help="Accuracy key for meta testing split.")
    parser.add_argument("--dev-key", default="dev", help="Accuracy key for real dev/test split.")
    parser.add_argument("--eval-inner-steps", type=int, default=5, help="Inner-loop steps at evaluation time.")
    parser.add_argument("--gen-max-new-tokens", type=int, default=256, help="Max new tokens for SQL generation.")
    parser.add_argument("--gen-temperature", type=float, default=0.0, help="Softmax temperature during decoding.")
    parser.add_argument("--gen-top-p", type=float, default=0.9, help="Nucleus sampling top-p.")
    parser.add_argument("--gen-repetition-penalty", type=float, default=1.0, help="Repetition penalty for decoding.")
    parser.add_argument("--save-meta-preds", action="store_true", help="Whether to persist meta-test predictions (default True).")
    parser.add_argument("--no-save-meta-preds", dest="save_meta_preds", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(save_meta_preds=True)
    parser.add_argument("--meta-reg-lambda", type=float, default=1e-4, help="L2 regularization on meta-parameters.")
    parser.add_argument("--meta-reg-beta", type=float, default=1e-3, help="KL-style meta regularizer on outer loss.")
    return parser.parse_args()


def build_tasks(
    models: Sequence[ModelSpec],
    splits: Mapping[str, PromptSplit],
    dev_split: PromptSplit,
    *,
    cache: EmbeddingCache,
    accuracy: ModelAccuracySource,
    meta_val_key: str,
    meta_test_key: str,
    dev_key: str,
    num_projections: int,
) -> List[ShiftDescriptorTask]:
    tasks: List[ShiftDescriptorTask] = []
    for model in tqdm(models, desc="Build tasks", leave=False):
        model_name = model.alias or model.model_id
        emb_meta_train = cache.load_or_compute(model, "meta_train", splits["meta_train"].prompts)
        emb_meta_val = cache.load_or_compute(model, "meta_val", splits["meta_val"].prompts)
        emb_meta_test = cache.load_or_compute(model, "meta_test", splits["meta_test"].prompts)
        emb_dev = cache.load_or_compute(model, dev_split.name, dev_split.prompts)

        support_desc = compute_shift_descriptor(
            model_name,
            "meta_train",
            emb_meta_train,
            "meta_val",
            emb_meta_val,
            num_projections=num_projections,
        )
        query_desc = compute_shift_descriptor(
            model_name,
            "meta_train",
            emb_meta_train,
            "meta_test",
            emb_meta_test,
            num_projections=num_projections,
        )
        transfer_desc = compute_shift_descriptor(
            model_name,
            "meta_train",
            emb_meta_train,
            dev_split.name,
            emb_dev,
            num_projections=num_projections,
        )

        support_label = accuracy.get(model_name, meta_val_key)
        query_label = accuracy.get(model_name, meta_test_key)
        transfer_label = accuracy.get(model_name, dev_key)

        task = ShiftDescriptorTask.from_descriptors(
            model_name=model_name,
            support=support_desc,
            support_label=support_label,
            query=query_desc,
            query_label=query_label,
            transfer=transfer_desc,
            transfer_label=transfer_label,
            device=cache.device,
            feature_order=DEFAULT_FEATURE_ORDER,
        )
        tasks.append(task)
        cache.clear_extractors()
    return tasks


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def sanitize_model_name(spec: ModelSpec) -> str:
    alias = spec.alias or spec.model_id
    return alias.replace("/", "-")


def create_dev_split(prompts: List[str], samples: List[dict]) -> PromptSplit:
    indices = list(range(len(prompts)))
    return PromptSplit(name="dev", prompts=prompts, indices=indices, samples=samples)


def evaluate_split_predictions(
    generator: SQLGenerator,
    split: PromptSplit,
    output_root: Path,
    model_name: str,
    *,
    db_root: Path | None = None,
) -> Dict[str, float]:
    preds = generator.generate(split.prompts)
    metrics, records = build_prediction_records(split.samples, preds, split.indices, db_root=db_root)
    payload = {
        "model": model_name,
        "split": split.name,
        "metrics": metrics,
        "sample_count": len(records),
        "predictions": records,
    }
    save_json(output_root / f"{split.name}.json", payload)
    return metrics


def maybe_load_cached_metrics(model_dir: Path, split_name: str, expected_samples: int) -> Dict[str, float] | None:
    path = model_dir / f"{split_name}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    records = payload.get("predictions")
    sample_count = payload.get("sample_count")
    if isinstance(records, list):
        count = len(records)
    elif isinstance(sample_count, int):
        count = sample_count
    else:
        return None
    if count != expected_samples:
        return None
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return metrics


def run_inference_for_models(
    models: Sequence[ModelSpec],
    splits: Mapping[str, PromptSplit],
    dev_split: PromptSplit,
    *,
    output_dir: Path,
    gen_settings: GenerationSettings,
    device: str | None = None,
    meta_val_key: str,
    meta_test_key: str,
    dev_key: str,
    lora_r: int | None = None,
    db_root: Path | None = None,
) -> Dict[str, Dict[str, float]]:
    predictions_root = output_dir / "predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)
    accuracy_summary: Dict[str, Dict[str, float]] = {}
    for model in models:
        model_name = model.alias or model.model_id
        print(f"[FusionSQL] Generating SQL for {model_name}...")
        model_dir = predictions_root / sanitize_model_name(model)
        model_dir.mkdir(exist_ok=True)
        model_accs: Dict[str, float] = {}
        split_plan = [
            (meta_val_key, splits["meta_val"]),
            (meta_test_key, splits["meta_test"]),
            (dev_key, dev_split),
        ]
        generator: SQLGenerator | None = None
        try:
            for key_name, split in split_plan:
                cached = maybe_load_cached_metrics(model_dir, split.name, len(split.indices))
                if cached is not None:
                    model_accs[key_name] = cached["execution_accuracy"]
                    continue
                if generator is None:
                    generator = SQLGenerator(model, gen_settings, device=device, lora_r=lora_r)
                metrics = evaluate_split_predictions(
                    generator,
                    split,
                    model_dir,
                    model_name,
                    db_root=db_root,
                )
                model_accs[key_name] = metrics["execution_accuracy"]
        finally:
            if generator is not None:
                generator.shutdown()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        accuracy_summary[model_name] = model_accs
    return accuracy_summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_raw_dataset(args.train_path)
    train_prompts = load_prompted_texts(
        args.train_path,
        template_path=args.prompt_template,
        context_fields=args.context_fields,
        use_plain_text=args.use_plain_text,
        text_field=args.text_field,
    )
    train_prompts, train_samples = maybe_cap_examples(train_prompts, train_samples, args.max_train_samples)
    
    train_prompts, train_samples = train_prompts[:1000], train_samples[:1000]

    splits = build_prompt_splits(train_prompts, samples=train_samples, ratios=args.split_ratios, seed=args.split_seed)

    dev_samples = load_raw_dataset(args.dev_path)
    dev_prompts = load_prompted_texts(
        args.dev_path,
        template_path=args.prompt_template,
        context_fields=args.context_fields,
        use_plain_text=args.use_plain_text,
        text_field=args.text_field,
    )
    dev_prompts, dev_samples = maybe_cap_examples(dev_prompts, dev_samples, args.max_dev_samples)
    dev_prompts, dev_samples = dev_prompts[:200], dev_samples[:200]
    
    dev_split = create_dev_split(dev_prompts, dev_samples)

    train_models = parse_model_specs(args.model_ids)
    test_models = parse_model_specs(args.test_model_ids) if args.test_model_ids else []
    all_models = train_models + [m for m in test_models if m not in train_models]

    gen_settings = GenerationSettings(
        max_new_tokens=args.gen_max_new_tokens,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        repetition_penalty=args.gen_repetition_penalty,
    )
    accuracy_map = run_inference_for_models(
        all_models,
        splits,
        dev_split,
        output_dir=output_dir,
        gen_settings=gen_settings,
        device=args.device,
        meta_val_key=args.meta_val_key,
        meta_test_key=args.meta_test_key,
        dev_key=args.dev_key,
        lora_r=args.lora_r if args.lora_r and args.lora_r > 0 else None,
        db_root=Path("data/database"),
    )
    save_json(output_dir / "model_accuracies.json", accuracy_map)
    accuracy_source = ModelAccuracySource(accuracy_map=accuracy_map)

    cache_cfg = EmbeddingCacheConfig(
        output_dir=Path(args.embedding_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
        lora_r=args.lora_r if args.lora_r > 0 else None,
        device=args.device,
        max_points_per_split=args.subsample_limit or None,
        subsample_seed=args.subsample_seed,
    )
    cache = EmbeddingCache(cache_cfg)

    tasks = build_tasks(
        train_models,
        splits,
        dev_split,
        cache=cache,
        accuracy=accuracy_source,
        meta_val_key=args.meta_val_key,
        meta_test_key=args.meta_test_key,
        dev_key=args.dev_key,
        num_projections=args.num_projections,
    )
    norm_mean, norm_std = _compute_descriptor_stats(tasks)
    _apply_descriptor_norm(tasks, norm_mean, norm_std)
    clear_cuda_cache()
    if len(tasks) > 1:
        val_size = max(1, len(tasks) // 5)
        val_tasks = tasks[:val_size]
        train_tasks = tasks[val_size:]
    else:
        train_tasks = tasks
        val_tasks = tasks

    model = FusionSQL(input_dim=len(DEFAULT_FEATURE_ORDER))
    meta_cfg = MetaLearningConfig(
        inner_lr=args.inner_lr,
        outer_lr=args.outer_lr,
        inner_steps=args.inner_steps,
        tasks_per_batch=args.tasks_per_batch,
        num_epochs=args.epochs,
        device=args.device,
        eval_inner_steps=args.eval_inner_steps,
        meta_reg_lambda=args.meta_reg_lambda,
        meta_reg_beta=args.meta_reg_beta,
    )
    meta_learner = FusionSQLMetaLearner(model, meta_cfg)

    print(f"[FusionSQL] Training on {len(train_tasks)} tasks for {args.epochs} epochs.")
    checkpoint_path = output_dir / "fusionsql_model.pt"
    history = meta_learner.meta_train(train_tasks, val_tasks=val_tasks, checkpoint_path=checkpoint_path)

    meta_results, meta_mae = meta_learner.evaluate(tasks, return_mae=True)
    print(f"[FusionSQL] Meta-test MAE: {meta_mae:.4f}")

    transfer_results = meta_learner.evaluate_transfer(tasks)
    transfer_mae = float(np.mean([entry["mae"] for entry in transfer_results])) if transfer_results else None
    if transfer_mae is not None:
        print(f"[FusionSQL] Real-test MAE: {transfer_mae:.4f}")

    metrics = {
        "meta_test_mae": meta_mae,
        "real_test_mae": transfer_mae,
        "epochs": args.epochs,
        "history": history,
        "num_tasks": len(tasks),
    }

    save_json(output_dir / "fusionsql_metrics.json", metrics)
    if args.save_meta_preds:
        save_json(output_dir / "fusionsql_meta_predictions.json", {"results": meta_results})
    save_json(output_dir / "fusionsql_dev_predictions.json", {"results": transfer_results})
    if test_models:
        print(f"[FusionSQL] Adapting to {len(test_models)} held-out model(s).")
        test_tasks = build_tasks(
            test_models,
            splits,
            dev_split,
            cache=cache,
            accuracy=accuracy_source,
            meta_val_key=args.meta_val_key,
            meta_test_key=args.meta_test_key,
            dev_key=args.dev_key,
            num_projections=args.num_projections,
        )
        _apply_descriptor_norm(test_tasks, norm_mean, norm_std)
        if args.ot_use_plan:
            print("[FusionSQL] Using OT mapping for held-out models.")
            support_refs = np.stack([task.support_descriptor.squeeze(0).cpu().numpy() for task in tasks])
            support_labels = np.array([task.support_label.squeeze().item() for task in tasks])
            support_model_names = [task.model_name for task in tasks]
            dev_refs = np.stack([task.transfer_descriptor.squeeze(0).cpu().numpy() for task in tasks if task.transfer_descriptor is not None])
            dev_labels = np.array([task.transfer_label.squeeze().item() for task in tasks if task.transfer_label is not None])
            dev_ref_models = [task.model_name for task in tasks if task.transfer_descriptor is not None]

            # Compute a single Sinkhorn transport plan from reference descriptors to the held-out descriptors
            support_targets = np.stack([task.support_descriptor.squeeze(0).cpu().numpy() for task in test_tasks])
            gamma_meta = sinkhorn_plan(support_refs, support_targets, reg=args.ot_epsilon, num_iter=200)
            
            """
            support_labels is a 1‑D array of length n_s (one label per source descriptor). barycentric_mapping expects Xt to be a 2‑D array with shape (n_s, d), so the matrix multiply gamma @ Xt works and stays 2‑D. 
            By reshaping to support_labels[:, None] we make it (n_s, 1), which lets gamma_meta.T (shape (n_t, n_s)) produce an (n_t, 1) column of predicted labels and keeps the subsequent division by row_sums well‑shaped. 
            Passing the 1‑D array directly would either error or broadcast incorrectly when divided by row_sums.
            """
            meta_pred = barycentric_mapping(gamma_meta.T, support_labels[:, None]).squeeze(-1)
            
            meta_top = np.argmax(gamma_meta, axis=0)
            true_meta = np.array([float(task.support_label.squeeze().item()) for task in test_tasks])

            ot_meta = []
            for idx, task in enumerate(test_tasks):
                ref_idx = int(meta_top[idx])
                ref_distance = float(np.linalg.norm(support_targets[idx] - support_refs[ref_idx]))
                ot_meta.append(
                    {
                        "model": task.model_name,
                        "predicted_accuracy": float(meta_pred[idx]),
                        "true_accuracy": true_meta[idx],
                        "ref_index": ref_idx,
                        "ref_model": support_model_names[ref_idx],
                        "distance": ref_distance,
                    }
                )
            ot_meta_mae = float(np.mean(np.abs(meta_pred - true_meta)))

            ot_dev = []
            dev_targets = [
                (
                    idx,
                    task.model_name,
                    task.transfer_descriptor.squeeze(0).cpu().numpy(),
                    float(task.transfer_label.squeeze().item()),
                )
                for idx, task in enumerate(test_tasks)
                if task.transfer_descriptor is not None and task.transfer_label is not None and len(dev_refs) > 0
            ]
            if dev_targets:
                dev_target_stack = np.stack([entry[2] for entry in dev_targets])
                gamma_dev = sinkhorn_plan(dev_refs, dev_target_stack, reg=args.ot_epsilon, num_iter=200)
                dev_pred = barycentric_mapping(gamma_dev.T, dev_labels[:, None]).squeeze(-1)
                dev_top = np.argmax(gamma_dev, axis=0)
                true_dev = np.array([entry[3] for entry in dev_targets])

                for j, entry in enumerate(dev_targets):
                    _, model_name, _, true_label = entry
                    ref_idx = int(dev_top[j])
                    ref_distance = float(np.linalg.norm(dev_target_stack[j] - dev_refs[ref_idx]))
                    ot_dev.append(
                        {
                            "model": model_name,
                            "predicted_accuracy": float(dev_pred[j]),
                            "true_accuracy": true_label,
                            "ref_index": ref_idx,
                            "ref_model": dev_ref_models[ref_idx],
                            "distance": ref_distance,
                        }
                    )
                ot_dev_mae = float(np.mean(np.abs(dev_pred - true_dev)))
            else:
                ot_dev_mae = None

            save_json(output_dir / "fusionsql_test_meta_predictions_ot.json", {"results": ot_meta, "mae": ot_meta_mae})
            save_json(output_dir / "fusionsql_test_dev_predictions_ot.json", {"results": ot_dev, "mae": ot_dev_mae})
        else:
            test_meta = meta_learner.evaluate(test_tasks)
            test_dev = meta_learner.evaluate_transfer(test_tasks)
            print(f"[FusionSQL] Held-out Meta-test MAE: {float(np.mean([entry['mae'] for entry in test_meta])):.4f}")
            print(f"[FusionSQL] Held-out Real-test MAE: {float(np.mean([entry['mae'] for entry in test_dev])):.4f}")
            save_json(output_dir / "fusionsql_test_meta_predictions.json", {"results": test_meta})
            save_json(output_dir / "fusionsql_test_dev_predictions.json", {"results": test_dev})


if __name__ == "__main__":
    main()
