"""Utility to migrate old output folder/file names to the new sanitized model_id-based naming."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


def sanitize(model_id: str) -> str:
    return model_id.replace("/", "-")


def find_candidate_dirs(root: Path) -> Iterable[Path]:
    for path in root.iterdir():
        if path.is_dir() and any(s in path.name for s in ("Llama-3.2", "Qwen", "deepseek", "TinyLlama", "stablelm", "XiYanSQL", "unsloth")):
            yield path


def rename_embeddings(embedding_dir: Path, old_name: str, new_name: str) -> None:
    for split in ("meta_train", "meta_val", "meta_test", "dev", "train", "test"):
        src = embedding_dir / f"{old_name}_{split}_embeddings.npz"
        dst = embedding_dir / f"{new_name}_{split}_embeddings.npz"
        if src.exists() and not dst.exists():
            print(f"Renaming {src} -> {dst}")
            src.rename(dst)


def rename_prediction_dirs(pred_root: Path, old_name: str, new_name: str) -> None:
    src_dir = pred_root / old_name
    dst_dir = pred_root / new_name
    if src_dir.exists() and not dst_dir.exists():
        print(f"Renaming folder {src_dir} -> {dst_dir}")
        src_dir.rename(dst_dir)


def copy_metrics_file(output_dir: Path, suffix: str, old_name: str, new_name: str) -> None:
    src = output_dir / f"{old_name}{suffix}"
    dst = output_dir / f"{new_name}{suffix}"
    if src.exists() and not dst.exists():
        print(f"Copying {src} -> {dst}")
        shutil.copy(src, dst)


def migrate_outputs(
    output_dir: Path,
    embedding_dir: Path | None,
    model_ids: Iterable[str],
) -> None:
    pred_root = output_dir / "predictions"
    for model_id in model_ids:
        new_name = sanitize(model_id)
        old_name = model_id.split("/")[-1]  # previous behavior used only the tail
        alias_only = new_name.split("-")[-1]  # handle cached runs where alias alone was used
        if embedding_dir:
            rename_embeddings(embedding_dir, old_name, new_name)
            rename_embeddings(embedding_dir, alias_only, new_name)
        rename_prediction_dirs(pred_root, old_name, new_name)
        rename_prediction_dirs(pred_root, alias_only, new_name)
        copy_metrics_file(output_dir, "_meta_predictions.json", old_name, new_name)
        copy_metrics_file(output_dir, "_dev_predictions.json", old_name, new_name)
        copy_metrics_file(output_dir, "_meta_predictions.json", alias_only, new_name)
        copy_metrics_file(output_dir, "_dev_predictions.json", alias_only, new_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/meta_evaluator", help="Root folder containing predictions/ and metrics files.")
    parser.add_argument("--embedding-dir", default="outputs/meta_evaluator/embeddings", help="Embedding cache folder (if present).")
    parser.add_argument("--model-ids", nargs="*", default=[
            # Llama 3.2 family (Meta)
            # "meta-llama/Llama-3.2-1B",
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
            
            "meta-llama/Llama-3.2-3B-Instruct",
            "Qwen/Qwen3-0.6B",
            "Gensyn/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "Qwen/Qwen3-0.6B-Base",
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            "Qwen/Qwen2-0.5B",
            "unsloth/Llama-3.2-1B-Instruct"
        ], help="List of model ids to migrate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    embedding_dir = Path(args.embedding_dir) if args.embedding_dir else None
    migrate_outputs(out_dir, embedding_dir, args.model_ids)


if __name__ == "__main__":
    main()
