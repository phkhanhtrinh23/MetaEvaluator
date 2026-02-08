"""Inspect a saved MetaEvaluator meta-learner checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from meta_evaluator.model import MetaEvaluator  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="outputs/meta_evaluator/meta_evaluator_model.pt", help="Path to MetaEvaluator checkpoint (.pt).")
    parser.add_argument("--input-dim", type=int, default=None, help="Input dimension (number of descriptor features). If omitted, infer from checkpoint.")
    parser.add_argument("--hidden1", type=int, default=None, help="First hidden layer size. If omitted, infer from checkpoint.")
    parser.add_argument("--hidden2", type=int, default=None, help="Second hidden layer size. If omitted, infer from checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")

    if args.input_dim is None or args.hidden1 is None or args.hidden2 is None:
        # Infer shapes from the state dict
        fc1_w = state["fc1.weight"]
        inferred_input = fc1_w.shape[1]
        inferred_h1 = fc1_w.shape[0]
        inferred_h2 = state["fc2.weight"].shape[0]
        args.input_dim = args.input_dim or inferred_input
        args.hidden1 = args.hidden1 or inferred_h1
        args.hidden2 = args.hidden2 or inferred_h2

    model = MetaEvaluator(input_dim=args.input_dim, hidden_dims=(args.hidden1, args.hidden2))
    model.load_state_dict(state)

    print("MetaEvaluator architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params} | Trainable: {trainable_params}")


if __name__ == "__main__":
    main()
