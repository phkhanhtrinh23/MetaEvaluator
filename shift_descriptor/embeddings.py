"""Embedding extraction utilities built on top of Hugging Face models."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _guess_lora_targets(model: torch.nn.Module) -> List[str]:
    candidate_names = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "wi",
        "wo",
        "dense",
        "fc1",
        "fc2",
    ]
    available = set()
    for name, _ in model.named_modules():
        parts = name.split(".")[-1]
        available.add(parts)
    targets = [name for name in candidate_names if name in available]
    if not targets:
        targets = candidate_names[:4]
    return targets


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover - macOS specific
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class EmbeddingConfig:
    model_id: str
    alias: str
    batch_size: int = 4
    max_length: int = 512
    trust_remote_code: bool = False
    revision: str | None = None
    dtype: torch.dtype | None = None
    lora_r: int | None = None
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


class EmbeddingExtractor:
    """Extract pooled embeddings from a decoder-only LLM."""

    def __init__(self, config: EmbeddingConfig, device: torch.device | None = None):
        self.cfg = config
        self.device = device or get_device()
        torch_dtype = config.dtype
        if torch_dtype is None and self.device.type == "cuda":
            torch_dtype = torch.float16
        elif torch_dtype is None:
            torch_dtype = torch.float32

        tokenizer_kwargs = dict(config.tokenizer_kwargs or {})
        if config.revision:
            tokenizer_kwargs.setdefault("revision", config.revision)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        model_kwargs = dict(config.model_kwargs or {})
        if "torch_dtype" not in model_kwargs and torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if config.revision:
            model_kwargs.setdefault("revision", config.revision)
        uses_device_map = "device_map" in model_kwargs and model_kwargs["device_map"] not in (None, "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            trust_remote_code=config.trust_remote_code,
            **model_kwargs,
        )
        if not uses_device_map:
            self.model.to(self.device)
        self.model_device = getattr(self.model, "device", self.device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        if hasattr(self.model.config, "output_hidden_states"):
            self.model.config.output_hidden_states = True
        if self.cfg.lora_r:
            self.model = self._apply_lora(self.model, self.cfg.lora_r)
            self.model.to(self.device)
            self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        """Return pooled embeddings for the provided texts."""

        vectors: List[np.ndarray] = []
        batch_texts: List[str] = []
        for text in texts:
            batch_texts.append(text)
            if len(batch_texts) >= self.cfg.batch_size:
                vectors.append(self._encode_batch(batch_texts))
                batch_texts = []

        if batch_texts:
            vectors.append(self._encode_batch(batch_texts))

        return np.vstack(vectors) if vectors else np.empty((0, self.model.config.hidden_size), dtype=np.float32)

    def _encode_batch(self, batch_texts: List[str]) -> np.ndarray:
        inputs = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.cfg.max_length,
        )
        inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=getattr(self.model.config, "output_hidden_states", False))
        hidden, is_sequence = self._select_hidden_tensor(outputs, batch_size=inputs["input_ids"].shape[0], seq_len=inputs["input_ids"].shape[1])

        if is_sequence:
            mask = inputs["attention_mask"].unsqueeze(-1)
            summed = torch.sum(hidden * mask, dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        else:
            pooled = hidden
        pooled = torch.nan_to_num(pooled, nan=0.0, posinf=1e4, neginf=-1e4)
        return pooled.detach().cpu().float().numpy()

    def _select_hidden_tensor(self, outputs, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, bool]:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is not None:
            return hidden, hidden.ndim == 3 and hidden.shape[1] == seq_len

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            hidden = hidden_states[-1]
            return hidden, hidden.ndim == 3 and hidden.shape[1] == seq_len

        for attr in ("state", "states"):
            state = getattr(outputs, attr, None)
            if state is None:
                continue
            if isinstance(state, (list, tuple)):
                state = state[-1]
            if torch.is_tensor(state):
                flattened = self._flatten_state_tensor(state, batch_size)
                if flattened is not None:
                    return flattened, False

        logits = getattr(outputs, "logits", None)
        if logits is not None:
            if logits.ndim == 3:
                if logits.shape[1] == seq_len:
                    return logits, True
                return logits.mean(dim=1), False
            if logits.ndim == 2:
                return logits, False

        raise AttributeError("Model output does not expose usable hidden states or logits.")

    def _flatten_state_tensor(self, tensor: torch.Tensor, batch_size: int) -> Optional[torch.Tensor]:
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        if tensor.shape[0] != batch_size:
            permute_dims = list(range(tensor.ndim))
            swapped = False
            for axis in range(tensor.ndim):
                if tensor.shape[axis] == batch_size:
                    permute_dims[0], permute_dims[axis] = permute_dims[axis], permute_dims[0]
                    tensor = tensor.permute(*permute_dims)
                    swapped = True
                    break
            if not swapped and tensor.shape[0] != batch_size:
                return None

        tensor = tensor.contiguous().view(batch_size, -1)
        return tensor

    def _apply_lora(self, model: torch.nn.Module, rank: int) -> torch.nn.Module:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("peft is required for LoRA support. Install via `pip install peft`.") from exc

        targets = _guess_lora_targets(model)
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
        )
        lora_model = get_peft_model(model, lora_config)
        lora_model.eval()
        return lora_model

    def save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, embeddings=embeddings)

    def shutdown(self) -> None:
        """Release model resources to avoid lingering GPU allocations."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
