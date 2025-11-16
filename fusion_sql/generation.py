"""SQL generation utilities for FusionSQL."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import List, Sequence

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from shift_descriptor.config import ModelSpec
from shift_descriptor.embeddings import get_device


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


@dataclass
class GenerationSettings:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 0.9
    repetition_penalty: float = 1.0


class SQLGenerator:
    """Lightweight wrapper around HF causal LMs for SQL decoding."""

    def __init__(
        self,
        model_spec: ModelSpec,
        settings: GenerationSettings,
        device: str | None = None,
        lora_r: int | None = None,
    ):
        self.spec = model_spec
        self.settings = settings
        self.device = get_device(device)
        tokenizer_kwargs = {}
        if model_spec.revision:
            tokenizer_kwargs["revision"] = model_spec.revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_spec.model_id,
            trust_remote_code=model_spec.trust_remote_code,
            **tokenizer_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"
        model_kwargs = {}
        if model_spec.revision:
            model_kwargs["revision"] = model_spec.revision
        if self.device.type == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["low_cpu_mem_usage"] = True
        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec.model_id,
            trust_remote_code=model_spec.trust_remote_code,
            **model_kwargs,
        )
        if lora_r and lora_r > 0:
            self.model = self._apply_lora(self.model, lora_r)
        self.model.to(self.device)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False
        self.model.eval()

    def generate(self, prompts: Sequence[str]) -> List[str]:
        outputs: List[str] = []
        for prompt in tqdm(prompts, desc="Generating SQL..."):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.settings.max_new_tokens,
                    temperature=max(self.settings.temperature, 1e-6),
                    top_p=self.settings.top_p,
                    do_sample=self.settings.temperature > 0.0,
                    repetition_penalty=self.settings.repetition_penalty,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=False,
                )
            gen_ids = generated[0, inputs["input_ids"].shape[1] :]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            # print(f"[SQLGenerator] Prompt: {prompt}")
            # print(f"[SQLGenerator] Generated: {text.strip()}")
            outputs.append(text.strip())
        return outputs

    def shutdown(self) -> None:
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _apply_lora(self, model: torch.nn.Module, rank: int) -> torch.nn.Module:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("peft is required for LoRA-enabled generation. Install via `pip install peft`.") from exc

        targets = _guess_lora_targets(model)
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            lora_dropout=0.0,
            bias="none",
            target_modules=targets,
            task_type=TaskType.CAUSAL_LM,
        )
        adapter = get_peft_model(model, lora_config)
        adapter.eval()
        return adapter
