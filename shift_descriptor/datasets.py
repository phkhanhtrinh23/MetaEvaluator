"""Dataset helpers for text2SQL SFT JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .prompts import default_template_path, load_prompt_template, render_prompt


def _load_json_list(json_path: str | Path) -> List[dict]:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in {json_path}, found {type(raw)}")
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {idx} in {json_path} is not a dict.")
    return raw


def load_text_field(json_path: str | Path, field: str = "text", fallback: Iterable[str] | None = ("question",)) -> List[str]:
    """Return a list of texts from the given JSON file."""

    raw = _load_json_list(json_path)
    results: List[str] = []
    fallback_fields = list(fallback or [])

    for idx, item in enumerate(raw):
        text = item.get(field)
        if not text:
            for fb in fallback_fields:
                text = item.get(fb)
                if text:
                    break
        if not text:
            raise ValueError(f"Entry {idx} missing '{field}' and fallbacks {fallback_fields}.")
        results.append(str(text))

    return results


def load_prompt_texts(
    json_path: str | Path,
    template_path: str | Path | None = None,
    context_fields: Iterable[str] | None = ("evidence", "matched_contents", "text"),
    system_prompt: str | None = None,
) -> List[str]:
    """Render prompt template per example to create Text-to-SQL inputs."""

    raw = _load_json_list(json_path)
    template = load_prompt_template(template_path or default_template_path())
    prompts: List[str] = []
    for item in raw:
        prompts.append(
            render_prompt(
                item,
                template=template,
                system_prompt=system_prompt,
                context_fields=context_fields,
            )
        )
    return prompts
