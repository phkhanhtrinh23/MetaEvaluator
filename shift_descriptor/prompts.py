"""Prompt templating helpers for Text-to-SQL inputs."""

from __future__ import annotations

import json
from pathlib import Path
from string import Template
from typing import Iterable, List, Sequence


DEFAULT_SYSTEM_PROMPT = (
    "You are a senior Text-to-SQL engineer. Translate natural-language questions into valid SQLite SQL over "
    "the provided schema. Study every table, respect column types, follow foreign keys, and return one SQL "
    "statement with no explanation."
)

_PACKAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_DIR.parent
_PROMPTS_DIR = _REPO_ROOT / "prompts"
_DEFAULT_TEMPLATE_PATH = _PROMPTS_DIR / "text2sql_prompt.tmpl"


def default_template_path() -> Path:
    return _DEFAULT_TEMPLATE_PATH


def load_prompt_template(path: str | Path | None = None) -> Template:
    template_path = Path(path) if path else default_template_path()
    return Template(template_path.read_text(encoding="utf-8"))


def _format_table(table: dict) -> str:
    name = table.get("table_name", "UNKNOWN_TABLE")
    pk = table.get("pk_indicators", [])
    col_names = table.get("column_names", [])
    col_types = table.get("column_types", [])
    col_comments = table.get("column_comments", [])
    column_lines = []
    for idx, (col, typ) in enumerate(zip(col_names, col_types)):
        comment = ""
        if idx < len(col_comments) and col_comments[idx]:
            comment = f" // {col_comments[idx]}"
        column_lines.append(f"        • {col} :: {typ}{comment}")

    contents = table.get("column_contents")
    content_str = ""
    if contents:
        try:
            preview = json.dumps(contents, ensure_ascii=False)
        except TypeError:
            preview = str(contents)
        content_str = f"\n    Example rows: {preview}"

    column_block = "\n".join(column_lines) if column_lines else "        • (no columns listed)"
    return (
        f"- Table {name}\n"
        f"    PK indicators: {pk}\n"
        f"    Columns:\n{column_block}"
        f"{content_str}"
    )


def format_schema(schema: dict | None) -> str:
    if not schema:
        return "None"
    items = schema.get("schema_items") or []
    if not items:
        return "None"
    return "\n".join(_format_table(item) for item in items)


def format_foreign_keys(foreign_keys: Iterable[Sequence[str]] | None) -> str:
    if not foreign_keys:
        return "None"
    formatted = []
    for fk in foreign_keys:
        if len(fk) != 4:
            formatted.append(f"- {fk}")
        else:
            formatted.append(f"- {fk[0]}.{fk[1]} → {fk[2]}.{fk[3]}")
    return "\n".join(formatted)


def format_context(sample: dict, context_fields: Iterable[str] | None = None) -> str:
    fields = list(context_fields) if context_fields else []
    parts: List[str] = []
    for field in fields:
        value = sample.get(field)
        if not value:
            continue
        if isinstance(value, (dict, list)):
            serialized = json.dumps(value, ensure_ascii=False)
        else:
            serialized = str(value)
        parts.append(f"{field}: {serialized}")
    return "\n".join(parts) if parts else "None"


def render_prompt(
    sample: dict,
    template: Template,
    system_prompt: str | None = None,
    context_fields: Iterable[str] | None = None,
) -> str:
    schema_overview = format_schema(sample.get("schema"))
    foreign_keys = format_foreign_keys((sample.get("schema") or {}).get("foreign_keys"))
    context = format_context(sample, context_fields)
    mapping = {
        "system_prompt": system_prompt or DEFAULT_SYSTEM_PROMPT,
        "db_id": sample.get("db_id", ""),
        "db_path": sample.get("db_path", ""),
        "schema_overview": schema_overview,
        "foreign_keys": foreign_keys,
        "question": sample.get("question") or sample.get("text") or "",
        "context": context,
    }
    return template.safe_substitute(mapping)
