"""Evaluation helpers for SQL prediction accuracy."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def normalize_sql(sql: str | None) -> str:
    if not sql:
        return ""
    cleaned = sql.strip().rstrip(";")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def compare_sql(predicted: str | None, reference: str | None) -> bool:
    return normalize_sql(predicted) == normalize_sql(reference)


def _normalize_rows(rows: Iterable[Sequence]) -> Tuple[Tuple, ...]:
    normalized = []
    for row in rows:
        normalized.append(tuple(row))
    return tuple(sorted(normalized))


class SQLiteExecutor:
    """Caches read-only SQLite connections and executes SQL statements."""

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self._connections: Dict[Path, sqlite3.Connection] = {}

    def _resolve(self, db_id: str | None, fallback_path: str | None) -> Path:
        candidates: List[Path] = []
        if self.base_dir and db_id:
            candidates.append((self.base_dir / db_id / f"{db_id}.sqlite").resolve())
        if fallback_path:
            path = Path(fallback_path)
            if not path.is_absolute():
                path = path.resolve()
            candidates.append(path)
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(f"Unable to locate database for db_id={db_id}, fallback={fallback_path}")

    def execute(self, db_id: str | None, fallback_path: str | None, sql: str) -> Tuple[bool, Tuple[Tuple, ...] | None, str | None, str | None]:
        sql = sql.strip()
        if not sql:
            return False, None, "empty SQL", None
        try:
            resolved = self._resolve(db_id, fallback_path)
        except FileNotFoundError as exc:
            return False, None, str(exc), None
        try:
            conn = self._connections.get(resolved)
            if conn is None:
                conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
                conn.row_factory = sqlite3.Row
                self._connections[resolved] = conn
            cursor = conn.execute(sql)
            rows = cursor.fetchall()
            normalized = _normalize_rows(rows)
            return True, normalized, None, str(resolved)
        except Exception as exc:  # pragma: no cover - relies on external DBs
            return False, None, str(exc), str(resolved)

    def close(self) -> None:
        for conn in self._connections.values():
            try:
                conn.close()
            except Exception:  # pragma: no cover - defensive
                pass
        self._connections.clear()


def build_prediction_records(
    samples: Sequence[dict],
    predictions: Sequence[str],
    indices: Sequence[int],
    *,
    db_root: str | Path | None = None,
) -> Tuple[Dict[str, float], List[dict]]:
    if len(samples) != len(predictions):
        raise ValueError("Sample and prediction counts must match.")
    total = len(samples)
    exact_correct = 0
    exec_correct = 0
    records: List[dict] = []
    executor = SQLiteExecutor(base_dir=db_root)
    try:
        for idx, (sample, prediction) in enumerate(zip(samples, predictions)):
            gold_sql = sample.get("sql") or ""
            exact_match = compare_sql(prediction, gold_sql)
            exact_correct += int(exact_match)

            exec_match = False
            exec_error = None
            db_path = sample.get("db_path")
            resolved_path = None
            db_id = sample.get("db_id")
            if (db_id or db_path) and gold_sql and prediction:
                gold_ok, gold_rows, gold_err, resolved = executor.execute(db_id, db_path, gold_sql)
                pred_ok, pred_rows, pred_err, resolved_pred = executor.execute(db_id, db_path, prediction)
                resolved_path = resolved_pred or resolved
                if gold_ok and pred_ok:
                    exec_match = gold_rows == pred_rows
                else:
                    exec_error = pred_err or gold_err
            else:
                exec_error = "missing db_path or sql"
            exec_correct += int(exec_match)

            records.append(
                {
                    "original_index": int(indices[idx]),
                    "db_id": sample.get("db_id"),
                    "db_path": resolved_path or db_path,
                    "question": sample.get("question"),
                    "reference_sql": gold_sql,
                    "predicted_sql": prediction,
                    "exact_match": exact_match,
                    "execution_correct": exec_match,
                    "execution_error": exec_error,
                }
            )
    finally:
        executor.close()

    metrics = {
        "exact_accuracy": exact_correct / total if total else 0.0,
        "execution_accuracy": exec_correct / total if total else 0.0,
    }
    return metrics, records
