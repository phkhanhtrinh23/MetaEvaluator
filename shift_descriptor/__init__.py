"""Utilities for measuring distribution shift via LLM embeddings."""

from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__"]


def __getattr__(name):
    if name == "__version__":
        try:
            return version("shift_descriptor")
        except PackageNotFoundError:  # pragma: no cover - local editable install
            return "0.0.0"
    raise AttributeError(name)
