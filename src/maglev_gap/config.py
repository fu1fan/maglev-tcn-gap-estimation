from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_single_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Config must be a mapping: {path}")

    base_paths = raw.pop("base_configs", [])
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    merged: dict[str, Any] = {}
    for rel_path in base_paths:
        base_path = (path.parent / rel_path).resolve()
        merged = deep_merge(merged, _load_single_config(base_path))

    merged = deep_merge(merged, raw)
    merged["config_path"] = str(path.resolve())
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    return _load_single_config(Path(path).resolve())


def clone_config(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(config)
