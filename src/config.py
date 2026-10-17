"""YAML configuration loading with single inheritance and dotted CLI overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherits", None)
    if parent:
        cfg = deep_merge(_load_yaml((path.parent / parent).resolve()), cfg)
    return cfg


def apply_override(cfg: dict, assignment: str) -> None:
    """Apply a ``a.b.c=value`` override; the value is parsed as YAML."""
    key, _, raw = assignment.partition("=")
    if not _:
        raise ValueError(f"Override must look like key=value, got {assignment!r}")
    node = cfg
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = _load_yaml(Path(path).resolve())
    for assignment in overrides or []:
        apply_override(cfg, assignment)
    env_out = os.environ.get("MC_OUTPUT_DIR")
    if env_out:
        cfg["experiment"]["output_dir"] = env_out
    return cfg


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
