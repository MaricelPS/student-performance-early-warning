"""Configuration loading.

All paths in the YAML file are resolved relative to the project root so the
CLI behaves the same regardless of the working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    root: Path

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else self.root / p

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def features(self) -> dict[str, Any]:
        return self.raw["features"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("EW_CONFIG", DEFAULT_CONFIG))
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=PROJECT_ROOT)
