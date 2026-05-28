"""Load CMAPSS pipeline configuration from Phase 1 YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_DIR = Path("configs")


def load_cmapss_config(
    dataset_id: str = "FD001",
    config_dir: str | Path = DEFAULT_CONFIG_DIR,
) -> dict[str, Any]:
    """Load `configs/cmapss_{dataset_id}.yaml`."""
    path = Path(config_dir) / f"cmapss_{dataset_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"CMAPSS config not found: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
