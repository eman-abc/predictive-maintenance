"""Ensure project root is on sys.path for all Streamlit pages."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ROOT_STR = str(_PROJECT_ROOT)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)


def _purge_stale_briefing_modules() -> None:
    """Drop renamed/updated briefing modules so Streamlit never uses stale .pyc."""
    if os.environ.get("_PM_BRIEFINGS_CACHE_PURGED"):
        return

    stale_keys = [
        k
        for k in list(sys.modules)
        if k.startswith("src.briefings") or k.endswith("prompt_templates")
    ]
    for key in stale_keys:
        del sys.modules[key]

    cache_dir = _PROJECT_ROOT / "src" / "briefings" / "__pycache__"
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir, ignore_errors=True)

    os.environ["_PM_BRIEFINGS_CACHE_PURGED"] = "1"


_purge_stale_briefing_modules()
