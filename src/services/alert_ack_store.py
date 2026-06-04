"""In-memory alert acknowledgements for demo (API process)."""

from __future__ import annotations

import threading

_lock = threading.Lock()
_acked: set[tuple[str, str, str]] = set()  # (dataset_id, asset_id, alert_level)


def ack_key(dataset_id: str, asset_id: str, alert_level: str) -> tuple[str, str, str]:
    return (dataset_id, asset_id, alert_level.lower())


def acknowledge(dataset_id: str, asset_id: str, alert_level: str) -> None:
    with _lock:
        _acked.add(ack_key(dataset_id, asset_id, alert_level))


def is_acknowledged(dataset_id: str, asset_id: str, alert_level: str) -> bool:
    with _lock:
        return ack_key(dataset_id, asset_id, alert_level) in _acked


def enrich_rows_with_ack(
    rows: list[dict],
    *,
    dataset_id: str,
) -> list[dict]:
    out = []
    for row in rows:
        level = str(row.get("alert_level", "warning"))
        asset = str(row.get("asset_id", ""))
        acked = is_acknowledged(dataset_id, asset, level)
        out.append({**row, "ack_status": "acknowledged" if acked else "open"})
    return out
