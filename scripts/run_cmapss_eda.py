#!/usr/bin/env python
"""Run CMAPSS Phase-1 EDA and write configs plus markdown summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.cmapss_eda import analyze_all, report_to_config_dict  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "cmapss"
CONFIG_DIR = ROOT / "configs"
DOCS_PATH = ROOT / "docs" / "cmapss_eda_summary.md"


def _fmt_list(items: list[str]) -> str:
    return ", ".join(f"`{x}`" for x in items) if items else "_none_"


def build_markdown(reports: dict) -> str:
    lines = [
        "# CMAPSS Phase 1 — Exploratory Data Analysis",
        "",
        "This document records **methodology**, **findings**, and **downstream decisions** "
        "from Phase 1 (EDA & sensor QC) for NASA C-MAPSS FD001–FD004. "
        "Configs generated from this analysis live in `configs/cmapss_FD00X.yaml`.",
        "",
        "## 1. Objectives",
        "",
        "Phase 1 answers four questions before feature engineering:",
        "",
        "1. **Fleet structure** — How many engines, cycles per engine, train vs test behavior?",
        "2. **Sensor quality** — Which sensors are constant or uninformative?",
        "3. **Operating conditions** — Are op settings static (FD001/003) or multi-condition (FD002/004)?",
        "4. **Test censorship** — Test run lengths and RUL-at-last-cycle from official label files.",
        "",
        "These decisions satisfy UC5 Component A prerequisites: justified data understanding "
        "before health indicators and target definition.",
        "",
        "## 2. Methodology",
        "",
        "### 2.1 Data source and schema",
        "",
        "- **Source:** NASA C-MAPSS turbofan simulation ([readme.txt](../data/raw/cmapss/readme.txt), "
        "Saxena et al., PHM08).",
        "- **Files per subset:** `train_FD00X.txt`, `test_FD00X.txt`, `RUL_FD00X.txt`.",
        "- **Schema:** 26 space-separated columns per row (see `src/ingestion/cmapss_loader.py`).",
        "- **Loader:** `pandas.read_csv(sep=r'\\s+')` with explicit column names (no header row).",
        "",
        "### 2.2 Trajectory statistics",
        "",
        "For each subset and split (train/test):",
        "",
        "- Count distinct `unit_id` (engines).",
        "- Per unit, record `max(cycle)` → run length distribution (min, max, median).",
        "- **Train:** trajectories run until simulated failure (RUL → 0 at last cycle).",
        "- **Test:** trajectories stop before failure; `RUL_FD00X.txt` gives true RUL at the last observed cycle.",
        "",
        "### 2.3 Train vs test trajectory behavior",
        "",
        "**Important:** Train and test engines are **different units** (same fleet type, not the same engines). "
        "We do **not** subtract train run length from test run length by `unit_id`.",
        "",
        "Instead we report:",
        "",
        "1. **Train run length** — cycles until simulated failure (RUL→0 at last row).",
        "2. **Test run length** — cycles observed before censorship.",
        "3. **RUL at last test cycle** — from `RUL_FD00X.txt` (competition ground truth).",
        "4. **Implied failure cycle** — `last_test_cycle + RUL` per test engine (sanity check on label file).",
        "",
        "### 2.4 Sensor variance QC",
        "",
        "On **train** and **test** separately, compute standard deviation per `sensor_1`…`sensor_21`:",
        "",
        "| Classification | Rule |",
        "|----------------|------|",
        "| **Constant** | `std == 0` |",
        "| **Near-constant** | `0 < std < 1e-6` |",
        "| **Informative** | `std ≥ 1e-6` |",
        "",
        "**Drop recommendation** = union of constant and near-constant sensors across **both** splits.",
        "For FD001, we also union the literature prior (Saxena et al.; common benchmark practice).",
        "",
        "### 2.5 Operating settings",
        "",
        "NASA labels each subset with a **scenario** count (1 or 6 operating conditions). "
        "Per-column `nunique` is high even for FD001 because settings differ across engines.",
        "",
        "We therefore report:",
        "",
        "- **Per-column nunique** — diagnostic only.",
        "- **Unique (op1, op2, op3) triplets** (rounded to 2 dp) — how many distinct setting vectors appear.",
        "- **Per-unit stability** — whether each engine holds fixed settings across its life.",
        "- **`cluster_for_normalization`** — `true` when NASA scenario has 6 conditions (FD002, FD004).",
        "",
        "### 2.6 Reproducibility",
        "",
        "- Script: `scripts/run_cmapss_eda.py`",
        "- Library: `src/ingestion/cmapss_eda.py`",
        "- Re-run: `python scripts/run_cmapss_eda.py`",
        "",
        "## 3. Decisions (locked for Phase 2+)",
        "",
        "| ID | Decision | Rationale |",
        "|----|----------|-----------|",
        "| D1 | **Primary target:** piecewise capped RUL (`cap=125`) | NASA benchmark convention; reduces healthy-plateau noise |",
        "| D2 | **Secondary targets:** `failure_30`, `failure_72` (`rul ≤ horizon`) | Maps UC5 failure-window requirement to cycle-based labels |",
        "| D3 | **Drop constant sensors** per EDA table below | Zero variance adds no information; reduces overfitting |",
        "| D4 | **FD001 first**, then FD003 (2 fault modes), then FD002/004 | Complexity ladder; UC5 requires ≥2 failure modes (FD003/004) |",
        "| D5 | **No row-level random split** | Official split is by engine; leakage if rows are shuffled across units |",
        "| D6 | **Op-setting clustering** for FD002/FD004 (6 NASA scenarios) | Condition-aware normalization when multiple flight regimes exist |",
        "| D7 | **Rolling windows** `[5, 10, 30]`, **lags** `[1, 3, 5]` | Short/medium/long degradation horizons for ~100–300 cycle runs |",
        "| D8 | **Train rows for RUL modeling:** focus on `rul ≤ 125` (Phase 4) | Standard practice; defers to config `train_row_filter_max_rul` |",
        "",
        "## 4. Results by dataset",
        "",
    ]

    for ds, r in reports.items():
        trl, terl = r.train_run_length, r.test_run_length
        lines.extend(
            [
                f"### {ds}",
                "",
                f"| Property | Value |",
                f"|----------|-------|",
                f"| Operating conditions | {r.meta['n_operating_conditions']} |",
                f"| Fault modes | {r.meta['n_fault_modes']} |",
                f"| Train engines × rows | {r.train_units} × {r.train_rows:,} |",
                f"| Test engines × rows | {r.test_units} × {r.test_rows:,} |",
                f"| Train run length (min / median / max) | {trl['min']:.0f} / {trl['median']:.0f} / {trl['max']:.0f} |",
                f"| Test run length (min / median / max) | {terl['min']:.0f} / {terl['median']:.0f} / {terl['max']:.0f} |",
                f"| RUL at last test cycle (min / median / max) | {r.rul_at_last_cycle['min']:.0f} / {r.rul_at_last_cycle['median']:.0f} / {r.rul_at_last_cycle['max']:.0f} |",
                f"| Implied failure cycle (median) | {r.implied_remaining_from_rul['median']:.0f} |",
                f"| NASA op scenarios | {r.meta['n_operating_conditions']} |",
                f"| Unique op triplets (train) | {r.n_unique_op_triplets} |",
                f"| Op settings stable within each unit? | **{'Yes' if r.per_unit_op_stable else 'No'}** |",
                f"| Cluster for normalization (Phase 3)? | **{'Yes' if r.cluster_for_normalization else 'No'}** |",
                "",
                "**Sensors to drop:** " + _fmt_list(r.recommended_drop_sensors),
                "",
                "**Sensors to keep:** " + _fmt_list(r.recommended_keep_sensors),
                "",
                f"*Train-only constant:* {_fmt_list(r.train_sensor_report.constant_sensors)}  ",
                f"*Test-only constant:* {_fmt_list(r.test_sensor_report.constant_sensors)}  ",
                "",
            ]
        )

    lines.extend(
        [
            "## 5. Cross-dataset comparison",
            "",
            "| Dataset | Fault modes | Op conditions | Keep sensors | Typical use |",
            "|---------|-------------|---------------|--------------|-------------|",
        ]
    )
    for ds, r in reports.items():
        lines.append(
            f"| {ds} | {r.meta['n_fault_modes']} | {r.meta['n_operating_conditions']} "
            f"| {len(r.recommended_keep_sensors)} | "
            f"{'Baseline' if ds == 'FD001' else 'Multi-mode' if r.meta['n_fault_modes'] == 2 else 'Multi-condition'} |"
        )

    lines.extend(
        [
            "",
            "## 6. UC5 alignment",
            "",
            "| UC5 requirement | Phase 1 outcome |",
            "|-----------------|-----------------|",
            "| Multi-unit time series | Confirmed 100–260 engines per subset |",
            "| ≥ 2 failure modes | **FD003, FD004** (2 modes); FD001/002 for ablation |",
            "| Justified preprocessing | Sensor drop lists + op-condition flags per subset |",
            "| RUL target | `rul_cap: 125` in configs; test RUL file validated |",
            "",
            "## 7. Next steps (Phase 2)",
            "",
            "1. Implement `compute_test_rul()` — align `RUL_FD00X.txt` to every test cycle.",
            "2. Build leakage-safe normalization (per unit; cluster for FD002/004).",
            "3. Add delta, slope, and spectral features per config.",
            "4. Persist `data/processed/cmapss_{FD00X}_{train,test}_features.parquet`.",
            "",
            "## 8. References",
            "",
            "- Saxena, A., et al. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* PHM08.",
            "- NASA C-MAPSS readme: `data/raw/cmapss/readme.txt`",
            "- Dataset overview: `docs/datasets/cmapss.md`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    reports = analyze_all(RAW_DIR)

    for ds, report in reports.items():
        cfg = report_to_config_dict(report)
        path = CONFIG_DIR / f"cmapss_{ds}.yaml"
        with path.open("w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    md = build_markdown(reports)
    DOCS_PATH.write_text(md, encoding="utf-8")

    snapshot = {
        ds: report_to_config_dict(r) for ds, r in reports.items()
    }
    (CONFIG_DIR / "cmapss_eda_snapshot.json").write_text(
        json.dumps(snapshot, indent=2), encoding="utf-8"
    )

    print(f"Wrote {DOCS_PATH}")
    for ds in reports:
        print(f"Wrote {CONFIG_DIR / f'cmapss_{ds}.yaml'}")


if __name__ == "__main__":
    main()
