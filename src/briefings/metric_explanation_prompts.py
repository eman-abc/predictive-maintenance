"""Prompt templates for Phase 3 metric explanations (Ollama / instant fallback)."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

SYSTEM_PROMPT = (
    "You explain CMAPSS Phase 3 predictive maintenance model metrics to plant engineers. "
    "Use plain language. Use ONLY numbers from the JSON provided — do not invent metrics. "
    "GBM (or the stated winner) is the production RUL model; Cox is a survival baseline only. "
    "Always start with the values in score_readings (quote the actual numbers shown in the dashboard table). "
    "Then apply interpretation_hints for what they mean operationally on this dataset. "
    "Write complete paragraphs with a clear ending. Section answers: 4–6 sentences. "
    "Summarizer may use short bullet lists."
)

SECTION_HEADLINE = "headline"
SECTION_RUL = "rul_comparison"
SECTION_COX = "cox_survival"
SECTION_FAILURE = "failure_classifiers"
SECTION_ANOMALY = "anomaly_detection"
SECTION_SUMMARIZE = "summarize_all"

ALL_SECTIONS = (
    SECTION_HEADLINE,
    SECTION_RUL,
    SECTION_COX,
    SECTION_FAILURE,
    SECTION_ANOMALY,
)

DATASET_PROFILES: dict[str, str] = {
    "FD001": "Single operating condition, one fault mode — easiest subset; Cox often ranks risk well.",
    "FD002": "Six operating conditions, one fault mode — harder generalization; Cox point RUL often poor.",
    "FD003": "Single operating condition, two fault modes — moderate difficulty.",
    "FD004": "Six operating conditions, two fault modes — hardest subset.",
}

PROJECT_RULES: list[str] = [
    "Winner is lowest validation NASA score (rul_score); tie-break RMSE then simpler model.",
    "Cox PH is not a winner candidate — parallel survival baseline for risk ranking.",
    "RMSE is cycles error; NASA score penalizes late RUL predictions more (safer for maintenance).",
    "Failure classifiers (≤30 / ≤72 cycles) drive operational alerts; test metrics are last-cycle snapshots.",
    "Anomaly uses Isolation Forest on healthy train rows; degradation AUC near 0.5 is weak separation.",
]

SECTION_TASKS: dict[str, str] = {
    SECTION_HEADLINE: (
        "Explain the headline test metrics and RUL winner. "
        "Open with the exact test RMSE, NASA score, and winner from score_readings."
    ),
    SECTION_RUL: (
        "Compare RF, GBM, LSTM, and Cox on validation metrics. "
        "Open with the winner and its validation RMSE/NASA from score_readings."
    ),
    SECTION_COX: (
        "Explain Cox survival for this dataset. "
        "First sentence: state validation concordance and its verdict from score_readings. "
        "Explain why RMSE/NASA may be missing (see metric_notes) — this is not a training failure on FD001. "
        "Compare to production RUL winner metrics in score_readings. "
        "End with a clear recommendation: risk ranking vs RUL scheduling."
    ),
    SECTION_FAILURE: (
        "Explain failure classifier test metrics at ≤30 and ≤72 cycle horizons. "
        "Which horizon is stronger for alerting and what do F1 / ROC-AUC imply?"
    ),
    SECTION_ANOMALY: (
        "Explain Isolation Forest anomaly metrics: mean score, pct flagged, degradation AUC. "
        "How useful is this as a secondary degradation signal?"
    ),
    SECTION_SUMMARIZE: (
        "Write an executive summary for this dataset only: "
        "(1) RUL headline and winner, (2) Cox role, (3) failure alerting, (4) anomaly, "
        "(5) recommended model per use case (RUL, alerts, risk ranking), (6) one caveat. "
        "Use short bullet lists where helpful."
    ),
}


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isinf(v) or math.isnan(v):
        return None
    return v


def _compact_metric(value: Any) -> Any:
    """JSON-safe metric; huge NASA scores become a short label."""
    v = _safe_float(value)
    if v is None:
        return None
    if abs(v) >= 1_000_000:
        return "very_large (poor point-RUL fit)"
    return round(v, 4)


def _missing_metric_note(value: Any, *, context: str = "cox") -> str | None:
    v = _safe_float(value)
    if v is not None and abs(v) < 1_000_000:
        return None
    if context == "cox":
        return (
            "not shown (—): median RUL undefined for many censored/healthy engines — "
            "common on Cox; use concordance for ranking, GBM for point RUL"
        )
    return "not available in summary"


def _concordance_verdict(value: Any) -> tuple[float | None, str, str]:
    """Return (value, label, operational hint) for concordance."""
    v = _safe_float(value)
    if v is None:
        return None, "not available", "Concordance was not logged for this split."
    if v >= 0.80:
        return round(v, 4), "strong (≥0.80)", (
            f"Concordance {v:.3f} ranks engine failure order well — useful to prioritize inspections."
        )
    if v >= 0.65:
        return round(v, 4), "moderate (0.65–0.79)", (
            f"Concordance {v:.3f} gives partial risk ranking — use cautiously alongside GBM RUL."
        )
    return round(v, 4), "weak (<0.65)", (
        f"Concordance {v:.3f} is too low for reliable risk ordering — do not use Cox for prioritization."
    )


def _roc_auc_verdict(value: Any) -> str:
    v = _safe_float(value)
    if v is None:
        return "not available"
    if v >= 0.90:
        return f"strong alert signal (ROC-AUC {v:.3f})"
    if v >= 0.75:
        return f"usable alert signal (ROC-AUC {v:.3f})"
    return f"weak alert signal (ROC-AUC {v:.3f})"


def _degradation_auc_verdict(value: Any) -> str:
    v = _safe_float(value)
    if v is None:
        return "not available"
    if v >= 0.70:
        return f"good degradation separation (AUC {v:.3f})"
    if v >= 0.55:
        return f"modest separation (AUC {v:.3f}) — secondary signal only"
    return f"weak separation (AUC {v:.3f}) — do not rely on alone"


def _dataset_profile(dataset_id: str) -> str:
    return DATASET_PROFILES.get(dataset_id, f"CMAPSS subset {dataset_id}.")


def _headline_block(summary: dict) -> dict[str, Any]:
    test = summary.get("test_metrics") or {}
    flags = []
    if summary.get("skip_lstm"):
        flags.append("LSTM skipped")
    if summary.get("skip_cox"):
        flags.append("Cox skipped")
    return {
        "test_rmse": _compact_metric(test.get("rmse")),
        "test_nasa": _compact_metric(test.get("rul_score")),
        "test_mae": _compact_metric(test.get("mae")),
        "rul_winner": (summary.get("winner") or "—").upper(),
        "training_notes": flags or ["full pipeline"],
    }


def _rul_comparison_rows(summary: dict) -> list[dict[str, Any]]:
    val = summary.get("val_metrics") or {}
    rows = []
    for model in ("rf", "gbm", "lstm", "cox"):
        m = val.get(model) or {}
        if not m:
            continue
        rows.append(
            {
                "model": model.upper(),
                "val_rmse": _compact_metric(m.get("rmse")),
                "val_nasa": _compact_metric(m.get("rul_score")),
                "val_mae": _compact_metric(m.get("mae")),
                "is_winner": model.upper() == (summary.get("winner") or "").upper(),
            }
        )
    return rows


def _cox_block(summary: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, key in (("validation", "cox_val_metrics"), ("test", "cox_test_metrics")):
        block = summary.get(key) or {}
        if not block:
            continue
        raw_rmse = block.get("rmse")
        raw_nasa = block.get("rul_score")
        raw_conc = block.get("concordance")
        entry: dict[str, Any] = {
            "concordance": _compact_metric(raw_conc),
            "rmse": _compact_metric(raw_rmse),
            "nasa": _compact_metric(raw_nasa),
        }
        if raw_conc is None and label == "test":
            entry["concordance_note"] = "not logged for test split in Phase 3 summary (table shows —)"
        rmse_note = _missing_metric_note(raw_rmse)
        if rmse_note:
            entry["rmse_note"] = rmse_note
        nasa_note = _missing_metric_note(raw_nasa)
        if nasa_note:
            entry["nasa_note"] = nasa_note
        if _safe_float(raw_rmse) is not None and abs(_safe_float(raw_rmse) or 0) >= 1_000_000:
            entry["rmse_note"] = "very poor point-RUL fit (RMSE extremely high)"
        out[label] = entry
    return out


def _score_readings(
    section_id: str,
    summary: dict,
    dataset_id: str,
) -> dict[str, Any]:
    """Deterministic snapshot the LLM must anchor on — matches dashboard table semantics."""
    headline = _headline_block(summary)
    winner = headline["rul_winner"]

    if section_id == SECTION_HEADLINE:
        return {
            "dataset": dataset_id,
            "test_rmse_cycles": headline["test_rmse"],
            "test_nasa_score": headline["test_nasa"],
            "test_mae_cycles": headline["test_mae"],
            "rul_winner": winner,
            "meaning": "Lower NASA and RMSE are better for production RUL scheduling.",
        }

    if section_id == SECTION_RUL:
        rows = _rul_comparison_rows(summary)
        winner_row = next((r for r in rows if r.get("is_winner")), None)
        return {
            "dataset": dataset_id,
            "rul_winner": winner,
            "winner_val_rmse": winner_row.get("val_rmse") if winner_row else None,
            "winner_val_nasa": winner_row.get("val_nasa") if winner_row else None,
            "all_models_validation": rows,
            "meaning": "Winner = lowest validation NASA; Cox row is baseline only.",
        }

    if section_id == SECTION_COX:
        val_block = summary.get("cox_val_metrics") or {}
        test_block = summary.get("cox_test_metrics") or {}
        conc_val, conc_label, conc_hint = _concordance_verdict(val_block.get("concordance"))
        test_rmse = headline["test_rmse"]
        return {
            "dataset": dataset_id,
            "validation_concordance": conc_val,
            "concordance_verdict": conc_label,
            "validation_rmse": _compact_metric(val_block.get("rmse")),
            "validation_nasa": _compact_metric(val_block.get("rul_score")),
            "test_concordance": _compact_metric(test_block.get("concordance")),
            "production_rul_winner": winner,
            "production_test_rmse": test_rmse,
            "production_test_nasa": headline["test_nasa"],
            "meaning": (
                f"{conc_hint} "
                f"RMSE/NASA dashes (—) on Cox mean median RUL was undefined, not a missing training run. "
                f"Schedule maintenance using {winner} test RMSE {test_rmse} cycles, not Cox point RUL."
            ),
        }

    if section_id == SECTION_FAILURE:
        block = summary.get("failure_clf_test_metrics") or {}
        readings: dict[str, Any] = {"dataset": dataset_id}
        for key, label in (("failure_30", "≤30 cycles"), ("failure_72", "≤72 cycles")):
            m = block.get(key) or {}
            readings[label] = {
                "f1": _compact_metric(m.get("f1")),
                "roc_auc": _compact_metric(m.get("roc_auc")),
                "precision": _compact_metric(m.get("precision")),
                "recall": _compact_metric(m.get("recall")),
                "alert_strength": _roc_auc_verdict(m.get("roc_auc")),
            }
        readings["meaning"] = "≤30 cycles drives short-horizon alerts; ≤72 is longer planning."
        return readings

    if section_id == SECTION_ANOMALY:
        val = summary.get("anomaly_val_metrics") or {}
        test = summary.get("anomaly_test_metrics") or {}
        return {
            "dataset": dataset_id,
            "validation": {
                "pct_flagged": _compact_metric(val.get("pct_flagged")),
                "degradation_auc": _compact_metric(val.get("degradation_roc_auc")),
                "verdict": _degradation_auc_verdict(val.get("degradation_roc_auc")),
            },
            "test": {
                "pct_flagged": _compact_metric(test.get("pct_flagged")),
                "degradation_auc": _compact_metric(test.get("degradation_roc_auc")),
                "verdict": _degradation_auc_verdict(test.get("degradation_roc_auc")),
            },
            "meaning": "Unsupervised secondary signal — not a substitute for RUL or failure probability.",
        }

    if section_id == SECTION_SUMMARIZE:
        val_block = summary.get("cox_val_metrics") or {}
        _, conc_label, _ = _concordance_verdict(val_block.get("concordance"))
        f30 = (summary.get("failure_clf_test_metrics") or {}).get("failure_30") or {}
        return {
            "dataset": dataset_id,
            "rul_winner": winner,
            "test_rmse": headline["test_rmse"],
            "test_nasa": headline["test_nasa"],
            "cox_concordance_verdict": conc_label,
            "failure_30_roc_auc": _compact_metric(f30.get("roc_auc")),
            "use_rul": winner,
            "use_alerts": "failure_30 classifier",
            "use_risk_ranking": "Cox if concordance ≥0.65, else GBM failure prob",
        }

    return {"dataset": dataset_id}


def _interpretation_hints(section_id: str, summary: dict, dataset_id: str) -> list[str]:
    hints = [_dataset_profile(dataset_id)]
    if section_id in (SECTION_COX, SECTION_SUMMARIZE):
        val_block = summary.get("cox_val_metrics") or {}
        _, _, conc_hint = _concordance_verdict(val_block.get("concordance"))
        hints.append(conc_hint)
        if _missing_metric_note(val_block.get("rmse")):
            hints.append(
                "Dashboard shows — for Cox RMSE/NASA when median RUL cannot be computed (censored engines)."
            )
    if section_id in (SECTION_RUL, SECTION_HEADLINE, SECTION_SUMMARIZE):
        hints.append(f"Production RUL model for {dataset_id} is {(summary.get('winner') or 'gbm').upper()}.")
    return hints


def _enrich_payload(payload: dict[str, Any], summary: dict, dataset_id: str) -> dict[str, Any]:
    section_id = payload.get("task", "")
    payload["score_readings"] = _score_readings(section_id, summary, dataset_id)
    payload["interpretation_hints"] = _interpretation_hints(section_id, summary, dataset_id)
    return payload


def _failure_rows(summary: dict, split: str = "failure_clf_test_metrics") -> list[dict[str, Any]]:
    block = summary.get(split) or {}
    rows = []
    for horizon in ("failure_30", "failure_72"):
        m = block.get(horizon) or {}
        if not m:
            continue
        rows.append(
            {
                "horizon": horizon.replace("failure_", "≤") + " cycles",
                "f1": _compact_metric(m.get("f1")),
                "roc_auc": _compact_metric(m.get("roc_auc")),
                "precision": _compact_metric(m.get("precision")),
                "recall": _compact_metric(m.get("recall")),
                "accuracy": _compact_metric(m.get("accuracy")),
            }
        )
    return rows


def _anomaly_rows(summary: dict) -> list[dict[str, Any]]:
    rows = []
    for split, key in (("validation", "anomaly_val_metrics"), ("test", "anomaly_test_metrics")):
        m = summary.get(key) or {}
        if not m:
            continue
        rows.append(
            {
                "split": split,
                "mean_score": _compact_metric(m.get("mean_anomaly_score")),
                "pct_flagged": _compact_metric(m.get("pct_flagged")),
                "degradation_auc": _compact_metric(m.get("degradation_roc_auc")),
            }
        )
    return rows


def build_section_payload(
    section_id: str,
    summary: dict,
    dataset_id: str,
    *,
    registry_entry: dict | None = None,
) -> dict[str, Any]:
    """Structured context for one metric block."""
    payload: dict[str, Any] = {
        "task": section_id,
        "dataset_id": dataset_id,
        "dataset_profile": _dataset_profile(dataset_id),
        "project_rules": PROJECT_RULES,
        "training_batch": summary.get("training_batch") or (registry_entry or {}).get("training_batch"),
    }

    if section_id == SECTION_HEADLINE:
        payload["headline"] = _headline_block(summary)
    elif section_id == SECTION_RUL:
        payload["rul_comparison"] = _rul_comparison_rows(summary)
        payload["headline"] = _headline_block(summary)
    elif section_id == SECTION_COX:
        payload["cox"] = _cox_block(summary)
        payload["headline"] = _headline_block(summary)
    elif section_id == SECTION_FAILURE:
        payload["failure_classifiers_test"] = _failure_rows(summary)
    elif section_id == SECTION_ANOMALY:
        payload["anomaly"] = _anomaly_rows(summary)
    elif section_id == SECTION_SUMMARIZE:
        payload["headline"] = _headline_block(summary)
        payload["rul_comparison"] = _rul_comparison_rows(summary)
        payload["cox"] = _cox_block(summary)
        payload["failure_classifiers_test"] = _failure_rows(summary)
        payload["anomaly"] = _anomaly_rows(summary)
    else:
        raise ValueError(f"Unknown section: {section_id}")

    return _enrich_payload(payload, summary, dataset_id)


def build_summarizer_payload(
    summary: dict,
    dataset_id: str,
    *,
    registry_entry: dict | None = None,
) -> dict[str, Any]:
    return build_section_payload(
        SECTION_SUMMARIZE,
        summary,
        dataset_id,
        registry_entry=registry_entry,
    )


def _payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


def build_explanation_prompt(section_id: str, payload: dict[str, Any]) -> str:
    task = SECTION_TASKS.get(section_id, "Explain these metrics.")
    return f"{task}\n\nMetrics JSON:\n{_payload_json(payload)}"


def build_summarizer_prompt(payload: dict[str, Any]) -> str:
    return build_explanation_prompt(SECTION_SUMMARIZE, payload)


def _winner_name(summary: dict) -> str:
    return (summary.get("winner") or "gbm").upper()


def _instant_headline(summary: dict, dataset_id: str) -> str:
    h = _headline_block(summary)
    winner = h["rul_winner"]
    rmse = h.get("test_rmse")
    nasa = h.get("test_nasa")
    profile = _dataset_profile(dataset_id)
    return (
        f"{dataset_id} RUL winner is {winner} with test RMSE {rmse} cycles and NASA score {nasa}. "
        f"Lower NASA is better for maintenance-safe RUL estimates. {profile} "
        f"Use {winner} RUL predictions for scheduling; Cox and anomaly are supporting signals."
    )


def _instant_rul(summary: dict) -> str:
    winner = _winner_name(summary)
    rows = _rul_comparison_rows(summary)
    parts = []
    for row in rows:
        tag = " (winner)" if row.get("is_winner") else ""
        parts.append(
            f"{row['model']}{tag}: val RMSE {row['val_rmse']}, NASA {row['val_nasa']}"
        )
    body = "; ".join(parts)
    return (
        f"Validation comparison: {body}. "
        f"{winner} has the best validation NASA score and is selected for test RUL. "
        "LSTM is often weaker on small CMAPSS subsets. Cox validation RMSE/NASA are not used for winner selection."
    )


def _instant_cox(summary: dict, dataset_id: str) -> str:
    readings = _score_readings(SECTION_COX, summary, dataset_id)
    val_block = summary.get("cox_val_metrics") or {}
    conc = readings.get("validation_concordance")
    verdict = readings.get("concordance_verdict", "")
    winner = readings.get("production_rul_winner", "GBM")
    test_rmse = readings.get("production_test_rmse")
    lines = [
        f"**{dataset_id} Cox (validation):** concordance **{conc}** — **{verdict}**.",
    ]
    if _missing_metric_note(val_block.get("rmse")):
        lines.append(
            "RMSE and NASA show as — because median RUL is undefined for many healthy engines; "
            "that is expected for Cox, not a failed training run."
        )
    elif _compact_metric(val_block.get("rmse")) == "very_large (poor point-RUL fit)":
        lines.append("Cox point RUL (RMSE/NASA) is very poor — do not use for maintenance timing.")
    test_block = summary.get("cox_test_metrics") or {}
    if test_block and test_block.get("concordance") is None:
        lines.append("Test split has no concordance in this summary — rely on validation concordance above.")
    lines.append(
        f"For RUL in cycles, use **{winner}** (test RMSE **{test_rmse}**). "
        "Use Cox only to rank relative failure risk when concordance is strong."
    )
    return " ".join(lines)


def _instant_failure(summary: dict) -> str:
    rows = _failure_rows(summary)
    if not rows:
        return "No failure classifier test metrics in this summary."
    parts = []
    for row in rows:
        auc = row.get("roc_auc")
        f1 = row.get("f1")
        strength = "strong" if isinstance(auc, (int, float)) and auc >= 0.85 else "moderate"
        parts.append(f"{row['horizon']}: F1 {f1}, ROC-AUC {auc} ({strength} alert signal)")
    return (
        f"Failure classifiers on test last-cycle: {'; '.join(parts)}. "
        "≤30 cycles aligns with operational short-horizon alerts; ≤72 cycles is a longer planning window. "
        "High ROC-AUC supports ranking engines for inspection priority."
    )


def _instant_anomaly(summary: dict) -> str:
    rows = _anomaly_rows(summary)
    if not rows:
        return "No anomaly metrics in this summary."
    parts = []
    for row in rows:
        auc = row.get("degradation_auc")
        flagged = row.get("pct_flagged")
        flagged_pct = f"{float(flagged) * 100:.1f}%" if flagged is not None else "—"
        note = "weak" if isinstance(auc, (int, float)) and auc < 0.6 else "modest"
        parts.append(f"{row['split']}: {flagged_pct} flagged, degradation AUC {auc} ({note} separation)")
    return (
        f"Isolation Forest anomaly: {'; '.join(parts)}. "
        "Treat as a secondary degradation hint — not a replacement for RUL or failure probability."
    )


def _instant_summarize(summary: dict, dataset_id: str) -> str:
    return (
        f"**{dataset_id} summary:** "
        f"{_instant_headline(summary, dataset_id)} "
        f"{_instant_cox(summary, dataset_id)} "
        f"{_instant_failure(summary)} "
        f"{_instant_anomaly(summary)}"
    )


def build_instant_explanation(
    section_id: str,
    summary: dict,
    dataset_id: str,
) -> str:
    """Rule-based explanation when Ollama is offline or user wants instant text."""
    if section_id == SECTION_HEADLINE:
        return _instant_headline(summary, dataset_id)
    if section_id == SECTION_RUL:
        return _instant_rul(summary)
    if section_id == SECTION_COX:
        return _instant_cox(summary, dataset_id)
    if section_id == SECTION_FAILURE:
        return _instant_failure(summary)
    if section_id == SECTION_ANOMALY:
        return _instant_anomaly(summary)
    if section_id == SECTION_SUMMARIZE:
        return _instant_summarize(summary, dataset_id)
    raise ValueError(f"Unknown section: {section_id}")


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Helper for tests — tables serialize the same way as summary builders."""
    if df.empty:
        return []
    return df.to_dict(orient="records")
