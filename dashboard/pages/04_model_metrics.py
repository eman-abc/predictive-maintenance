"""MLflow experiment comparison view."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os

import pandas as pd
import streamlit as st

from dashboard.components.phase3_metrics import render_phase3_summary, render_registry_banner
from dashboard.data_loader import (
    list_available_models,
    load_phase3_summary,
    load_training_registry,
    render_dataset_selector,
)
from dashboard.page_init import init_page

st.set_page_config(page_title="Model Metrics", layout="wide")
init_page()
st.title("Model Metrics")

registry = load_training_registry()
render_registry_banner(registry)

st.divider()

dataset_id = render_dataset_selector()
summary = load_phase3_summary(dataset_id)

models_on_disk = list_available_models(dataset_id)
if models_on_disk:
    st.caption(
        "Models on disk: "
        + ", ".join(f"`{name}`" for name in models_on_disk)
    )

if summary:
    render_phase3_summary(summary, dataset_id)
else:
    st.warning(
        f"No Phase 3 summary for **{dataset_id}**. Import Colab outputs:\n"
        "`python scripts/import_cmapss_colab_outputs.py --force`"
    )

st.divider()
st.subheader("MLflow runs (local or Databricks)")

tracking = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "predictive_maintenance")

if tracking == "databricks":
    st.info(
        f"Tracking URI is **databricks**. View runs in your workspace under "
        f"**Experiments → {experiment}**. Set env vars locally to query runs here."
    )
else:
    mlruns_path = Path(tracking)
    if not mlruns_path.exists() or not any(mlruns_path.iterdir()):
        st.warning(
            "No local `mlruns/` folder. Train or import Colab outputs; "
            "Databricks runs are in the workspace UI."
        )
    else:
        try:
            import mlflow

            mlflow.set_tracking_uri(str(mlruns_path))
            exp = mlflow.get_experiment_by_name(experiment)
            if exp:
                runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], max_results=50)
                phase3 = runs[
                    runs["tags.mlflow.runName"].astype(str).str.endswith("_phase3_summary", na=False)
                ]
                st.dataframe(
                    phase3 if not phase3.empty else runs,
                    use_container_width=True,
                )
            else:
                st.info(f"Experiment '{experiment}' not found in {mlruns_path}.")
        except Exception as exc:
            st.error(f"Could not load MLflow data: {exc}")
