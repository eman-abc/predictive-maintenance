"""MLflow experiment comparison view."""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dashboard.data_loader import load_phase3_summary, render_dataset_selector

st.set_page_config(page_title="Model Metrics", layout="wide")
st.title("Model Metrics")

dataset_id = render_dataset_selector()
summary = load_phase3_summary(dataset_id)
if summary:
    st.caption(f"Summary from **artifacts/cmapss_{dataset_id}_phase3_summary.json**")
    st.json(summary)

mlruns_path = Path(os.getenv("MLFLOW_TRACKING_URI", "./mlruns"))

if not mlruns_path.exists() or not any(mlruns_path.iterdir()):
    st.warning("No MLflow runs found. Train models with `python -m src.models.train` first.")
    st.markdown(
        """
        Expected metrics after training:
        | Model | Metric | Target |
        |-------|--------|--------|
        | RUL Regressor | RMSE | < 20 |
        | RUL Regressor | NASA Score | < 500 |
        | Failure Classifier | F1 | > 0.85 |
        | Failure Classifier | ROC-AUC | > 0.90 |
        """
    )
else:
    try:
        import mlflow

        mlflow.set_tracking_uri(str(mlruns_path))
        experiment = mlflow.get_experiment_by_name("predictive_maintenance")
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            st.dataframe(runs, use_container_width=True)
        else:
            st.info("Experiment 'predictive_maintenance' not found yet.")
    except Exception as exc:
        st.error(f"Could not load MLflow data: {exc}")
