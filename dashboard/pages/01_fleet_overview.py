"""Fleet overview — all assets health scores."""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Fleet Overview", layout="wide")
st.title("Fleet Overview")

@st.cache_data
def load_fleet_data() -> pd.DataFrame:
    """Load fleet health data or generate demo data."""
    np.random.seed(42)
    n_assets = 20
    return pd.DataFrame({
        "asset_id": [f"ENG-{i:03d}" for i in range(1, n_assets + 1)],
        "health_score": np.random.uniform(40, 98, n_assets).round(1),
        "rul": np.random.randint(5, 120, n_assets),
        "failure_prob": np.random.uniform(0.01, 0.95, n_assets).round(2),
        "status": np.random.choice(["normal", "warning", "critical"], n_assets, p=[0.7, 0.2, 0.1]),
    })

df = load_fleet_data()

col1, col2, col3 = st.columns(3)
col1.metric("Average Health", f"{df['health_score'].mean():.1f}%")
col2.metric("Critical Assets", int((df["status"] == "critical").sum()))
col3.metric("Warning Assets", int((df["status"] == "warning").sum()))

st.dataframe(
    df.style.background_gradient(subset=["health_score"], cmap="RdYlGn"),
    use_container_width=True,
    hide_index=True,
)

st.bar_chart(df.set_index("asset_id")["health_score"])
