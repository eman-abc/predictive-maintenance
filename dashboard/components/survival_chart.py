"""Cox PH survival curve for Asset Detail."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.utils.visualizations import plot_survival_curve


def render_survival_chart(
    curve_df: pd.DataFrame,
    asset_id: str,
    *,
    rul_pred_cox: float | None = None,
    survival_prob_30: float | None = None,
) -> None:
    """Plot conditional survival from current cycle forward."""
    if curve_df is None or curve_df.empty:
        st.info("Survival curve unavailable for this asset.")
        return

    fig = plot_survival_curve(curve_df, asset_id)
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(2)
    if rul_pred_cox is not None and pd.notna(rul_pred_cox):
        cols[0].metric("Cox median RUL", f"{rul_pred_cox:.0f} cycles")
    if survival_prob_30 is not None and pd.notna(survival_prob_30):
        cols[1].metric("P(survive 30+ cycles)", f"{survival_prob_30:.0%}")
    st.caption(
        "From Cox proportional hazards (lifelines): probability the engine remains "
        "operational beyond each future cycle, given its current sensor state."
    )
