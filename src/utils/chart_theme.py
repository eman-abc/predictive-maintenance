"""Predictive maintenance color system — palette + semantic roles + Plotly layout."""

from __future__ import annotations

# Brand palette (UC5 dashboard)
RIP_VAN_PERIWINKLE = "#91A4D7"
PRISMARINE = "#0F727A"
PALACE_PURPLE = "#68477C"
HAWAIIAN_MALASADA = "#9D2C0B"
TELOPEA = "#30223D"
BLACK_SHEEP = "#110B0F"

# Semantic (industry-standard meaning, palette-backed)
COLOR_HEALTHY = PRISMARINE
COLOR_CAUTION = RIP_VAN_PERIWINKLE  # no amber in palette; light periwinkle = trending/caution
COLOR_CRITICAL = HAWAIIAN_MALASADA
COLOR_NOMINAL = PRISMARINE

# Chart lines on BLACK_SHEEP — lightened for ≥3:1 contrast vs dark background
COLOR_HEALTHY_LINE = "#6EC4CC"
COLOR_CAUTION_LINE = RIP_VAN_PERIWINKLE
COLOR_CRITICAL_LINE = "#E8957A"
COLOR_ACCENT_LINE = "#B49AD4"  # palace purple lightened
COLOR_SURVIVAL_LINE = COLOR_HEALTHY_LINE
COLOR_RUL_LINE = COLOR_CAUTION_LINE

COLOR_BG = BLACK_SHEEP
COLOR_SURFACE = TELOPEA
COLOR_TEXT = "#EDE8F2"
COLOR_TEXT_MUTED = RIP_VAN_PERIWINKLE
COLOR_GRID = "#4A3D55"

SENSOR_LINE_COLORS = [
    COLOR_HEALTHY_LINE,
    COLOR_CAUTION_LINE,
    COLOR_ACCENT_LINE,
    "#C9B8E8",
    COLOR_CRITICAL_LINE,
]

ALERT_LEVEL_COLORS = {
    "normal": COLOR_HEALTHY,
    "healthy": COLOR_HEALTHY,
    "warning": COLOR_CAUTION,
    "caution": COLOR_CAUTION,
    "critical": COLOR_CRITICAL,
}

PLOTLY_LAYOUT: dict = {
    "template": "plotly_dark",
    "paper_bgcolor": COLOR_BG,
    "plot_bgcolor": COLOR_SURFACE,
    "font": {"family": "Segoe UI, system-ui, sans-serif", "color": COLOR_TEXT, "size": 13},
    "title": {"font": {"size": 16, "color": COLOR_TEXT}},
    "xaxis": {
        "gridcolor": COLOR_GRID,
        "linecolor": COLOR_GRID,
        "zerolinecolor": COLOR_GRID,
        "tickfont": {"color": COLOR_TEXT_MUTED},
    },
    "yaxis": {
        "gridcolor": COLOR_GRID,
        "linecolor": COLOR_GRID,
        "zerolinecolor": COLOR_GRID,
        "tickfont": {"color": COLOR_TEXT_MUTED},
    },
    "legend": {"bgcolor": "rgba(17,11,15,0.8)", "bordercolor": COLOR_GRID, "font": {"color": COLOR_TEXT}},
    "margin": {"l": 48, "r": 24, "t": 56, "b": 48},
}


def apply_plotly_theme(fig):
    """Apply PM layout and default trace styling."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def health_gauge_color(score: float) -> str:
    if score >= 70:
        return COLOR_HEALTHY
    if score >= 40:
        return COLOR_CAUTION
    return COLOR_CRITICAL


def health_bar_color(score: float) -> str:
    return health_gauge_color(score)


def alert_level_color(level: str) -> str:
    return ALERT_LEVEL_COLORS.get(str(level).lower(), COLOR_TEXT_MUTED)
