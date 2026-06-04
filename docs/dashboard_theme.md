# Dashboard theme — predictive maintenance palette

## Palette

| Name | Hex | Role |
|------|-----|------|
| Black Sheep | `#110B0F` | App background |
| Telopea | `#30223D` | Cards, plot area, sidebar |
| Prismarine | `#0F727A` | Healthy / nominal |
| Rip Van Periwinkle | `#91A4D7` | Caution / trending (no amber in brand palette) |
| Hawaiian Malasada | `#9D2C0B` | Critical / immediate action |
| Palace Purple | `#68477C` | Borders, grid, accents |

## Semantic mapping (industry)

| State | UI color | Chart line (on dark bg) |
|-------|----------|-------------------------|
| Healthy / nominal | Prismarine `#0F727A` | `#6EC4CC` |
| Caution / trending | Periwinkle `#91A4D7` | `#91A4D7` |
| Critical | Malasada `#9D2C0B` | `#E8957A` (lightened for 3:1 contrast) |

Chart line colors are lightened vs UI fills so sensor/RUL traces meet **≥3:1 contrast** on `#110B0F` / `#30223D`.

## Code

- `src/utils/chart_theme.py` — Plotly layout + semantic constants
- `dashboard/theme.py` — Streamlit CSS + dataframe styling
- `.streamlit/config.toml` — base theme
- `dashboard/page_init.py` — applies theme on every page

Restart Streamlit after theme changes.
