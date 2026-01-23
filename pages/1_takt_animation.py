import time
import math
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Takt-Animation", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def build_layout(n_areas: int, cols: int):
    """area_id 1..n_areas -> (row, col) grid position"""
    cols = max(1, cols)
    rows = math.ceil(n_areas / cols)
    layout = {}
    for i in range(n_areas):
        area = i + 1
        r = i // cols
        c = i % cols
        layout[area] = (r, c)
    return layout, rows, cols

def simulate_positions(t: int, n_areas: int, trades: list[str], start_offsets: list[int], takt_time: int):
    """
    Each trade moves one area every takt_time steps.
    Returns trade -> area (0 if not started/finished)
    """
    pos = {}
    for trade, off in zip(trades, start_offsets):
        if t < off:
            pos[trade] = 0
            continue
        k = (t - off) // takt_time
        area = 1 + k
        pos[trade] = area if 1 <= area <= n_areas else 0
    return pos

def build_takttreppe_series_upto(
    n_areas: int,
    trades: list[str],
    start_offsets: list[int],
    takt_time: int,
    t_upto: int,
):
    """
    Build step-lines ONLY up to current time t_upto.
    """
    series = {}
    for trade, off in zip(trades, start_offsets):
        xs, ys = [], []
        if t_upto < off:
            series[trade] = (xs, ys)
            continue

        # We add horizontal segments for each area that has started by t_upto
        for area in range(1, n_areas + 1):
            t_start = off + (area - 1) * takt_time
            t_end = t_start + takt_time
            if t_start > t_upto:
                break

            # segment is [t_start, min(t_end, t_upto)]
            xs.extend([t_start, min(t_end, t_upto)])
            ys.extend([area, area])

        series[trade] = (xs, ys)
    return series

def make_color_map(trades: list[str]):
    """
    Consistent colors across layout + takttreppe.
    (Plotly-like palette; extend if needed)
    """
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#4c78a8", "#f58518",
    ]
    return {tr: palette[i % len(palette)] for i, tr in enumerate(trades)}

# -----------------------------
# UI
# -----------------------------
st.title("Takt-Animation: Gewerkezug + Takttreppe")

with st.sidebar:
    st.header("Einstellungen")

    n_areas = st.slider("Anzahl Taktbereiche", 6, 60, 24, 1)
    grid_cols = st.slider("Layout-Spalten (Kachel-Grid)", 3, 12, 8, 1)

    trades_default = ["Trockenbau", "Elektro", "HLS", "Estrich", "Maler", "Fliesen", "Boden", "Schreiner"]
    n_trades = st.slider("Anzahl Gewerke", 2, 12, 8, 1)
    trades = trades_default[:n_trades]

    takt_time = st.slider("Taktzeit (Zeitschritte je Bereich)", 1, 10, 3, 1)

    st.subheader("Versatz (Start) je Gewerk")
    auto_offsets = st.checkbox("Automatisch staffeln (klassischer Gewerkezug)", value=True)
    if auto_offsets:
        start_offsets = [i * takt_time for i in range(n_trades)]
        st.caption(f"Start-Offsets: {start_offsets}")
    else:
        start_offsets = []
        for i, tr in enumerate(trades):
            start_offsets.append(
                st.number_input(
                    f"{tr}: Start bei t=",
                    min_value=0,
                    max_value=5000,
                    value=i * takt_time,
                    step=1,
                )
            )

    st.subheader("Animation")
    speed_ms = st.slider("Geschwindigkeit (ms pro Frame)", 150, 1500, 450, 50)
    t_max_default = max(120, n_areas * takt_time + n_trades * takt_time)
    t_max = st.slider("Simulationsdauer (Zeitschritte)", 30, 2000, t_max_default, 10)

# -----------------------------
# Session state for animation
# -----------------------------
if "run" not in st.session_state:
    st.session_state.run = False
if "t" not in st.session_state:
    st.session_state.t = 0

c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
with c1:
    if st.button("▶ Play", use_container_width=True):
        st.session_state.run = True
with c2:
    if st.button("⏸ Pause", use_container_width=True):
        st.session_state.run = False
with c3:
    if st.button("↺ Reset", use_container_width=True):
        st.session_state.run = False
        st.session_state.t = 0
with c4:
    st.write(f"**Zeit t = {st.session_state.t}** / {t_max}")

# Tick (real animation)
if st.session_state.run:
    time.sleep(speed_ms / 1000.0)
    st.session_state.t += 1
    if st.session_state.t > t_max:
        st.session_state.t = 0
    st.rerun()

# -----------------------------
# Compute current state
# -----------------------------
t_current = int(st.session_state.t)
layout_map, rows, cols_eff = build_layout(n_areas, grid_cols)
positions = simulate_positions(t_current, n_areas, trades, start_offsets, takt_time)
color_map = make_color_map(trades)

# -----------------------------
# Figure A: Layout with moving trades (colored)
# -----------------------------
fig_layout = go.Figure()
tile_w, tile_h = 1.0, 1.0

for area, (r, c) in layout_map.items():
    x0 = c * tile_w
    y0 = (rows - 1 - r) * tile_h  # invert y so first row is top
    x1 = x0 + tile_w
    y1 = y0 + tile_h

    fig_layout.add_shape(
        type="rect",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line=dict(width=1),
        fillcolor="rgba(0,0,0,0)",
        layer="below",
    )
    fig_layout.add_annotation(
        x=x0 + 0.5 * tile_w,
        y=y0 + 0.5 * tile_h,
        text=str(area),
        showarrow=False,
        font=dict(size=11),
        opacity=0.35,
    )

# One marker per trade with its own color
xs, ys, labels, colors = [], [], [], []
for tr in trades:
    area = positions.get(tr, 0)
    if area == 0:
        continue
    r, c = layout_map[area]
    xs.append(c * tile_w + 0.5 * tile_w)
    ys.append((rows - 1 - r) * tile_h + 0.5 * tile_h)
    labels.append(tr)
    colors.append(color_map[tr])

fig_layout.add_trace(
    go.Scatter(
        x=xs,
        y=ys,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=18, color=colors, line=dict(width=1)),
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
    )
)

fig_layout.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=40, b=10),
    title="Layout: Taktbereiche (Kacheln) + aktueller Standort der Gewerke",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
)
fig_layout.update_yaxes(scaleanchor="x", scaleratio=1)

# -----------------------------
# Figure B: Takttreppe up to current time + current points
# -----------------------------
series = build_takttreppe_series_upto(
    n_areas=n_areas,
    trades=trades,
    start_offsets=start_offsets,
    takt_time=takt_time,
    t_upto=t_current,
)

fig_step = go.Figure()

# Step-lines only up to current time, with consistent colors
for tr, (x, y) in series.items():
    fig_step.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=tr,
            line=dict(color=color_map[tr], width=2),
            hovertemplate=f"{tr}<br>t=%{{x}}<br>Taktbereich=%{{y}}<extra></extra>",
        )
    )

# Current position markers at x = t_current
cur_xs, cur_ys, cur_cols, cur_text = [], [], [], []
for tr in trades:
    area = positions.get(tr, 0)
    if area == 0:
        continue
    cur_xs.append(t_current)
    cur_ys.append(area)
    cur_cols.append(color_map[tr])
    cur_text.append(tr)

fig_step.add_trace(
    go.Scatter(
        x=cur_xs,
        y=cur_ys,
        mode="markers",
        marker=dict(size=10, color=cur_cols, line=dict(width=1)),
        text=cur_text,
        hovertemplate="%{text}<br>t=%{x}<br>Taktbereich=%{y}<extra></extra>",
        showlegend=False,
    )
)

# Current time vertical line
fig_step.add_vline(x=t_current, line_width=2)

fig_step.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=40, b=10),
    title="Takttreppe: Zeit (x) vs. Taktbereich (y) — nur bis aktuelle Zeit t",
    xaxis_title="Zeit (Zeitschritte)",
    yaxis_title="Taktbereich",
)

# Keep x-axis stable to t_max (optional). If you prefer "growing x-axis", set range=[0, max(1,t_current)]
fig_step.update_xaxes(range=[0, max(1, t_max)])

fig_step.update_yaxes(
    autorange="reversed",  # optional: oben = Bereich 1
    tickmode="linear",
    tick0=1,
    dtick=1,
)

# -----------------------------
# Render side-by-side
# -----------------------------
left, right = st.columns([1, 1])
with left:
    st.plotly_chart(fig_layout, use_container_width=True)
with right:
    st.plotly_chart(fig_step, use_container_width=True)

st.caption("Hinweis: Navigation erfolgt über Streamlit 'pages/'. Bitte kein eigenes Sidebar-Routing in app.py verwenden.")
