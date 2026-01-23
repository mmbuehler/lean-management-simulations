import time
import math
import random
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Takt-Animation", layout="wide")

# -----------------------------
# Helpers: simple takt flow model
# -----------------------------
def build_layout(n_areas: int, cols: int):
    """
    Returns dict: area_id -> (row, col) in a grid.
    area_id is 1..n_areas
    """
    cols = max(1, cols)
    rows = math.ceil(n_areas / cols)
    layout = {}
    for i in range(n_areas):
        area = i + 1
        r = i // cols
        c = i % cols
        layout[area] = (r, c)
    return layout, rows, cols

def simulate_positions(
    t: int,
    n_areas: int,
    trades: list[str],
    start_offsets: list[int],
    takt_time: int,
):
    """
    Deterministic takt: each trade moves one area every takt_time steps.
    Returns positions dict trade -> area (or 0 if not started / finished).
    """
    pos = {}
    max_area = n_areas
    for trade, off in zip(trades, start_offsets):
        if t < off:
            pos[trade] = 0  # not started
            continue
        # how many takt moves have happened since start
        k = (t - off) // takt_time
        area = 1 + k
        if area > max_area:
            pos[trade] = 0  # finished (hide)
        else:
            pos[trade] = area
    return pos

def build_takttreppe_series(
    n_areas: int,
    trades: list[str],
    start_offsets: list[int],
    takt_time: int,
    t_max: int,
):
    """
    Series for Takttreppe: for each trade, create x(time), y(area) arrays.
    Step-like by repeating points.
    """
    series = {}
    for trade, off in zip(trades, start_offsets):
        xs = []
        ys = []
        # area 1..n_areas each takt_time
        for area in range(1, n_areas + 1):
            t_start = off + (area - 1) * takt_time
            t_end = t_start + takt_time
            if t_start > t_max:
                break
            # step: hold y constant across [t_start, t_end)
            xs.extend([t_start, min(t_end, t_max)])
            ys.extend([area, area])
        series[trade] = (xs, ys)
    return series

# -----------------------------
# UI: Controls
# -----------------------------
st.title("Takt-Animation: Gewerkezug + Takttreppe")

with st.sidebar:
    st.header("Einstellungen")

    n_areas = st.slider("Anzahl Taktbereiche", 6, 60, 24, 1)
    cols = st.slider("Layout-Spalten (Kachel-Grid)", 3, 12, 8, 1)

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
            start_offsets.append(st.number_input(f"{tr}: Start bei t=", min_value=0, max_value=500, value=i * takt_time, step=1))

    st.subheader("Animation")
    speed_ms = st.slider("Geschwindigkeit (ms pro Frame)", 150, 1500, 450, 50)
    t_max = st.slider("Simulationsdauer (Zeitschritte)", 30, 600, max(120, n_areas * takt_time + n_trades * takt_time), 10)

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

# Auto-refresh tick (real animation)
if st.session_state.run:
    st_autorefresh = st.experimental_data_editor  # dummy to avoid lint confusion
    st.experimental_rerun  # keep reference

    # Use Streamlit's autorefresh via a small trick:
    # We sleep and rerun. This works reliably both local + cloud.
    time.sleep(speed_ms / 1000.0)
    st.session_state.t += 1
    if st.session_state.t > t_max:
        st.session_state.t = 0
    st.rerun()

# -----------------------------
# Build figures
# -----------------------------
layout_map, rows, cols_eff = build_layout(n_areas, cols)

positions = simulate_positions(
    t=st.session_state.t,
    n_areas=n_areas,
    trades=trades,
    start_offsets=start_offsets,
    takt_time=takt_time,
)

# ---- Figure A: Layout with moving trades
fig_layout = go.Figure()

# Draw tiles as shapes
tile_w = 1.0
tile_h = 1.0

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
        x=x0 + 0.5*tile_w,
        y=y0 + 0.5*tile_h,
        text=str(area),
        showarrow=False,
        font=dict(size=11),
        opacity=0.65,
    )

# Plot trade positions
xs, ys, labels = [], [], []
for tr in trades:
    area = positions.get(tr, 0)
    if area == 0:
        continue
    r, c = layout_map[area]
    x = c * tile_w + 0.5 * tile_w
    y = (rows - 1 - r) * tile_h + 0.5 * tile_h
    xs.append(x)
    ys.append(y)
    labels.append(tr)

fig_layout.add_trace(
    go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=18),
        hovertemplate="%{text}<extra></extra>",
    )
)

fig_layout.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=40, b=10),
    title="Layout: Taktbereiche (Kacheln) + aktueller Standort der Gewerke",
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
)

# lock aspect ratio
fig_layout.update_yaxes(scaleanchor="x", scaleratio=1)

# ---- Figure B: Takttreppe
series = build_takttreppe_series(
    n_areas=n_areas,
    trades=trades,
    start_offsets=start_offsets,
    takt_time=takt_time,
    t_max=t_max,
)

fig_step = go.Figure()
for tr, (x, y) in series.items():
    fig_step.add_trace(
        go.Scatter(
            x=x, y=y,
            mode="lines",
            name=tr,
            hovertemplate=f"{tr}<br>t=%{{x}}<br>Taktbereich=%{{y}}<extra></extra>"
        )
    )

# add "current time" vertical line
fig_step.add_vline(x=st.session_state.t, line_width=2)

fig_step.update_layout(
    height=520,
    margin=dict(l=10, r=10, t=40, b=10),
    title="Takttreppe: Zeit (x) vs. Taktbereich (y)",
    xaxis_title="Zeit (Zeitschritte)",
    yaxis_title="Taktbereich",
)

fig_step.update_yaxes(
    autorange="reversed",  # optional: oben = Bereich 1
    tickmode="linear",
    tick0=1,
    dtick=1
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
