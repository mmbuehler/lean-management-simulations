# pages/1_takt_animation.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dataclasses import dataclass, asdict

# -----------------------------
# 1) Streamlit setup
# -----------------------------
st.set_page_config(page_title="Takt Animation", layout="wide")
st.title("Takt-Control – Animation (Prototype)")

# -----------------------------
# 2) Config / Data model
# -----------------------------
@dataclass
class Trade:
    name: str
    base_duration: int      # in "work units"
    crew: int
    var: float = 0.0        # variability (0..)
    rework_p: float = 0.0   # probability of rework step

@dataclass
class Config:
    seed: int = 1
    takt_time: int = 3
    rows: int = 2               # e.g., floors
    cols: int = 8               # e.g., rooms per floor
    steps: int = 30
    harmonized: bool = False
    speed_ms: int = 200

def default_trades(harmonized: bool) -> list[Trade]:
    # TODO: later: load from YAML/JSON
    if not harmonized:
        return [
            Trade("Trockenbau", 3, 1, var=0.3),
            Trade("Elektro", 4, 2, var=0.2),
            Trade("HLS", 3, 2, var=0.2),
            Trade("Maler", 2, 1, var=0.1),
        ]
    else:
        # Harmonisierung: z.B. näher an takt_time ziehen
        return [
            Trade("Trockenbau", 3, 1, var=0.15),
            Trade("Elektro", 3, 2, var=0.15),
            Trade("HLS", 3, 2, var=0.15),
            Trade("Maler", 3, 1, var=0.15),
        ]

# Sidebar controls
st.sidebar.header("Einstellungen")
cfg = Config(
    seed=st.sidebar.number_input("Random seed", 0, 10_000, 7),
    takt_time=st.sidebar.slider("Taktzeit (Einheiten)", 1, 10, 3),
    rows=st.sidebar.slider("Taktbereiche – Reihen", 1, 6, 2),
    cols=st.sidebar.slider("Taktbereiche – Spalten", 2, 20, 8),
    steps=st.sidebar.slider("Simulationstakte", 5, 200, 40),
    harmonized=st.sidebar.checkbox("Harmonisiert", value=False),
    speed_ms=st.sidebar.slider("Abspielgeschwindigkeit (ms)", 50, 800, 200),
)

trades = default_trades(cfg.harmonized)

# -----------------------------
# 3) Simulation engine
# -----------------------------
def init_state(cfg: Config, trades: list[Trade]) -> dict:
    rng = np.random.default_rng(cfg.seed)
    # Each trade starts before first cell (pos = -1)
    return {
        "rng": rng,
        "trade_pos": {t.name: -1 for t in trades},        # linear index 0..(rows*cols-1)
        "trade_rem": {t.name: 0 for t in trades},         # remaining work in current cell
        "done_cells": {t.name: set() for t in trades},    # completed cells
    }

def sample_duration(rng, trade: Trade, takt_time: int) -> int:
    # Simple variability model (you can improve later)
    base = trade.base_duration
    if trade.var <= 0:
        return base
    delta = int(np.round(rng.normal(0, trade.var * base)))
    d = max(1, base + delta)
    return d

def step(state: dict, cfg: Config, trades: list[Trade], t: int) -> list[dict]:
    """One takt step. Returns event rows."""
    rng = state["rng"]
    events = []
    n_cells = cfg.rows * cfg.cols

    # occupancy: which trade is currently in which cell
    occ = {}
    for tr in trades:
        pos = state["trade_pos"][tr.name]
        if pos >= 0 and pos < n_cells:
            occ[pos] = tr.name

    # process trades in order (trade train)
    for tr in trades:
        name = tr.name
        pos = state["trade_pos"][name]

        # if not yet started, try to enter cell 0
        if pos == -1:
            next_pos = 0
        else:
            next_pos = pos + 1

        # If currently working in a cell, decrement remaining
        if pos >= 0 and state["trade_rem"][name] > 0:
            state["trade_rem"][name] -= cfg.takt_time
            status = "working"
            if state["trade_rem"][name] <= 0:
                state["done_cells"][name].add(pos)
                status = "completed"
            events.append({"takt": t, "trade": name, "pos": pos, "status": status})
            continue

        # If completed current cell (or idle), try to move forward
        if next_pos >= n_cells:
            events.append({"takt": t, "trade": name, "pos": pos, "status": "finished_all"})
            continue

        # Blocking: next cell occupied by another trade
        if next_pos in occ:
            events.append({"takt": t, "trade": name, "pos": pos, "status": "blocked"})
            continue

        # Move into next cell and start work
        state["trade_pos"][name] = next_pos
        occ[next_pos] = name
        dur = sample_duration(rng, tr, cfg.takt_time)
        state["trade_rem"][name] = dur
        events.append({"takt": t, "trade": name, "pos": next_pos, "status": "start"})
    return events

def run_sim(cfg: Config, trades: list[Trade]) -> pd.DataFrame:
    state = init_state(cfg, trades)
    all_events = []
    for t in range(cfg.steps):
        all_events.extend(step(state, cfg, trades, t))
    return pd.DataFrame(all_events)

timeline = run_sim(cfg, trades)

# -----------------------------
# 4) KPIs
# -----------------------------
def compute_kpis(timeline: pd.DataFrame, cfg: Config) -> dict:
    k = {}
    k["events"] = len(timeline)
    k["blocked"] = int((timeline["status"] == "blocked").sum())
    k["starts"] = int((timeline["status"] == "start").sum())
    k["completed"] = int((timeline["status"] == "completed").sum())
    return k

kpis = compute_kpis(timeline, cfg)

# -----------------------------
# 5) Visualization
# -----------------------------
def grid_frame(timeline: pd.DataFrame, cfg: Config, takt: int) -> np.ndarray:
    """Return grid with trade index or -1."""
    grid = -1 * np.ones((cfg.rows, cfg.cols), dtype=int)
    df = timeline[timeline["takt"] == takt]
    trade_names = [t.name for t in trades]
    idx = {name: i for i, name in enumerate(trade_names)}
    for _, r in df.iterrows():
        if r["pos"] >= 0:
            rr = r["pos"] // cfg.cols
            cc = r["pos"] % cfg.cols
            grid[rr, cc] = idx[r["trade"]]
    return grid

def plot_grid(grid: np.ndarray, trades: list[Trade], title: str):
    # Plotly heatmap with numeric categories
    fig = go.Figure(data=go.Heatmap(
        z=grid,
        coloraxis="coloraxis",
        showscale=False,
        hovertemplate="Row %{y}<br>Col %{x}<br>Trade %{z}<extra></extra>",
    ))
    fig.update_layout(
        title=title,
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis=dict(colorscale="Viridis"),
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False, autorange="reversed"),
    )
    return fig

# -----------------------------
# 6) UI
# -----------------------------
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("KPIs")
    st.metric("Starts", kpis["starts"])
    st.metric("Blocked", kpis["blocked"])
    st.metric("Completed", kpis["completed"])
    st.caption("Prototype-KPIs – werden im nächsten Schritt präzisiert.")

with col1:
    takt = st.slider("Takt", 0, cfg.steps - 1, 0)
    grid = grid_frame(timeline, cfg, takt)
    fig = plot_grid(grid, trades, title=f"Takt {takt}")
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Timeline (debug)"):
    st.dataframe(timeline, use_container_width=True)
