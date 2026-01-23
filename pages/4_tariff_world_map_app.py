#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Demonstrator: Tariffs vs Coalition (Toy Model) + echte Weltkarte
=========================================================================

Was du bekommst:
- Interaktives Streamlit-Dashboard mit Slidern
- "Echte" Weltkarte als Choropleth (Plotly) mit Ländern gefärbt nach Strategie:
    C = Coalition / koordinierte Gegenmaßnahmen
    B = Bilateral Deal / separater Deal
- Iterative Dynamik (Best Response), sodass du Kipp-Punkte siehst
- KPIs + Zeitreihen (Koalitionsgröße, Durchschnittsschaden) + A-Schaden (Aggressor)

Install (lokal):
    pip install streamlit numpy pandas plotly

Run:
    streamlit run tariff_world_map_app.py

Deploy (Streamlit Cloud):
- Datei ins Repo legen
- requirements.txt: streamlit numpy pandas plotly

Hinweis:
- Das Modell ist absichtlich "stylized": es zeigt Anreizlogik / Koalitions-Kipppunkte,
  nicht reale Länderpolitik. Du kannst es im Kurs als "Tariff Bully vs Coalition" labeln.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Default country set (ISO-3)
# -----------------------------
# Gute didaktische Mischung: große Player + viele mittelgroße
DEFAULT_ISO3 = [
    "USA", "CAN", "MEX", "BRA", "ARG", "CHL", "COL", "PER",
    "GBR", "IRL", "FRA", "DEU", "ESP", "PRT", "ITA", "NLD", "BEL", "CHE", "AUT", "SWE", "NOR", "DNK", "FIN", "POL",
    "TUR", "UKR",
    "EGY", "ZAF", "NGA", "KEN", "MAR",
    "SAU", "ARE", "ISR", "IRN",
    "IND", "PAK", "BGD",
    "CHN", "JPN", "KOR", "TWN",
    "THA", "VNM", "IDN", "MYS", "SGP", "PHL",
    "AUS", "NZL",
]


# Optional: grobe "Exposure" (Exportabhängigkeit) per Land als Startwert.
# Du kannst das später durch echte Daten ersetzen.
# Werte sind Indizes (nicht Milliarden), nur relative Größenordnung.
EXPOSURE_SEED: Dict[str, float] = {
    "CHN": 18, "DEU": 14, "JPN": 13, "KOR": 12, "IND": 11,
    "MEX": 12, "CAN": 11, "USA": 10, "GBR": 10, "FRA": 10,
    "ITA": 9, "ESP": 8, "NLD": 8, "BRA": 8, "AUS": 9,
    "TUR": 7, "IDN": 8, "VNM": 8, "THA": 7, "SGP": 9,
    "SAU": 7, "ARE": 6, "ZAF": 6, "NGA": 5, "EGY": 5,
    "SWE": 6, "NOR": 6, "DNK": 6, "FIN": 5, "POL": 6,
}


# -----------------------------
# Model
# -----------------------------
@dataclass
class Params:
    tau: float          # tariff rate (0..0.5) as fraction
    epsilon: float      # sensitivity to tariffs

    d: float            # deal discount (effective tariff tau*(1-d))
    c: float            # concession cost for deal

    K: float            # coalition coordination cost
    rho: float          # coalition effectiveness
    alpha: float        # coalition nonlinearity (>=1)

    # "Aggressor" cost parameters (for A-loss)
    A_trade_value: float    # baseline benefit of imposing tariffs (per unit exposure hit)
    A_retaliation_cost: float  # how costly coalition retaliation is for A
    beta: float             # nonlinearity for A retaliation cost

    update_mode: str    # "sync" or "async"
    steps: int
    seed: int


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def shield(m: int, N: int, rho: float, alpha: float) -> float:
    """
    Shield factor in [0, 1.2]:
      coalition reduces damage as m grows:
        L_coal = L * shield(m)
    """
    if N <= 0:
        return 1.0
    frac = clamp(m / N, 0.0, 1.0)
    s = 1.0 - rho * (frac ** alpha)
    return clamp(s, 0.0, 1.2)


def base_loss(exposure: float, tau: float, epsilon: float) -> float:
    # simple linear loss proxy
    return float(exposure * epsilon * tau)


def loss_deal(exposure: float, tau: float, epsilon: float, d: float, c: float) -> float:
    tau_eff = tau * (1.0 - d)
    return base_loss(exposure, tau_eff, epsilon) + c


def loss_coal(
    exposure: float, tau: float, epsilon: float, m: int, N: int, rho: float, alpha: float, K: float
) -> float:
    return base_loss(exposure, tau, epsilon) * shield(m, N, rho, alpha) + K


def aggressor_loss(
    exposures: np.ndarray,
    strategies: np.ndarray,
    tau: float,
    epsilon: float,
    rho: float,
    alpha: float,
    A_trade_value: float,
    A_retaliation_cost: float,
    beta: float,
) -> float:
    """
    A "loss" (negative utility) proxy:
    - A gains from tariff impact on trade partners (trade value captured / leverage)
    - but coalition retaliation imposes costs increasing with coalition size m

    We model:
      benefit ~ A_trade_value * sum_i base_loss_i(tau)   (the bigger the shock, the "more leverage")
      retaliation cost ~ A_retaliation_cost * (m/N)^beta * sum_i base_loss_i(tau)
    net utility ~ benefit - retaliation_cost
    We report "A_loss" = -net_utility (so smaller is better for A).
    """
    N = len(exposures)
    m = int(strategies.sum())
    shock = float(np.sum(exposures * epsilon * tau))

    frac = clamp(m / N, 0.0, 1.0)
    ret = A_retaliation_cost * (frac ** beta) * shock
    benefit = A_trade_value * shock

    net = benefit - ret
    return float(-net)


def best_response_one(
    exposure: float,
    current_strat_i: int,
    m_current: int,
    N: int,
    p: Params,
) -> int:
    # if i chooses coalition, coalition size increases by 1 if currently bilateral
    m_if_coal = m_current + (1 if current_strat_i == 0 else 0)

    Lb = loss_deal(exposure, p.tau, p.epsilon, p.d, p.c)
    Lc = loss_coal(exposure, p.tau, p.epsilon, m_if_coal, N, p.rho, p.alpha, p.K)

    return 1 if Lc < Lb else 0  # 1=C, 0=B


def run_dynamics(
    iso3: List[str],
    exposures: np.ndarray,
    p: Params,
) -> Tuple[List[np.ndarray], pd.DataFrame]:
    """
    Returns:
      strategies_hist: list of (N,) arrays with 1=C, 0=B
      metrics_df: dataframe per step with m, avg_loss, A_loss
    """
    random.seed(p.seed)
    np.random.seed(p.seed)

    N = len(iso3)
    strat = np.zeros(N, dtype=int)  # start: all bilateral

    hist: List[np.ndarray] = []
    rows = []

    for t in range(p.steps):
        hist.append(strat.copy())

        m = int(strat.sum())
        losses = np.zeros(N, dtype=float)
        for i in range(N):
            if strat[i] == 1:
                losses[i] = loss_coal(exposures[i], p.tau, p.epsilon, m, N, p.rho, p.alpha, p.K)
            else:
                losses[i] = loss_deal(exposures[i], p.tau, p.epsilon, p.d, p.c)

        avg_loss = float(losses.mean())
        A_loss = aggressor_loss(
            exposures=exposures,
            strategies=strat,
            tau=p.tau,
            epsilon=p.epsilon,
            rho=p.rho,
            alpha=p.alpha,
            A_trade_value=p.A_trade_value,
            A_retaliation_cost=p.A_retaliation_cost,
            beta=p.beta,
        )

        rows.append({"t": t, "m": m, "avg_loss": avg_loss, "A_loss": A_loss})

        # update
        if p.update_mode.lower() == "sync":
            new = strat.copy()
            m_current = int(strat.sum())
            for i in range(N):
                new[i] = best_response_one(exposures[i], strat[i], m_current, N, p)
            strat = new
        else:
            i = random.randrange(N)
            m_current = int(strat.sum())
            strat[i] = best_response_one(exposures[i], strat[i], m_current, N, p)

    metrics_df = pd.DataFrame(rows)
    return hist, metrics_df


# -----------------------------
# UI helpers
# -----------------------------
def make_exposures(iso3: List[str], dispersion: float, seed: int) -> np.ndarray:
    """
    Create per-country exposure index.
    - start from EXPOSURE_SEED if available else mid value
    - add controlled noise/dispersion (slider)
    """
    rng = np.random.default_rng(seed)
    base = np.array([EXPOSURE_SEED.get(code, 8.0) for code in iso3], dtype=float)
    noise = rng.normal(0.0, dispersion, size=len(iso3))
    exp = np.clip(base + noise, 2.0, None)
    return exp


def map_dataframe(
    iso3: List[str],
    exposures: np.ndarray,
    strat: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "iso_alpha": iso3,
            "Exposure": exposures,
            "Strategy": np.where(strat == 1, "C (Coalition)", "B (Bilateral)"),
            "Coalition": (strat == 1).astype(int),
        }
    )


# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Tariff vs Coalition (World Map)", layout="wide")

st.title("Tariff vs Coalition — Weltkarten-Demonstrator (Toy Model)")
st.caption(
    "Stylisiertes Zoll-/Koalitionsmodell: Länder wählen zwischen bilateralem Deal (B) und Koalition (C). "
    "Die Koalition wirkt umso stärker, je mehr Länder mitmachen (Kipppunktlogik)."
)

with st.sidebar:
    st.header("Länder-Set & Dynamik")
    use_default = st.checkbox("Default Länder-Set nutzen", value=True)
    if use_default:
        iso3 = DEFAULT_ISO3.copy()
        st.write(f"Anzahl Länder: **{len(iso3)}**")
    else:
        raw = st.text_area("ISO-3 Codes (kommagetrennt)", value="DEU,FRA,ITA,ESP,NLD,POL,USA,CHN,JPN,KOR,IND,GBR,CAN,MEX,BRA,AUS,ZAF,TUR")
        iso3 = [x.strip().upper() for x in raw.split(",") if x.strip()]
        st.write(f"Anzahl Länder: **{len(iso3)}**")

    update_mode = st.selectbox("Update Mode", ["async", "sync"], index=0)
    steps = st.slider("Iterations (steps)", 20, 250, 100, 5)
    seed = st.slider("Random seed", 1, 999, 7, 1)

    st.divider()
    st.header("Zoll & Deal")
    tau = st.slider("Zollsatz τ (%)", 0, 50, 25, 1) / 100.0
    epsilon = st.slider("Tariff Sensitivity ε", 0.2, 3.0, 1.2, 0.05)

    d = st.slider("Deal-Rabatt d (Zollreduktion)", 0.0, 0.95, 0.60, 0.05)
    c = st.slider("Konzessionskosten c", 0.0, 15.0, 4.0, 0.5)

    st.divider()
    st.header("Koalition")
    K = st.slider("Koordinationskosten K", 0.0, 15.0, 2.5, 0.5)
    rho = st.slider("Koalitionsstärke ρ", 0.0, 1.5, 0.90, 0.05)
    alpha = st.slider("Skalierung α (Kipppunkt)", 1.0, 4.0, 1.9, 0.1)

    st.divider()
    st.header("Aggressor A (nur KPI/Plot)")
    A_trade_value = st.slider("A: Nutzen pro Zoll-Schock", 0.0, 2.0, 1.0, 0.05)
    A_retaliation_cost = st.slider("A: Retaliation-Kostenfaktor", 0.0, 3.0, 1.2, 0.05)
    beta = st.slider("A: Nichtlinearität β", 1.0, 4.0, 2.0, 0.1)

    st.divider()
    st.header("Exposure (Heterogenität)")
    dispersion = st.slider("Exposure-Streuung", 0.0, 6.0, 1.0, 0.2)

    run = st.button("Simulation neu berechnen", type="primary")


# Cache to avoid recompute on every minor UI change unless button pressed
@st.cache_data(show_spinner=False)
def cached_run(
    iso3: Tuple[str, ...],
    dispersion: float,
    tau: float,
    epsilon: float,
    d: float,
    c: float,
    K: float,
    rho: float,
    alpha: float,
    A_trade_value: float,
    A_retaliation_cost: float,
    beta: float,
    update_mode: str,
    steps: int,
    seed: int,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    iso3_list = list(iso3)
    exposures = make_exposures(iso3_list, dispersion=dispersion, seed=seed)

    p = Params(
        tau=tau,
        epsilon=epsilon,
        d=d,
        c=c,
        K=K,
        rho=rho,
        alpha=alpha,
        A_trade_value=A_trade_value,
        A_retaliation_cost=A_retaliation_cost,
        beta=beta,
        update_mode=update_mode,
        steps=steps,
        seed=seed,
    )

    hist, metrics = run_dynamics(iso3_list, exposures, p)
    final_strat = hist[-1]
    df_map = map_dataframe(iso3_list, exposures, final_strat)
    return final_strat, metrics, df_map


# Trigger compute
if run:
    st.cache_data.clear()

final_strat, metrics_df, df_map = cached_run(
    iso3=tuple(iso3),
    dispersion=dispersion,
    tau=tau,
    epsilon=epsilon,
    d=d,
    c=c,
    K=K,
    rho=rho,
    alpha=alpha,
    A_trade_value=A_trade_value,
    A_retaliation_cost=A_retaliation_cost,
    beta=beta,
    update_mode=update_mode,
    steps=steps,
    seed=seed,
)

N = len(iso3)
m_final = int(final_strat.sum())
avg_loss_final = float(metrics_df["avg_loss"].iloc[-1])
A_loss_final = float(metrics_df["A_loss"].iloc[-1])

# -----------------------------
# Layout
# -----------------------------
col1, col2 = st.columns([1.25, 1.0], gap="large")

with col1:
    st.subheader("Weltkarte: Koalition (C) vs Bilateral (B)")

    # Choropleth: Coalition=1 vs 0
    fig_map = px.choropleth(
        df_map,
        locations="iso_alpha",
        color="Coalition",
        hover_name="iso_alpha",
        hover_data={"Strategy": True, "Exposure": ":.1f", "Coalition": False},
        color_continuous_scale="Blues",
        range_color=(0, 1),
        projection="natural earth",
        title="Countries colored by strategy (1=Coalition, 0=Bilateral)",
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    st.subheader("Länderübersicht")
    st.dataframe(
        df_map.sort_values(["Coalition", "Exposure"], ascending=[False, False]).reset_index(drop=True),
        use_container_width=True,
        height=280,
    )

with col2:
    st.subheader("KPIs")
    k1, k2, k3 = st.columns(3)
    k1.metric("Koalition m", f"{m_final}/{N}")
    k2.metric("Ø Länder-Schaden", f"{avg_loss_final:.2f}")
    k3.metric("A-Loss (Proxy)", f"{A_loss_final:.2f}")

    st.caption(
        "Interpretation: Länder minimieren ihren eigenen erwarteten 'Loss'. "
        "Koalition wird attraktiv, wenn Koalitionswirkung (ρ, α) die Koordinationskosten (K) überkompensiert."
    )

    st.subheader("Dynamik über Zeit")
    fig_m = px.line(metrics_df, x="t", y="m", title="Koalitionsgröße m(t)")
    fig_m.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_m, use_container_width=True)

    fig_loss = px.line(metrics_df, x="t", y=["avg_loss", "A_loss"], title="Ø Länder-Schaden & A-Loss")
    fig_loss.update_layout(margin=dict(l=0, r=0, t=40, b=0), legend_title_text="")
    st.plotly_chart(fig_loss, use_container_width=True)

st.divider()
st.subheader("Didaktische Hinweise (kurz)")
st.markdown(
    """
- **Warum kippt das System?**  
  Bei kleinen m ist Koalition oft zu teuer (K) → Länder wählen bilateral. Ab einem Punkt wirkt Retaliation/Koordination stark genug (ρ, α), dann wird Koalition **selbsttragend**.
- **Was ist der wichtigste Slider?**  
  Meist die Kombination aus **K** (Koordinationskosten) und **ρ/α** (Koalitionswirksamkeit).
- **Wie zeigst du „bilateral lohnt sich nicht“?**  
  Erhöhe **c** (Konzessionskosten) und/oder **d** etwas senken (kleiner Deal-Rabatt) → dann wird sichtbar, dass bilaterale Deals teuer sind und Koalition stabiler wird.
"""
)
