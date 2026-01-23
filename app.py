import streamlit as st

st.set_page_config(
    page_title="Lean Management Simulations",
    page_icon="📌",
    layout="wide",
)

# ---------- minimal CSS for a clean, academic look ----------
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      .muted {color: rgba(49,51,63,0.65); font-size: 0.95rem;}
      .card {
        border: 1px solid rgba(49,51,63,0.12);
        border-radius: 14px;
        padding: 16px 18px;
        background: white;
        height: 100%;
      }
      .card h4 {margin: 0 0 6px 0;}
      .pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        background: rgba(49,51,63,0.06);
        font-size: 0.85rem;
        margin-right: 6px;
      }
      hr {margin: 1.2rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- header ----------
st.title("Lean Management & Game Theory Simulations")
st.markdown(
    "<div class='muted'>Interactive teaching demonstrators for Lean Construction, queueing dynamics, and strategic interaction.</div>",
    unsafe_allow_html=True,
)

st.divider()

# ---------- overview cards ----------
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown(
        """
        <div class="card">
          <span class="pill">Lean Construction</span>
          <span class="pill">Takt</span>
          <h4>Takt Control</h4>
          <div class="muted">
            Explore takt time, variability and basic KPI outputs. Designed for quick classroom demos.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        """
        <div class="card">
          <span class="pill">Queues</span>
          <span class="pill">SimPy</span>
          <h4>Concrete Queue</h4>
          <div class="muted">
            Delivery arrivals, waiting time, utilization and bottlenecks. Links directly to Little’s Law intuition.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        """
        <div class="card">
          <span class="pill">Game Theory</span>
          <span class="pill">Trade</span>
          <h4>Tariffs & Coalition</h4>
          <div class="muted">
            Coalition formation vs. bilateral deals. Parameter-driven tipping dynamics and outcome KPIs.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

st.markdown(
    """
**Navigation:** Use the left sidebar to open a module (Streamlit multipage menu).  
**Tip for teaching:** Start with *Concrete Queue* → then *Takt* → finish with *Tariffs & Coalition* for strategic interpretation.
"""
)
