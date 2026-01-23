import streamlit as st

st.set_page_config(page_title="Lean Management Simulations", layout="wide")

st.sidebar.title("Simulationen")

module = st.sidebar.radio(
    "Wähle ein Modul:",
    [
        "Start",
        "Warteschlangen & Takt",
        "Spieltheorie: Trump vs. Welt",
        "Zölle & Koalitionen (Weltkarte)",
    ],
)

if module == "Start":
    st.title("Lean Management & Game Theory Simulations")
    st.markdown("""
    ### 🎯 Zweck
    Interaktive Demonstratoren für:
    - Warteschlangen & Taktplanung (Lean Construction)
    - Spieltheorie & internationale Kooperation
    - Zölle, Retaliation & Koalitionen
    """)
    st.success("Streamlit App läuft korrekt 🚀")

elif module == "Warteschlangen & Takt":
    from sim.queue_demo import show
    show()

elif module == "Spieltheorie: Trump vs. Welt":
    from sim.trump_game import show
    show()

elif module == "Zölle & Koalitionen (Weltkarte)":
    from sim.tariff_world_map_app import show
    show()
