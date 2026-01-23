import simpy
import random
import pandas as pd
import plotly.graph_objects as go


# -----------------------------
# Konfiguration
# -----------------------------
CONFIG = {
    "seed": 3,
    "num_workfaces": 8,          # Anzahl Taktbereiche (z. B. Wohnungen / Achsen / Räume)
    "takt_time_target": 3,       # Soll-Taktzeit (Zeiteinheiten)
    "crews": [
        # name, min_dur, max_dur, rework_prob
        ("Rohinstallation", 2, 4, 0.10),
        ("Trockenbau",      2, 5, 0.15),
        ("Spachtel/Maler",  2, 4, 0.08),
        ("Bodenbelag",      1, 4, 0.05),
    ],
    "disturbance_prob": 0.12,    # Wahrscheinlichkeit für Störung je Task
    "disturbance_delay": (1, 3), # Verzögerung (min,max)
    "max_wait_before_support": 4,# Steuerung: wenn Kolonne >x wartet -> Support
    "support_time_reduction": 1, # Support verkürzt Restdauer um x (mind. 1)
}


# -----------------------------
# Hilfsfunktionen
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def log_segment(log, crew, workface, seg_type, start, finish, detail=""):
    log.append({
        "Crew": crew,
        "Taktbereich": workface,
        "Type": seg_type,     # Work / Waiting / Disturbance / Rework / Support
        "Start": float(start),
        "Finish": float(finish),
        "Dauer": float(finish - start),
        "Detail": detail
    })


# -----------------------------
# Steuerungslogik / "Make ready"
# -----------------------------
def is_ready_to_start(env, crew_idx, wf_idx, done_matrix):
    """
    Simple Make-Ready Regel:
    Crew darf in Workface wf_idx nur starten, wenn:
      - Vorgängercrew im selben Workface fertig ist (crew_idx-1 abgeschlossen)
      - UND die Crew im vorherigen Workface (wf_idx-1) fertig ist (damit Fluss gewährleistet)
    """
    if crew_idx > 0 and not done_matrix[crew_idx - 1][wf_idx]:
        return False
    if wf_idx > 0 and not done_matrix[crew_idx][wf_idx - 1]:
        return False
    return True


# -----------------------------
# Crew-Prozess
# -----------------------------
def crew_process(env, crew_idx, crew_name, workfaces, done_matrix, log, cfg):
    num_wf = len(workfaces)

    for wf_idx in range(num_wf):
        wf = workfaces[wf_idx]

        # Taktsteuerung: erst "release", wenn ready
        while not is_ready_to_start(env, crew_idx, wf_idx, done_matrix):
            # Das ist taktsteuerungsrelevantes Warten (Constraint nicht erfüllt)
            t0 = env.now
            yield env.timeout(1)
            t1 = env.now
            log_segment(log, crew_name, f"WF{wf_idx+1}", "Waiting", t0, t1, "not-ready (constraints)")

        # Ressource Workface holen (Kapazität 1)
        wait_start = env.now
        with wf.request() as req:
            yield req
            wait_end = env.now

            if wait_end > wait_start:
                log_segment(log, crew_name, f"WF{wf_idx+1}", "Waiting", wait_start, wait_end, "workface busy")

            # Wenn sehr lange gewartet -> Support-Team eingreifen (Steuerung)
            waited = wait_end - wait_start
            support_applied = False

            # Basisdauer
            _, dmin, dmax, rework_prob = cfg["crews"][crew_idx]
            duration = random.randint(dmin, dmax)
            planned = duration

            # Störung?
            if random.random() < cfg["disturbance_prob"]:
                delay = random.randint(*cfg["disturbance_delay"])
                t0 = env.now
                yield env.timeout(delay)
                t1 = env.now
                log_segment(log, crew_name, f"WF{wf_idx+1}", "Disturbance", t0, t1, f"delay={delay}")

            # Support-Regel: bei langem Warten verkürzen wir die Arbeit etwas (z. B. zusätzliche Leute/Material)
            if waited >= cfg["max_wait_before_support"]:
                duration = clamp(duration - cfg["support_time_reduction"], 1, 9999)
                support_applied = True

            # Arbeit
            t0 = env.now
            yield env.timeout(duration)
            t1 = env.now
            detail = f"planned={planned}, actual={duration}"
            log_segment(log, crew_name, f"WF{wf_idx+1}", "Work", t0, t1, detail)

            if support_applied:
                # Support-Segment als Meta-Info (0 Dauer) oder als eigener Block (hier 0.5)
                s0 = t0
                s1 = t0 + 0.5
                log_segment(log, crew_name, f"WF{wf_idx+1}", "Support", s0, s1, "support applied")

            # Rework?
            if random.random() < rework_prob:
                rework_dur = 1
                r0 = env.now
                yield env.timeout(rework_dur)
                r1 = env.now
                log_segment(log, crew_name, f"WF{wf_idx+1}", "Rework", r0, r1, "quality issue")

            # Done setzen (für constraints)
            done_matrix[crew_idx][wf_idx] = True


# -----------------------------
# Simulation
# -----------------------------
def run_simulation(cfg):
    random.seed(cfg["seed"])
    env = simpy.Environment()

    # Workfaces (Taktbereiche) als Ressourcen (Kapazität 1)
    workfaces = [simpy.Resource(env, capacity=1) for _ in range(cfg["num_workfaces"])]

    # Done-Matrix: crews x workfaces
    done_matrix = [[False for _ in range(cfg["num_workfaces"])] for _ in range(len(cfg["crews"]))]

    log = []

    # Start aller Crews
    for i, (name, *_rest) in enumerate(cfg["crews"]):
        env.process(crew_process(env, i, name, workfaces, done_matrix, log, cfg))

    env.run()
    return pd.DataFrame(log)


# -----------------------------
# Visualisierung: robustes Gantt (numerisch)
# -----------------------------
def plot_gantt(df, show_types=("Work", "Waiting", "Disturbance", "Rework")):
    df = df.copy()
    df = df[df["Type"].isin(show_types)]

    if df.empty:
        raise RuntimeError("Keine Daten für Gantt vorhanden (Filter zu streng?).")

    fig = go.Figure()

    # Wir zeichnen jede Zeile als eigenen horizontalen Balken
    for _, r in df.iterrows():
        fig.add_trace(
            go.Bar(
                y=[r["Crew"]],
                x=[r["Dauer"]],
                base=r["Start"],
                orientation="h",
                hovertext=(
                    f"{r['Crew']}<br>"
                    f"{r['Taktbereich']}<br>"
                    f"{r['Type']}<br>"
                    f"Start: {r['Start']} – Ende: {r['Finish']}<br>"
                    f"{r['Detail']}"
                ),
                hoverinfo="text",
                name=r["Type"],
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Taktplanung & Taktsteuerung – Gantt (SimPy, numerisch)",
        barmode="overlay",  # overlay zeigt Warte-/Störsegmente gut
        xaxis=dict(title="Zeit (Einheiten)", type="linear"),
        yaxis=dict(title="Kolonne", autorange="reversed"),
        height=500,
    )

    fig.write_html("gantt.html", auto_open=True)
    print("✅ Gantt gespeichert als gantt.html")


# -----------------------------
# KPIs
# -----------------------------
def compute_kpis(df, cfg):
    df = df.copy()

    # Durchlaufzeit je Crew (von min Start bis max Finish)
    cycle = df.groupby("Crew").agg(
        start=("Start", "min"),
        finish=("Finish", "max")
    )
    cycle["Durchlaufzeit"] = cycle["finish"] - cycle["start"]

    # Arbeitszeit & Wartezeit je Crew
    work = df[df["Type"] == "Work"].groupby("Crew")["Dauer"].sum()
    wait = df[df["Type"] == "Waiting"].groupby("Crew")["Dauer"].sum()
    dist = df[df["Type"] == "Disturbance"].groupby("Crew")["Dauer"].sum()
    rewk = df[df["Type"] == "Rework"].groupby("Crew")["Dauer"].sum()

    cycle["Arbeitszeit"] = work
    cycle["Wartezeit"] = wait
    cycle["Störung"] = dist
    cycle["Rework"] = rewk
    cycle = cycle.fillna(0)

    # Flow Efficiency = Arbeitszeit / Durchlaufzeit
    cycle["FlowEfficiency"] = (cycle["Arbeitszeit"] / cycle["Durchlaufzeit"]).round(3)

    # Taktabweichung: Work-Dauer vs Soll-Takt
    work_df = df[df["Type"] == "Work"].copy()
    work_df["Abweichung"] = work_df["Dauer"] - cfg["takt_time_target"]
    takt_stats = work_df.groupby("Crew")["Abweichung"].agg(["mean", "std"]).fillna(0)
    takt_stats.columns = ["Taktabweichung_Mittel", "Taktabweichung_Std"]

    kpis = cycle.join(takt_stats, how="left").fillna(0)
    return kpis.reset_index()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df = run_simulation(CONFIG)
    print("\nEventlog (head):\n", df.head(20))

    # Gantt (zeige Work + Waiting + Disturbance + Rework)
    plot_gantt(df, show_types=("Work", "Waiting", "Disturbance", "Rework"))

    # KPIs
    kpis = compute_kpis(df, CONFIG)
    print("\nKPIs:\n", kpis)
