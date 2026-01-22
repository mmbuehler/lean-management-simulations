import simpy
import random
import pandas as pd
import plotly.graph_objects as go


# -----------------------------
# SimPy Modell
# -----------------------------
def crew(env, name, workfaces, log):
    """Eine Kolonne, die durch alle Taktbereiche arbeitet und Events loggt."""
    for wf in workfaces:
        # Warten bis Bereich frei
        with wf.request() as req:
            wait_start = env.now
            yield req
            wait_end = env.now

            if wait_end > wait_start:
                log.append(
                    {
                        "Crew": name,
                        "Task": f"Warten ({wf.name})",
                        "Taktbereich": wf.name,
                        "Start": wait_start,
                        "Finish": wait_end,
                        "Type": "Waiting",
                    }
                )

            # Arbeit
            start = env.now
            duration = random.randint(2, 4)
            finish = start + duration

            log.append(
                {
                    "Crew": name,
                    "Task": f"Arbeit ({wf.name})",
                    "Taktbereich": wf.name,
                    "Start": start,
                    "Finish": finish,
                    "Type": "Work",
                }
            )

            print(f"{env.now:>3} | {name} startet in {wf.name}")
            yield env.timeout(duration)
            print(f"{env.now:>3} | {name} verlässt {wf.name} (Dauer {duration})")


def run_simulation(seed=2):
    random.seed(seed)
    env = simpy.Environment()
    log = []

    # 3 Taktbereiche mit Kapazität 1
    workfaces = [simpy.Resource(env, capacity=1) for _ in range(3)]
    for i, wf in enumerate(workfaces):
        wf.name = f"Taktbereich {chr(65 + i)}"  # A, B, C

    env.process(crew(env, "Kolonne 1", workfaces, log))
    env.process(crew(env, "Kolonne 2", workfaces, log))

    env.run()
    return pd.DataFrame(log)


# -----------------------------
# Robustes Gantt (numerisch)
# -----------------------------
def plot_gantt(df):
    df = df.copy()

    # nur Arbeitsblöcke anzeigen (Waiting später optional)
    df = df[df["Type"] == "Work"]

    df["Start"] = df["Start"].astype(float)
    df["Finish"] = df["Finish"].astype(float)
    df["Dauer"] = df["Finish"] - df["Start"]

    if df.empty:
        raise RuntimeError("Keine Daten für Gantt vorhanden.")

    fig = go.Figure()

    for _, r in df.iterrows():
        fig.add_trace(
            go.Bar(
                y=[r["Crew"]],
                x=[r["Dauer"]],
                base=r["Start"],
                orientation="h",
                name=r["Taktbereich"],
                hovertext=(
                    f"{r['Crew']}<br>"
                    f"{r['Task']}<br>"
                    f"{r['Taktbereich']}<br>"
                    f"Start: {r['Start']} – Ende: {r['Finish']}"
                ),
                hoverinfo="text",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Taktplanung – Gantt (SimPy, numerische Zeit)",
        barmode="stack",
        xaxis=dict(title="Zeit (Einheiten)", type="linear"),
        yaxis=dict(title="Kolonne", autorange="reversed"),
        height=400,
    )

    fig.write_html("gantt.html", auto_open=True)
    print("✅ Gantt gespeichert als gantt.html")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    df = run_simulation(seed=2)
    print("\nEventlog:\n", df)
    plot_gantt(df)
