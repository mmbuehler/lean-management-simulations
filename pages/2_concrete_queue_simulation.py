import random
import simpy
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Beton-LKW Warteschlange", layout="wide")

# ----------------------------
# SimPy-Modell
# ----------------------------
def truck(env, name, pump, log, service_min, service_max):
    arrival = env.now
    with pump.request() as req:
        yield req
        start_service = env.now
        wait = start_service - arrival

        service = random.randint(service_min, service_max)
        end_service = start_service + service

        log["trucks"].append(
            dict(
                Truck=name,
                Arrival=arrival,
                StartService=start_service,
                EndService=end_service,
                Wait=wait,
                Service=service,
            )
        )
        yield env.timeout(service)

def truck_arrivals(env, pump, log, mean_interarrival, service_min, service_max):
    i = 0
    while True:
        inter = random.expovariate(1.0 / mean_interarrival)
        yield env.timeout(inter)
        i += 1
        env.process(truck(env, f"LKW {i}", pump, log, service_min, service_max))

def monitor(env, pump, log, step):
    while True:
        q = len(pump.queue)
        s = pump.count
        log["timeline"].append(dict(t=env.now, queue=q, in_service=s, system=q+s))
        yield env.timeout(step)

def run_sim(sim_time, step, mean_interarrival, service_min, service_max, num_pumps, seed):
    random.seed(seed)
    env = simpy.Environment()
    pump = simpy.Resource(env, capacity=num_pumps)
    log = {"timeline": [], "trucks": []}

    env.process(truck_arrivals(env, pump, log, mean_interarrival, service_min, service_max))
    env.process(monitor(env, pump, log, step))

    env.run(until=sim_time)
    return pd.DataFrame(log["timeline"]), pd.DataFrame(log["trucks"])

# ----------------------------
# UI
# ----------------------------
st.title("Baustelle: Lieferbeton-LKW Warteschlange (SimPy)")

with st.sidebar:
    st.header("Parameter")
    sim_time = st.slider("Simulationsdauer (Min)", 30, 480, 120, 10)
    step = st.slider("Monitoring-Schrittweite (Min)", 1, 10, 1, 1)
    mean_interarrival = st.slider("Ø Interarrival (Min/LKW)", 1, 20, 6, 1)
    service_min = st.slider("Service min (Min)", 1, 30, 8, 1)
    service_max = st.slider("Service max (Min)", 1, 40, 14, 1)
    num_pumps = st.slider("Anzahl Pumpen", 1, 4, 1, 1)
    seed = st.slider("Seed", 1, 999, 7, 1)

if service_max < service_min:
    st.error("Service max muss ≥ Service min sein.")
    st.stop()

timeline, trucks = run_sim(sim_time, step, mean_interarrival, service_min, service_max, num_pumps, seed)

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("LKW gesamt", int(len(trucks)))
if len(trucks) > 0:
    col2.metric("Ø Wartezeit (Min)", f"{trucks['Wait'].mean():.2f}")
    col3.metric("Max Wartezeit (Min)", f"{trucks['Wait'].max():.2f}")
else:
    col2.metric("Ø Wartezeit (Min)", "-")
    col3.metric("Max Wartezeit (Min)", "-")

col4.metric("Max System (LKW)", int(timeline["system"].max()) if len(timeline) else 0)

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(timeline["t"], timeline["queue"], label="Warteschlange (Queue)")
ax.plot(timeline["t"], timeline["system"], label="Im System (Queue+Service)")
ax.set_xlabel("Zeit (Min)")
ax.set_ylabel("Anzahl LKWs")
ax.set_title("Zeitverlauf Warteschlange / System")
ax.legend()
st.pyplot(fig)

with st.expander("Rohdaten anzeigen"):
    st.subheader("Timeline")
    st.dataframe(timeline)
    st.subheader("Trucks")
    st.dataframe(trucks)
