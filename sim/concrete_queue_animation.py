import random
import simpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =====================================================
# Parameter (Baustellenlogik)
# =====================================================
SEED = 7

SIM_TIME = 120            # Minuten
TIME_STEP = 1             # Auflösung Animation

MEAN_INTERARRIVAL = 6     # Ø Minuten zwischen LKWs
SERVICE_TIME_MIN = 8      # Pumpzeit min
SERVICE_TIME_MAX = 14     # Pumpzeit max

NUM_PUMPS = 1             # Anzahl Betonpumpen


# =====================================================
# SimPy-Model
# =====================================================
def truck(env, name, pump, log):
    arrival = env.now

    with pump.request() as req:
        yield req
        start_service = env.now
        wait = start_service - arrival

        service = random.randint(SERVICE_TIME_MIN, SERVICE_TIME_MAX)
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


def truck_arrivals(env, pump, log):
    i = 0
    while True:
        inter = random.expovariate(1.0 / MEAN_INTERARRIVAL)
        yield env.timeout(inter)

        i += 1
        env.process(truck(env, f"LKW {i}", pump, log))


def monitor(env, pump, log):
    while True:
        q = len(pump.queue)
        s = pump.count
        log["timeline"].append(
            dict(
                t=env.now,
                queue=q,
                in_service=s,
                system=q + s,
            )
        )
        yield env.timeout(TIME_STEP)


def run_simulation():
    random.seed(SEED)

    env = simpy.Environment()
    pump = simpy.Resource(env, capacity=NUM_PUMPS)

    log = {"timeline": [], "trucks": []}

    env.process(truck_arrivals(env, pump, log))
    env.process(monitor(env, pump, log))

    env.run(until=SIM_TIME)
    return log


# =====================================================
# Animation (stabil für macOS)
# =====================================================
def animate_queue(log):
    timeline = log["timeline"]
    if not timeline:
        raise RuntimeError("Timeline leer – Monitoring fehlgeschlagen")

    t = [r["t"] for r in timeline]
    q = [r["queue"] for r in timeline]
    s = [r["in_service"] for r in timeline]
    sys = [r["system"] for r in timeline]

    ymax = max(5, max(sys) + 2)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Baustelle – Lieferbeton Warteschlange (SimPy)")
    ax.set_xlabel("Zeit [Min]")
    ax.set_ylabel("Anzahl LKWs")
    ax.set_xlim(0, SIM_TIME)
    ax.set_ylim(0, ymax)

    line_q, = ax.plot([], [], lw=2, label="Warteschlange")
    line_sys, = ax.plot([], [], lw=2, label="Im System")
    scatter = ax.scatter([], [], s=80, color="black")

    status = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")
    ax.legend()

    def init():
        line_q.set_data([], [])
        line_sys.set_data([], [])
        scatter.set_offsets(np.empty((0, 2)))
        status.set_text("")
        return line_q, line_sys, scatter, status

    def update(frame):
        x = t[: frame + 1]
        line_q.set_data(x, q[: frame + 1])
        line_sys.set_data(x, sys[: frame + 1])

        n = sys[frame]
        if n > 0:
            xs = [t[frame]] * n
            ys = list(range(1, n + 1))
            scatter.set_offsets(np.column_stack([xs, ys]))
        else:
            scatter.set_offsets(np.empty((0, 2)))

        status.set_text(
            f"t={t[frame]:.0f} min | "
            f"Queue={q[frame]} | "
            f"In Service={s[frame]} | "
            f"System={sys[frame]}"
        )

        return line_q, line_sys, scatter, status

    anim = FuncAnimation(
        fig,
        update,
        frames=len(t),
        init_func=init,
        interval=120,
        blit=False,      # wichtig für macOS
        repeat=False,
    )

    plt.show()


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    log = run_simulation()

    waits = [r["Wait"] for r in log["trucks"]]
    if waits:
        print(f"LKW gesamt: {len(waits)}")
        print(f"Ø Wartezeit: {np.mean(waits):.2f} min")
        print(f"Max Wartezeit: {max(waits):.2f} min")

    animate_queue(log)
