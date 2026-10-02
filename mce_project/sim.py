"""
sim.py  -  Step 1: a simple crowd simulator (social force model).

Idea in plain words:
  Every pedestrian wants to walk to a goal at a comfortable speed.
  Other pedestrians and walls "push" them away when they get too close.
  We repeat this tiny calculation every 0.05 seconds and record
  where everybody is, how fast they move, and how hard they are pushed.

Five scenarios (same as the paper):
  unidirectional, counterflow, crossing, bottleneck, evacuation
"""
import numpy as np

# ---------- model parameters (these go into Section IV-A of your paper) ----------
DT = 0.05          # time step (s)
SNAP_EVERY = 20    # save a snapshot every 20 steps = 1 second
TAU = 0.5          # relaxation time (s): how fast people reach their wanted speed
RADIUS = 0.25      # body radius (m)
A_PED, B_PED = 3.0, 0.3   # pedestrian-pedestrian repulsion strength (m/s^2) and range (m)
A_WALL, B_WALL = 5.0, 0.2  # wall repulsion
VMAX_FACTOR = 1.5  # nobody may move faster than 1.5 x their wanted speed
T_MAX = 120        # maximum run length (s)


def pairwise_forces(pos):
    """Repulsion from all other pedestrians: f_ij = A*exp((r_i+r_j-d_ij)/B) * n_ij  (paper Eq. 4)."""
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1)) + 1e-9
    np.fill_diagonal(dist, np.inf)
    n = diff / dist[..., None]
    mag = A_PED * np.exp((2 * RADIUS - dist) / B_PED)
    mag[dist > 3.0] = 0.0  # ignore far-away people (saves time)
    return (mag[..., None] * n).sum(1)  # total push on each pedestrian, shape (N, 2)


def wall_forces(pos, walls):
    """Repulsion from wall segments. walls = list of (x1, y1, x2, y2)."""
    total = np.zeros_like(pos)
    for (x1, y1, x2, y2) in walls:
        p1 = np.array([x1, y1]); p2 = np.array([x2, y2])
        seg = p2 - p1
        t = np.clip(((pos - p1) @ seg) / (seg @ seg), 0, 1)
        nearest = p1 + t[:, None] * seg
        diff = pos - nearest
        dist = np.sqrt((diff ** 2).sum(1)) + 1e-9
        mag = A_WALL * np.exp((RADIUS - dist) / B_WALL)
        mag[dist > 1.5] = 0.0
        total += mag[:, None] * diff / dist[:, None]
    return total


# ---------------------------------------------------------------------------
# Scenario builders. Each returns a dictionary describing the run.
# ---------------------------------------------------------------------------
def build_scenario(name, rng):
    if name == "unidirectional":
        n = rng.integers(100, 220)
        pos = np.column_stack([rng.uniform(0, 8, n), rng.uniform(0.5, 5.5, n)])
        goal = np.column_stack([np.full(n, 40.0), pos[:, 1]])
        walls = [(0, 0, 40, 0), (0, 6, 40, 6)]
        return dict(pos=pos, goal=goal, walls=walls, exit_x=30.0, box=(30, 6))

    if name == "counterflow":
        n1, n2 = rng.integers(60, 120), rng.integers(60, 120)
        left = np.column_stack([rng.uniform(0, 8, n1), rng.uniform(0.5, 5.5, n1)])
        right = np.column_stack([rng.uniform(22, 30, n2), rng.uniform(0.5, 5.5, n2)])
        pos = np.vstack([left, right])
        goal = np.vstack([np.column_stack([np.full(n1, 40.0), left[:, 1]]),
                          np.column_stack([np.full(n2, -10.0), right[:, 1]])])
        walls = [(-10, 0, 40, 0), (-10, 6, 40, 6)]
        return dict(pos=pos, goal=goal, walls=walls, exit_x=None, exit_both=(0.0, 30.0), box=(30, 6))

    if name == "crossing":
        n1, n2 = rng.integers(70, 130), rng.integers(70, 130)
        a = np.column_stack([rng.uniform(0, 8, n1), rng.uniform(7, 13, n1)])      # walks right
        b = np.column_stack([rng.uniform(7, 13, n2), rng.uniform(0, 8, n2)])      # walks up
        pos = np.vstack([a, b])
        goal = np.vstack([np.column_stack([np.full(n1, 30.0), a[:, 1]]),
                          np.column_stack([b[:, 0], np.full(n2, 30.0)])])
        return dict(pos=pos, goal=goal, walls=[], exit_x=None, exit_xy=(20.0, 20.0), box=(20, 20))

    if name == "bottleneck":
        n = rng.integers(120, 240)
        pos = np.column_stack([rng.uniform(0.5, 9, n), rng.uniform(0.5, 11.5, n)])
        gap = rng.uniform(0.9, 1.4)
        gy = 6.0
        walls = [(0, 0, 30, 0), (0, 12, 30, 12), (10, 0, 10, gy - gap / 2), (10, gy + gap / 2, 10, 12)]
        # waypoint: first the gap, afterwards the far side
        goal = np.column_stack([np.full(n, 10.0), np.full(n, gy)])
        return dict(pos=pos, goal=goal, walls=walls, exit_x=20.0, box=(20, 12),
                    waypoint=(10.0, gy), final_x=30.0)

    if name == "evacuation":
        n = rng.integers(120, 240)
        pos = rng.uniform([0.5, 0.5], [19.5, 19.5], size=(n, 2))
        gap = rng.uniform(1.5, 2.2)
        gy = 10.0
        walls = [(0, 0, 20, 0), (0, 20, 20, 20), (0, 0, 0, 20),
                 (20, 0, 20, gy - gap / 2), (20, gy + gap / 2, 20, 20)]
        goal = rng.uniform([1, 1], [19, 19], size=(n, 2))  # calm phase: random wandering
        return dict(pos=pos, goal=goal, walls=walls, exit_x=20.5, box=(20, 20),
                    exit_point=(21.0, gy), onset=15.0)
    raise ValueError(name)


def run_scenario(name, seed):
    """Run one simulation. Returns snapshots: times, list of (pos, vel, push_magnitude)."""
    rng = np.random.default_rng(seed)
    sc = build_scenario(name, rng)
    pos, goal, walls = sc["pos"].copy(), sc["goal"].copy(), sc["walls"]
    n = len(pos)
    vel = np.zeros((n, 2))
    v0 = np.clip(rng.normal(1.3, 0.2, n), 0.8, 1.8)  # wanted speeds
    active = np.ones(n, bool)
    snaps = []
    steps = int(T_MAX / DT)
    onset = sc.get("onset")
    for step in range(steps):
        t = step * DT
        idx = np.where(active)[0]
        if len(idx) < 3:
            break
        P, V = pos[idx], vel[idx]

        # --- scenario-specific goals ---
        G = goal[idx].copy()
        want = v0[idx].copy()
        if name == "bottleneck":
            passed = P[:, 0] > 10.4
            G[passed] = np.column_stack([np.full(passed.sum(), sc["final_x"]), P[passed, 1]])
            near_gap = (~passed) & (np.abs(P[:, 0] - 10) < 2.0)
            G[near_gap] = sc["waypoint"]
            far = (~passed) & (~near_gap)
            G[far] = sc["waypoint"]
        if name == "evacuation":
            if t < onset:       # calm: wander to random targets, change target when reached
                reached = np.linalg.norm(G - P, axis=1) < 1.0
                if reached.any():
                    goal[idx[reached]] = rng.uniform([1, 1], [19, 19], size=(reached.sum(), 2))
                    G = goal[idx].copy()
                want = want * 0.6
            else:               # panic onset: everyone heads for the exit, faster
                G = np.tile(sc["exit_point"], (len(idx), 1))
                want = want * 1.5

        # --- social force step ---
        d = G - P
        dn = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        drive = (want[:, None] * dn - V) / TAU
        fped = pairwise_forces(P)
        fwall = wall_forces(P, walls)
        acc = drive + fped + fwall
        V = V + acc * DT
        sp = np.linalg.norm(V, axis=1, keepdims=True)
        cap = VMAX_FACTOR * want[:, None]
        V = np.where(sp > cap, V / (sp + 1e-9) * cap, V)
        P = P + V * DT
        pos[idx], vel[idx] = P, V

        # --- remove pedestrians who reached the exit ---
        if name in ("unidirectional",):
            gone = pos[idx, 0] > sc["exit_x"]
        elif name == "counterflow":
            gone = (pos[idx, 0] > 30.0) | (pos[idx, 0] < 0.0)
        elif name == "crossing":
            gone = (pos[idx, 0] > 20.0) | (pos[idx, 1] > 20.0)
        elif name == "bottleneck":
            gone = pos[idx, 0] > sc["exit_x"]
        else:
            gone = pos[idx, 0] > sc["exit_x"]
        active[idx[gone]] = False

        if step % SNAP_EVERY == 0:
            idx2 = np.where(active)[0]
            if len(idx2) >= 3:
                fm = np.linalg.norm(pairwise_forces(pos[idx2]), axis=1)
                snaps.append((t, pos[idx2].copy(), vel[idx2].copy(), fm))
    return dict(name=name, seed=seed, box=sc["box"], snaps=snaps, n_total=n)


if __name__ == "__main__":
    import time
    for s in ["unidirectional", "counterflow", "crossing", "bottleneck", "evacuation"]:
        t0 = time.time()
        r = run_scenario(s, 1)
        print(f"{s:15s} people={r['n_total']:4d}  snapshots={len(r['snaps']):3d}  took {time.time()-t0:.1f}s")
