"""
features.py  -  Step 2: turn raw walking data into the paper's features and labels.

For every snapshot (1 s window) and every cell of the map we compute:
  rho   : density                       (Eq. 1)
  Hdir  : directional entropy           (Eq. 2)
  Hspd  : speed-variation entropy       (Eq. 3)
  F     : interaction-force intensity   (Eq. 4)
  trends: change over the last W windows (Eq. 5)
plus the numbers the 3 baselines need (mean speed along x, velocity variance, mean speed).
"""
import numpy as np
import pandas as pd

CELL = 4.0          # cell size (m)
K_BINS = 8          # angular bins for directional entropy
M_BINS = 6          # speed bins for speed entropy
V_RANGE = (0.0, 2.5)
MIN_PEOPLE = 3      # a cell is only evaluated if it has >= 3 people
MOVING = 0.05       # people slower than this have no meaningful heading -> ignored in Hdir
W_TREND = 3         # trend = value now - value 3 windows ago
HORIZON = 10        # label looks this many seconds into the future
CONTACT_DIST = 0.5  # two bodies touch if centres are closer than 2*radius
CRIT_RATIO = 0.20   # >= 20% of people in the cell are in body contact -> danger event


def entropy(counts, k):
    s = counts.sum()
    if s == 0:
        return 0.0
    p = counts[counts > 0] / s
    return float(-(p * np.log(p)).sum() / np.log(k))


def contact_flags(pos):
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return (d < CONTACT_DIST).any(1)


def snapshot_grid(pos, vel, fm, box):
    nx, ny = int(np.ceil(box[0] / CELL)), int(np.ceil(box[1] / CELL))
    cx = np.clip((pos[:, 0] // CELL).astype(int), 0, nx - 1)
    cy = np.clip((pos[:, 1] // CELL).astype(int), 0, ny - 1)
    cell_id = cx * ny + cy
    touching = contact_flags(pos)
    speed = np.linalg.norm(vel, axis=1)
    out = {}
    for c in np.unique(cell_id):
        m = cell_id == c
        n = int(m.sum())
        if n < MIN_PEOPLE:
            continue
        v, sp = vel[m], speed[m]
        mv = sp > MOVING
        if mv.sum() >= MIN_PEOPLE:
            ang = np.arctan2(v[mv, 1], v[mv, 0])
            hd = entropy(np.histogram(ang, bins=K_BINS, range=(-np.pi, np.pi))[0], K_BINS)
        else:
            hd = 0.0
        hs = entropy(np.histogram(sp, bins=M_BINS, range=V_RANGE)[0], M_BINS)
        rho = n / (CELL * CELL)
        out[c] = dict(N=n, rho=rho, Hdir=hd, Hspd=hs, F=float(fm[m].mean()),
                      vx=float(v[:, 0].mean()), vbar=float(sp.mean()),
                      vvar=float(v[:, 0].var() + v[:, 1].var()),
                      contact=float(touching[m].mean()))
    return out, nx * ny


def run_to_table(run, scenario, run_id):
    """Convert one simulation run into a table with one row per (window, cell)."""
    snaps = run["snaps"]
    grids, ncell = [], 0
    for (t, pos, vel, fm) in snaps:
        g, ncell = snapshot_grid(pos, vel, fm, run["box"])
        grids.append(g)
    keys = ["rho", "Hdir", "Hspd", "F", "vx", "vbar", "vvar"]
    T = len(grids)
    # dense arrays for trends and for future-contact labels (empty cell -> 0)
    arr = {k: np.zeros((T, ncell)) for k in keys + ["contact", "N"]}
    for ti, g in enumerate(grids):
        for c, d in g.items():
            for k in keys + ["contact", "N"]:
                arr[k][ti, c] = d[k]
    rows = []
    for ti in range(T):
        # future danger: worst contact ratio in the next HORIZON windows
        fut = arr["contact"][ti: ti + HORIZON + 1].max(0)
        for c in grids[ti]:
            row = dict(run=run_id, scenario=scenario, t=snaps[ti][0], cell=c, N=arr["N"][ti, c])
            for k in keys:
                row[k] = arr[k][ti, c]
                prev = arr[k][ti - W_TREND, c] if ti >= W_TREND else arr[k][0, c]
                row["d_" + k] = arr[k][ti, c] - prev
            row["contact_now"] = arr["contact"][ti, c]
            row["future_contact"] = fut[c]
            rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    # ---- labels (do NOT use any model input feature) ----
    # critical : a danger event (>=20% of people touching) happens within the next 10 s
    # congested: some body contact happens (but below the danger level)
    # safe     : no body contact in the next 10 s
    df["label"] = np.where(df.future_contact >= CRIT_RATIO, 2,
                           np.where(df.future_contact > 0, 1, 0))
    return df
