import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
from sim import run_scenario
fig, axs = plt.subplots(2, 3, figsize=(13, 7))
cfg = [("unidirectional", 10), ("counterflow", 10), ("crossing", 8),
       ("bottleneck", 25), ("evacuation", 25)]
for ax, (s, tt) in zip(axs.ravel(), cfg):
    r = run_scenario(s, 1000)
    t, pos, vel, fm = r["snaps"][min(tt, len(r["snaps"]) - 1)]
    sp = np.linalg.norm(vel, axis=1)
    ax.scatter(pos[:, 0], pos[:, 1], c=sp, cmap="viridis", s=14, vmin=0, vmax=2)
    ax.quiver(pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], angles="xy", scale_units="xy", scale=3, width=0.003, alpha=.5)
    ax.set_title(f"{s} (t={t:.0f}s, {len(pos)} people)", fontsize=10)
    ax.set_aspect("equal")
axs[1, 2].axis("off")
plt.tight_layout(); plt.savefig("snapshots.png", dpi=90)
