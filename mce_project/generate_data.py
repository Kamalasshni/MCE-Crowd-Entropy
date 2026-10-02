"""generate_data.py - Step 3: run all simulations and save one big table (data.csv)."""
import pandas as pd, time
from sim import run_scenario
from features import run_to_table

SCENARIOS = ["unidirectional", "counterflow", "crossing", "bottleneck", "evacuation"]
RUNS_PER_SCENARIO = 20

tables = []
t0 = time.time()
for s in SCENARIOS:
    for k in range(RUNS_PER_SCENARIO):
        run = run_scenario(s, seed=1000 + k)      # same seeds -> same results every time
        tables.append(run_to_table(run, s, f"{s}_{k:02d}"))
    print(f"{s} done  ({time.time()-t0:.0f}s)", flush=True)
df = pd.concat(tables, ignore_index=True)
df.to_csv("data.csv", index=False)
print("rows:", len(df), " runs:", df.run.nunique())
print(df.label.value_counts().sort_index().rename({0: "safe", 1: "congested", 2: "critical"}))
