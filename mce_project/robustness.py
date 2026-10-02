"""
robustness.py - Day 3: two extra checks, so the results are not an accident.

Check 1: change the danger threshold used for the labels (10%, 20%, 30% of people touching).
Check 2: leave-one-scenario-out - train on 4 scenarios, test on the 5th (never seen before).
Uses data.csv only (no new simulation needed).
"""
import numpy as np, pandas as pd, warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
SEED = 0
df = pd.read_csv("data.csv")
df["P"] = df.rho * df.vvar
df["d_P"] = df.P - (df.rho - df.d_rho) * (df.vvar - df.d_vvar)

SETS = {
    "MCE (proposed)": ["rho", "Hdir", "Hspd", "F", "d_rho", "d_Hdir", "d_Hspd", "d_F"],
    "Unidirectional entropy (x-direction)": ["rho", "vx", "d_rho", "d_vx"],
    "Crowd pressure": ["P", "d_P"],
}


def model():
    return RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


def far(y, p):
    neg = y != 2
    return ((p == 2) & neg).sum() / max(neg.sum(), 1)


lines = []
def say(s=""):
    print(s); lines.append(s)

# ------------------------------------------------------------ check 1: thresholds
say("=" * 70)
say("CHECK 1: different danger thresholds for the labels (5-fold, run-level CV)")
say("=" * 70)
strat = df.scenario.astype("category").cat.codes.values
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, strat, df.run.values))
rows = []
for thr in [0.10, 0.20, 0.30]:
    y = np.where(df.future_contact >= thr, 2, np.where(df.future_contact > 0, 1, 0))
    share = (y == 2).mean() * 100
    for name, cols in SETS.items():
        pred = np.zeros(len(df), int)
        for tr, te in folds:
            pred[te] = model().fit(df.iloc[tr][cols], y[tr]).predict(df.iloc[te][cols])
        acc, f = (pred == y).mean() * 100, far(y, pred) * 100
        rows.append(dict(Threshold=thr, Method=name, Accuracy=round(acc, 1), FAR=round(f, 1),
                         CriticalShare=round(share, 1)))
        say(f"threshold={thr:.2f} (critical={share:4.1f}% of windows)  {name:38s} acc={acc:5.1f}%  FAR={f:4.1f}%")
    say()
pd.DataFrame(rows).to_csv("table_thresholds.csv", index=False)

# ------------------------------------------------------------ check 2: leave one scenario out
say("=" * 70)
say("CHECK 2: leave-one-scenario-out (train on 4 scenarios, test on the unseen one)")
say("=" * 70)
y = df.label.values
rows = []
for sc in df.scenario.unique():
    te = (df.scenario == sc).values
    row = {"Held-out scenario": sc}
    for name, cols in SETS.items():
        m = model().fit(df[~te][cols], y[~te])
        row[name] = round((m.predict(df[te][cols]) == y[te]).mean() * 100, 1)
    rows.append(row)
loso = pd.DataFrame(rows)
loso.loc[len(loso)] = ["MEAN"] + [round(loso[c].mean(), 1) for c in SETS]
say(loso.to_string(index=False))
loso.to_csv("table_leave_one_scenario_out.csv", index=False)
open("results_robustness.txt", "w").write("\n".join(lines))
print("\nSaved: results_robustness.txt, table_thresholds.csv, table_leave_one_scenario_out.csv")
