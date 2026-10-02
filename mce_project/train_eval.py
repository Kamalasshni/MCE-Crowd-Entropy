"""
train_eval.py  -  Day 2: train MCE and the baselines, measure everything.

What this script does (plain words):
  1. Reads data.csv (made by generate_data.py).
  2. Builds the feature sets for MCE and for each baseline.
  3. Trains the SAME Random Forest for every method (fair comparison).
  4. Uses 5-fold cross-validation where whole RUNS are kept together
     (no run is ever in both training and testing -> no leakage).
  5. Reports accuracy, false-alarm rate, lead time, ablation, significance.
All numbers are printed and saved to results.txt, tables are saved as CSV files.
"""
import numpy as np, pandas as pd, warnings, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, f1_score
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
SEED = 0
N_FOLDS = 5
CRIT_RATIO = 0.20     # same danger level used for labels
LOOKBACK = 30         # an alert only counts if it is at most 30 s before the danger event
NAMES = {0: "safe", 1: "congested", 2: "critical"}

df = pd.read_csv("data.csv")
print(f"Loaded {len(df)} rows, {df.run.nunique()} runs")

# ---------------------------------------------------------------- feature sets
# baseline indices; trend = value now - value one step earlier (prev = now - d)
df["P"] = df.rho * df.vvar
df["J"] = df.rho * df.vbar
df["d_P"] = df.P - (df.rho - df.d_rho) * (df.vvar - df.d_vvar)
df["d_J"] = df.J - (df.rho - df.d_rho) * (df.vbar - df.d_vbar)

FEATURES = {
    "MCE (proposed)": ["rho", "Hdir", "Hspd", "F", "d_rho", "d_Hdir", "d_Hspd", "d_F"],
    "Unidirectional entropy (x-direction)": ["rho", "vx", "d_rho", "d_vx"],
    "Unidirectional entropy (speed magnitude)": ["rho", "vbar", "d_rho", "d_vbar"],
    "Crowd pressure": ["P", "d_P"],
    "Crowd flow": ["J", "d_J"],
}
ABLATION = {
    "Full MCE": FEATURES["MCE (proposed)"],
    "Without directional entropy": ["rho", "Hspd", "F", "d_rho", "d_Hspd", "d_F"],
    "Without speed-variation entropy": ["rho", "Hdir", "F", "d_rho", "d_Hdir", "d_F"],
    "Without interaction force": ["rho", "Hdir", "Hspd", "d_rho", "d_Hdir", "d_Hspd"],
    "Without all three (density only)": ["rho", "d_rho"],
}

# ---------------------------------------------------------------- cross-validation
y = df.label.values
groups = df.run.values
strat = df.scenario.astype("category").cat.codes.values   # keep scenarios balanced in folds
folds = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
             .split(df, strat, groups))


def make_model():
    return RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=5,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)


def cross_validate(cols):
    pred = np.zeros(len(df), int)
    fold_acc = []
    for tr, te in folds:
        m = make_model().fit(df.iloc[tr][cols], y[tr])
        pred[te] = m.predict(df.iloc[te][cols])
        fold_acc.append((pred[te] == y[te]).mean())
    return pred, np.array(fold_acc)


def metrics(pred, mask=None):
    yt, yp = (y, pred) if mask is None else (y[mask], pred[mask])
    acc = (yt == yp).mean()
    pos_t, pos_p = yt == 2, yp == 2
    tp, fn = (pos_t & pos_p).sum(), (pos_t & ~pos_p).sum()
    fp, tn = (~pos_t & pos_p).sum(), (~pos_t & ~pos_p).sum()
    return dict(acc=acc, sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                far=fp / max(fp + tn, 1), f1_crit=f1_score(pos_t, pos_p),
                f1_macro=f1_score(yt, yp, average="macro"))


# ---------------------------------------------------------------- lead time
# danger event = first time a cell has >= 20% of people touching.
# alert = first time the model predicts "critical" in that cell, at most 30 s before.
def lead_times(pred):
    d = df[["run", "cell", "t"]].copy()
    d["pred"] = pred
    d["event"] = df.contact_now >= CRIT_RATIO
    out = {}
    for (run, cell), g in d.groupby(["run", "cell"]):
        ev = g[g.event]
        if ev.empty:
            continue
        onset = ev.t.min()
        al = g[(g.pred == 2) & (g.t <= onset) & (g.t >= onset - LOOKBACK)]
        out[(run, cell)] = (onset - al.t.min()) if not al.empty else np.nan
    return pd.Series(out)


# ---------------------------------------------------------------- run everything
lines = []
def say(s=""):
    print(s); lines.append(s)

say("=" * 70); say("TABLE I-style: main comparison (5-fold, run-level cross-validation)")
say("=" * 70)
main_rows, preds, leads = [], {}, {}
for name, cols in FEATURES.items():
    pred, fa = cross_validate(cols)
    preds[name] = pred
    m = metrics(pred)
    lt = lead_times(pred)
    leads[name] = lt
    det = lt.notna().mean()
    main_rows.append(dict(Method=name, Accuracy=m["acc"] * 100, Acc_std_over_folds=fa.std() * 100,
                          FalseAlarmRate=m["far"] * 100, Sensitivity=m["sens"] * 100,
                          Specificity=m["spec"] * 100, F1_critical=m["f1_crit"],
                          F1_macro=m["f1_macro"], EventsDetected_pct=det * 100,
                          MeanLead_s=lt.mean(), N_events=len(lt)))
    say(f"{name:42s} acc={m['acc']*100:5.1f}%  FAR={m['far']*100:5.1f}%  "
        f"sens={m['sens']*100:5.1f}%  lead={lt.mean():4.1f}s  detected={det*100:4.1f}%")
main = pd.DataFrame(main_rows).round(2)
main.to_csv("table_main.csv", index=False)

say(); say("=" * 70); say("Accuracy by flow scenario (MCE vs baselines)"); say("=" * 70)
sc_rows = []
for sc in df.scenario.unique():
    mk = (df.scenario == sc).values
    row = {"Scenario": sc}
    for name in FEATURES:
        row[name] = metrics(preds[name], mk)["acc"] * 100
    sc_rows.append(row)
sc_tab = pd.DataFrame(sc_rows).round(1)
say(sc_tab.to_string(index=False))
sc_tab.to_csv("table_by_scenario.csv", index=False)

say(); say("=" * 70); say("Confusion matrix for MCE (rows = true, columns = predicted)"); say("=" * 70)
cm = confusion_matrix(y, preds["MCE (proposed)"])
say(pd.DataFrame(cm, index=[f"true {NAMES[i]}" for i in range(3)],
                 columns=[f"pred {NAMES[i]}" for i in range(3)]).to_string())

say(); say("=" * 70); say("Ablation (same folds, same classifier)"); say("=" * 70)
ab_rows = []
for name, cols in ABLATION.items():
    pred, fa = cross_validate(cols)
    m = metrics(pred)
    ab_rows.append(dict(Variant=name, Accuracy=m["acc"] * 100, FalseAlarmRate=m["far"] * 100,
                        F1_critical=m["f1_crit"]))
    say(f"{name:36s} acc={m['acc']*100:5.1f}%  FAR={m['far']*100:5.1f}%  F1(critical)={m['f1_crit']:.3f}")
ab = pd.DataFrame(ab_rows).round(2)
ab["Change_vs_full"] = (ab.Accuracy - ab.Accuracy.iloc[0]).round(2)
ab.to_csv("table_ablation.csv", index=False)

say(); say("=" * 70)
say("Significance: MCE vs each baseline, paired Wilcoxon test over the 100 runs")
say("(each run = one paired accuracy value; windows inside a run are NOT treated as independent)")
say("=" * 70)
run_acc = {n: pd.Series(preds[n] == y).groupby(df.run.values).mean() for n in FEATURES}
sig_rows = []
for name in list(FEATURES)[1:]:
    a, b = run_acc["MCE (proposed)"], run_acc[name]
    p_acc = wilcoxon(a, b).pvalue if (a != b).any() else 1.0
    both = leads["MCE (proposed)"].notna() & leads[name].notna()
    la, lb = leads["MCE (proposed)"][both], leads[name][both]
    p_lead = wilcoxon(la, lb).pvalue if (la != lb).any() else 1.0
    sig_rows.append(dict(Comparison=f"MCE vs {name}", p_accuracy=p_acc, p_leadtime=p_lead,
                         n_events_paired=int(both.sum())))
    say(f"MCE vs {name:42s} p(acc)={p_acc:.2e}  p(lead)={p_lead:.2e}  (paired events: {both.sum()})")
pd.DataFrame(sig_rows).to_csv("table_significance.csv", index=False)

say(); say("=" * 70); say("Cross-validation fold-by-fold accuracy (mean +/- std)"); say("=" * 70)
cv_rows = []
for name, cols in FEATURES.items():
    _, fa = cross_validate(cols)
    cv_rows.append(dict(Method=name, MeanAcc=fa.mean() * 100, Std=fa.std() * 100))
    say(f"{name:42s} {fa.mean()*100:5.1f}% +/- {fa.std()*100:3.1f}")
pd.DataFrame(cv_rows).round(2).to_csv("table_cv.csv", index=False)

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(figsize=(8, 4.2))
w = 0.2
names = ["MCE (proposed)", "Unidirectional entropy (x-direction)", "Crowd pressure", "Crowd flow"]
labels = ["MCE", "Unidir. entropy", "Crowd pressure", "Crowd flow"]
for i, (n, l) in enumerate(zip(names, labels)):
    ax.bar(np.arange(len(sc_tab)) + (i - 1.5) * w, sc_tab[n], w, label=l)
ax.set_xticks(range(len(sc_tab))); ax.set_xticklabels(sc_tab.Scenario, fontsize=8)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_accuracy_by_scenario.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(ab.Variant[::-1], ab.Accuracy[::-1])
ax.set_xlabel("Accuracy (%)"); ax.set_xlim(0, 100)
plt.tight_layout(); plt.savefig("fig_ablation.png", dpi=150); plt.close()

open("results.txt", "w").write("\n".join(lines))
print("\nSaved: results.txt, table_*.csv, fig_*.png")
