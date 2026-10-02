# MCE: Multidirectional Crowd Entropy

Code and data for the paper **"A Multidirectional Crowd Entropy (MCE) Framework
for Early Detection of Critical Crowd States"**.

MCE combines directional entropy, speed-variation entropy and
pedestrian interaction-force intensity, and uses a Random Forest to classify
each map cell as *safe*, *congested* or *critical*. It is compared with a
simplified unidirectional entropy model, crowd pressure and crowd flow.

**Important:** all results come from a social-force simulation.
No real pedestrian data were used.

## Requirements

- Python 3.10 or newer
- `pip install numpy pandas scikit-learn matplotlib scipy`

## How to run

```
cd mce_project
python generate_data.py     # runs 100 simulations, creates data.csv (about 5 minutes)
python train_eval.py        # main comparison, ablation, significance tests
python robustness.py        # danger-threshold and leave-one-scenario-out checks
python plot_snapshot.py     # optional: draws one snapshot per scenario
```

`data.csv` (the file used in the paper) is already included, so you can run
`train_eval.py` and `robustness.py` directly without re-running the simulation.

## Files

| File | What it does |
|---|---|
| `sim.py` | Social force simulator with 5 scenarios (unidirectional, counter-flow, crossing, bottleneck, evacuation onset) |
| `features.py` | Computes density, directional entropy, speed entropy, interaction force, trends, and labels |
| `generate_data.py` | Runs 20 simulations per scenario (100 runs) and writes `data.csv` |
| `train_eval.py` | Trains MCE and baselines (same Random Forest), run-level 5-fold cross-validation, ablation, Wilcoxon tests, figures |
| `robustness.py` | Different danger thresholds (10/20/30%) and leave-one-scenario-out test |
| `plot_snapshot.py` | Draws a snapshot of each scenario |

## Output files

`results.txt`, `results_robustness.txt`, `table_*.csv`, `fig_accuracy_by_scenario.png`, `fig_ablation.png`.

## Main settings

- Time step 0.05 s, one snapshot per second, cells 4 m x 4 m, minimum 3 pedestrians per cell
- Social force: A = 3.0 m/s^2, B = 0.3 m, body radius 0.25 m
- Heading bins K = 8, speed bins M = 6, trend over the last 3 windows
- Labels from future body contact within 10 s:
  critical = at least 20% of people in the cell touching, congested = some contact, safe = none
- Random Forest: 150 trees, max depth 12, min 5 samples per leaf, balanced class weights, seed 0
- Cross-validation keeps all windows of one run in the same fold (no leakage)

## `data.csv` columns

`run`, `scenario`, `t` (time in s), `cell`, `N` (people in cell), `rho` (density),
`Hdir`, `Hspd`, `F`, `vx`, `vbar`, `vvar`, trend columns starting with `d_`,
`contact_now`, `future_contact`, and `label` (0 = safe, 1 = congested, 2 = critical).

## Main results (from the paper, simulated data)

| Method | Accuracy | False-alarm rate |
|---|---|---|
| MCE (proposed) | 81.2% | 5.1% |
| Unidirectional entropy (x-direction) | 75.9% | 6.5% |
| Crowd pressure | 71.9% | 7.2% |
| Crowd flow | 54.5% | 23.9% |

Interaction force is the most influential component in the ablation. Early-warning lead
time was about 1 s for all methods and was not significantly better than the
entropy baselines. On scenario types unseen in training, the advantage of MCE is small
(mean 68.2% vs 64.1%).

## Limitations

- Simulation only; no validation on real pedestrian data yet.
- Labels are based on body contact, which is related to the interaction-force feature.
- The unidirectional baseline is a simplified re-implementation, not the exact model of the original reference.
- Results can differ very slightly between computers because the simulation is chaotic.

## Citation

[Authors]. "A Multidirectional Crowd Entropy (MCE) Framework for Early Detection
of Critical Crowd States: A Simulation Study." [Conference name, year].

## License

MIT License (add a `LICENSE` file to the repository).
