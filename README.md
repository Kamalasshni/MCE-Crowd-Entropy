# MCE-Crowd-Entropy
# MCE: Multidirectional Crowd Entropy (simulation code)

Code for the paper "A Multidirectional Crowd Entropy (MCE) Framework for
Early Detection of Critical Crowd States".

Run in this order:
1. pip install numpy pandas scikit-learn matplotlib scipy
2. python generate_data.py   (runs the simulation, creates data.csv)
3. python train_eval.py      (main results, ablation, significance)
4. python robustness.py      (threshold and leave-one-scenario-out checks)

Note: results can differ very slightly between computers because the
simulation is chaotic. The data.csv used in the paper is included.
