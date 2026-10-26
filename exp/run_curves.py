# -*- coding: utf-8 -*-
"""Data-efficiency learning curves for the zero-shot task (holdout = asym).

Fractions of the training pool x {MLP, GradNet, ScalarEnergy} x 3 seeds.
Validation rows stay fixed (drawn from the full pool, never subsampled).
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401  (WMI hang workaround, before numpy/scipy)

import numpy as np
import torch

from exp.data import load_all, make_split  # noqa: E402
from exp.run_benchmark import Runner, metrics  # noqa: E402

FRACTIONS = (0.05, 0.10, 0.25, 0.50, 1.0)
SEEDS = (0, 1, 2)


def main():
    X, Y, K = load_all()
    runner = Runner(X, Y, K)
    split = make_split(X, Y, K, "asym", variant="base")
    Fte, dphite, rows_te = split["test"]
    out = {}
    for frac in FRACTIONS:
        for model in ("mlp", "gradnet", "energy"):
            vals = []
            for seed in SEEDS:
                if frac >= 1.0 and seed != 0:
                    continue  # full data has no subsampling variance
                rng = np.random.default_rng(100 + seed)
                sub = np.ones(len(split["fit_idx"]), bool)
                if frac < 1.0:
                    for k in np.unique(K[split["fit_idx"]]):
                        idx = np.where(K[split["fit_idx"]] == k)[0]
                        keep = max(1, int(round(frac * len(idx))))
                        drop = rng.choice(idx, size=len(idx) - keep, replace=False)
                        sub[drop] = False
                import copy
                sp = copy.deepcopy(split)
                sp["fit"] = (split["fit"][0][sub], split["fit"][1][sub],
                             split["fit"][2][sub])
                t0 = time.time()
                _, predict, _ = runner.fit_standardized(sp, model, seed=seed)
                mm = metrics(predict(Fte, dphite), Y[rows_te])
                vals.append({"flux_nrmse": mm["flux_nrmse"],
                             "torque_r2": mm["torque_r2"],
                             "n_train": int(sub.sum()),
                             "seconds": round(time.time() - t0, 1)})
                print(f"frac {frac} {model} seed {seed}: "
                      f"nrmse {mm['flux_nrmse']:.3f} T_r2 {mm['torque_r2']:.3f}",
                      flush=True)
            out[f"{frac}/{model}"] = vals
            with open(ROOT / "exp" / "results_curves.json", "w") as f:
                json.dump(out, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
