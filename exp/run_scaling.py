# -*- coding: utf-8 -*-
"""Coverage scaling: how many random per-coil patterns does zero-shot need?

Trains MLP and StructuredEnergy on {standard connections} x {0,2,8,24,48}
random-pattern pools (P0-era datasets for 2/8/24, new file for 48) and tests
zero-shot on asym and delta. Fixed seed, current training protocol.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

from exp.run_benchmark import Runner, metrics  # noqa: E402
from exp.models import make_features, theta_feature_grad  # noqa: E402
from exp.data import load_checked  # noqa: E402

POOLS = (("0", None), ("2", "coil_dataset_random.npz"),
         ("8", "coil_dataset_random8.npz"),
         ("24", "coil_dataset_random24.npz"),
         ("48", "coil_dataset_random48.npz"))
HOLDOUTS = ("asym", "delta")


def build_split(X, Y, K, holdout, rnd_path, val_frac=0.12, seed=0):
    rng = np.random.default_rng(seed)
    if rnd_path is not None:
        d = load_checked(rnd_path)
        X = np.concatenate([X, d["X"]])
        Y = np.concatenate([Y, d["Y"]])
        K = np.concatenate([K, np.array(["random"] * len(d["X"]))])
    in_train = (np.isin(K, ("4S", "2S2P", "delta", "asym", "zero"))
                & (K != holdout)) | (K == "random")
    tr_pool = np.where(in_train)[0]
    val_mask = np.zeros(len(tr_pool), bool)
    for k in np.unique(K[tr_pool]):
        idx = np.where(K[tr_pool] == k)[0]
        n_val = max(4, int(round(val_frac * len(idx))))
        val_mask[rng.choice(idx, size=n_val, replace=False)] = True
    val_idx, fit_idx = tr_pool[val_mask], tr_pool[~val_mask]
    te_idx = np.where(K == holdout)[0]

    def pack(rows):
        return (make_features(X[rows, 0], X[rows, 1:]).astype(np.float32),
                theta_feature_grad(X[rows, 0]).astype(np.float32), rows)

    return X, Y, K, {"fit": pack(fit_idx), "val": pack(val_idx),
                     "test": pack(te_idx)}


def main():
    d_full = load_checked("coil_dataset_full.npz")
    d_zero = load_checked("coil_zero.npz")
    X0 = np.concatenate([d_full["X"], d_zero["X"]])
    Y0 = np.concatenate([d_full["Y"], d_zero["Y"]])
    K0 = np.concatenate([np.array([str(k) for k in d_full["kind"]]),
                         np.array(["zero"] * len(d_zero["X"]))])

    out = {}
    for holdout in HOLDOUTS:
        for tag, rnd in POOLS:
            X, Y, K, split = build_split(X0, Y0, K0, holdout, rnd)
            runner = Runner(X, Y, K)
            for model in ("mlp", "energy"):
                t0 = time.time()
                _, predict, _ = runner.fit_standardized(split, model)
                Fte, dphite, rows_te = split["test"]
                mm = metrics(predict(Fte, dphite), Y[rows_te])
                key = f"{holdout}/{tag}pat/{model}"
                out[key] = {"flux_nrmse": mm["flux_nrmse"],
                            "torque_r2": mm["torque_r2"],
                            "n_train": int(len(split["fit"][2])),
                            "seconds": round(time.time() - t0, 1)}
                print(key, "nrmse %.3f T_r2 %.3f" %
                      (mm["flux_nrmse"], mm["torque_r2"]), flush=True)
                with open(ROOT / "exp" / "results_scaling.json", "w") as f:
                    json.dump(out, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
