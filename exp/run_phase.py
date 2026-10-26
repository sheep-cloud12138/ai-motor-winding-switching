# -*- coding: utf-8 -*-
"""Phase-coverage ablation: magnitude-random patterns vs phase-random patterns
on the hard hold-outs (openB, tap50, delta)."""

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

# pools: none | amp (48 magnitude patterns) | ph (24 phase-random) | amp+ph
POOLS = {
    "none": [],
    "amp48": [("coil_dataset_random48.npz", "random")],
    "ph24": [("coil_dataset_random3.npz", "random3")],
    "amp48+ph24": [("coil_dataset_random48.npz", "random"),
                   ("coil_dataset_random3.npz", "random3")],
}
HOLDOUTS = ("openB", "tap50", "delta")


def main():
    d_full = np.load(ROOT / "coil_dataset_full.npz", allow_pickle=True)
    d_zero = np.load(ROOT / "coil_zero.npz")
    d_ext = np.load(ROOT / "coil_dataset_ext.npz", allow_pickle=True)
    X0 = np.concatenate([d_full["X"], d_ext["X"], d_zero["X"]])
    Y0 = np.concatenate([d_full["Y"], d_ext["Y"], d_zero["Y"]])
    K0 = np.concatenate([np.array([str(k) for k in d_full["kind"]]),
                         np.array([str(k) for k in d_ext["kind"]]),
                         np.array(["zero"] * len(d_zero["X"]))])
    out = {}
    for holdout in HOLDOUTS:
        for tag, pools in POOLS.items():
            X, Y, K = X0, Y0, K0
            for path, kind in pools:
                d = np.load(ROOT / path, allow_pickle=True)
                X = np.concatenate([X, d["X"]])
                Y = np.concatenate([Y, d["Y"]])
                K = np.concatenate([K, np.array([kind] * len(d["X"]))])
            in_train = ((np.isin(K, ("4S", "2S2P", "delta", "asym", "zero"))
                         & (K != holdout))
                        | (K == "random") | (K == "random3"))
            tr_pool = np.where(in_train)[0]
            rng = np.random.default_rng(0)
            val_mask = np.zeros(len(tr_pool), bool)
            for k in np.unique(K[tr_pool]):
                idx = np.where(K[tr_pool] == k)[0]
                n_val = max(4, int(round(0.12 * len(idx))))
                val_mask[rng.choice(idx, size=n_val, replace=False)] = True
            split = {"fit": None, "val": None, "test": None}

            def pack(rows):
                return (make_features(X[rows, 0], X[rows, 1:]).astype(np.float32),
                        theta_feature_grad(X[rows, 0]).astype(np.float32), rows)

            split["fit"] = pack(tr_pool[~val_mask])
            split["val"] = pack(tr_pool[val_mask])
            split["test"] = pack(np.where(K == holdout)[0])
            runner = Runner(X, Y, K)
            for model in ("mlp", "energy"):
                t0 = time.time()
                _, predict, _ = runner.fit_standardized(split, model)
                Fte, dphite, rows_te = split["test"]
                mm = metrics(predict(Fte, dphite), Y[rows_te])
                key = f"{holdout}/{tag}/{model}"
                out[key] = {"flux_nrmse": mm["flux_nrmse"],
                            "torque_r2": mm["torque_r2"],
                            "n_train": int(len(split["fit"][2])),
                            "seconds": round(time.time() - t0, 1)}
                print(key, "nrmse %.3f T_r2 %.3f" %
                      (mm["flux_nrmse"], mm["torque_r2"]), flush=True)
                with open(ROOT / "exp" / "results_phase.json", "w") as f:
                    json.dump(out, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
