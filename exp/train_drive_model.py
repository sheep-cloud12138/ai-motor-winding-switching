# -*- coding: utf-8 -*-
"""Train the drive-cycle surrogate on the dense-beta pool:
standard connections + dense current-angle sweep + random patterns + zero.
Saves ckpt_mlp_drive.pt and ckpt_energy_drive.pt (all connections seen)."""

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

from exp.models import make_features, theta_feature_grad  # noqa: E402
from exp.run_benchmark import Runner, output_transform  # noqa: E402
from exp.data import load_checked  # noqa: E402

def main():
    Xs, Ys, Ks = [], [], []
    for name in ("coil_dataset_full.npz", "coil_dataset_densebeta.npz",
                 "coil_dataset_sync.npz", "coil_dataset_random48.npz",
                 "coil_dataset_random3.npz", "coil_zero.npz"):
        d = load_checked(name)
        tag = "densebeta" if "densebeta" in name else \
              ("sync" if "sync" in name else
               ("random" if "random48" in name else
                ("random3" if "random3" in name else
                 ("zero" if "zero" in name else None))))
        K = np.array([str(k) for k in d["kind"]]) if tag is None \
            else np.array([tag] * len(d["X"]))
        Xs.append(d["X"]); Ys.append(d["Y"]); Ks.append(K)
    X, Y, K = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ks)
    print("pool", {k: int((K == k).sum()) for k in np.unique(K)}, flush=True)

    rng = np.random.default_rng(0)
    tr_pool = np.arange(len(X))
    val_mask = np.zeros(len(tr_pool), bool)
    for k in np.unique(K):
        idx = np.where(K == k)[0]
        n_val = max(6, int(round(0.10 * len(idx))))
        val_mask[rng.choice(idx, size=n_val, replace=False)] = True
    val_idx, fit_idx = tr_pool[val_mask], tr_pool[~val_mask]

    def pack(rows):
        return (make_features(X[rows, 0], X[rows, 1:]).astype(np.float32),
                theta_feature_grad(X[rows, 0]).astype(np.float32), rows)

    split = {"fit": pack(fit_idx), "val": pack(val_idx), "test": pack(fit_idx[:1])}
    runner = Runner(X, Y, K)
    for model in ("mlp", "energy"):
        t0 = time.time()
        m, predict, fit_s = runner.fit_standardized(split, model)
        # persist weights WITH the output standardisation so downstream users
        # get physical units directly.
        mu, sd = output_transform(model, Y[split["fit"][2]])
        torch.save({"state": m.state_dict(),
                    "mu": torch.tensor(mu, dtype=torch.float32),
                    "sd": torch.tensor(sd, dtype=torch.float32)},
                   ROOT / "exp" / f"ckpt_{model}_drive.pt")
        from exp.run_benchmark import metrics
        Fv, dphiv, rows_v = split["val"]
        mm = metrics(predict(Fv, dphiv), Y[rows_v])
        print(model, "val flux_nrmse %.4f torque_r2 %.4f (%.0fs)" %
              (mm["flux_nrmse"], mm["torque_r2"], time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
