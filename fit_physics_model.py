# -*- coding: utf-8 -*-
"""Fit shared physics-structured flux models and test zero-shot on a held-out connection."""

import argparse
import json

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HARMONICS = [2, 4, 6]


def harmonic_features(theta_deg):
    th = np.deg2rad(theta_deg)
    cols = [np.ones_like(th)]
    for k in HARMONICS:
        cols += [np.sin(k * th), np.cos(k * th)]
    return np.stack(cols, axis=1)


def build_flux_design(X, symmetric=True, diagonal=False):
    theta = X[:, 0]
    ic = X[:, 1:]
    n = len(X)
    harm = harmonic_features(theta)
    H = harm.shape[1]
    if diagonal:
        pairs = [(i, i) for i in range(12)]
    elif symmetric:
        pairs = [(j, k) for j in range(12) for k in range(j, 12)]
    else:
        pairs = [(j, k) for j in range(12) for k in range(12)]
    D = np.zeros((n * 12, 12 * H + len(pairs) * H))
    for s in range(n):
        for i in range(12):
            row = s * 12 + i
            D[row, i * H : (i + 1) * H] = harm[s]
            for pidx, (j, k) in enumerate(pairs):
                if diagonal:
                    coef = ic[s, i] if i == j else 0.0
                elif symmetric:
                    if i == j and i == k:
                        coef = ic[s, j]
                    elif i == j:
                        coef = ic[s, k]
                    elif i == k:
                        coef = ic[s, j]
                    else:
                        coef = 0.0
                else:
                    coef = ic[s, k] if i == j else 0.0
                if coef != 0.0:
                    D[row, 12 * H + pidx * H : 12 * H + (pidx + 1) * H] = coef * harm[s]
    return D


def build_torque_design(X):
    theta = X[:, 0]
    ic = X[:, 1:]
    harm = harmonic_features(theta)
    H = harm.shape[1]
    cols = [harm]
    for i in range(12):
        cols.append(harm * ic[:, i : i + 1])
        cols.append(harm * (ic[:, i : i + 1] ** 2))
    return np.concatenate(cols, axis=1)


def eval_flux(pred, true):
    err = pred - true
    rmse = np.sqrt(np.mean(err**2, axis=0))
    nrmse = rmse / np.maximum(np.std(true, axis=0), 1e-9)
    r2 = 1 - np.sum(err**2, axis=0) / np.maximum(np.sum((true - true.mean(0)) ** 2, 0), 1e-12)
    return {
        "rmse_mean": float(rmse.mean()),
        "nrmse_mean": float(nrmse.mean()),
        "r2_mean": float(r2.mean()),
        "per_coil_rmse": rmse.tolist(),
        "per_coil_nrmse": nrmse.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="coil_dataset_full.npz")
    ap.add_argument("--zero", default="coil_zero.npz")
    ap.add_argument("--holdout", default="asym")
    ap.add_argument("--out", default="physics_zero_shot.json")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    z = np.load(args.zero)
    X = np.concatenate([d["X"], z["X"]], axis=0)
    Y = np.concatenate([d["Y"], z["Y"]], axis=0)
    kind = np.concatenate([d["kind"], np.array(["zero"] * len(z["X"]))])
    tr = kind != args.holdout
    te = kind == args.holdout

    # Choose ridge alpha on a theta-based validation split inside training.
    # The concatenated zero samples make d["theta"] shorter; rebuild theta from X.
    th_all = X[:, 0]
    val = tr & np.isin(th_all, [45.0, 135.0])
    fit = tr & ~val
    fit_rows = np.repeat(fit, 12)
    val_rows = np.repeat(val, 12)
    tr_rows = np.repeat(tr, 12)
    te_rows = np.repeat(te, 12)

    results = {}
    for name, kwargs in [
        ("diagonal", dict(diagonal=True, symmetric=False)),
        ("symmetric", dict(symmetric=True, diagonal=False)),
        ("full", dict(symmetric=False, diagonal=False)),
    ]:
        D = build_flux_design(X, **kwargs)
        best = None
        for alpha in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(D[fit_rows], Y[fit, :12].reshape(-1))
            pv = model.predict(D[val_rows]).reshape(-1, 12)
            ev = eval_flux(pv, Y[val, :12])
            if best is None or ev["nrmse_mean"] < best[0]:
                best = (ev["nrmse_mean"], alpha)
        alpha = best[1]
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(D[tr_rows], Y[tr, :12].reshape(-1))
        pred = model.predict(D[te_rows]).reshape(-1, 12)
        ev = eval_flux(pred, Y[te, :12])
        ev["alpha"] = alpha
        ev["val_nrmse"] = best[0]
        results[name] = ev
        print(name, "alpha", alpha, "test nrmse", ev["nrmse_mean"], "r2", ev["r2_mean"])

    # Torque model.
    Tf = build_torque_design(X)
    best = None
    for alpha in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(Tf[fit], Y[fit, 12])
        pv = model.predict(Tf[val])
        rmse = np.sqrt(np.mean((pv - Y[val, 12]) ** 2))
        if best is None or rmse < best[0]:
            best = (rmse, alpha)
    alpha = best[1]
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(Tf[tr], Y[tr, 12])
    pt = model.predict(Tf[te])
    true = Y[te, 12]
    results["torque"] = {
        "alpha": alpha,
        "val_rmse": float(best[0]),
        "rmse": float(np.sqrt(np.mean((pt - true) ** 2))),
        "nrmse_std": float(np.sqrt(np.mean((pt - true) ** 2)) / max(np.std(true), 1e-9)),
        "r2": float(1 - np.sum((pt - true) ** 2) / max(np.sum((true - true.mean()) ** 2), 1e-12)),
    }
    print("torque", json.dumps(results["torque"], indent=1))

    out = {
        "data": args.data,
        "holdout": args.holdout,
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
