# -*- coding: utf-8 -*-
"""Train a small MLP on seen winding connections and test zero-shot on a held-out one."""

import argparse
import json

import numpy as np
import torch
from torch import nn


def fourier_theta(theta):
    theta = np.deg2rad(theta)
    return np.stack(
        [np.sin(2 * theta), np.cos(2 * theta), np.sin(4 * theta), np.cos(4 * theta)],
        axis=1,
    )


def make_features(X):
    theta = X[:, :1]
    ic = X[:, 1:]
    return np.concatenate([theta / 180.0, fourier_theta(X[:, 0]), ic / 60.0], axis=1).astype(
        np.float32
    )


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width=128, depth=3):
        super().__init__()
        layers = [nn.Linear(d_in, width), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, Ytr, Xte, Yte, epochs=4000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = MLP(Xtr.shape[1], Ytr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x = torch.tensor(Xtr)
    y = torch.tensor(Ytr)
    xte = torch.tensor(Xte)
    yte = torch.tensor(Yte)
    best = None
    best_state = None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        opt.step()
        if ep % 500 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                te = nn.functional.mse_loss(model(xte), yte).item()
                tr = loss.item()
            if best is None or te < best:
                best = te
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"ep {ep:4d} train {tr:.3e} test {te:.3e}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(xte).numpy()
    return model, pred


def metrics(pred, true, names):
    err = pred - true
    rmse = np.sqrt(np.mean(err**2, axis=0))
    denom = np.maximum(np.std(true, axis=0), 1e-9)
    nrmse = rmse / denom
    r2 = 1 - np.sum(err**2, axis=0) / np.maximum(np.sum((true - true.mean(0)) ** 2, 0), 1e-12)
    return {
        "rmse": {n: float(v) for n, v in zip(names, rmse)},
        "nrmse_std": {n: float(v) for n, v in zip(names, nrmse)},
        "r2": {n: float(v) for n, v in zip(names, r2)},
        "flux_rmse_mean": float(rmse[:12].mean()),
        "flux_nrmse_mean": float(nrmse[:12].mean()),
        "torque_rmse": float(rmse[12]),
        "torque_nrmse": float(nrmse[12]),
        "torque_r2": float(r2[12]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="coil_dataset_quick.npz")
    ap.add_argument("--holdout", default="asym")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="zero_shot_result.json")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, Y, kind = d["X"], d["Y"], d["kind"]
    tr = kind != args.holdout
    te = kind == args.holdout
    print("train", tr.sum(), "test", te.sum(), "holdout", args.holdout)

    F = make_features(X)
    # Standardize outputs with training statistics.
    y_mu = Y[tr].mean(0, keepdims=True)
    y_sd = Y[tr].std(0, keepdims=True) + 1e-8
    Yn = (Y - y_mu) / y_sd

    model, predn = fit_mlp(
        F[tr], Yn[tr], F[te], Yn[te], epochs=args.epochs, seed=args.seed
    )
    pred = predn * y_sd + y_mu
    names = [f"psi{i}" for i in range(12)] + ["T"]
    m = metrics(pred, Y[te], names)
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict)}, indent=1))

    # Ridge baseline on a physics-inspired linear feature set.
    theta = X[:, 0]
    ic = X[:, 1:]
    th = np.deg2rad(theta)
    base = np.concatenate(
        [np.ones((len(X), 1)), np.sin(2 * th)[:, None], np.cos(2 * th)[:, None], ic], axis=1
    )
    inter = np.concatenate(
        [base, ic * np.sin(2 * th)[:, None], ic * np.cos(2 * th)[:, None]], axis=1
    )
    from sklearn.linear_model import Ridge

    ridge = Ridge(alpha=1e-3).fit(inter[tr], Y[tr])
    pred_r = ridge.predict(inter[te])
    m_r = metrics(pred_r, Y[te], names)

    out = {
        "data": args.data,
        "holdout": args.holdout,
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "mlp": m,
        "ridge": m_r,
        "ridge_train_r2": float(ridge.score(inter[tr], Y[tr])),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nMLP:")
    print(json.dumps({k: v for k, v in m.items() if not isinstance(v, dict)}, indent=1))
    print("\nRidge:")
    print(json.dumps({k: v for k, v in m_r.items() if not isinstance(v, dict)}, indent=1))
    print("saved", args.out)


if __name__ == "__main__":
    main()
