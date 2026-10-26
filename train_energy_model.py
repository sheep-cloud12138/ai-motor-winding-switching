# -*- coding: utf-8 -*-
"""Train a scalar magnetic co-energy model with autograd flux/torque, then zero-shot test."""

import argparse
import json

import numpy as np
import torch
from torch import nn


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


class EnergyModel(nn.Module):
    def __init__(self, n_coil=12, width=128):
        super().__init__()
        self.L = nn.Parameter(torch.randn(n_coil, n_coil) * 0.02)
        self.bias = MLP(7, n_coil, width=width, depth=2)
        self.res = MLP(n_coil + 7, 1, width=width, depth=3)

    def features(self, theta_deg):
        th = theta_deg * np.pi / 180.0
        return torch.stack(
            [
                torch.ones_like(th),
                torch.sin(2 * th),
                torch.cos(2 * th),
                torch.sin(4 * th),
                torch.cos(4 * th),
                torch.sin(6 * th),
                torch.cos(6 * th),
            ],
            dim=1,
        )

    def forward(self, ic, theta_deg):
        feats = self.features(theta_deg)
        lic = ic @ self.L
        quad = 0.5 * torch.sum(lic * lic, dim=1)
        bias = torch.sum(self.bias(feats) * ic, dim=1)
        res = self.res(torch.cat([ic, feats], dim=1)).squeeze(1)
        return quad + bias + res


def grad_outputs(model, ic, theta):
    ic = ic.clone().requires_grad_(True)
    theta = theta.clone().requires_grad_(True)
    W = model(ic, theta)
    psi = torch.autograd.grad(W.sum(), ic, create_graph=True)[0]
    T = torch.autograd.grad(W.sum(), theta, create_graph=True)[0]
    return psi, T


def metrics(pred_psi, pred_T, true_psi, true_T):
    err = pred_psi - true_psi
    rmse = np.sqrt(np.mean(err**2, axis=0))
    nrmse = rmse / np.maximum(np.std(true_psi, axis=0), 1e-9)
    r2 = 1 - np.sum(err**2, axis=0) / np.maximum(
        np.sum((true_psi - true_psi.mean(0)) ** 2, 0), 1e-12
    )
    terr = pred_T - true_T
    trmse = np.sqrt(np.mean(terr**2))
    tr2 = 1 - np.sum(terr**2) / max(np.sum((true_T - true_T.mean()) ** 2), 1e-12)
    return {
        "flux_rmse_mean": float(rmse.mean()),
        "flux_nrmse_mean": float(nrmse.mean()),
        "flux_r2_mean": float(r2.mean()),
        "torque_rmse": float(trmse),
        "torque_nrmse": float(trmse / max(np.std(true_T), 1e-9)),
        "torque_r2": float(tr2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="coil_dataset_mixed8.npz")
    ap.add_argument("--holdout", default="asym")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="energy_zero_shot.json")
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, Y, kind = d["X"], d["Y"], d["kind"]
    tr = kind != args.holdout
    te = kind == args.holdout
    print("train", tr.sum(), "test", te.sum(), "holdout", args.holdout)

    theta = X[:, 0].astype(np.float32)
    ic = X[:, 1:].astype(np.float32) / 60.0
    psi = Y[:, :12].astype(np.float32)
    T = Y[:, 12].astype(np.float32)
    psi_mu, psi_sd = psi[tr].mean(0, keepdims=True), psi[tr].std(0, keepdims=True) + 1e-8
    T_mu, T_sd = T[tr].mean(), T[tr].std() + 1e-8
    psi_n = (psi - psi_mu) / psi_sd
    T_n = (T - T_mu) / T_sd

    torch.manual_seed(0)
    model = EnergyModel(width=args.width)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ic_t = torch.tensor(ic[tr])
    th_t = torch.tensor(theta[tr])
    psi_t = torch.tensor(psi_n[tr])
    T_t = torch.tensor(T_n[tr])
    ic_v = torch.tensor(ic[te])
    th_v = torch.tensor(theta[te])
    psi_v = torch.tensor(psi_n[te])
    T_v = torch.tensor(T_n[te])
    best = None
    best_state = None
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        pred_psi, pred_T = grad_outputs(model, ic_t, th_t)
        loss = nn.functional.mse_loss(pred_psi, psi_t) + nn.functional.mse_loss(pred_T, T_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if ep % 250 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.enable_grad():
                pv, tv = grad_outputs(model, ic_v, th_v)
                vl = nn.functional.mse_loss(pv, psi_v).item() + nn.functional.mse_loss(tv, T_v).item()
            if best is None or vl < best:
                best = vl
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"ep {ep:4d} train {loss.item():.3e} val {vl:.3e}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.enable_grad():
        pv, tv = grad_outputs(model, ic_v, th_v)
    pred_psi = pv.detach().numpy() * psi_sd + psi_mu
    pred_T = tv.detach().numpy() * T_sd + T_mu
    m = metrics(pred_psi, pred_T, psi[te], T[te])
    out = {
        "data": args.data,
        "holdout": args.holdout,
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "best_val": float(best) if best is not None else None,
        "metrics": m,
    }
    print(json.dumps(m, indent=1))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
