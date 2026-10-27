# -*- coding: utf-8 -*-
"""Main zero-shot benchmark: leave-one-connection-out + unseen connections.

Produces exp/results_benchmark.json with, for every (holdout, variant, model):
flux nRMSE / R2, torque nRMSE / R2 on the held-out connection, plus
seen-connection validation metrics and Jacobian-consistency probes.
Saves the EXT-protocol GradNet and MLP checkpoints for the drive-cycle study.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401  (WMI hang workaround, before numpy/scipy)

import numpy as np
import torch

from exp.data import EXT, STANDARD, load_all, make_split  # noqa: E402
from exp.models import (DirectTrainer, EnergyTrainer, GradNet, GradNetTrainer,  # noqa: E402
                        MLP, StructuredEnergy, jacobian_asymmetry)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def metrics(pred, true):
    err = pred - true
    rmse = np.sqrt(np.mean(err**2, axis=0))
    sd = np.maximum(np.std(true, axis=0), 1e-9)
    nrmse = rmse / sd
    r2 = 1 - np.sum(err**2, axis=0) / np.maximum(
        np.sum((true - true.mean(0)) ** 2, 0), 1e-12)
    return {
        "flux_nrmse": float(nrmse[:12].mean()),
        "flux_r2": float(r2[:12].mean()),
        "flux_nrmse_worst_coil": float(nrmse[:12].max()),
        "torque_nrmse": float(nrmse[12]),
        "torque_r2": float(r2[12]),
    }


class Runner:
    def __init__(self, X, Y, K):
        self.X, self.Y, self.K = X, Y, K

    def to_batch(self, pack):
        F, dphi, rows = pack
        return (torch.tensor(F, device=DEVICE),
                torch.tensor(dphi, device=DEVICE),
                torch.tensor(rows, device=DEVICE))

    def fit_standardized(self, split, model_name, seed=0, verbose=False,
                         train_kinds=None):
        Ff, dphif, rows_f = split["fit"]
        Fv, dphiv, rows_v = split["val"]
        if train_kinds is not None:  # e.g. per-connection baseline
            keep = np.isin(self.K[rows_f], train_kinds)
            Ff, dphif, rows_f = Ff[keep], dphif[keep], rows_f[keep]
        Yf = self.Y[rows_f]
        mu = Yf.mean(0, keepdims=True)
        sd = Yf.std(0, keepdims=True) + 1e-8

        def b(F, dphi, rows):
            y = (self.Y[rows] - mu) / sd
            return (torch.tensor(F, device=DEVICE), torch.tensor(dphi, device=DEVICE),
                    torch.tensor(y, dtype=torch.float32, device=DEVICE))

        torch.manual_seed(seed)
        if model_name == "mlp" or model_name == "perconn":
            model = MLP()
            trainer = DirectTrainer(model)
            tmax, pat, lr = 8000, 800, 1e-3
        elif model_name == "gradnet":
            model = GradNet()
            trainer = GradNetTrainer(model)
            tmax, pat, lr = 8000, 800, 1e-3
        else:
            model = StructuredEnergy()
            trainer = EnergyTrainer(model)
            tmax, pat, lr = 12000, 1500, 2e-3
        trainer.max_epochs, trainer.patience = tmax, pat
        trainer.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        trainer.max_epochs, trainer.patience = tmax, pat
        t0 = time.time()
        trainer.fit(b(Ff, dphif, rows_f), b(Fv, dphiv, rows_v), verbose=verbose)
        fit_s = time.time() - t0

        def predict(F, dphi):
            x = torch.tensor(F, device=DEVICE)
            dp = torch.tensor(dphi, device=DEVICE)
            with torch.no_grad() if model_name not in ("energy",) else torch.enable_grad():
                if model_name in ("mlp", "perconn"):
                    out = model(x).cpu().numpy()
                elif model_name == "gradnet":
                    psi, T = model.psi_T(x, dp)
                    out = torch.cat([psi, T[:, None]], 1).detach().cpu().numpy()
                else:
                    psi, T = model.psi_T(x[:, :12], x[:, 12:], dp)
                    out = torch.cat([psi.detach(), T.detach()[:, None]], 1).cpu().numpy()
            return out * sd + mu

        return model, predict, fit_s


def physics_ridge(runner, split, holdout):
    """Linear reciprocity-structured surrogate: psi = L(theta) i + pm(theta),
    L(theta) symmetric by parameterization (vectorized ridge)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    H = 7  # theta harmonics
    pairs = [(j, k) for j in range(12) for k in range(j, 12)]
    ncol = 12 * H + len(pairs) * H

    def design(F, rows):
        ic = runner.X[rows, 1:] / 60.0
        phi = F[:, 12:]
        n = len(rows)
        D = np.zeros((n * 12, ncol), dtype=np.float32)
        for s in range(n):
            base = s * 12
            for i in range(12):
                D[base + i, i * H:(i + 1) * H] = phi[s]
            for p, (j, k) in enumerate(pairs):
                c0 = 12 * H + p * H
                if j == k:
                    D[base + j, c0:c0 + H] = ic[s, j] * phi[s]
                else:
                    D[base + j, c0:c0 + H] = ic[s, k] * phi[s]
                    D[base + k, c0:c0 + H] = ic[s, j] * phi[s]
        return D

    Ff, _, rows_f = split["fit"]
    Fv, _, rows_v = split["val"]
    Fte, _, rows_te = split["test"]
    Yf, Yv, Yte = runner.Y[rows_f], runner.Y[rows_v], runner.Y[rows_te]
    Df, Dv, Dte = design(Ff, rows_f), design(Fv, rows_v), design(Fte, rows_te)
    yf = Yf[:, :12].reshape(-1)
    best = None
    for alpha in (1e-3, 1e-2, 1e-1, 1.0):
        sc = StandardScaler().fit(Df)
        r = Ridge(alpha=alpha).fit(sc.transform(Df), yf)
        pv = r.predict(sc.transform(Dv)).reshape(-1, 12)
        ev = np.sqrt(np.mean(((pv - Yv[:, :12]) /
                              np.maximum(Yv[:, :12].std(0), 1e-9)) ** 2))
        if best is None or ev < best[0]:
            best = (ev, alpha, sc, r)
    _, alpha, sc, r = best
    pte = r.predict(sc.transform(Dte)).reshape(-1, 12)
    err = pte - Yte[:, :12]
    nrmse = np.sqrt(np.mean(err**2, 0)) / np.maximum(Yte[:, :12].std(0), 1e-9)
    return {"flux_nrmse": float(nrmse.mean()), "alpha": alpha}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="discard saved benchmark results and retrain everything")
    args = ap.parse_args()
    X, Y, K = load_all()
    runner = Runner(X, Y, K)
    out_path = ROOT / "exp" / "results_benchmark.json"
    torque_sha = hashlib.sha256(
        np.ascontiguousarray(Y[:, 12]).tobytes()).hexdigest()
    meta = {"n_total": len(X),
            "counts": {k: int((K == k).sum()) for k in np.unique(K)},
            "device": DEVICE, "torque_sha256": torque_sha,
            "torque_model": "functional_v2"}
    if out_path.exists() and not args.force:
        with open(out_path) as f:
            results = json.load(f)
        if results.get("meta", {}).get("torque_sha256") != torque_sha:
            print("torque labels changed; discarding stale benchmark", flush=True)
            results = {"meta": meta}
    else:
        results = {"meta": meta}
    print(json.dumps(results["meta"], indent=1), flush=True)

    holdouts = list(STANDARD) + list(EXT)
    for holdout in holdouts:
        for variant in ("base", "norandom", "withz"):
            if variant == "norandom" and holdout in EXT and holdout != "4P":
                continue  # norandom is hopeless on OOD connections; skip except 4P
            key = f"{holdout}/{variant}"
            if key in results:
                print("skip", key, flush=True)
                continue
            split = make_split(X, Y, K, holdout, variant=variant)
            res = {}
            for model in ("mlp", "gradnet", "energy"):
                t0 = time.time()
                m, predict, fit_s = runner.fit_standardized(
                    split, model, verbose=(model == "gradnet" and variant == "base"))
                Fte, dphite, rows_te = split["test"]
                mm = metrics(predict(Fte, dphite), Y[rows_te])
                # Seen-connection validation metrics (E2), grouped by kind.
                Fv, dphiv, rows_v = split["val"]
                pv = predict(Fv, dphiv)
                per_kind = {}
                for k in np.unique(K[rows_v]):
                    sel = K[rows_v] == k
                    per_kind[k] = metrics(pv[sel], Y[rows_v[sel]])
                mm["val_per_kind"] = {
                    k: {"flux_nrmse": v["flux_nrmse"], "torque_r2": v["torque_r2"]}
                    for k, v in per_kind.items()}
                mm["fit_seconds"] = round(fit_s, 1)
                res[model] = mm
                print(key, model, "flux_nrmse %.3f torque_r2 %.3f (%.0fs)" % (
                    mm["flux_nrmse"], mm["torque_r2"], time.time() - t0), flush=True)
                if variant == "base":
                    torch.save(m.state_dict(), ROOT / "exp" /
                               f"ckpt_{model}_{holdout}.pt")
            if holdout in ("asym",) and variant == "base":
                # Per-connection baseline and physics ridge on the primary holdout.
                m, predict, fit_s = runner.fit_standardized(
                    split, "perconn", train_kinds=["4S"])
                Fte, dphite, rows_te = split["test"]
                res["perconn4S"] = metrics(predict(Fte, dphite), Y[rows_te])
                print(key, "perconn4S", res["perconn4S"]["flux_nrmse"], flush=True)
                res["ridge"] = physics_ridge(runner, split, holdout)
                print(key, "ridge", res["ridge"], flush=True)
            results[key] = res
            with open(ROOT / "exp" / "results_benchmark.json", "w") as f:
                json.dump(results, f, indent=1)

    # Jacobian reciprocity probes on the asym holdout, base variant.
    split = make_split(X, Y, K, "asym", variant="base")
    Fte, _, rows_te = split["test"]
    probe = np.linspace(0, len(Fte) - 1, 24).astype(int)
    xprobe = Fte[probe]
    probe_out = {}
    for model_name in ("mlp", "gradnet", "energy"):
        state = torch.load(ROOT / "exp" / f"ckpt_{model_name}_asym.pt",
                           weights_only=True)
        if model_name == "mlp":
            m = MLP(); m.load_state_dict(state); m.to(DEVICE).eval()
            pred = lambda F: m(torch.tensor(F, device=DEVICE)).detach()[:, :12].cpu().numpy()
        elif model_name == "gradnet":
            m = GradNet(); m.load_state_dict(state); m.to(DEVICE).eval()
            pred = lambda F: m(torch.tensor(F, device=DEVICE)).detach()[:, :12].cpu().numpy()
        else:
            m = StructuredEnergy(); m.load_state_dict(state); m.to(DEVICE).eval()
            def pred(F, m=m):
                x = torch.tensor(F, device=DEVICE)
                psi, _ = m.psi_T(x[:, :12].requires_grad_(True), x[:, 12:],
                                 torch.zeros(len(F), 7, device=DEVICE))
                return psi.detach().cpu().numpy()
        asym, mineig = jacobian_asymmetry(pred, xprobe)
        probe_out[model_name] = {
            "asym_mean": float(asym.mean()), "asym_max": float(asym.max()),
            "min_eig_mean": float(mineig.mean())}
        print("jacobian", model_name, probe_out[model_name], flush=True)
    results["jacobian_probe_asym"] = probe_out
    with open(ROOT / "exp" / "results_benchmark.json", "w") as f:
        json.dump(results, f, indent=1)
    print("done")


if __name__ == "__main__":
    main()
