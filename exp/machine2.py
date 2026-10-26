# -*- coding: utf-8 -*-
"""Cross-machine replication on a 6-slot/4-pole machine.

Generates the same dataset families as the 12s4p study (standard
connections, magnitude-random, phase-random, PM-only) and re-runs the two
core coverage ablations:
  A1  hold-out tap50 with {standard, +magnitude, +phase} pools
  A2  hold-out openB with the same pools
A direct-MLP model, identical training protocol to the main study.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import exp.wmi_fix  # noqa: F401
from coil_fea2 import connection_currents, solve_coils  # noqa: E402

THETAS = np.arange(0, 180, 15)
AMPS = np.array([10.0, 20.0, 40.0, 60.0])
BETAS = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
KINDS = ("2S", "2P", "tap50", "openB")


def run_conn(case):
    kind, th, a, b = case
    ip = a * np.array([np.cos(b), np.cos(b - 2 * np.pi / 3),
                       np.cos(b + 2 * np.pi / 3)])
    ic = connection_currents(kind, ip)
    r = solve_coils(float(th), ic, nr=40, nt=48, pm_on=True,
                    linear_iron=False, tol=2e-4)
    return np.concatenate([[th], ic, r["psi"], [r["T"]]])


def run_rand(case):
    th, a, b, seed, mode = case
    rng = np.random.default_rng(seed)
    if mode == "mag":
        ip = a * np.array([np.cos(b), np.cos(b - 2 * np.pi / 3),
                           np.cos(b + 2 * np.pi / 3)])
        base = connection_currents("2S", ip)
        scale = rng.uniform(0.2, 1.0, size=6)
        ic = base * scale
        rms = np.sqrt(np.mean(ic ** 2))
        if rms > 1e-9:
            ic *= a / rms
    else:  # phase-random
        amp = a * rng.uniform(0.2, 1.0)
        phi = rng.uniform(0, 2 * np.pi, size=3)
        z = a * rng.uniform(-0.3, 0.3)
        ip = amp * np.cos(phi) + z
        ic = connection_currents("2S", ip)
        rms = np.sqrt(np.mean(ic ** 2))
        if rms > 1e-9:
            ic *= a / rms
    r = solve_coils(float(th), ic, nr=40, nt=48, pm_on=True,
                    linear_iron=False, tol=2e-4)
    return np.concatenate([[th], ic, r["psi"], [r["T"]]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    pool = mp.Pool(args.workers)

    cases = [(k, float(th), float(a), float(b))
             for k in KINDS for th in THETAS for a in AMPS for b in BETAS]
    rows = pool.map(run_conn, cases, chunksize=4)
    tagc = np.array([c[0] for c in cases])
    print("conn:", len(rows), flush=True)

    cases_r = []
    s = 30000
    for mode, groups in (("mag", 24), ("ph", 12)):
        for th in THETAS:
            for a in AMPS:
                for b in ([np.pi / 4] if mode == "mag" else [0.0, np.pi / 2]):
                    for _ in range(groups):
                        cases_r.append((float(th), float(a), float(b), s,
                                        mode))
                        s += 1
    rows_r = pool.map(run_rand, cases_r, chunksize=4)
    tagr = np.array([c[4] for c in cases_r])
    print("rand:", len(rows_r), flush=True)

    zrows = []
    for th in THETAS:
        r = solve_coils(float(th), np.zeros(6), nr=40, nt=48)
        zrows.append(np.concatenate([[th], np.zeros(6), r["psi"], [r["T"]]]))
    pool.close(); pool.join()
    zrows = np.array(zrows, dtype=np.float32)

    # assemble pools
    R = lambda x: np.array(x, dtype=np.float32)
    std = R(rows); rnd = R(rows_r)
    mag = rnd[tagr == "mag"]; ph = rnd[tagr == "ph"]
    labels = np.concatenate([tagc, np.array(["mag"] * len(mag)),
                             np.array(["ph"] * len(ph)),
                             np.array(["zero"] * len(zrows))])
    Xall = np.concatenate([std, mag, ph, zrows])
    Yall = Xall[:, 7:14]
    Xin = np.concatenate([Xall[:, :1], Xall[:, 1:7]], axis=1)
    print("total", len(Xin), "elapsed %.0fs" % (time.time() - t0), flush=True)

    # ---- coverage ablations (direct MLP, same protocol) ----
    import torch
    from torch import nn
    from exp.models import make_features, theta_feature_grad

    class MLP2(nn.Module):
        def __init__(self):
            super().__init__()
            layers = [nn.Linear(13, 128), nn.GELU()]
            for _ in range(2):
                layers += [nn.Linear(128, 128), nn.GELU()]
            layers += [nn.Linear(128, 7)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def train_eval(holdout, pools):
        rng = np.random.default_rng(0)
        in_tr = np.isin(labels, [k for k in KINDS if k != holdout] + ["zero"])
        for p in pools:
            in_tr |= labels == p
        pool_idx = np.where(in_tr)[0]
        val_mask = np.zeros(len(pool_idx), bool)
        for k in np.unique(labels[pool_idx]):
            ix = np.where(labels[pool_idx] == k)[0]
            val_mask[rng.choice(ix, size=max(3, int(0.12 * len(ix))),
                                replace=False)] = True
        fi, vi = pool_idx[~val_mask], pool_idx[val_mask]
        te = np.where(labels == holdout)[0]

        def pack(rows):
            return (torch.tensor(make_features(Xin[rows, 0],
                                               Xin[rows, 1:]), device=dev),
                    torch.tensor(Xin[rows], device=dev), rows)

        xf, _, _ = pack(fi); xv, _, _ = pack(vi); xt, _, _ = pack(te)
        yf = torch.tensor((Yall[fi] - Yall[fi].mean(0)) /
                          (Yall[fi].std(0) + 1e-8), dtype=torch.float32,
                          device=dev)
        yv = torch.tensor((Yall[vi] - Yall[fi].mean(0)) /
                          (Yall[fi].std(0) + 1e-8), dtype=torch.float32,
                          device=dev)
        torch.manual_seed(0)
        m = MLP2().to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(6000):
            m.train(); opt.zero_grad()
            loss = nn.functional.mse_loss(m(xf), yf)
            loss.backward(); opt.step()
            if ep % 50 == 0:
                m.eval()
                with torch.no_grad():
                    vl = nn.functional.mse_loss(m(xv), yv).item()
                if vl < best - 1e-9:
                    best, bad = vl, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in m.state_dict().items()}
                else:
                    bad += 50
                    if bad >= 800:
                        break
        m.load_state_dict(best_state); m.eval()
        with torch.no_grad():
            pred = m(xt).cpu().numpy() * Yall[fi].std(0) + Yall[fi].mean(0)
        true = Yall[te]
        err = pred - true
        nrmse = (np.sqrt(np.mean(err ** 2, 0)) /
                 np.maximum(np.std(true, 0), 1e-9))
        r2 = 1 - np.sum(err ** 2, 0) / np.maximum(
            np.sum((true - true.mean(0)) ** 2, 0), 1e-12)
        return float(nrmse[:6].mean()), float(r2[6])

    out = {"machine": "6s4p", "n_total": int(len(Xin)),
           "seconds": time.time() - t0}
    for holdout in ("tap50", "openB"):
        for pname, pools in (("standard", []),
                             ("+mag", ["mag"]),
                             ("+ph", ["ph"]),
                             ("+mag+ph", ["mag", "ph"])):
            fnr, tr2 = train_eval(holdout, pools)
            out[f"{holdout}/{pname}"] = {"flux_nrmse": fnr, "torque_r2": tr2}
            print(f"{holdout}/{pname}: nrmse {fnr:.3f} T_r2 {tr2:.3f}",
                  flush=True)
    with open(ROOT / "exp" / "results_machine2.json", "w") as f:
        json.dump(out, f, indent=1)
    print("saved")


if __name__ == "__main__":
    main()
