# -*- coding: utf-8 -*-
"""Synchronous-path dataset: coil currents along the actual dq trajectories the
drive-cycle study queries, i_c = C_m @ ip(2*theta+gamma), for the four
switchable modes with corrected standard phase axes."""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp.wmi_fix  # noqa: F401
from coil_fea import solve_coils  # noqa: E402
from exp.drive_cycle import connection_matrix  # noqa: E402
from exp.energy_torque import functional_torque  # noqa: E402

MODES = ("4S", "2S2P", "delta", "4P")
THETAS = np.arange(0, 180, 15)
GAMMAS = np.linspace(0.0, 2 * np.pi, 36, endpoint=False)
AMPS = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])


def run_case(case):
    mode, theta, amp, gamma = case
    C = connection_matrix(mode)
    s = 2 * np.deg2rad(theta) + gamma
    ip = amp * np.array([np.cos(s), np.cos(s - 2 * np.pi / 3),
                         np.cos(s + 2 * np.pi / 3)])
    ic = C @ ip
    r = solve_coils(float(theta), ic, nr=40, nt=96, pm_on=True,
                    linear_iron=False, tol=2e-4)
    r["T"] = functional_torque(
        float(theta), ic, nr=40, nt=96, pm_on=True, linear_iron=False
    )
    return {"mode": mode, "theta": float(theta), "amp": amp,
            "gamma": float(gamma), "ic": ic, "psi": r["psi"], "T": r["T"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="coil_dataset_sync.npz")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cases = [(m, float(th), float(a), float(g))
             for m in MODES for th in THETAS for a in AMPS for g in GAMMAS]
    print(f"cases: {len(cases)}", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        rows = pool.map(run_case, cases, chunksize=4)
    X = np.array([[r["theta"], *r["ic"]] for r in rows], dtype=np.float32)
    Y = np.array([[*r["psi"], r["T"]] for r in rows], dtype=np.float32)
    np.savez_compressed(
        args.out, X=X, Y=Y,
        kind=np.array([r["mode"] for r in rows]),
        theta=np.array([r["theta"] for r in rows], dtype=np.float32))
    meta = {"n": len(rows), "modes": MODES, "gammas": len(GAMMAS),
            "amps": AMPS.tolist(), "seconds": time.time() - t0}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1),
                                                   encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
