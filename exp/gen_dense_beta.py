# -*- coding: utf-8 -*-
"""Dense-beta dataset for the drive-cycle study: standard connections over a
dense current-angle grid covering one electrical period (beta in [0, 2pi/3),
12 values), so synchronous dq trajectories stay inside the training support."""

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
from connections_ext import connection_currents_ext  # noqa: E402
from coil_fea import solve_coils  # noqa: E402

KINDS = ("4S", "2S2P", "delta", "4P")
THETAS = np.arange(0, 180, 15)
AMPS = np.array([10.0, 20.0, 40.0, 60.0])
BETAS = np.linspace(0.0, 2 * np.pi / 3, 12, endpoint=False)


def run_case(case):
    kind, theta, amp, beta, nr, nt = case
    ip = amp * np.array([np.cos(beta), np.cos(beta - 2 * np.pi / 3),
                         np.cos(beta + 2 * np.pi / 3)])
    ic = connection_currents_ext(kind, ip)
    r = solve_coils(theta, ic, nr=nr, nt=nt, pm_on=True, linear_iron=False,
                    tol=2e-4)
    return {"kind": kind, "theta": theta, "amp": amp, "beta": beta,
            "ic": ic, "psi": r["psi"], "T": r["T"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="coil_dataset_densebeta.npz")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cases = [(k, float(th), float(a), float(b), 40, 96)
             for k in KINDS for th in THETAS for a in AMPS for b in BETAS]
    print(f"cases: {len(cases)}", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        rows = pool.map(run_case, cases, chunksize=4)
    X = np.array([[r["theta"], *r["ic"]] for r in rows], dtype=np.float32)
    Y = np.array([[*r["psi"], r["T"]] for r in rows], dtype=np.float32)
    np.savez_compressed(
        args.out, X=X, Y=Y, kind=np.array([r["kind"] for r in rows]),
        theta=np.array([r["theta"] for r in rows], dtype=np.float32),
        amp=np.array([r["amp"] for r in rows], dtype=np.float32),
        beta=np.array([r["beta"] for r in rows], dtype=np.float32))
    meta = {"n": len(rows), "kinds": KINDS, "betas": len(BETAS),
            "seconds": time.time() - t0}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1),
                                                   encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
