# -*- coding: utf-8 -*-
"""Random 3-phase excitation patterns: per-phase amplitude AND phase both
random (covers zero-/negative-sequence and unbalanced phase relations that
magnitude-only random patterns cannot reach)."""

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
from gen_coil_dataset import connection_currents  # noqa: E402


def run_case(case):
    theta, amp, seed, nr, nt = case
    rng = np.random.default_rng(seed)
    # random balanced-with-zero-sequence 3-phase vector: common amplitude and
    # angle, plus a random zero-sequence component (sums need not vanish).
    a = amp * rng.uniform(0.2, 1.0)
    phi = rng.uniform(0, 2 * np.pi, size=3)
    z = amp * rng.uniform(-0.3, 0.3)
    ip = a * np.cos(phi) + z
    ic = connection_currents("4S", ip)
    rms = np.sqrt(np.mean(ic**2))
    if rms > 1e-9:
        ic *= amp / rms
    r = solve_coils(theta, ic, nr=nr, nt=nt, pm_on=True, linear_iron=False,
                    tol=2e-4)
    return {"theta": theta, "amp": amp, "seed": seed, "ic": ic,
            "psi": r["psi"], "T": r["T"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="coil_dataset_random3.npz")
    ap.add_argument("--patterns", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    thetas = np.arange(0, 180, 15)
    amps = np.array([10.0, 20.0, 40.0, 60.0])
    cases = []
    seed = 20000
    for th in thetas:
        for a in amps:
            for _ in range(args.patterns):
                cases.append((float(th), float(a), seed, 40, 96))
                seed += 1
    print(f"cases: {len(cases)}", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        rows = pool.map(run_case, cases, chunksize=4)
    X = np.array([[r["theta"], *r["ic"]] for r in rows], dtype=np.float32)
    Y = np.array([[*r["psi"], r["T"]] for r in rows], dtype=np.float32)
    np.savez_compressed(args.out, X=X, Y=Y,
                        kind=np.array(["random3"] * len(rows)),
                        theta=np.array([r["theta"] for r in rows],
                                       dtype=np.float32))
    meta = {"n": len(rows), "patterns": args.patterns,
            "seconds": time.time() - t0}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=1),
                                                   encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
