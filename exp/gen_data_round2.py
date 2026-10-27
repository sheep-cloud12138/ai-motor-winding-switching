# -*- coding: utf-8 -*-
"""Round-2 data generation for the paper.

Generates:
  A. coil_dataset_random48.npz : 48 random per-coil patterns, scale U(0.2,1), seeds 9000+
  B. coil_dataset_random24z.npz: 24 random patterns, scale U(0,1) (zero-support ablation), seeds 12000+
  C. coil_dataset_ext.npz      : hold-out connections 4P / tap50 / openB, full grid
  D. coil_fine_check.npz       : re-solve a 64-point validation subset on a 80x192 mesh

Run:  python exp/gen_data_round2.py --workers 14
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Must be set before numpy/scipy load their BLAS: each worker should stay
# single-threaded or 14 workers x 24 threads exhausts memory.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from exp.wmi_fix import *  # noqa: F401,F403  (must run before scipy imports)
import exp.wmi_fix  # noqa: F401  (defensive: explicit import)

import gen_coil_dataset as gcd
import gen_random_dataset as grd
from coil_fea import solve_coils
from exp.energy_torque import functional_torque
from connections_ext import connection_currents_ext

# Make gen_coil_dataset.run_case understand the new kinds (also applied in
# spawned workers, because this assignment runs at import time of this module).
gcd.connection_currents = connection_currents_ext

THETAS = np.arange(0, 180, 15)
AMPS_STD = np.array([5, 10, 20, 30, 40, 50, 60.0])
AMPS_RND = np.array([10.0, 20.0, 40.0, 60.0])
BETAS_STD = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
BETAS_RND = np.array([0.0, np.pi / 2])


def run_random_z(case):
    """Random per-coil pattern with scale support including zero (U(0,1))."""
    theta, amp, beta, seed, nr, nt = case
    rng = np.random.default_rng(seed)
    ip = amp * np.array(
        [np.cos(beta), np.cos(beta - 2 * np.pi / 3), np.cos(beta + 2 * np.pi / 3)]
    )
    base = connection_currents_ext("4S", ip)
    scale = rng.uniform(0.0, 1.0, size=12)
    ic = base * scale
    rms = np.sqrt(np.mean(ic**2))
    if rms > 1e-9:
        ic *= amp / rms
    r = solve_coils(theta, ic, nr=nr, nt=nt, pm_on=True, linear_iron=False, tol=2e-4)
    r["T"] = functional_torque(
        theta, ic, nr=nr, nt=nt, pm_on=True, linear_iron=False
    )
    return {
        "theta": theta, "amp": amp, "beta": beta, "seed": seed,
        "ic": ic, "psi": r["psi"], "T": r["T"],
    }


def save_rows(rows, out, kind):
    X = np.array([[r["theta"], *r["ic"]] for r in rows], dtype=np.float32)
    Y = np.array([[*r["psi"], r["T"]] for r in rows], dtype=np.float32)
    np.savez_compressed(
        out, X=X, Y=Y,
        kind=np.array([kind] * len(rows)),
        theta=np.array([r["theta"] for r in rows], dtype=np.float32),
        amp=np.array([r["amp"] for r in rows], dtype=np.float32),
        beta=np.array([r["beta"] for r in rows], dtype=np.float32),
    )
    return {"n": len(rows)}


def run_fine_case(case):
    theta, ic, nr, nt = case
    r = solve_coils(theta, ic, nr=nr, nt=nt, pm_on=True, linear_iron=False, tol=2e-4)
    r["T"] = functional_torque(
        theta, ic, nr=nr, nt=nt, pm_on=True, linear_iron=False
    )
    return {"theta": theta, "psi": r["psi"], "T": r["T"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent

    t_all = time.time()
    print("creating pool ...", flush=True)
    pool = mp.Pool(args.workers)
    print("pool created", flush=True)

    # A. random48, scale U(0.2,1.0), seed base 9000 (no overlap with 1000..3303).
    cases = []
    seed = 9000
    for th in THETAS:
        for a in AMPS_RND:
            for b in BETAS_RND:
                for _ in range(48):
                    cases.append((float(th), float(a), float(b), seed, 40, 96))
                    seed += 1
    t0 = time.time()
    rows = pool.map(grd.run_case, cases, chunksize=4)
    meta_a = save_rows(rows, root / "coil_dataset_random48.npz", "random")
    meta_a.update(seconds=time.time() - t0, patterns=48, scale="U(0.2,1.0)")
    print("A random48:", meta_a, flush=True)

    # B. random24z, scale U(0,1), seed base 12000.
    cases = []
    seed = 12000
    for th in THETAS:
        for a in AMPS_RND:
            for b in BETAS_RND:
                for _ in range(24):
                    cases.append((float(th), float(a), float(b), seed, 40, 96))
                    seed += 1
    t0 = time.time()
    rows = pool.map(run_random_z, cases, chunksize=4)
    meta_b = save_rows(rows, root / "coil_dataset_random24z.npz", "randomz")
    meta_b.update(seconds=time.time() - t0, patterns=24, scale="U(0,1.0)")
    print("B random24z:", meta_b, flush=True)

    # C. extended hold-out connections on the standard grid.
    cases = [
        (k, float(th), float(a), float(b), 40, 96)
        for k in ("4P", "tap50", "openB")
        for th in THETAS
        for a in AMPS_STD
        for b in BETAS_STD
    ]
    t0 = time.time()
    rows = pool.map(gcd.run_case, cases, chunksize=4)
    X = np.array([[r["theta"], *r["ic"]] for r in rows], dtype=np.float32)
    Y = np.array([[*r["psi"], r["T"]] for r in rows], dtype=np.float32)
    np.savez_compressed(
        root / "coil_dataset_ext.npz", X=X, Y=Y,
        kind=np.array([r["kind"] for r in rows]),
        theta=np.array([r["theta"] for r in rows], dtype=np.float32),
        amp=np.array([r["amp"] for r in rows], dtype=np.float32),
        beta=np.array([r["beta"] for r in rows], dtype=np.float32),
    )
    meta_c = {"n": len(rows), "seconds": time.time() - t0}
    print("C ext:", meta_c, flush=True)

    # D. fine-mesh check on a fixed 64-point subset (nr=80 nt=192, in the pool).
    rng = np.random.default_rng(7)
    fine_cases = []
    fine_src = []
    full = np.load(root / "coil_dataset_full.npz", allow_pickle=True)
    ext = np.load(root / "coil_dataset_ext.npz", allow_pickle=True)
    r48 = np.load(root / "coil_dataset_random48.npz", allow_pickle=True)
    for tag, d, n in (("full", full, 28), ("ext", ext, 20), ("random48", r48, 16)):
        idx = rng.choice(len(d["X"]), size=n, replace=False)
        for j in idx:
            fine_cases.append((float(d["X"][j, 0]), d["X"][j, 1:].astype(float), 80, 192))
            fine_src.append((tag, int(j)))
    t0 = time.time()
    fine = pool.map(run_fine_case, fine_cases, chunksize=1)
    np.savez_compressed(
        root / "coil_fine_check.npz",
        src=np.array([s[0] for s in fine_src]),
        row=np.array([s[1] for s in fine_src]),
        theta=np.array([f["theta"] for f in fine], dtype=np.float32),
        psi=np.array([f["psi"] for f in fine], dtype=np.float32),
        T=np.array([f["T"] for f in fine], dtype=np.float32),
    )
    meta_d = {"n": len(fine), "seconds": time.time() - t0}
    print("D fine:", meta_d, flush=True)

    pool.close()
    pool.join()

    meta = {"A_random48": meta_a, "B_random24z": meta_b, "C_ext": meta_c,
            "D_fine": meta_d, "total_seconds": time.time() - t_all}
    (root / "exp" / "gen_data_round2.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
