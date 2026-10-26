# -*- coding: utf-8 -*-
"""Generate random per-coil current patterns to cover non-uniform winding excitations."""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from coil_fea import solve_coils
from gen_coil_dataset import connection_currents


def run_case(case):
    theta, amp, beta, seed, nr, nt = case
    rng = np.random.default_rng(seed)
    ip = amp * np.array(
        [np.cos(beta), np.cos(beta - 2 * np.pi / 3), np.cos(beta + 2 * np.pi / 3)]
    )
    base = connection_currents("4S", ip)
    scale = rng.uniform(0.2, 1.0, size=12)
    ic = base * scale
    # Keep the RMS current comparable to the requested amplitude.
    rms = np.sqrt(np.mean(ic**2))
    if rms > 1e-9:
        ic *= amp / rms
    r = solve_coils(theta, ic, nr=nr, nt=nt, pm_on=True, linear_iron=False, tol=2e-4)
    return {
        "theta": theta,
        "amp": amp,
        "beta": beta,
        "seed": seed,
        "ic": ic,
        "psi": r["psi"],
        "T": r["T"],
        "iters": r["iters"],
        "rel": r["rel"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="coil_dataset_random.npz")
    ap.add_argument("--nr", type=int, default=40)
    ap.add_argument("--nt", type=int, default=96)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--patterns", type=int, default=2)
    args = ap.parse_args()

    thetas = np.arange(0, 180, 15)
    amps = np.array([10.0, 20.0, 40.0, 60.0])
    betas = np.array([0.0, np.pi / 2])
    cases = []
    seed = 1000
    for th in thetas:
        for a in amps:
            for b in betas:
                for _ in range(args.patterns):
                    cases.append((float(th), float(a), float(b), seed, args.nr, args.nt))
                    seed += 1

    print(f"cases: {len(cases)}", flush=True)
    t0 = time.time()
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            rows = pool.map(run_case, cases, chunksize=2)
    else:
        rows = [run_case(c) for c in cases]
    print(f"solved in {time.time() - t0:.1f} s", flush=True)

    X = np.array([[r["theta"], *r["ic"]] for r in rows], dtype=np.float32)
    Y = np.array([[*r["psi"], r["T"]] for r in rows], dtype=np.float32)
    np.savez_compressed(
        args.out,
        X=X,
        Y=Y,
        kind=np.array(["random"] * len(rows)),
        theta=np.array([r["theta"] for r in rows], dtype=np.float32),
        amp=np.array([r["amp"] for r in rows], dtype=np.float32),
        beta=np.array([r["beta"] for r in rows], dtype=np.float32),
    )
    meta = {
        "n": len(rows),
        "thetas": thetas.tolist(),
        "amps": amps.tolist(),
        "betas": betas.tolist(),
        "patterns": args.patterns,
        "seconds": time.time() - t0,
    }
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
