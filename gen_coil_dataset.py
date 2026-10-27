# -*- coding: utf-8 -*-
"""Generate a small coil-level FEA dataset for zero-shot winding-topology tests."""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from coil_fea import solve_coils
from exp.energy_torque import functional_torque


PHASE_COILS = {
    "A": ([0, 3, 6, 9], np.array([1.0, -1.0, 1.0, -1.0])),
    "B": ([2, 5, 8, 11], np.array([1.0, -1.0, 1.0, -1.0])),
    "C": ([1, 4, 7, 10], np.array([1.0, -1.0, 1.0, -1.0])),
}


def connection_currents(kind, ip):
    """Map three phase currents to twelve coil currents for a given connection."""
    ic = np.zeros(12)
    ia, ib, icp = ip

    if kind == "4S":
        for p, key in enumerate("ABC"):
            coils, signs = PHASE_COILS[key]
            ic[coils] = signs * ip[p]
    elif kind == "2S2P":
        for p, key in enumerate("ABC"):
            coils, signs = PHASE_COILS[key]
            ic[coils] = 0.5 * signs * ip[p]
    elif kind == "4P":
        for p, key in enumerate("ABC"):
            coils, signs = PHASE_COILS[key]
            ic[coils] = 0.25 * signs * ip[p]
    elif kind == "delta":
        # Balanced delta: winding currents from line currents.
        i_ab = (ia - ib) / 3.0
        i_bc = (ib - icp) / 3.0
        i_ca = (icp - ia) / 3.0
        for p, key in enumerate("ABC"):
            coils, signs = PHASE_COILS[key]
            iph = [i_ab, i_bc, i_ca][p]
            ic[coils] = signs * iph
    elif kind == "asym":
        # Phase A: 3 series + 1 parallel; B: 2+2; C: 1+3.
        splits = {"A": (3, 1), "B": (2, 2), "C": (1, 3)}
        for p, key in enumerate("ABC"):
            coils, signs = PHASE_COILS[key]
            n1, n2 = splits[key]
            # Branch currents are inversely proportional to branch resistance.
            i1 = ip[p] * n2 / (n1 + n2)
            i2 = ip[p] * n1 / (n1 + n2)
            ic[coils[:n1]] = signs[:n1] * i1
            ic[coils[n1:]] = signs[n1:] * i2
    else:
        raise ValueError(kind)
    return ic


def run_case(case):
    kind, theta, amp, beta, nr, nt = case
    ip = amp * np.array(
        [np.cos(beta), np.cos(beta - 2 * np.pi / 3), np.cos(beta + 2 * np.pi / 3)]
    )
    ic = connection_currents(kind, ip)
    r = solve_coils(theta, ic, nr=nr, nt=nt, pm_on=True, linear_iron=False, tol=2e-4)
    r["T"] = functional_torque(
        theta, ic, nr=nr, nt=nt, pm_on=True, linear_iron=False
    )
    return {
        "kind": kind,
        "theta": theta,
        "amp": amp,
        "beta": beta,
        "ip": ip,
        "ic": ic,
        "psi": r["psi"],
        "T": r["T"],
        "iters": r["iters"],
        "rel": r["rel"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="coil_dataset_quick.npz")
    ap.add_argument("--nr", type=int, default=40)
    ap.add_argument("--nt", type=int, default=96)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    kinds = ["4S", "2S2P", "asym"] if args.quick else ["4S", "2S2P", "delta", "asym"]
    thetas = np.arange(0, 180, 22.5) if args.quick else np.arange(0, 180, 15)
    amps = np.array([10.0, 20.0, 40.0, 60.0]) if args.quick else np.array([5, 10, 20, 30, 40, 50, 60.0])
    betas = np.array([0.0, np.pi / 3]) if args.quick else np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])

    cases = [
        (k, float(th), float(a), float(b), args.nr, args.nt)
        for k in kinds
        for th in thetas
        for a in amps
        for b in betas
    ]
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
    kinds_arr = np.array([r["kind"] for r in rows])
    theta_arr = np.array([r["theta"] for r in rows], dtype=np.float32)
    amp_arr = np.array([r["amp"] for r in rows], dtype=np.float32)
    beta_arr = np.array([r["beta"] for r in rows], dtype=np.float32)
    np.savez_compressed(
        args.out,
        X=X,
        Y=Y,
        kind=kinds_arr,
        theta=theta_arr,
        amp=amp_arr,
        beta=beta_arr,
    )
    meta = {
        "n": len(rows),
        "kinds": kinds,
        "thetas": thetas.tolist(),
        "amps": amps.tolist(),
        "betas": betas.tolist(),
        "nr": args.nr,
        "nt": args.nt,
        "seconds": time.time() - t0,
        "max_iters": max(r["iters"] for r in rows),
        "max_rel": max(r["rel"] for r in rows),
    }
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
