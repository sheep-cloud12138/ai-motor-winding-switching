"""Compare finite-angle torque estimators on the fixed 64-point mesh check."""

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401,E402
from exp.energy_torque import functional_torque  # noqa: E402


def evaluate(case):
    theta, current, nr, nt, dtheta = case
    return functional_torque(
        theta, current, nr=nr, nt=nt, dtheta_deg=dtheta,
    )


def metrics(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return {
        "correlation": float(np.corrcoef(a, b)[0, 1]),
        "r2": float(1 - np.sum((a - b) ** 2) /
                    np.sum((b - b.mean()) ** 2)),
        "norm_ratio": float(np.linalg.norm(a) / np.linalg.norm(b)),
        "median_abs_relative": float(np.median(
            np.abs(a - b) / np.maximum(np.abs(b), 1e-9)
        )),
        "mae": float(np.mean(np.abs(a - b))),
    }


def main():
    with np.load(ROOT / "coil_fine_check.npz", allow_pickle=True) as fine:
        rows = fine["row"].astype(int)
        src = np.array([str(x) for x in fine["src"]])
        theta = fine["theta"].astype(float)
    pools = {
        name: np.load(ROOT / f"coil_dataset_{name}.npz", allow_pickle=True)
        for name in ("full", "ext", "random48")
    }
    currents = np.asarray([
        pools[s]["X"][row, 1:].astype(float) for s, row in zip(src, rows)
    ])
    coarse_7p5 = np.asarray([
        pools[s]["Y"][row, 12] for s, row in zip(src, rows)
    ], dtype=float)
    with mp.Pool(8) as pool:
        fine_7p5 = pool.map(
            evaluate,
            [(float(t), i, 80, 192, 7.5)
             for t, i in zip(theta, currents)],
            chunksize=1,
        )
        coarse_15 = pool.map(
            evaluate,
            [(float(t), i, 40, 96, 15.0)
             for t, i in zip(theta, currents)],
            chunksize=1,
        )
        fine_15 = pool.map(
            evaluate,
            [(float(t), i, 80, 192, 15.0)
             for t, i in zip(theta, currents)],
            chunksize=1,
        )
    result = {
        "comparison": "coarse 40x96 versus fine 80x192",
        "coarse_7p5_vs_fine_7p5": metrics(coarse_7p5, fine_7p5),
        "coarse_15_vs_fine_15": metrics(coarse_15, fine_15),
        "coarse_15_vs_fine_7p5": metrics(coarse_15, fine_7p5),
    }
    out = ROOT / "exp" / "results_theta_step_audit.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
