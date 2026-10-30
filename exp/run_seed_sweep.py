"""Three-seed uncertainty and compute-cost check for key unseen connections."""

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

import numpy as np  # noqa: E402

from exp.data import load_all, make_split  # noqa: E402
from exp.run_benchmark import Runner, metrics  # noqa: E402


def summarize(values):
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    # t_{0.975,2} for a small three-seed interval.
    half = 4.303 * sd / math.sqrt(len(a))
    return {
        "mean": mean,
        "std": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def main():
    X, Y, K = load_all()
    runner = Runner(X, Y, K)
    out = {"seeds": [0, 1, 2], "holdouts": {}}
    for holdout in ("4P", "tap50", "openB"):
        split = make_split(X, Y, K, holdout, variant="base")
        Fte, dphite, rows_te = split["test"]
        out["holdouts"][holdout] = {}
        for model_name in ("mlp", "energy"):
            rows = []
            for seed in out["seeds"]:
                m, predict, fit_s = runner.fit_standardized(
                    split, model_name, seed=seed
                )
                mm = metrics(predict(Fte, dphite), Y[rows_te])
                rows.append({"seed": seed, **mm, "fit_seconds": fit_s})
                print(holdout, model_name, seed,
                      "flux", round(mm["flux_nrmse"], 4),
                      "T_R2", round(mm["torque_r2"], 4),
                      "fit_s", round(fit_s, 1), flush=True)
            out["holdouts"][holdout][model_name] = {
                "runs": rows,
                "flux_nrmse": summarize([r["flux_nrmse"] for r in rows]),
                "torque_r2": summarize([r["torque_r2"] for r in rows]),
                "fit_seconds": summarize([r["fit_seconds"] for r in rows]),
            }
    path = ROOT / "exp" / "results_seed_sweep.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
