# -*- coding: utf-8 -*-
"""Generate zero-current (PM-only) coil flux samples."""

import json

import numpy as np

from coil_fea import solve_coils
from exp.energy_torque import functional_torque


def main():
    rows = []
    for theta in np.arange(0, 180, 15):
        r = solve_coils(float(theta), np.zeros(12), nr=40, nt=96, pm_on=True)
        r["T"] = functional_torque(
            float(theta), np.zeros(12), nr=40, nt=96, pm_on=True
        )
        rows.append({"theta": float(theta), "psi": r["psi"], "T": r["T"]})
    X = np.array([[r["theta"], *np.zeros(12)] for r in rows], dtype=np.float32)
    Y = np.array([[*r["psi"], r["T"]] for r in rows], dtype=np.float32)
    np.savez_compressed("coil_zero.npz", X=X, Y=Y)
    print(json.dumps({"n": len(rows), "T": [r["T"] for r in rows]}, indent=1))


if __name__ == "__main__":
    main()
