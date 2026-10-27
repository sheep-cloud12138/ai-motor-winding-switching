import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from coil_fea import solve_coils
from gen_coil_dataset import connection_currents


def run(label, theta, pm_on, linear_iron, nr=80, nt=192):
    ip = 20.0 * np.array([
        np.cos(np.pi / 4),
        np.cos(np.pi / 4 - 2 * np.pi / 3),
        np.cos(np.pi / 4 + 2 * np.pi / 3),
    ])
    ic = connection_currents("4S", ip)

    def solve(th, cur):
        return solve_coils(th, cur, nr=nr, nt=nt, pm_on=pm_on,
                           linear_iron=linear_iron, tol=2e-4)

    dtheta = 1.0
    pplus = solve(theta + dtheta, ic)["psi"]
    pminus = solve(theta - dtheta, ic)["psi"]
    dpsi = (pplus - pminus) / (2 * np.deg2rad(dtheta))
    dTdi = []
    di = 0.25
    for j in range(12):
        step = np.zeros(12)
        step[j] = di
        dTdi.append((solve(theta, ic + step)["T"] -
                     solve(theta, ic - step)["T"]) / (2 * di))
    dTdi = np.asarray(dTdi)
    print(label)
    print(" dpsi", np.round(dpsi, 6).tolist())
    print(" dTdi", np.round(dTdi, 6).tolist())
    print(" corr", np.corrcoef(dpsi, dTdi)[0, 1],
          "norm ratio", np.linalg.norm(dTdi) / np.linalg.norm(dpsi))


run("nonlinear/noPM/highres", 0.0, False, False, 120, 288)
