import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from coil_fea import solve_coils


def solve(theta, ic):
    return solve_coils(theta, ic, nr=80, nt=192, pm_on=True,
                       linear_iron=True, tol=2e-4)


theta, dtheta, di = 30.0, 1.0, 0.5
dpsi = np.zeros(12)
dTdi = np.zeros(12)
for j in range(12):
    e = np.zeros(12)
    e[j] = di
    pplus = solve(theta + dtheta, e)["psi"][j]
    pminus = solve(theta - dtheta, e)["psi"][j]
    dpsi[j] = (pplus - pminus) / (2 * np.deg2rad(dtheta))
    dTdi[j] = (solve(theta, e)["T"] - solve(theta, -e)["T"]) / (2 * di)
print("dpsi", np.round(dpsi, 6).tolist())
print("dTdi", np.round(dTdi, 6).tolist())
print("corr", np.corrcoef(dpsi, dTdi)[0, 1],
      "norm ratio", np.linalg.norm(dTdi) / np.linalg.norm(dpsi))
