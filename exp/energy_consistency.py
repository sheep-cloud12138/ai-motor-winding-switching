"""Check the discrete Maxwell relation d(psi)/d(theta) = dT/di."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from coil_fea import solve_coils
from exp.energy_torque import functional_torque
from gen_coil_dataset import connection_currents

THETA = 30.0
DTHETA = 7.5
DI = 0.5
ip = 40.0 * np.array([
    np.cos(np.pi / 4),
    np.cos(np.pi / 4 - 2 * np.pi / 3),
    np.cos(np.pi / 4 + 2 * np.pi / 3),
])
ic = connection_currents("4S", ip)


def solve(theta, current):
    return solve_coils(theta, current, nr=80, nt=192, pm_on=True,
                       linear_iron=False, tol=2e-4)


p_plus = solve(THETA + DTHETA, ic)["psi"]
p_minus = solve(THETA - DTHETA, ic)["psi"]
dpsi_dtheta = (p_plus - p_minus) / (2 * np.deg2rad(DTHETA))

dT_di = np.zeros(12)
for j in range(12):
    step = np.zeros(12)
    step[j] = DI
    tp = functional_torque(THETA, ic + step, nr=80, nt=192,
                           dtheta_deg=DTHETA)
    tm = functional_torque(THETA, ic - step, nr=80, nt=192,
                           dtheta_deg=DTHETA)
    dT_di[j] = (tp - tm) / (2 * DI)

rel = float(np.linalg.norm(dT_di - dpsi_dtheta) /
            np.linalg.norm(dpsi_dtheta))
result = {
    "theta_deg": THETA,
    "dtheta_deg": DTHETA,
    "current_step_A": DI,
    "dpsi_dtheta": dpsi_dtheta.tolist(),
    "dT_di": dT_di.tolist(),
    "relative_norm_residual": rel,
    "correlation": float(np.corrcoef(dT_di, dpsi_dtheta)[0, 1]),
}
path = ROOT / "exp" / "results_energy_consistency.json"
path.write_text(json.dumps(result, indent=1), encoding="utf-8")
print(json.dumps({k: result[k] for k in
                  ("relative_norm_residual", "correlation")}, indent=1))
