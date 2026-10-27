"""Validate functional torque labels for the 6-slot replication machine."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import coil_fea2
from coil_fea2 import connection_currents, solve_coils
from exp.energy_torque import functional_torque

theta, dtheta, di = 30.0, 1.0, 0.5
ip = 20.0 * np.array([
    np.cos(np.pi / 4),
    np.cos(np.pi / 4 - 2 * np.pi / 3),
    np.cos(np.pi / 4 + 2 * np.pi / 3),
])
ic = connection_currents("2S", ip)


def field(th, current):
    return solve_coils(th, current, nr=40, nt=48, pm_on=True,
                       linear_iron=False, tol=2e-4)


pplus = field(theta + dtheta, ic)["psi"]
pminus = field(theta - dtheta, ic)["psi"]
dpsi = (pplus - pminus) / (2 * np.deg2rad(dtheta))
dTdi = np.zeros(6)
for j in range(6):
    step = np.zeros(6)
    step[j] = di
    dTdi[j] = (
        functional_torque(theta, ic + step, nr=40, nt=48,
                          pm_on=True, linear_iron=False, solver=coil_fea2)
        - functional_torque(theta, ic - step, nr=40, nt=48,
                            pm_on=True, linear_iron=False, solver=coil_fea2)
    ) / (2 * di)
out = {
    "dpsi_dtheta": dpsi.tolist(),
    "dT_di": dTdi.tolist(),
    "correlation": float(np.corrcoef(dpsi, dTdi)[0, 1]),
    "relative_norm_residual": float(
        np.linalg.norm(dTdi - dpsi) / np.linalg.norm(dpsi)
    ),
}
(ROOT / "exp" / "results_energy_torque_validation2.json").write_text(
    json.dumps(out, indent=1), encoding="utf-8"
)
print(json.dumps({k: out[k] for k in
                  ("correlation", "relative_norm_residual")}, indent=1))
