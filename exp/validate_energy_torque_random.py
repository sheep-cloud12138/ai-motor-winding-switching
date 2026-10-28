"""Stress-test the discrete Maxwell relation beyond one 4S operating point."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from coil_fea import solve_coils  # noqa: E402
from exp.energy_torque import functional_torque  # noqa: E402
from gen_coil_dataset import connection_currents  # noqa: E402

DTHETA = 7.5
DI = 0.5


def check(name, theta, ic):
    pp = solve_coils(theta + DTHETA, ic, nr=40, nt=96, tol=2e-4)["psi"]
    pm = solve_coils(theta - DTHETA, ic, nr=40, nt=96, tol=2e-4)["psi"]
    dpsi = (pp - pm) / (2 * np.deg2rad(DTHETA))
    dTdi = np.zeros(12)
    for j in range(12):
        step = np.zeros(12)
        step[j] = DI
        dTdi[j] = (
            functional_torque(theta, ic + step, nr=40, nt=96)
            - functional_torque(theta, ic - step, nr=40, nt=96)
        ) / (2 * DI)
    return {
        "name": name,
        "theta_deg": float(theta),
        "current_rms_A": float(np.sqrt(np.mean(ic ** 2))),
        "correlation": float(np.corrcoef(dpsi, dTdi)[0, 1]),
        "relative_norm_residual": float(
            np.linalg.norm(dTdi - dpsi) / np.linalg.norm(dpsi)
        ),
    }


def main():
    rng = np.random.default_rng(0)
    ip = 40.0 * np.array([
        np.cos(np.pi / 4),
        np.cos(np.pi / 4 - 2 * np.pi / 3),
        np.cos(np.pi / 4 + 2 * np.pi / 3),
    ])
    i_4s = connection_currents("4S", ip)
    random_30 = rng.normal(size=12)
    random_30 *= 30.0 / np.sqrt(np.mean(random_30 ** 2))
    random_50 = rng.normal(size=12)
    random_50 *= 50.0 / np.sqrt(np.mean(random_50 ** 2))
    cases = [
        ("4S_40A_theta45", 45.0, i_4s),
        ("random_30A_theta45", 45.0, random_30),
        ("random_50A_theta105", 105.0, random_50),
    ]
    cases += [
        (f"random_30A_theta{int(theta)}", float(theta), random_30)
        for theta in np.arange(0, 180, 15)
    ]
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        current = rng.normal(size=12)
        current *= 30.0 / np.sqrt(np.mean(current ** 2))
        cases.append((f"random_seed{seed}_theta105", 105.0, current))
    result = {"dtheta_deg": DTHETA, "current_step_A": DI,
              "cases": [check(*case) for case in cases]}
    out = ROOT / "exp" / "results_energy_torque_random.json"
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
