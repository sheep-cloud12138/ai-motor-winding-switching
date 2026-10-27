"""Torque from the discrete magnetostatic energy functional."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from coil_fea import (
    MU0, MU_R0, B_SAT, H0, L_STACK, PPAIR, build, solve_coils,
)


def magnetic_functional(theta, current, pm_on=True, linear_iron=False,
                        nr=80, nt=192):
    """Return field energy, source work and their difference."""
    result = solve_coils(
        theta, current, nr=nr, nt=nt, pm_on=pm_on,
        linear_iron=linear_iron, tol=2e-4, return_fields=True,
    )
    g = build(nr, nt)
    pole = np.tanh(
        np.sin(PPAIR * (g["th2"] - np.deg2rad(theta))) / 0.12
    )
    M_r = np.where(g["is_mag"],
                   (1.2 / MU0 if pm_on else 0.0) * pole, 0.0)
    src_pm = -np.gradient(M_r, g["dth"], axis=1) / g["r_c"][:, None]
    source = result["J"] + src_pm
    B = result["Bmag"]
    density = np.empty_like(B)
    lossless = g["is_air"] | g["is_mag"]
    density[lossless] = B[lossless] ** 2 / (2 * MU0)
    if linear_iron:
        density[g["is_iron"]] = (
            B[g["is_iron"]] ** 2 / (2 * MU0 * MU_R0)
        )
    else:
        bi = B[g["is_iron"]]
        density[g["is_iron"]] = (
            bi ** 2 / (2 * MU0 * MU_R0)
            + H0 * bi ** 7 / (7 * B_SAT ** 6)
        )
    area = np.broadcast_to(g["r2"], B.shape) * g["dr"] * g["dth"]
    field = L_STACK * np.sum(density * area)
    source_work = L_STACK * np.sum(result["Az"] * source * area)
    return float(field), float(source_work), float(field - source_work)


def functional_torque(theta, current, dtheta_deg=1.0, **kwargs):
    """Differentiate the discrete energy functional at fixed currents."""
    plus = magnetic_functional(theta + dtheta_deg, current, **kwargs)[2]
    minus = magnetic_functional(theta - dtheta_deg, current, **kwargs)[2]
    return float((plus - minus) / (2 * np.deg2rad(dtheta_deg)))
