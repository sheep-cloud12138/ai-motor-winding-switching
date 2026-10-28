"""Torque from the discrete magnetostatic energy functional."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_PM_TORQUE_CACHE = {}
TORQUE_MODEL = "functional_v5_pmrotor"


def magnetic_functional(theta, current, pm_on=True, linear_iron=False,
                        nr=80, nt=192, solver=None):
    """Return magnetic energy, source work and magnetic co-energy."""
    if solver is None:
        import coil_fea as solver
    result = solver.solve_coils(
        theta, current, nr=nr, nt=nt, pm_on=pm_on,
        linear_iron=linear_iron, tol=2e-4, return_fields=True,
    )
    g = solver.build(nr, nt)
    src_pm = -np.gradient(
        result["M_r"], g["dth"], axis=1
    ) / g["r_c"][:, None]
    source = result["J"] + src_pm
    B = result["Bmag"]
    density = np.empty_like(B)
    lossless = g["is_air"] | g["in_pm_ring"]
    density[lossless] = B[lossless] ** 2 / (2 * solver.MU0)
    if linear_iron:
        density[g["is_iron"]] = (
            B[g["is_iron"]] ** 2 / (2 * solver.MU0 * solver.MU_R0)
        )
    else:
        bi = B[g["is_iron"]]
        density[g["is_iron"]] = (
            bi ** 2 / (2 * solver.MU0 * solver.MU_R0)
            + solver.H0 * bi ** 7 / (7 * solver.B_SAT ** 6)
        )
    area = np.broadcast_to(g["r2"], B.shape) * g["dr"] * g["dth"]
    field = solver.L_STACK * np.sum(density * area)
    source_work = solver.L_STACK * np.sum(result["Az"] * source * area)
    return float(field), float(source_work), float(source_work - field)


def functional_torque(theta, current, dtheta_deg=7.5, **kwargs):
    """Differentiate the discrete energy functional at fixed currents."""
    if kwargs.get("pm_on", True) and not np.any(current):
        return 0.0
    plus = magnetic_functional(theta + dtheta_deg, current, **kwargs)[2]
    minus = magnetic_functional(theta - dtheta_deg, current, **kwargs)[2]
    torque = float((plus - minus) / (2 * np.deg2rad(dtheta_deg)))
    if not kwargs.get("pm_on", True):
        return torque
    solver = kwargs.get("solver")
    solver_name = "__default__" if solver is None else solver.__name__
    key = (
        solver_name, kwargs.get("nr", 80), kwargs.get("nt", 192),
        float(theta), float(dtheta_deg), kwargs.get("linear_iron", False),
    )
    if key not in _PM_TORQUE_CACHE:
        zero = np.zeros_like(np.asarray(current, dtype=float))
        zp = magnetic_functional(
            theta + dtheta_deg, zero,
            **{k: v for k, v in kwargs.items() if k != "solver"},
            solver=solver,
        )[2]
        zm = magnetic_functional(
            theta - dtheta_deg, zero,
            **{k: v for k, v in kwargs.items() if k != "solver"},
            solver=solver,
        )[2]
        _PM_TORQUE_CACHE[key] = float(
            (zp - zm) / (2 * np.deg2rad(dtheta_deg))
        )
    return torque - _PM_TORQUE_CACHE[key]
