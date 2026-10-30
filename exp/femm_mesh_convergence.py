"""Three-level FEMM mesh check for representative connection/current cases."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

import numpy as np  # noqa: E402

from coil_fea import solve_coils  # noqa: E402
from exp.connections_ext import connection_currents_ext  # noqa: E402
from exp.energy_torque import functional_torque  # noqa: E402
from exp.femm_validate import (  # noqa: E402
    Femm, build_geometry, coil_currents, solve_psi, weighted_torque,
)
from exp.drive_cycle import connection_matrix  # noqa: E402
from gen_coil_dataset import connection_currents  # noqa: E402


def current_vector(kind, theta, amp=40.0, gamma=0.0):
    s = 2.0 * np.deg2rad(theta) + gamma
    ip = amp * np.array([
        np.cos(s), np.cos(s - 2 * np.pi / 3),
        np.cos(s + 2 * np.pi / 3),
    ])
    if kind in ("4S", "2S2P", "4P"):
        return connection_currents(kind, ip)
    if kind == "3S1P":
        return ip @ connection_matrix(kind).T
    return connection_currents_ext(kind, ip)


def main():
    cases = [
        ("4S", 30.0),
        ("4P", 45.0),
        ("tap50", 45.0),
        ("3S1P", 45.0),
    ]
    levels = [
        {"mesh_size_m": 1.5e-3, "arc_seg_deg": 10.0, "min_angle": 30.0},
        {"mesh_size_m": 1.0e-3, "arc_seg_deg": 6.0, "min_angle": 20.0},
        {"mesh_size_m": 0.7e-3, "arc_seg_deg": 3.0, "min_angle": 15.0},
    ]
    f = Femm()
    out = {"levels": levels, "cases": {}}
    for kind, theta in cases:
        ic = current_vector(kind, theta)
        ref = solve_coils(theta, ic, nr=40, nt=96, tol=2e-4)["psi"]
        ref_t = functional_torque(theta, ic, nr=40, nt=96)
        rows = []
        for level in levels:
            mesh_size = level["mesh_size_m"]
            t0 = time.time()
            build_geometry(
                f, theta, pm_on=True, linear_iron=False,
                mesh_size=mesh_size,
                arc_seg_deg=level["arc_seg_deg"],
                min_angle=level["min_angle"],
            )
            coil_currents(f, ic)
            psi = solve_psi(f, f"mesh_{kind}_{mesh_size}")
            torque = weighted_torque(f)
            f.mi_close()
            rows.append({
                **level,
                "psi": psi.tolist(),
                "torque": float(torque),
                "psi_relnorm_vs_ref": float(
                    np.linalg.norm(psi - ref) / np.linalg.norm(ref)
                ),
                "torque_rel_vs_ref": float(
                    abs(torque - ref_t) / max(abs(ref_t), 1e-12)
                ),
                "seconds": time.time() - t0,
            })
            print(kind, mesh_size, rows[-1]["psi_relnorm_vs_ref"],
                  rows[-1]["torque_rel_vs_ref"], flush=True)
        out["cases"][kind] = {
            "theta_deg": theta,
            "currents": ic.tolist(),
            "reference_torque": float(ref_t),
            "levels": rows,
            "finest_pair_psi_change": float(
                np.linalg.norm(
                    np.asarray(rows[-1]["psi"]) - np.asarray(rows[-2]["psi"])
                ) / np.linalg.norm(rows[-1]["psi"])
            ),
            "finest_pair_torque_change": float(
                abs(rows[-1]["torque"] - rows[-2]["torque"]) /
                max(abs(rows[-1]["torque"]), 1e-12)
            ),
        }
    path = ROOT / "exp" / "results_femm_mesh_convergence.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
