"""Independent FEMM check of an unseen drive-cycle connection."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

import numpy as np

from exp.drive_cycle import connection_matrix, load_model, model_psi_T
from exp.femm_unseen_withz import compare, solve_femm, solve_reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="3S1P")
    ap.add_argument("--reuse-femm", action="store_true",
                    help="Reuse FEMM arrays from an existing result JSON.")
    args = ap.parse_args()

    theta_g, gamma_g, amp_g = np.meshgrid(
        np.array([0.0, 45.0, 90.0, 135.0]),
        np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3]),
        np.array([20.0, 40.0, 60.0]),
        indexing="ij",
    )
    theta, gamma, amp = theta_g.ravel(), gamma_g.ravel(), amp_g.ravel()
    s = 2 * np.deg2rad(theta) + gamma
    ip = amp[:, None] * np.stack(
        [np.cos(s), np.cos(s - 2 * np.pi / 3), np.cos(s + 2 * np.pi / 3)],
        axis=1,
    )
    ic = ip @ connection_matrix(args.kind).T
    psi_pred, torque_pred = model_psi_T(load_model("mlp"), theta, ic)
    pred = np.c_[psi_pred, torque_pred]
    out = ROOT / "exp" / f"results_{args.kind.lower()}_femm.json"
    old = json.loads(out.read_text()) if args.reuse_femm and out.exists() else None
    if old is None:
        psi_femm, torque_femm = solve_femm(theta, ic)
    else:
        psi_femm = np.asarray(old["psi_femm"])
        torque_femm = np.asarray(old["torque_femm"])
    psi_ref, torque_ref = solve_reference(theta, ic)
    result = {
        "kind": args.kind,
        "n_points": len(theta),
        "theta": theta.tolist(),
        "gamma": gamma.tolist(),
        "amp": amp.tolist(),
        "ic": ic.tolist(),
        "psi_model": psi_pred.tolist(),
        "torque_model": torque_pred.tolist(),
        "psi_femm": psi_femm.tolist(),
        "torque_femm": torque_femm.tolist(),
        "psi_ref": psi_ref.tolist(),
        "torque_ref": torque_ref.tolist(),
        "vs_femm": compare(pred, psi_femm, torque_femm),
        "vs_reference": compare(pred, psi_ref, torque_ref),
        "femm_vs_reference": compare(
            np.c_[psi_ref, torque_ref], psi_femm, torque_femm,
        ),
    }
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(args.kind, "vs FEMM", {
        k: result["vs_femm"][k] for k in
        ("flux_corr", "flux_shape_reldiff", "torque_corr",
         "torque_shape_reldiff", "torque_mae")
    })


if __name__ == "__main__":
    main()
