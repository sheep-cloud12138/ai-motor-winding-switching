"""Validate coverage models for an unseen connection against FEMM."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

import numpy as np
import torch

from coil_fea import solve_coils
from exp.data import load_all, make_split
from exp.connections_ext import connection_currents_ext
from exp.energy_torque import functional_torque
from exp.femm_validate import (
    Femm, build_geometry, coil_currents, solve_psi, weighted_torque,
)
from exp.models import make_features
from exp.run_benchmark import Runner, metrics

def trajectory(kind):
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
    # Use the held-out benchmark's exact connection matrix. In particular,
    # its phase-C axis differs from the corrected drive-cycle axis.
    ic = np.asarray([connection_currents_ext(kind, x) for x in ip])
    dphi = np.stack([
        np.zeros_like(theta),
        2 * np.cos(2 * np.deg2rad(theta)),
        -2 * np.sin(2 * np.deg2rad(theta)),
        4 * np.cos(4 * np.deg2rad(theta)),
        -4 * np.sin(4 * np.deg2rad(theta)),
        6 * np.cos(6 * np.deg2rad(theta)),
        -6 * np.sin(6 * np.deg2rad(theta)),
    ], axis=1)
    return theta, gamma, amp, ic, dphi


def solve_reference(theta, ic):
    psi, torque = [], []
    for th, currents in zip(theta, ic):
        r = solve_coils(float(th), currents, nr=40, nt=96, pm_on=True,
                        linear_iron=False, tol=2e-4)
        psi.append(r["psi"])
        torque.append(functional_torque(
            float(th), currents, nr=40, nt=96, pm_on=True,
            linear_iron=False,
        ))
    return np.asarray(psi), np.asarray(torque)


def solve_femm(theta, ic):
    femm = Femm()
    psi, torque = [], []
    for i, (th, currents) in enumerate(zip(theta, ic)):
        build_geometry(femm, float(th), pm_on=True, linear_iron=False)
        coil_currents(femm, currents)
        psi.append(solve_psi(femm, f"tapwithz{i}"))
        torque.append(weighted_torque(femm))
        femm.mi_close()
        if (i + 1) % 6 == 0:
            print("solved", i + 1, "/", len(theta), flush=True)
    return np.asarray(psi), np.asarray(torque)


def compare(pred, psi_femm, torque_femm):
    psi = pred[:, :12]
    torque = pred[:, 12]
    scale_psi = float(np.sum(psi * psi_femm) / np.sum(psi ** 2))
    scale_t = float(np.sum(torque * torque_femm) / np.sum(torque ** 2))
    return {
        "flux_corr": float(np.corrcoef(psi.ravel(), psi_femm.ravel())[0, 1]),
        "flux_shape_reldiff": float(
            np.linalg.norm(scale_psi * psi - psi_femm) /
            np.linalg.norm(psi_femm)
        ),
        "torque_corr": float(np.corrcoef(torque, torque_femm)[0, 1]),
        "torque_shape_reldiff": float(
            np.linalg.norm(scale_t * torque - torque_femm) /
            np.linalg.norm(torque_femm)
        ),
        "torque_mae": float(np.mean(np.abs(torque - torque_femm))),
        "torque_scale_to_femm": scale_t,
        "torque_raw_reldiff": float(
            np.linalg.norm(torque - torque_femm) /
            np.linalg.norm(torque_femm)
        ),
        "psi_model": psi.tolist(),
        "torque_model": torque.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", choices=("4P", "tap50"), default="tap50")
    ap.add_argument("--reuse-femm", action="store_true",
                    help="Reuse the FEMM arrays already stored in results JSON.")
    args = ap.parse_args()

    out_path = ROOT / "exp" / f"results_{args.holdout.lower()}_femm_withz.json"
    theta, gamma, amp, ic, dphi = trajectory(args.holdout)
    old = json.loads(out_path.read_text()) if args.reuse_femm and out_path.exists() else None
    if old is not None:
        psi_femm = np.asarray(old["psi_femm"])
        torque_femm = np.asarray(old["torque_femm"])
    else:
        psi_femm, torque_femm = solve_femm(theta, ic)
    psi_ref, torque_ref = solve_reference(theta, ic)

    X, Y, K = load_all()
    split = make_split(X, Y, K, args.holdout, variant="withz")
    runner = Runner(X, Y, K)
    fit_rows = split["fit"][2]
    mu, sd = Y[fit_rows].mean(0), Y[fit_rows].std(0) + 1e-8
    result = {
        "n_points": len(theta),
        "theta": theta.tolist(),
        "gamma": gamma.tolist(),
        "amp": amp.tolist(),
        "ic": ic.tolist(),
        "psi_femm": psi_femm.tolist(),
        "torque_femm": torque_femm.tolist(),
        "psi_ref": psi_ref.tolist(),
        "torque_ref": torque_ref.tolist(),
        "models": {},
    }
    result["femm_vs_ref"] = compare(
        np.c_[psi_ref, torque_ref], psi_femm, torque_femm,
    )
    for name in ("mlp", "energy"):
        model, predict, fit_s = runner.fit_standardized(split, name)
        Fte, dphite, rows = split["test"]
        bench = metrics(predict(Fte, dphite), Y[rows])
        ckpt = ROOT / "exp" / f"ckpt_{name}_{args.holdout.lower()}_withz.pt"
        torch.save({"state": model.state_dict(),
                    "mu": torch.tensor(mu, dtype=torch.float32),
                    "sd": torch.tensor(sd, dtype=torch.float32)}, ckpt)
        comparison = compare(
            predict(make_features(theta, ic), dphi), psi_femm, torque_femm,
        )
        comparison_vs_ref = compare(
            predict(make_features(theta, ic), dphi), psi_ref, torque_ref,
        )
        comparison["vs_reference"] = comparison_vs_ref
        comparison.update({
            "fit_seconds": fit_s,
            "benchmark_flux_nrmse": bench["flux_nrmse"],
            "benchmark_torque_r2": bench["torque_r2"],
        })
        result["models"][name] = comparison
        brief = {k: v for k, v in comparison.items()
                 if isinstance(v, (int, float))}
        brief["vs_reference"] = {
            k: v for k, v in comparison["vs_reference"].items()
            if isinstance(v, (int, float))
        }
        print(name, json.dumps(brief, indent=1), flush=True)

    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
