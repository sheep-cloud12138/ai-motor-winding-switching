"""Propagate torque/flux label uncertainty through the drive-cycle comparison."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

import numpy as np  # noqa: E402

import exp.drive_cycle as dc  # noqa: E402


def perturbed(data, torque_scale, flux_scale):
    """Scale only the model channels affected by label uncertainty."""
    out = {}
    for mode, values in data.items():
        d = {k: (v.copy() if isinstance(v, np.ndarray) else v)
             for k, v in values.items()}
        d["T_avg"] *= torque_scale
        d["E_unit"] *= flux_scale
        d["P_fe_unit"] *= flux_scale ** 2
        d["P_fe_eddy"] *= flux_scale ** 2
        out[mode] = d
    return out


def evaluate(ev, tm, omega):
    fixed = {m: dc.simulate(ev, [m] * len(tm), tm, omega)
             for m in dc.MODES}
    complete = {m: r for m, r in fixed.items()
                if r["unmet_Nm_s"] < 1.0}
    best = min(complete, key=lambda m: complete[m]["E_Wh"]) if complete else None
    seq, _ = dc.dp_schedule(ev, tm, omega)
    dp = dc.simulate(ev, seq, tm, omega)
    saving = None
    if best is not None:
        saving = 100.0 * (1.0 - dp["E_Wh"] / complete[best]["E_Wh"])
    return {
        "fixed": {m: {"E_Wh": r["E_Wh"],
                      "unmet_Nm_s": r["unmet_Nm_s"],
                      "efficiency": r["traction_efficiency"]}
                  for m, r in fixed.items()},
        "best_fixed_mode": best,
        "best_fixed_Wh": None if best is None else complete[best]["E_Wh"],
        "dp_Wh": dp["E_Wh"],
        "dp_switches": dp["switches"],
        "dp_unmet_Nm_s": dp["unmet_Nm_s"],
        "dp_saving_vs_best_fixed_pct": saving,
    }


def main():
    model = dc.load_model("mlp")
    data = dc.build_mode_data(model)
    _, wheel, omega = dc.cycle_demands("class1")
    demand = dc.machine_torque(wheel)
    scenarios = {
        "low": (0.90, 0.92),
        "base": (1.00, 1.00),
        "high": (1.10, 1.08),
    }
    out = {"torque_flux_perturbation": {}, "params": {
        "model": "ckpt_mlp_drive.pt", "cycle": "WLTC class 1",
        "torque_scales": [v[0] for v in scenarios.values()],
        "flux_scales": [v[1] for v in scenarios.values()],
    }}
    for name, (ts, fs) in scenarios.items():
        ev = dc.ModeEval(perturbed(data, ts, fs))
        out["torque_flux_perturbation"][name] = {
            "torque_scale": ts, "flux_scale": fs, **evaluate(ev, demand, omega)
        }
        print(name, out["torque_flux_perturbation"][name], flush=True)
    path = ROOT / "exp" / "results_drive_uncertainty.json"
    path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
