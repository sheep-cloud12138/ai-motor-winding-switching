"""Map where winding reconfiguration is necessary versus merely optional."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

import exp.drive_cycle as dc  # noqa: E402
from exp.energy_torque import TORQUE_MODEL  # noqa: E402


def evaluate(ev, cycle, v_dc, gear):
    dc.V_DC = float(v_dc)
    dc.V_SV_MAX = dc.V_DC / np.sqrt(3)
    dc.GEAR = float(gear)
    _, wheel_torque, omega = dc.cycle_demands(cycle)
    demand = dc.machine_torque(wheel_torque)
    fixed = {
        mode: dc.simulate(ev, [mode] * len(demand), demand, omega)
        for mode in dc.MODES
    }
    complete = {
        mode: result for mode, result in fixed.items()
        if result["unmet_fraction"] < 1e-6
    }
    best = min(complete, key=lambda mode: complete[mode]["E_Wh"]) \
        if complete else None
    sequence, _ = dc.dp_schedule(ev, demand, omega)
    dp = dc.simulate(ev, sequence, demand, omega)
    saving = None if best is None else 100 * (
        1 - dp["E_Wh"] / complete[best]["E_Wh"]
    )
    if dp["unmet_fraction"] >= 1e-6:
        status = "infeasible"
    elif best is None:
        status = "reconfiguration_required"
    elif saving >= 1.0:
        status = "reconfiguration_saves"
    else:
        status = "fixed_sufficient"
    return {
        "status": status,
        "best_fixed_mode": best,
        "best_fixed_energy_Wh": None if best is None
        else complete[best]["E_Wh"],
        "dp_energy_Wh": dp["E_Wh"],
        "dp_saving_pct": saving,
        "dp_switches": dp["switches"],
        "dp_unmet_fraction": dp["unmet_fraction"],
        "fixed": {
            mode: {
                "energy_Wh": result["E_Wh"],
                "unmet_fraction": result["unmet_fraction"],
                "peak_unmet_Nm": result["peak_unmet_Nm"],
            }
            for mode, result in fixed.items()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", nargs="+",
                    default=["class1", "class2", "class3b"])
    ap.add_argument("--vdc", nargs="+", type=float,
                    default=[180, 220, 260, 320, 400, 500, 700])
    ap.add_argument("--gears", nargs="+", type=float,
                    default=[8, 10, 13, 16, 20])
    args = ap.parse_args()

    model = dc.load_model("mlp")
    ev = dc.ModeEval(dc.build_mode_data(model))
    out = {
        "vehicle": {
            "mass_kg": dc.M_VEH, "motors": dc.N_MOTORS,
            "wheel_radius_m": dc.R_WHEEL, "CdA_m2": dc.CDA,
            "Crr": dc.CRR, "gear_efficiency": dc.ETA_G,
        },
        "grid": {"cycles": args.cycles, "vdc": args.vdc, "gears": args.gears},
        "provenance": {
            "model": "ckpt_mlp_drive.pt",
            "torque_model": TORQUE_MODEL,
            "V_CE0": dc.V_CE0, "R_CE": dc.R_CE, "E_SW_DEV": dc.E_SW_DEV,
        },
        "points": {},
    }
    for cycle in args.cycles:
        for v_dc in args.vdc:
            for gear in args.gears:
                key = f"{cycle}/V{int(v_dc)}/g{gear:g}"
                result = evaluate(ev, cycle, v_dc, gear)
                out["points"][key] = result
                print(
                    f"{key:24s} {result['status']:24s} "
                    f"fixed={result['best_fixed_mode']} "
                    f"DP={result['dp_energy_Wh']:.1f} Wh "
                    f"saving={result['dp_saving_pct']}",
                    flush=True,
                )
                (ROOT / "exp" / "results_drive_regimes.json").write_text(
                    json.dumps(out, indent=1, default=float), encoding="utf-8"
                )


if __name__ == "__main__":
    main()
