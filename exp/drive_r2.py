# -*- coding: utf-8 -*-
"""Reviewer-response experiments for the drive-cycle study.

R1  Fixed-connection baselines for ALL modes -> best fixed connection.
R2  Changeover-energy sweep over a physically meaningful range (5 mJ .. 100 J),
    plus a winding stored-energy estimate that bounds realistic E_sw.
R3  Zero-shot mode in the DP: add tap50 (never in the drive training pool)
    to the switchable set.
R4  Connection-design screening: cycle energy for 8 candidate connections
    evaluated zero-shot by the same surrogate (design-space exploration).
R5  Inverter rescaling sensitivity: current-class-matched device parameters
    for high-current modes (r_CE halved, V_CE0 +10%), best-fixed vs DP.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

import exp.drive_cycle as dc  # noqa: E402

OUT = ROOT / "exp" / "results_drive_r2.json"


def main():
    msd = dc.load_model("mlp")
    data = dc.build_mode_data(msd)
    ev = dc.ModeEval(data)
    v, T_wheel, omega_m = dc.cycle_demands("class1")
    Tm = dc.machine_torque(T_wheel)
    N = len(v)
    dist = float(np.trapezoid(v / 3.6, dx=1.0) / 1000.0)
    out = {"distance_km": dist}

    # ---------- R1: fixed baselines for every mode ----------
    print("R1 fixed baselines", flush=True)
    fixed = {}
    for m in dc.MODES:
        r = dc.simulate(ev, [m] * N, Tm, omega_m)
        fixed[m] = r
        print(" ", m, "E=%.1f unmet=%.1f" % (r["E_Wh"], r["unmet_Nm_s"]), flush=True)
    complete = {m: r for m, r in fixed.items() if r["unmet_Nm_s"] < 1.0}
    best_fixed = min(complete, key=lambda m: complete[m]["E_Wh"])
    out["fixed"] = {m: {
        "E_Wh": r["E_Wh"], "unmet": r["unmet_Nm_s"],
        "unmet_fraction": r["unmet_fraction"],
        "peak_unmet_Nm": r["peak_unmet_Nm"],
        "efficiency": r["traction_efficiency"],
        "switches": r["switches"],
    } for m, r in fixed.items()}
    out["best_fixed"] = {"mode": best_fixed,
                         "E_Wh": complete[best_fixed]["E_Wh"]}
    print(" best fixed:", out["best_fixed"], flush=True)

    # DP reference (rerun to have the sequence locally)
    seq_dp, J = dc.dp_schedule(ev, Tm, omega_m)
    r_dp = dc.simulate(ev, seq_dp, Tm, omega_m)
    out["dp"] = {"E_Wh": r_dp["E_Wh"], "switches": r_dp["switches"],
                 "unmet": r_dp["unmet_Nm_s"],
                 "unmet_fraction": r_dp["unmet_fraction"],
                 "peak_unmet_Nm": r_dp["peak_unmet_Nm"],
                 "efficiency": r_dp["traction_efficiency"]}
    saving_best = 100 * (1 - r_dp["E_Wh"] / complete[best_fixed]["E_Wh"])
    out["dp_saving_vs_best_fixed_pct"] = saving_best
    print(" DP vs best fixed saving %.1f%%" % saving_best, flush=True)

    # ---------- R2: wide-range changeover energy sweep ----------
    print("R2 E_sw sweep", flush=True)
    sweep = {}
    for e in (5e-3, 25e-3, 0.1, 1.0, 10.0, 100.0):
        rd = dc.simulate(ev, seq_dp, Tm, omega_m, e_sw=e)
        rg = dc.simulate(ev, None if False else [m for m in
                          np.array(seq_dp)], Tm, omega_m, e_sw=e)  # same seq
        # greedy seq needs recomputation with the same e to be fair:
        greedy = []
        for t in range(N):
            cands = [(ev.best(m, Tm[t], omega_m[t]), m) for m in dc.MODES]
            cands = [(c, m) for c, m in cands if c is not None]
            greedy.append(min(cands, key=lambda cm: cm[0]["P"])[1] if cands else "4P")
        rg = dc.simulate(ev, greedy, Tm, omega_m, e_sw=e)
        sweep[f"{e:.3g}J"] = {"dp_Wh": rd["E_Wh"], "dp_sw": rd["switches"],
                              "greedy_Wh": rg["E_Wh"], "greedy_sw": rg["switches"]}
        print("  E_sw=%s: DP %.1f Wh / %d sw ; greedy %.1f Wh / %d sw" %
              (e, rd["E_Wh"], rd["switches"], rg["E_Wh"], rg["switches"]), flush=True)
    out["e_sw_sweep"] = sweep
    # winding stored energy at typical commutation currents (coenergy bound):
    # psi ~ 0.05 Wb per coil at 30 A coil current -> W ~ 0.5*psi*I*12 coils
    W_est = 0.5 * 0.05 * 30 * 12
    out["winding_stored_energy_J_est"] = W_est

    # ---------- R3: zero-shot tap50 in the DP ----------
    print("R3 zero-shot tap50 in DP", flush=True)
    dc.MODES = ("4S", "2S2P", "delta", "4P", "tap50")
    data5 = dc.build_mode_data(msd)
    ev5 = dc.ModeEval(data5)
    seq5, _ = dc.dp_schedule(ev5, Tm, omega_m)
    r5 = dc.simulate(ev5, seq5, Tm, omega_m)
    out["dp5"] = {"E_Wh": r5["E_Wh"], "switches": r5["switches"],
                  "unmet": r5["unmet_Nm_s"],
                  "mode_counts": {m: int(sum(1 for x in seq5 if x == m))
                                  for m in dc.MODES}}
    print(" DP5:", out["dp5"], flush=True)
    dc.MODES = ("4S", "2S2P", "delta", "4P")

    # ---------- R4: design screening over candidate connections ----------
    print("R4 design screening", flush=True)
    cands = ["4S", "2S2P", "delta", "4P", "3S1P", "tap75", "tap50", "tap25"]
    screening = {}
    for m in cands:
        dc.MODES = (m,)
        d1 = dc.build_mode_data(msd)
        e1 = dc.ModeEval(d1)
        r1 = dc.simulate(e1, [m] * N, Tm, omega_m)
        screening[m] = {"E_Wh": r1["E_Wh"], "unmet": r1["unmet_Nm_s"]}
        print("  %-6s E=%.1f Wh unmet=%.1f" % (m, r1["E_Wh"], r1["unmet_Nm_s"]),
              flush=True)
    dc.MODES = ("4S", "2S2P", "delta", "4P")
    out["screening"] = screening

    # ---------- R5: inverter rescaling sensitivity ----------
    print("R5 inverter rescaling", flush=True)
    # current-class-matched devices: halve r_CE, +10% V_CE0 for all modes
    old = dc.R_CE, dc.V_CE0
    dc.R_CE, dc.V_CE0 = 0.5 * old[0], 1.1 * old[1]
    dataR = dc.build_mode_data(msd)
    evR = dc.ModeEval(dataR)
    seqR, _ = dc.dp_schedule(evR, Tm, omega_m)
    rR_dp = dc.simulate(evR, seqR, Tm, omega_m)
    fixedR = {}
    for m in dc.MODES:
        fixedR[m] = dc.simulate(evR, [m] * N, Tm, omega_m)
    okR = {m: r for m, r in fixedR.items() if r["unmet_Nm_s"] < 1.0}
    bestR = min(okR, key=lambda m: okR[m]["E_Wh"])
    out["inverter_rescale"] = {
        "device": "r_CE x0.5, V_CE0 +10%",
        "dp_Wh": rR_dp["E_Wh"], "dp_sw": rR_dp["switches"],
        "best_fixed_mode": bestR, "best_fixed_Wh": okR[bestR]["E_Wh"],
        "saving_pct": 100 * (1 - rR_dp["E_Wh"] / okR[bestR]["E_Wh"])}
    print(" ", out["inverter_rescale"], flush=True)
    dc.R_CE, dc.V_CE0 = old

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("saved", OUT)


if __name__ == "__main__":
    main()
