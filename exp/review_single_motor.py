import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401
import exp.drive_cycle as dc

model = dc.load_model("mlp")
dc.N_MOTORS = 1
dc.GEAR = 13.0
dc.V_DC = 320.0
dc.V_SV_MAX = dc.V_DC / 3 ** 0.5
for fill, core_mass in ((0.4, 0.5), (0.5, 0.5), (0.5, 0.2)):
    dc.A_COPPER_PER_COIL = fill * dc.SLOT_AREA
    dc.R_COIL = (dc.RHO_CU * dc.N_TURN * 2 * (dc.L_STACK + 0.0217)
                 / dc.A_COPPER_PER_COIL)
    dc.M_FE_PER_COIL = core_mass
    ev = dc.ModeEval(dc.build_mode_data(model))
    v, tw, omega = dc.cycle_demands("class1")
    tm = dc.machine_torque(tw)
    fixed = {m: dc.simulate(ev, [m] * len(tm), tm, omega)
             for m in dc.MODES}
    ok = {m: r for m, r in fixed.items() if r["unmet_Nm_s"] < 1.0}
    best = min(ok, key=lambda m: ok[m]["E_Wh"]) if ok else None
    seq, _ = dc.dp_schedule(ev, tm, omega)
    rd = dc.simulate(ev, seq, tm, omega)
    print("fill", fill, "core", core_mass, "best", best,
          round(fixed[best]["E_Wh"], 1) if best else None,
          round(fixed[best]["traction_efficiency"], 3) if best else None,
          "DP", round(rd["E_Wh"], 1),
          "eff", round(rd["traction_efficiency"], 3),
          "unmet", round(rd["unmet_Nm_s"], 1),
          "save", round((1 - rd["E_Wh"] / fixed[best]["E_Wh"]) * 100, 2)
          if best else None)
