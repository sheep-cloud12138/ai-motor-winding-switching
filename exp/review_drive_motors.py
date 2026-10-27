import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401
import exp.drive_cycle as dc

model = dc.load_model("mlp")
for motors, gear, voltage in (
    (1, 13.0, 320.0),
    (1, 13.0, 400.0),
    (1, 16.0, 400.0),
    (2, 13.0, 400.0),
    (4, 13.0, 400.0),
):
    dc.N_MOTORS = motors
    dc.GEAR = gear
    dc.V_DC = voltage
    dc.V_SV_MAX = voltage / 3 ** 0.5
    ev = dc.ModeEval(dc.build_mode_data(model))
    v, tw, omega = dc.cycle_demands("class1")
    tm = dc.machine_torque(tw)
    fixed = {m: dc.simulate(ev, [m] * len(tm), tm, omega)
             for m in dc.MODES}
    ok = {m: r for m, r in fixed.items() if r["unmet_Nm_s"] < 1.0}
    best = min(ok, key=lambda m: ok[m]["E_Wh"]) if ok else None
    seq, _ = dc.dp_schedule(ev, tm, omega)
    rd = dc.simulate(ev, seq, tm, omega)
    print("motors", motors, "gear", gear, "V", voltage,
          "best", best,
          round(fixed[best]["E_Wh"], 1) if best else None,
          round(fixed[best]["traction_efficiency"], 3) if best else None,
          "DP", round(rd["E_Wh"], 1),
          "unmet", round(rd["unmet_Nm_s"], 1),
          "eff", round(rd["traction_efficiency"], 3),
          "save", round((1 - rd["E_Wh"] / fixed[best]["E_Wh"]) * 100, 2)
          if best else None)
