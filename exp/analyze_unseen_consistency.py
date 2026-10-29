"""Compare one surrogate with the reference solver and FEMM consistently."""

import json
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def metrics(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    scale = float(np.sum(pred * true) / np.sum(pred ** 2))
    r2 = 1 - np.sum((pred - true) ** 2) / np.sum((true - true.mean()) ** 2)
    return {
        "correlation": float(np.corrcoef(pred, true)[0, 1]),
        "shape_relative": float(np.linalg.norm(scale * pred - true) /
                                np.linalg.norm(true)),
        "r2_raw": float(r2),
        "mae": float(np.mean(np.abs(pred - true))),
    }


def main():
    out = {"model": "per-connection stored MLP", "connections": {}}
    for kind, path in (
        ("4P", "results_4p_femm_withz.json"),
        ("tap50", "results_tap50_femm_withz.json"),
        ("3S1P", "results_3s1p_femm.json"),
        ("tap75", "results_tap75_femm.json"),
    ):
        data = json.loads((ROOT / "exp" / path).read_text())
        if "models" in data:
            pred = np.c_[data["models"]["mlp"]["psi_model"],
                         data["models"]["mlp"]["torque_model"]]
        else:
            pred = np.c_[data["psi_model"], data["torque_model"]]
        psi_ref = np.asarray(data["psi_ref"])
        torque_ref = np.asarray(data["torque_ref"])
        psi_femm = np.asarray(data["psi_femm"])
        torque_femm = np.asarray(data["torque_femm"])
        out["connections"][kind] = {
            "proxy_vs_reference": {
                "flux": metrics(pred[:, :12].ravel(), psi_ref.ravel()),
                "torque": metrics(pred[:, 12], torque_ref),
            },
            "reference_vs_femm": {
                "flux": metrics(psi_ref.ravel(), psi_femm.ravel()),
                "torque": metrics(torque_ref, torque_femm),
            },
            "proxy_vs_femm": {
                "flux": metrics(pred[:, :12].ravel(), psi_femm.ravel()),
                "torque": metrics(pred[:, 12], torque_femm),
            },
        }
    path = ROOT / "exp" / "results_unseen_consistency.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
