"""Compare MLP and co-energy models on the same unseen trajectories."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

import numpy as np  # noqa: E402

import exp.drive_cycle as dc  # noqa: E402


def stats(pred, true):
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    scale = float(np.sum(pred * true) / np.sum(pred ** 2))
    return {
        "correlation": float(np.corrcoef(pred, true)[0, 1]),
        "shape_relative": float(
            np.linalg.norm(scale * pred - true) / np.linalg.norm(true)
        ),
        "r2_raw": float(
            1 - np.sum((pred - true) ** 2) /
            np.sum((true - true.mean()) ** 2)
        ),
        "mae": float(np.mean(np.abs(pred - true))),
    }


def compare(pred, data):
    ref = np.c_[data["psi_ref"], data["torque_ref"]]
    femm = np.c_[data["psi_femm"], data["torque_femm"]]
    return {
        "proxy_vs_reference": {
            "flux": stats(pred[:, :12], ref[:, :12]),
            "torque": stats(pred[:, 12], ref[:, 12]),
        },
        "proxy_vs_femm": {
            "flux": stats(pred[:, :12], femm[:, :12]),
            "torque": stats(pred[:, 12], femm[:, 12]),
        },
    }


def load_model_prediction_pairs():
    pairs = {}
    for kind, path in (
        ("4P", "results_4p_femm_withz.json"),
        ("tap50", "results_tap50_femm_withz.json"),
    ):
        data = json.loads((ROOT / "exp" / path).read_text(encoding="utf-8"))
        pairs[kind] = {
            "data": data,
            "mlp": np.c_[
                data["models"]["mlp"]["psi_model"],
                data["models"]["mlp"]["torque_model"],
            ],
            "energy": np.c_[
                data["models"]["energy"]["psi_model"],
                data["models"]["energy"]["torque_model"],
            ],
        }
    for kind, path in (
        ("3S1P", "results_3s1p_femm.json"),
        ("tap75", "results_tap75_femm.json"),
    ):
        data = json.loads((ROOT / "exp" / path).read_text(encoding="utf-8"))
        theta = np.asarray(data["theta"])
        ic = np.asarray(data["ic"])
        pairs[kind] = {"data": data}
        for name, ckpt in (("mlp", "ckpt_mlp_drive.pt"),
                           ("energy", "ckpt_energy_drive.pt")):
            psi, torque = dc.model_psi_T(
                dc.load_model(name, ckpt), theta, ic
            )
            pairs[kind][name] = np.c_[psi, torque]
    return pairs


def main():
    out = {}
    for kind, pair in load_model_prediction_pairs().items():
        mlp = pair["mlp"]
        energy = pair["energy"]
        out[kind] = {
            "mlp": compare(mlp, pair["data"]),
            "energy": compare(energy, pair["data"]),
            "model_agreement": {
                "flux": stats(mlp[:, :12], energy[:, :12]),
                "torque": stats(mlp[:, 12], energy[:, 12]),
            },
        }
        print(kind, json.dumps(out[kind], indent=1), flush=True)
    path = ROOT / "exp" / "results_cross_model_unseen.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("saved", path)


if __name__ == "__main__":
    main()
