"""Compare one surrogate with the reference solver and FEMM consistently."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from exp.models import MLP, make_features


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
    ck = torch.load(ROOT / "exp" / "ckpt_mlp_tap50_withz.pt",
                    map_location="cpu", weights_only=True)
    model = MLP()
    model.load_state_dict(ck["state"])
    model.eval()
    mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
    out = {"model": "single withz MLP", "connections": {}}
    for kind, path in (
        ("4P", "results_4p_femm_withz.json"),
        ("tap50", "results_tap50_femm_withz.json"),
        ("3S1P", "results_3s1p_femm.json"),
    ):
        data = json.loads((ROOT / "exp" / path).read_text())
        theta = np.asarray(data["theta"])
        ic = np.asarray(data["ic"])
        with torch.no_grad():
            raw = model(torch.tensor(make_features(theta, ic),
                                     dtype=torch.float32)).numpy()
        pred = raw * sd + mu
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
