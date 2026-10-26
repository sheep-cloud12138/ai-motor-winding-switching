# -*- coding: utf-8 -*-
"""Paper figures. Reads exp/results_*.json + checkpoints + datasets, writes
exp/figs/*.pdf. Style: single-column width 3.5in, two-column 7.1in."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

from exp.data import EXT, STANDARD, load_all, make_split  # noqa: E402
from exp.models import MLP, StructuredEnergy, make_features  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans"})
FIGS = ROOT / "exp" / "figs"
FIGS.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
C_MLP, C_EN, C_GN = "#1f77b4", "#d62728", "#7f7f7f"
LABEL = {"mlp": "Direct MLP", "energy": "Co-energy net (ours)",
         "gradnet": "Monotone vector field"}


def load_json(name):
    p = ROOT / "exp" / name
    return json.loads(p.read_text()) if p.exists() else {}


def predictions_for(holdout):
    """Recompute test-set predictions of the saved MLP/energy checkpoints."""
    X, Y, K = load_all()
    split = make_split(X, Y, K, holdout, variant="base")
    Fte, dphite, rows_te = split["fit"] and split["test"]
    Yte = Y[rows_te]
    preds = {}
    for name, ctor in (("mlp", MLP), ("energy", StructuredEnergy)):
        m = ctor()
        m.load_state_dict(torch.load(ROOT / "exp" / f"ckpt_{name}_{holdout}.pt",
                                     weights_only=True))
        m = m.to(DEVICE).eval()
        x = torch.tensor(Fte, device=DEVICE)
        dp = torch.tensor(dphite, device=DEVICE)
        # standardization must match training; recompute from train rows
        mu = Y[split["fit"][2]].mean(0, keepdims=True)
        sd = Y[split["fit"][2]].std(0, keepdims=True) + 1e-8
        with torch.no_grad() if name == "mlp" else torch.enable_grad():
            if name == "mlp":
                out = m(x).cpu().numpy()
            else:
                psi, T = m.psi_T(x[:, :12], x[:, 12:], dp)
                out = torch.cat([psi.detach(), T.detach()[:, None]], 1).cpu().numpy()
        preds[name] = out * sd + mu
    return Yte, preds


def fig_parity():
    Yte, preds = predictions_for("asym")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3))
    for ax, (name, pred) in zip(axes[:2], preds.items()):
        ax.scatter(Yte[:, :12].ravel(), pred[:, :12].ravel(), s=2, alpha=0.25,
                   color=C_MLP if name == "mlp" else C_EN, lw=0)
        lim = [-0.16, 0.16]
        ax.plot(lim, lim, "k--", lw=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("FEA coil flux linkage (Wb)")
        if name == "mlp":
            ax.set_ylabel("Predicted flux linkage (Wb)")
        err = pred[:, :12] - Yte[:, :12]
        nrmse = np.sqrt((err**2).mean(0)) / np.maximum(Yte[:, :12].std(0), 1e-9)
        ax.set_title(f"{LABEL[name]}\nflux nRMSE {nrmse.mean():.3f}")
    tlim = [-30, 30]
    for name, pred in preds.items():
        axes[2].scatter(Yte[:, 12], pred[:, 12], s=3, alpha=0.4, lw=0,
                        color=C_MLP if name == "mlp" else C_EN,
                        label=LABEL[name])
    axes[2].plot(tlim, tlim, "k--", lw=0.6)
    axes[2].set_xlim(tlim); axes[2].set_ylim(tlim)
    axes[2].set_xlabel("FEA torque (N m)")
    axes[2].set_ylabel("Predicted torque (N m)")
    axes[2].legend(loc="lower right", frameon=False)
    axes[2].set_title("Torque, hold-out connection")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_parity.pdf")
    plt.close(fig)


def fig_scaling():
    sc = load_json("results_scaling.json")
    if not sc:
        return
    pats = [0, 2, 8, 24, 48]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2))
    for holdout, ax in (("asym", axes[0]), ("delta", axes[1])):
        for model, color, mk in (("mlp", C_MLP, "o"), ("energy", C_EN, "s")):
            ys = [sc[f"{holdout}/{p}pat/{model}"]["flux_nrmse"] for p in pats]
            ax.plot(pats, ys, marker=mk, ms=3.5, color=color,
                    label=LABEL[model])
        ax.set_xlabel("Random per-coil current patterns in training")
        ax.set_ylabel(r"Flux nRMSE, hold-out %s" % holdout)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_scaling.pdf")
    plt.close(fig)


def fig_curves():
    cv = load_json("results_curves.json")
    if not cv:
        return
    fracs = [0.05, 0.10, 0.25, 0.50, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2))
    for model, color, mk in (("mlp", C_MLP, "o"), ("energy", C_EN, "s")):
        mu_f, sd_f, mu_t, sd_t = [], [], [], []
        for f in fracs:
            rows = cv[f"{f}/{model}"]
            mu_f.append(np.mean([r["flux_nrmse"] for r in rows]))
            sd_f.append(np.std([r["flux_nrmse"] for r in rows]))
            mu_t.append(np.mean([r["torque_r2"] for r in rows]))
            sd_t.append(np.std([r["torque_r2"] for r in rows]))
        mu_f, sd_f = np.array(mu_f), np.array(sd_f)
        mu_t, sd_t = np.array(mu_t), np.array(sd_t)
        axes[0].errorbar(np.array(fracs) * 100, mu_f, yerr=sd_f, marker=mk,
                         ms=3.5, color=color, capsize=2, label=LABEL[model])
        axes[1].errorbar(np.array(fracs) * 100, mu_t, yerr=sd_t, marker=mk,
                         ms=3.5, color=color, capsize=2, label=LABEL[model])
    for ax, ylab in ((axes[0], "Flux nRMSE (hold-out asym)"),
                     (axes[1], r"Torque $R^2$ (hold-out asym)")):
        ax.set_xlabel("Training-pool fraction (%)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_curves.pdf")
    plt.close(fig)


def fig_jacobian():
    res = load_json("results_benchmark.json")
    jp = res.get("jacobian_probe_asym")
    if not jp:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    names = ["mlp", "energy"]
    vals = [jp[n]["asym_mean"] for n in names]
    bars = ax.bar([LABEL[n] for n in names], vals,
                  color=[C_MLP, C_EN], width=0.5)
    ax.set_yscale("log")
    ax.set_ylabel(r"Jacobian asymmetry $\|J-J^\top\|_F/\|J\|_F$")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.1e}",
                ha="center", fontsize=7)
    ax.grid(alpha=0.3, axis="y", lw=0.4)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_jacobian.pdf")
    plt.close(fig)


def fig_drive():
    dr = load_json("results_drive.json")
    dr2 = load_json("results_drive_r2.json")
    arr = np.load(ROOT / "exp" / "drive_arrays.npz") \
        if (ROOT / "exp" / "drive_arrays.npz").exists() else None
    if not dr or arr is None:
        return
    modes = [str(m) for m in arr["modes"]]
    tmin = np.arange(len(arr["v"])) / 60.0
    fig = plt.figure(figsize=(7.1, 4.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.45,
                          wspace=0.28)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(tmin, arr["v"], color="k", lw=0.7, label="vehicle speed (km/h)")
    ax0.set_ylabel("Speed (km/h)")
    ax0.set_xlabel("Time (min)")
    ax2 = ax0.twinx()
    seq = arr["dp"]
    for j, m in enumerate(modes):
        mask = seq == j
        ax2.fill_between(tmin, 0, 1, where=mask, step="mid", alpha=0.55,
                         label=f"DP: {m}", lw=0)
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", ncols=4, frameon=False, fontsize=6)
    ax0.set_title("WLTC class 1 and DP mode schedule", fontsize=8)

    ax1 = fig.add_subplot(gs[1, 0])
    fixed = dr2["fixed"]
    E = [fixed[m]["E_Wh"] for m in ("4S", "2S2P", "delta", "4P")]
    E.append(dr2["dp"]["E_Wh"])
    labels = []
    for m in ("4S", "2S2P", "delta", "4P"):
        gap = 100 * fixed[m]["unmet_fraction"]
        labels.append(f"{m}\n$\\Delta${gap:.1f}%" if gap else f"{m}\nok")
    labs = labels + ["DP\nok"]
    colors = ["#bbbbbb", "#c8c8c8", "#b8c7d9", "#8fb8de", "#2e7ebc"]
    ax1.bar(labs, E, color=colors, width=0.62)
    ax1.set_ylabel("Cycle energy (Wh)")
    ax1.grid(alpha=0.3, axis="y", lw=0.4)
    for i, e in enumerate(E):
        ax1.text(i, e + 8, f"{e:.1f}", ha="center", fontsize=6)
    ax1.set_ylim(0, 525)

    ax2 = fig.add_subplot(gs[1, 1])
    strat = ["fixed4P", "greedy", "dp"]
    labs = ["Fixed 4P", "Greedy", "DP (ours)"]
    keys = ["mech", "cu", "fe", "inv", "sw"]
    nice = {"mech": "Mechanical", "cu": "Copper", "fe": "Iron",
            "inv": "Inverter", "sw": "Changeover"}
    bottom = np.zeros(len(strat))
    for k in keys:
        vals = np.array([dr[s]["breakdown_Wh"][k] for s in strat])
        ax2.bar(labs, vals, bottom=bottom, width=0.6, label=nice[k])
        bottom += vals
    ax2.set_ylabel("Energy over cycle (Wh)")
    ax2.legend(frameon=False, fontsize=6, ncols=2)
    ax2.tick_params(axis="x", rotation=12)

    ax3 = fig.add_subplot(gs[2, :])
    sens = dr2.get("e_sw_sweep", {})
    if sens:
        xs = [float(k[:-1]) for k in sens]
        ax3.plot(xs, [sens[k]["dp_Wh"] for k in sens], "o-", ms=3.5,
                 color="#2e7ebc", label="DP")
        ax3.plot(xs, [sens[k]["greedy_Wh"] for k in sens], "s--", ms=3.5,
                 color="#f0b27a", label="Greedy")
        ax3.axvline(dr2.get("winding_stored_energy_J_est", 9.0),
                    color="0.35", lw=0.8, ls=":", label="Winding stored energy")
        ax3.set_xscale("log")
        ax3.set_xlabel("Changeover energy per event (J)")
        ax3.set_ylabel("Cycle energy (Wh)")
        ax3.legend(frameon=False)
        ax3.grid(alpha=0.3, lw=0.4)
    fig.savefig(FIGS / "fig_drive.pdf")
    plt.close(fig)


def fig_mesh():
    f = ROOT / "exp" / "coil_fine_check.npz"
    if not f.exists():
        return
    d = np.load(f)
    full = np.load(ROOT / "coil_dataset_full.npz", allow_pickle=True)
    ext = np.load(ROOT / "coil_dataset_ext.npz", allow_pickle=True)
    r48 = np.load(ROOT / "coil_dataset_random48.npz", allow_pickle=True)
    pools = {"full": full, "ext": ext, "random48": r48}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.1))
    psi_c, psi_f, T_c, T_f = [], [], [], []
    for s, row, psi, T in zip(d["src"], d["row"], d["psi"], d["T"]):
        psi_c.append(pools[str(s)]["Y"][int(row), :12])
        T_c.append(pools[str(s)]["Y"][int(row), 12])
        psi_f.append(psi)
        T_f.append(T)
    psi_c, psi_f = np.array(psi_c), np.array(psi_f)
    T_c, T_f = np.array(T_c), np.array(T_f)
    axes[0].scatter(psi_c.ravel(), psi_f.ravel(), s=3, alpha=0.4, lw=0)
    lim = [-0.16, 0.16]
    axes[0].plot(lim, lim, "k--", lw=0.6)
    axes[0].set_xlim(lim); axes[0].set_ylim(lim)
    axes[0].set_xlabel("Coarse mesh 40x96 (Wb)")
    axes[0].set_ylabel("Fine mesh 80x192 (Wb)")
    rel = np.abs(T_f - T_c) / np.maximum(np.abs(T_c), 1e-6)
    axes[1].hist(np.clip(rel, 0, 0.15), bins=24, color="#8fb8de")
    axes[1].set_xlabel("Relative torque difference")
    axes[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_mesh.pdf")
    plt.close(fig)


def fig_benchmark_bar():
    res = load_json("results_benchmark.json")
    if not res:
        return
    holds = ["4S", "2S2P", "delta", "asym", "4P", "tap50", "openB"]
    x = np.arange(len(holds))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.3))
    for i, (model, color, mk) in enumerate((("mlp", C_MLP, "o"),
                                            ("energy", C_EN, "s"))):
        f = [res[f"{h}/base"][model]["flux_nrmse"] for h in holds
             if f"{h}/base" in res and model in res[f"{h}/base"]]
        t = [res[f"{h}/base"][model]["torque_r2"] for h in holds
             if f"{h}/base" in res and model in res[f"{h}/base"]]
        if len(f) == len(holds):
            axes[0].plot(x, f, marker=mk, ms=4.5, color=color, lw=1.2,
                         label=LABEL[model])
            axes[1].plot(x, np.clip(t, -1, 1), marker=mk, ms=4.5, color=color,
                         lw=1.2)
    axes[0].set_xticks(x, holds)
    axes[0].set_ylabel("Flux nRMSE")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    axes[1].set_xticks(x, holds)
    axes[1].set_ylabel(r"Torque $R^2$ (clipped at $-1$)")
    axes[1].axhline(0.95, color="k", lw=0.5, ls="--")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y", lw=0.4)
        ax.set_xlabel("Hold-out connection (zero-shot)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_benchmark.pdf")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    if "all" in which or "parity" in which:
        fig_parity()
    if "all" in which or "scaling" in which:
        fig_scaling()
    if "all" in which or "curves" in which:
        fig_curves()
    if "all" in which or "jacobian" in which:
        fig_jacobian()
    if "all" in which or "drive" in which:
        fig_drive()
    if "all" in which or "mesh" in which:
        fig_mesh()
    if "all" in which or "benchmark" in which:
        fig_benchmark_bar()
    print("figures written to", FIGS)
