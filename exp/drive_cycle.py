# -*- coding: utf-8 -*-
"""Drive-cycle case study: surrogate-in-the-loop loss maps + DP mode scheduling.

Machine : 12-slot/4-pole SPM from the FEA study (parameters mirror coil_fea.py).
Cycle   : WLTC class 1 (official JRC trace, checksum-verified).
Vehicle : light L7E-class EV (declared parameters below).

Per operating point (mode m, torque demand T*, speed omega):
  copper      : sum_c i_c^2 R_c                     (exact, from coil currents)
  iron        : Bertotti proxy on per-coil AC flux amplitude (declared coefficients)
  inverter    : IGBT conduction + switching (declared device parameters)
  changeover  : E_sw per reconfiguration event (swept in sensitivity)
Strategies: fixed-4P (only single mode spanning the whole cycle), 2-mode rule
(speed threshold, threshold swept), pointwise greedy, DP (min energy with
switching cost and minimum dwell time).
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401  (WMI hang workaround, before numpy/scipy)

import numpy as np
import torch

from coil_fea import L_STACK, N_TURN  # noqa: E402
from exp.connections_ext import connection_currents_ext  # noqa: E402
from exp.models import MLP, StructuredEnergy, make_features  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- machine / powertrain parameters (declared case study) -----
RHO_CU = 1.68e-8
SLOT_AREA = (0.0475 - 0.0355) * (2 * np.pi / 12 * 0.52) * (0.0355 + 0.0475) / 2
A_COPPER_PER_COIL = 0.40 * SLOT_AREA             # 40% slot fill
R_COIL = RHO_CU * N_TURN * 2 * (L_STACK + 0.0217) / A_COPPER_PER_COIL
I_COIL_MAX = 60.0                                # A peak coil current (FEA range)
V_DC = 320.0
V_SV_MAX = V_DC / np.sqrt(3)                     # SVPWM linear region
# Bertotti proxy, M250-35A-class steel
K_H, K_C = 0.026, 1.2e-4                         # W/(kg Hz T^2), W/(kg Hz^2 T^2)
M_FE_PER_COIL = 0.5                              # kg equivalent core mass/coil
A_TOOTH_FE = 0.012 * L_STACK                     # m^2 tooth cross-section
# Inverter (declared device parameters)
V_CE0, R_CE = 0.85, 0.009
E_SW_DEV = 2.7e-3                                # J per device period @ refs
F_SW = 10e3
E_CHANGEOVER = 25e-3                             # J per winding reconfiguration
# Vehicle (light EV with four identical wheel motors)
M_VEH, CDA, CRR, R_WHEEL, GEAR, ETA_G = 400.0, 0.50, 0.008, 0.25, 13.0, 0.96
N_MOTORS = 4
P_AUX = 250.0
RHO_AIR = 1.20
MODES = ("4S", "2S2P", "delta", "4P")

I_GRID = np.linspace(1.0, 260.0, 130)
G_GRID = np.linspace(0.0, 2 * np.pi, 145)
TH_T = np.arange(0, 180, 15).astype(float)       # torque averaging
TH_V = np.linspace(0.0, 180.0, 73)[:-1]          # voltage FD, 2.5 deg steps


# ---------------- surrogate -------------------------------------------------


def load_model(name="mlp", ckpt="ckpt_mlp_drive.pt"):
    """Returns (model, mu, sd); model outputs are de-standardised inside."""
    ck = torch.load(ROOT / "exp" / ckpt, weights_only=True)
    m = MLP() if name == "mlp" else StructuredEnergy()
    if isinstance(ck, dict) and "state" in ck:
        m.load_state_dict(ck["state"])
        mu, sd = ck["mu"].numpy(), ck["sd"].numpy()
    else:
        m.load_state_dict(ck)
        mu, sd = np.zeros(13), np.ones(13)
    return m.to(DEVICE).eval(), mu, sd


@torch.no_grad()
def model_psi_T(msd, theta_deg, ic):
    """theta (n,), ic (n,12) -> psi (n,12), T (n,). msd = (model, mu, sd)."""
    model, mu, sd = msd
    theta_deg = np.asarray(theta_deg, float)
    t = np.deg2rad(theta_deg)
    dphi = np.stack([np.zeros_like(t), 2 * np.cos(2 * t), -2 * np.sin(2 * t),
                     4 * np.cos(4 * t), -4 * np.sin(4 * t),
                     6 * np.cos(6 * t), -6 * np.sin(6 * t)], axis=1)
    F = torch.tensor(make_features(theta_deg, ic), device=DEVICE)
    dp = torch.tensor(dphi, dtype=torch.float32, device=DEVICE)
    if isinstance(model, StructuredEnergy):
        with torch.enable_grad():
            psi, T = model.psi_T(F[:, :12], F[:, 12:], dp)
        out = torch.cat([psi.detach(), T.detach()[:, None]], 1).cpu().numpy()
    else:
        out = model(F).cpu().numpy()
    return out[:, :12] * sd[:12] + mu[:12], out[:, 12] * sd[12] + mu[12]


def connection_matrix(kind):
    """C in R^{12x3} with i_c = C i_p, for synchronous drive-cycle operation.

    Port axes follow the standard 0/120/240-deg electrical layout. The demo
    machine's original phase-C sign pattern placed its axis at 60 deg (a
    nonstandard grouping); since the surrogate is connection-invariant we
    simply query it with corrected signs -- no retraining involved.
    """
    signs = {"A": (1, -1, 1, -1), "B": (1, -1, 1, -1), "C": (-1, 1, -1, 1)}
    slots = {"A": (0, 3, 6, 9), "B": (2, 5, 8, 11), "C": (1, 4, 7, 10)}
    base = np.zeros((12, 3))
    for p, ph in enumerate("ABC"):
        for s, sg in zip(slots[ph], signs[ph]):
            base[s, p] = sg
    if kind == "4S":
        return base
    if kind == "2S2P":
        return 0.5 * base
    if kind == "4P":
        return 0.25 * base
    if kind == "delta":
        # winding (branch) currents from line currents, standard delta:
        T = np.array([[1, -1, 0], [0, 1, -1], [-1, 0, 1]]) / 3.0
        return base @ T
    if kind in ("tap50", "tap25", "tap75"):
        # series tap: only the first n of 4 coils per phase carry current
        keep = {"tap50": 2, "tap25": 1, "tap75": 3}[kind]
        C = base.copy()
        for p, ph in enumerate("ABC"):
            for s in slots[ph][keep:]:
                C[s, p] = 0.0
        return C
    if kind == "3S1P":
        # 3-series branch (R) vs 1-coil branch (3R): currents ip/4 vs 3ip/4
        C = base.copy()
        for p, ph in enumerate("ABC"):
            sl = slots[ph]
            for s in sl[:3]:
                C[s, p] *= 0.25
            C[sl[3], p] *= 0.75
        return C
    raise ValueError(kind)


def _coil_currents(kind, I, b):
    ip = I * np.array([np.cos(b), np.cos(b - 2 * np.pi / 3), np.cos(b + 2 * np.pi / 3)])
    return connection_currents_ext(kind, ip)


def build_mode_data(model, iron_scale=1.0):
    """Grid data per mode over (I, gamma): SYNCHRONOUS average torque, losses,
    peak port voltage. Phase currents rotate with the rotor:
    ip(theta) = I [cos(2t+g), cos(2t+g-2pi/3), cos(2t+g+2pi/3)] with t the
    mechanical angle in rad; 2t spans one electrical period over the 180-deg
    mechanical theta grid. Port voltage is split as V = omega*E_unit + I*R_v.
    """
    II, GG = np.meshgrid(I_GRID, G_GRID, indexing="ij")
    II, GG = II.ravel(), GG.ravel()
    n = len(II)
    th_T_rad = np.deg2rad(TH_T)[None, :]                  # (1, TH)
    th_V_rad = np.deg2rad(TH_V)[None, :]                  # (1, nV)
    out = {}
    for m in MODES:
        C = connection_matrix(m)
        R_port = np.diag(C.T @ (np.eye(12) * R_COIL) @ C)

        def ip_of(s, I):
            # s: electrical angle array (n, L); I: (n,1) -> (n, L, 3)
            return I[..., None] * np.stack(
                [np.cos(s), np.cos(s - 2 * np.pi / 3),
                 np.cos(s + 2 * np.pi / 3)], axis=-1)

        s_T = 2.0 * th_T_rad + GG[:, None]                # (n, TH)
        ip_T = ip_of(s_T, II[:, None])                    # (n, TH, 3)
        ic_T = np.einsum("jk,ntk->ntj", C, ip_T)          # (n, TH, 12)
        theta_T = np.broadcast_to(th_T_rad, (n, len(TH_T))) * 180.0 / np.pi
        psi_T, T_e = model_psi_T(model, theta_T.ravel(), ic_T.reshape(-1, 12))
        T_avg = T_e.reshape(n, len(TH_T)).mean(1)
        P_cu = (ic_T ** 2).sum(2).mean(1) * R_COIL

        s_V = 2.0 * th_V_rad + GG[:, None]                # (n, nV)
        ip_V = ip_of(s_V, II[:, None])                    # (n, nV, 3)
        ic_V = np.einsum("jk,ntk->ntj", C, ip_V)
        theta_V = np.broadcast_to(th_V_rad, (n, len(TH_V))) * 180.0 / np.pi
        psi_V, _ = model_psi_T(model, theta_V.ravel(), ic_V.reshape(-1, 12))
        psi_V = psi_V.reshape(n, len(TH_V), 12)
        psi_p = psi_V @ C                                 # port flux (n, nV, 3)
        # Fundamental fit psi ~ a + b cos(2t) + c sin(2t) per port/coil:
        # robust to interpolation ripple, unlike numerical differentiation.
        t2 = 2.0 * th_V_rad.ravel()                        # (nV,)
        A = np.stack([np.ones_like(t2), np.cos(t2), np.sin(t2)], 1)  # (nV,3)
        coef, *_ = np.linalg.lstsq(
            A, psi_p.transpose(0, 2, 1).reshape(n * 3, len(TH_V)).T,
            rcond=None)                                        # (3, n*3)
        b, c = coef[1].reshape(n, 3), coef[2].reshape(n, 3)
        # For balanced sinusoidal steady state the port-voltage space vector
        # is v_sv = omega_e * psi_hat_phase; take the worst port (conservative
        # under unbalance). E_unit multiplies omega_e = 2*omega_m in points().
        E_unit = np.sqrt(b ** 2 + c ** 2).max(1)
        # resistive drop: constant space vector amplitude per amp
        R_v = np.full(n, float(R_port.max()))

        coef_c, *_ = np.linalg.lstsq(
            A, psi_V.transpose(0, 2, 1).reshape(n * 12, len(TH_V)).T,
            rcond=None)
        amp = np.sqrt(coef_c[1] ** 2 + coef_c[2] ** 2).reshape(n, 12)
        B_hat = amp / (N_TURN * A_TOOTH_FE)
        P_fe_unit = M_FE_PER_COIL * K_H * iron_scale * (B_hat ** 2).sum(1)
        P_fe_eddy = M_FE_PER_COIL * K_C * iron_scale * (B_hat ** 2).sum(1)

        P_inv = 6 * (V_CE0 * II / np.pi + R_CE * II ** 2 / 4) \
            + 3 * F_SW * E_SW_DEV * np.clip(II / 100.0, 0.04, None)
        ic_peak = np.abs(ic_V).max(2).max(1)
        cap_ok = ic_peak <= I_COIL_MAX

        out[m] = dict(I=II, gamma=GG, T_avg=T_avg, P_cu=P_cu, P_inv=P_inv,
                      E_unit=E_unit, R_v=R_v, P_fe_unit=P_fe_unit,
                      P_fe_eddy=P_fe_eddy, cap_ok=cap_ok, C=C)
    return out


class ModeEval:
    """Per-mode operating-point evaluation with 1-D inversion along I."""

    def __init__(self, data):
        self.data = data
        self._cols = {m: self._precolumns(m) for m in MODES}

    def _precolumns(self, m):
        d = self.data[m]
        cols = []
        for g in np.unique(d["gamma"]):
            sel = (d["gamma"] == g) & d["cap_ok"]
            I = d["I"][sel]
            order = np.argsort(I)
            I = I[order]
            vals = {k: d[k][sel][order] for k in
                    ("T_avg", "P_cu", "P_inv", "E_unit", "R_v",
                     "P_fe_unit", "P_fe_eddy")}
            cols.append((g, I, vals))
        return cols

    def points(self, m, T_star, omega_m):
        """Feasible (voltage-OK) operating points meeting T_avg = T_star."""
        f_e = 2.0 * omega_m / (2 * np.pi)
        out = []
        for g, I, v in self._cols[m]:
            Tb = v["T_avg"]
            lo, hi = Tb[0], Tb[-1]
            if not (min(lo, hi) - 1e-9 <= T_star <= max(lo, hi) + 1e-9):
                continue
            if abs(hi - lo) < 1e-12:
                continue
            Tl = Tb if lo <= hi else Tb[::-1]
            Il = I if lo <= hi else I[::-1]
            I_star = float(np.interp(T_star, Tl, Il))

            def lerp(k):
                col = v[k] if lo <= hi else v[k][::-1]
                return float(np.interp(T_star, Tl, col))

            P = (lerp("P_cu") + lerp("P_inv")
                 + lerp("P_fe_unit") * f_e + lerp("P_fe_eddy") * f_e ** 2)
            V = lerp("E_unit") * 2.0 * omega_m + I_star * lerp("R_v")
            if V <= V_SV_MAX:
                out.append({"P": P, "V": V, "I": I_star, "gamma": float(g),
                            "P_cu": lerp("P_cu"), "P_inv": lerp("P_inv"),
                            "P_fe": lerp("P_fe_unit") * f_e
                                    + lerp("P_fe_eddy") * f_e ** 2})
        return out

    def best(self, m, T_star, omega_m):
        pts = self.points(m, T_star, omega_m)
        return min(pts, key=lambda p: p["P"]) if pts else None

    def T_max(self, m, omega_m):
        d = self.data[m]
        f_e = 2.0 * omega_m / (2 * np.pi)
        V = d["E_unit"] * 2.0 * omega_m + d["I"] * d["R_v"]
        ok = d["cap_ok"] & (V <= V_SV_MAX)
        if not ok.any():
            return 0.0
        P_fe = d["P_fe_unit"] * f_e + d["P_fe_eddy"] * f_e ** 2
        # max torque subject to voltage: pick the feasible point with max |T|
        T = np.where(ok, np.abs(d["T_avg"]), 0.0)
        return float(T.max())


# ---------------- vehicle / cycle -------------------------------------------


def read_V_file(path):
    txt = Path(path).read_text()
    return np.array([float(m) for m in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", txt)])


def cycle_demands(cycle="class1"):
    v = read_V_file(ROOT / "exp" / f"V_{cycle}.txt")
    v = v[:1611] if cycle == "class1" else v[:1800]
    a = np.gradient(v) / 3.6
    F = M_VEH * a + CRR * M_VEH * 9.81 + 0.5 * RHO_AIR * CDA * (v / 3.6) ** 2
    return v, F * R_WHEEL, v / 3.6 / R_WHEEL * GEAR


def machine_torque(T_wheel):
    return np.where(T_wheel >= 0, T_wheel / (GEAR * ETA_G * N_MOTORS),
                    T_wheel * ETA_G / (GEAR * N_MOTORS))


def simulate(ev, mode_seq, Tm_dem, omega_m, e_sw=E_CHANGEOVER):
    """Battery energy for one mode command applied to all wheel motors."""
    machine_energy = 0.0
    aux_energy = 0.0
    sw = 0
    br = {"cu": 0.0, "fe": 0.0, "inv": 0.0, "mech": 0.0, "sw": 0.0}
    unmet = 0.0
    prev = None
    for t in range(len(Tm_dem)):
        m = mode_seq[t]
        T_star = Tm_dem[t]
        r = ev.best(m, T_star, omega_m[t])
        if r is None:
            T_cap = ev.T_max(m, omega_m[t])
            if T_cap <= abs(T_star):
                unmet += abs(T_star) - T_cap
            T_star = np.sign(T_star) * min(abs(T_star), 0.999 * T_cap)
            r = ev.best(m, T_star, omega_m[t]) if abs(T_star) > 1e-6 else None
        if r is None:
            # speed infeasible in this mode (or zero demand): no machine power
            aux_energy += P_AUX
            prev = m
            continue
        P_mech = T_star * omega_m[t]
        machine_energy += P_mech + r["P_cu"] + r["P_fe"] + r["P_inv"]
        aux_energy += P_AUX
        br["cu"] += r["P_cu"]; br["fe"] += r["P_fe"]
        br["inv"] += r["P_inv"]; br["mech"] += P_mech
        if prev is not None and m != prev:
            sw += 1
        prev = m
    for k in ("cu", "fe", "inv", "mech"):
        br[k] *= N_MOTORS
    br["sw"] = sw * e_sw
    E = N_MOTORS * machine_energy + aux_energy + br["sw"]
    return {"E_Wh": E / 3600.0, "switches": sw,
            "breakdown_Wh": {k: x / 3600.0 for k, x in br.items()},
            "unmet_Nm_s": N_MOTORS * unmet}


def dp_schedule(ev, Tm_dem, omega_m, e_sw=E_CHANGEOVER, dwell=10):
    """DP over modes with minimum dwell. State: (mode, steps since last
    switch, capped at dwell). Backpointers store the full predecessor
    (mode, state) to avoid ambiguity when several predecessors reach the
    same capped state."""
    N = len(Tm_dem)
    nm, S = len(MODES), dwell + 1
    INF = np.inf
    cost = np.full((N, nm), INF)
    for t in range(N):
        for j, m in enumerate(MODES):
            r = ev.best(m, Tm_dem[t], omega_m[t])
            if r is not None:
                cost[t, j] = r["P"] + P_AUX
            elif abs(Tm_dem[t]) < 1e-3:
                cost[t, j] = P_AUX
    J = np.full((nm, S), INF)
    # start: "ready to switch" (s = dwell) in every mode
    for j in range(nm):
        if np.isfinite(cost[0, j]):
            J[j, dwell] = cost[0, j]
    back = np.full((N, nm, S), -1, dtype=np.int32)
    for t in range(1, N):
        Jn = np.full((nm, S), INF)
        for m2 in range(nm):
            c = cost[t, m2]
            if not np.isfinite(c):
                continue
            # stay in m2: s -> min(s+1, dwell)
            for s in range(S):
                v = J[m2, s]
                if not np.isfinite(v):
                    continue
                s2 = min(s + 1, dwell)
                if v + c < Jn[m2, s2]:
                    Jn[m2, s2] = v + c
                    back[t, m2, s2] = m2 * S + s
            # switch into m2 from any m1 at s = dwell
            for m1 in range(nm):
                if m1 == m2:
                    continue
                v = J[m1, dwell]
                if not np.isfinite(v):
                    continue
                if v + e_sw + c < Jn[m2, 0]:
                    Jn[m2, 0] = v + e_sw + c
                    back[t, m2, 0] = m1 * S + dwell
        J = Jn
    if not np.isfinite(J).any():
        return ["4P"] * N, float("inf")
    seq = [0] * N
    m_star, s_star = np.unravel_index(np.argmin(J), J.shape)
    seq[-1] = m_star
    for t in range(N - 1, 0, -1):
        code = back[t, seq[t], s_star]
        if code < 0:  # unreachable state on the optimal path: fall back
            code = seq[t] * S + max(s_star - 1, 0)
        seq[t - 1] = code // S
        s_star = code % S
    return [MODES[i] for i in seq], float(J.min())


def main():
    msd = load_model("mlp")
    data = build_mode_data(msd)
    ev = ModeEval(data)
    v, T_wheel, omega_m = cycle_demands("class1")
    Tm_dem = machine_torque(T_wheel)

    diag = {"T_machine_max_demand": float(np.abs(Tm_dem).max()),
            "rpm_max": float(omega_m.max() * 60 / (2 * np.pi)),
            "R_COIL": R_COIL}
    for m in MODES:
        diag[f"T_max_{m}_at_rpm_max"] = ev.T_max(m, omega_m.max())
        diag[f"T_max_{m}_at_half_speed"] = ev.T_max(m, omega_m.max() / 2)
    print(json.dumps(diag, indent=1), flush=True)

    results = {"diagnostics": diag, "params": {
        "R_COIL": R_COIL, "V_DC": V_DC, "GEAR": GEAR, "M_VEH": M_VEH,
        "N_MOTORS": N_MOTORS,
        "CDA": CDA, "CRR": CRR, "R_WHEEL": R_WHEEL, "ETA_G": ETA_G,
        "P_AUX": P_AUX, "E_CHANGEOVER": E_CHANGEOVER, "K_H": K_H, "K_C": K_C,
        "cycle": "WLTC class 1", "n_steps": len(v)}}

    N = len(v)
    strategies = {"fixed4P": ["4P"] * N}
    best_rule = None
    for vth in np.arange(10, 60, 2.5):
        seq = ["4S" if vv <= vth else "4P" for vv in v]
        r = simulate(ev, seq, Tm_dem, omega_m)
        if best_rule is None or r["E_Wh"] < best_rule[1]["E_Wh"]:
            best_rule = (float(vth), r, seq)
    strategies["rule2"] = best_rule[2]
    greedy = []
    for t in range(N):
        cands = [(ev.best(m, Tm_dem[t], omega_m[t]), m) for m in MODES]
        cands = [(c, m) for c, m in cands if c is not None]
        greedy.append(min(cands, key=lambda cm: cm[0]["P"])[1] if cands else "4P")
    strategies["greedy"] = greedy

    seq_dp, J_opt = dp_schedule(ev, Tm_dem, omega_m)
    strategies["dp"] = seq_dp

    dist = float(np.trapezoid(v / 3.6, dx=1.0) / 1000.0)
    results["distance_km"] = dist
    for name, seq in strategies.items():
        r = simulate(ev, seq, Tm_dem, omega_m)
        r["Wh_per_km"] = r["E_Wh"] / dist
        r["mode_counts"] = {m: int(sum(1 for x in seq if x == m)) for m in MODES}
        if name == "rule2":
            r["threshold_kmh"] = best_rule[0]
        results[name] = r
        print(name, "E=%.1f Wh (%.2f Wh/km), switches=%d" %
              (r["E_Wh"], r["Wh_per_km"], r["switches"]), flush=True)

    results["dp"]["unmet_check"] = float(results["dp"]["unmet_Nm_s"])

    # sensitivity: changeover energy on the DP and greedy sequences
    sens = {"e_sw": {}}
    for e in (5e-3, 25e-3, 100e-3, 400e-3):
        rd = simulate(ev, seq_dp, Tm_dem, omega_m, e_sw=e)
        rg = simulate(ev, greedy, Tm_dem, omega_m, e_sw=e)
        sens["e_sw"][f"{e*1e3:.0f}mJ"] = {
            "dp_Wh": rd["E_Wh"], "dp_switches": rd["switches"],
            "greedy_Wh": rg["E_Wh"], "greedy_switches": rg["switches"]}
    # iron-loss scale on the DP sequence
    sens["iron"] = {}
    for scale in (0.5, 2.0):
        ev2 = ModeEval(build_mode_data(msd, iron_scale=scale))
        rd = simulate(ev2, seq_dp, Tm_dem, omega_m)
        rf = simulate(ev2, ["4P"] * N, Tm_dem, omega_m)
        sens["iron"][f"x{scale:g}"] = {"dp_Wh": rd["E_Wh"], "fixed4P_Wh": rf["E_Wh"]}
    results["sensitivity"] = sens

    # surrogate robustness: same schedule evaluated with the energy surrogate
    try:
        en = load_model("energy", "ckpt_energy_drive.pt")
        ev_en = ModeEval(build_mode_data(en))
        r_en = simulate(ev_en, seq_dp, Tm_dem, omega_m)
        r_4p_en = simulate(ev_en, ["4P"] * N, Tm_dem, omega_m)
        results["energy_check"] = {"dp_Wh": r_en["E_Wh"],
                                   "fixed4P_Wh": r_4p_en["E_Wh"],
                                   "saving_pct": 100 * (1 - r_en["E_Wh"] / r_4p_en["E_Wh"])}
        print("energy check:", results["energy_check"], flush=True)
    except Exception as e:
        results["energy_check"] = {"error": str(e)}

    with open(ROOT / "exp" / "results_drive.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    # dump arrays for figures
    np.savez_compressed(
        ROOT / "exp" / "drive_arrays.npz",
        v=v, omega=omega_m, Tm=Tm_dem,
        dp=np.array([MODES.index(x) for x in seq_dp]),
        greedy=np.array([MODES.index(x) for x in greedy]),
        rule=np.array([MODES.index(x) for x in strategies["rule2"]]),
        modes=np.array(MODES))
    print("saved results_drive.json + drive_arrays.npz")


if __name__ == "__main__":
    main()
