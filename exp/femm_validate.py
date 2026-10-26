# -*- coding: utf-8 -*-
"""Independent validation of the in-house solver against FEMM 4.2 (ActiveX).

Builds the SAME machine (same radii, wedge slots, 0.8-pole-arc PMs, stack
length) in FEMM and compares:
  V1  linear mutual-inductance matrix L (12x12, one coil excited at a time,
      PM removed, linear iron mu_r = 4000)
  V2  PM-only coil flux linkage psi(theta) over 12 rotor positions
  V3  one nonlinear saturated case (Fröhlich BH curve imported into FEMM)
Reference: coil_linear_reciprocity.json, coil_zero.npz, coil_fea.solve_coils.
Radial magnetisation approximated by 9 sub-arcs per pole.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401

MU0 = 4 * np.pi * 1e-7
RIN, ROUT = 0.005, 0.065
R_SHAFT, R_ROTOR, R_BORE = 0.012, 0.0295, 0.0310
R_SLOT_IN, R_SLOT_OUT = 0.0355, 0.0475
N_SLOT, N_POLE = 12, 4
SLOT_W = 2 * np.pi / N_SLOT * 0.52
L_STACK = 0.1
N_TURN = 40
HC = 1.2 / MU0


def lua_lit(v):
    if isinstance(v, str):
        return '"' + v + '"'
    if isinstance(v, (list, tuple, np.ndarray)):
        return "{" + ",".join(lua_lit(x) for x in v) + "}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return repr(float(v))


class Femm:
    """Minimal Python binding for FEMM 4.2 ActiveX (mlab2femm protocol)."""

    def __init__(self):
        import win32com.client
        self.f = win32com.client.Dispatch("femm.ActiveFEMM")

    def raw(self, s):
        z = self.f.mlab2femm(s)
        if z and z[:1] == 'e':
            raise RuntimeError("FEMM: " + z)
        return z

    def __getattr__(self, name):
        def call(*args):
            s = name + "(" + ",".join(lua_lit(a) for a in args) + ")"
            z = self.raw(s)
            if z is None or z == "":
                return None
            z = z.replace("[", " ").replace("]", " ")
            try:
                return float(z)
            except ValueError:
                return [float(x) for x in z.replace("I", "i").split()]
        return call


def build_geometry(f, theta_deg, pm_on, linear_iron):
    f.newdocument(0)
    f.mi_probdef(0, "meters", "planar", 1e-8, L_STACK, 30, 0)
    f.mi_addmaterial("Air", 1, 1, 0, 0)
    if linear_iron:
        f.mi_addmaterial("IronLin", 4000, 4000, 0, 0)
    else:
        # Froehlich curve of coil_fea.py as BH points
        bs = np.linspace(0.01, 2.2, 45)
        h = bs / (MU0 * 4000) + (bs / 1.6) ** 6 * 6000.0
        f.mi_addmaterial("IronNL", 2000, 2000, 0, 0)
        for b, hi in zip(bs, h):
            f.mi_addbhpoint("IronNL", float(b), float(hi))
    if pm_on:
        f.mi_addmaterial("PMp", 1.0, 1.0, HC, 0)
        f.mi_addmaterial("PMm", 1.0, 1.0, -HC, 0)
    f.mi_addboundprop("azero", 0, 0, 0, 0, 0, 0, 0, 0, 0)
    for k in range(N_SLOT):
        f.mi_addcircprop(f"c{k}", 0, 1)

    def circle(r, extra_angs=()):
        """Circle as a closed polyline of arcs with nodes at given angles
        (degrees) plus the four quadrant points, so radial segment endpoints
        land on arc nodes (FEMM never splits arcs implicitly)."""
        angs = sorted({a % 360.0 for a in
                       list(extra_angs) + [0.0, 90.0, 180.0, 270.0]})
        P = []
        for a in angs:
            ar = np.deg2rad(a)
            x, y = r * np.cos(ar), r * np.sin(ar)
            f.mi_addnode(x, y)
            P.append((x, y))
        for i in range(len(P)):
            x1, y1 = P[i]
            x2, y2 = P[(i + 1) % len(P)]
            da = (angs[(i + 1) % len(angs)] - angs[i]) % 360.0
            f.mi_addarc(x1, y1, x2, y2, da, 3)

    # wedge corner angles (degrees), shared by circles and wedges
    wedge_angs = []
    for k in range(N_SLOT):
        c = 15 + 30 * k
        wedge_angs += [c - np.degrees(SLOT_W / 2), c + np.degrees(SLOT_W / 2)]

    for r in (RIN, R_SHAFT, R_BORE, ROUT):
        circle(r)
    for r in (R_SLOT_IN, R_SLOT_OUT):
        circle(r, wedge_angs)

    # Close the finite domain explicitly. FEMM does not infer a zero-vector-
    # potential boundary from the outer circle alone.
    for radius in (RIN, ROUT):
        for a in (45.0, 135.0, 225.0, 315.0):
            ar = np.deg2rad(a)
            f.mi_selectarcsegment(radius * np.cos(ar), radius * np.sin(ar))
            f.mi_setarcsegmentprop(10.0, "azero", 0, 0)
    f.mi_clearselected()

    # PM ring: one annular sector per pole (parallel magnetisation at the
    # pole-centre angle) + one air sector per inter-pole gap. All shared
    # edges use bit-exact identical coordinates.
    th = np.deg2rad(theta_deg)

    def sector(r1, r2, a1, a2, radials=True):
        """Draw annular sector edges (idempotent, bit-exact shared edges)."""
        seg = max(2, int(np.degrees(a2 - a1) / 6))
        deg = np.degrees(a2 - a1)
        for r in (r1, r2):
            x1, y1 = r * np.cos(a1), r * np.sin(a1)
            x2, y2 = r * np.cos(a2), r * np.sin(a2)
            f.mi_addarc(x1, y1, x2, y2, deg, seg)
        if radials:
            for a in (a1, a2):
                x1, y1 = r1 * np.cos(a), r1 * np.sin(a)
                x2, y2 = r2 * np.cos(a), r2 * np.sin(a)
                f.mi_addnode(x1, y1)
                f.mi_addnode(x2, y2)
                f.mi_addsegment(x1, y1, x2, y2)

    pm_r1, pm_r2 = R_ROTOR - 0.004, R_ROTOR
    if not pm_on:
        # The un-magnetised magnet pocket is a complete annulus bounded by
        # its inner radius and R_ROTOR.
        circle(pm_r1)
        circle(pm_r2)
    else:
        # Approximate the smooth radial magnetisation of the reference
        # solver with nine radial sub-arcs per pole.
        pm_angles_deg = []
        for p in range(N_POLE):
            start = np.degrees(th) + 90.0 * p + 9.0
            pm_angles_deg += [(start + 8.0 * k) % 360 for k in range(10)]
        circle(pm_r1, pm_angles_deg)
        circle(pm_r2, pm_angles_deg)
        for a_deg in pm_angles_deg:
            a = np.deg2rad(a_deg)
            x1, y1 = pm_r1 * np.cos(a), pm_r1 * np.sin(a)
            x2, y2 = pm_r2 * np.cos(a), pm_r2 * np.sin(a)
            f.mi_addsegment(x1, y1, x2, y2)

    # slot wedges: radial edges only (corner nodes/arcs already exist)
    for k in range(N_SLOT):
        c = np.deg2rad(15 + 30 * k)
        h = SLOT_W / 2
        for sgn in (-1, 1):
            a = c + sgn * h
            x1, y1 = R_SLOT_IN * np.cos(a), R_SLOT_IN * np.sin(a)
            x2, y2 = R_SLOT_OUT * np.cos(a), R_SLOT_OUT * np.sin(a)
            f.mi_addsegment(x1, y1, x2, y2)

    def label(x, y, mat, circ="", magdir=0, group=0, turns=0):
        f.mi_addblocklabel(x, y)
        f.mi_selectlabel(x, y)
        f.mi_setblockprop(mat, 1, 0.001, circ, magdir, group, turns)
        f.mi_clearselected()

    label(0.5 * RIN, 0.0, "Air")
    label((RIN + R_SHAFT) / 2, 0.0, "Air")
    label((R_SHAFT + R_ROTOR - 0.004) / 2 * np.cos(.02),
          (R_SHAFT + R_ROTOR - 0.004) / 2 * np.sin(.02),
          "IronLin" if linear_iron else "IronNL", group=1)
    label((R_ROTOR + R_BORE) / 2 * np.cos(np.pi / 7),
          (R_ROTOR + R_BORE) / 2 * np.sin(np.pi / 7), "Air")
    if pm_on:
        for p in range(N_POLE):
            rm = (pm_r1 + pm_r2) / 2
            mat = "PMp" if p % 2 == 0 else "PMm"
            start = th + p * np.pi / 2 + np.pi / 20
            for s in range(9):
                center = start + (s + 0.5) * 4 * np.pi / 90
                label(rm * np.cos(center), rm * np.sin(center), mat,
                      magdir=float(np.degrees(center) % 360), group=1)
            gap_center = th + (p + 1) * np.pi / 2
            label(rm * np.cos(gap_center), rm * np.sin(gap_center), "Air",
                  group=1)
    iron = "IronLin" if linear_iron else "IronNL"
    for a in (0.0,):
        label((R_BORE + R_SLOT_IN) / 2 * np.cos(a),
              (R_BORE + R_SLOT_IN) / 2 * np.sin(a), iron)
        label((R_SLOT_OUT + ROUT) / 2 * np.cos(a),
              (R_SLOT_OUT + ROUT) / 2 * np.sin(a), iron)
    # teeth: centred at 30k deg (slots are at 15+30k)
    for k in range(N_SLOT):
        at = np.deg2rad(30 * k)
        label((R_SLOT_IN + R_SLOT_OUT) / 2 * np.cos(at),
              (R_SLOT_IN + R_SLOT_OUT) / 2 * np.sin(at), iron)
    # PM-ring inter-pole gaps (and the whole ring when PMs are off)
    rm = (R_ROTOR - 0.004 + R_ROTOR) / 2
    if not pm_on:
        label(rm * np.cos(np.pi / 4), rm * np.sin(np.pi / 4), "Air")
    # coil blocks: one per slot, referenced to circuit c{0..11}
    for k in range(N_SLOT):
        c = np.deg2rad(15 + 30 * k)
        label((R_SLOT_IN + R_SLOT_OUT) / 2 * np.cos(c),
              (R_SLOT_IN + R_SLOT_OUT) / 2 * np.sin(c), "Air",
              circ=f"c{k}", turns=N_TURN)
    f.mi_zoomnatural()


def coil_currents(f, ic):
    for k in range(N_SLOT):
        f.mi_modifycircprop(f"c{k}", 1, float(ic[k]))


def solve_psi(f, tag):
    import tempfile, os
    fem_path = os.path.join(tempfile.gettempdir(), f"femmval_{tag}.fem").replace(chr(92), "/")
    f.mi_saveas(fem_path)
    f.mi_analyze(1)
    f.mi_loadsolution()
    psi = np.array([f.mo_getcircuitproperties(f"c{k}")[2]
                    for k in range(N_SLOT)])
    return psi


def lorentz_torque(f):
    f.mo_seteditmode("area")
    f.mo_clearblock()
    for k in range(N_SLOT):
        c = np.deg2rad(15 + 30 * k)
        r = (R_SLOT_IN + R_SLOT_OUT) / 2
        f.mo_selectblock(r * np.cos(c), r * np.sin(c))
    return float(f.mo_blockintegral(15))


def weighted_torque(f):
    f.mo_seteditmode("area")
    f.mo_clearblock()
    f.mo_groupselectblock(1)
    return float(f.mo_blockintegral(22))


def validate_tap50(f):
    from coil_fea import solve_coils
    from exp.drive_cycle import connection_matrix, load_model, model_psi_T

    theta = np.arange(0.0, 180.0, 30.0)
    gamma = 0.0
    amp = 40.0
    C = connection_matrix("tap50")
    ip = amp * np.stack([
        np.cos(2 * np.deg2rad(theta) + gamma),
        np.cos(2 * np.deg2rad(theta) + gamma - 2 * np.pi / 3),
        np.cos(2 * np.deg2rad(theta) + gamma + 2 * np.pi / 3),
    ], axis=1)
    ic = ip @ C.T

    psi_ref, psi_femm, psi_mlp = [], [], []
    t_ref, t_femm = [], []
    for th, currents in zip(theta, ic):
        ref = solve_coils(float(th), currents, nr=40, nt=96, pm_on=True,
                          linear_iron=False, tol=2e-4)
        build_geometry(f, float(th), pm_on=True, linear_iron=False)
        coil_currents(f, currents)
        psi_ref.append(ref["psi"])
        psi_femm.append(solve_psi(f, f"tap{int(th)}"))
        t_ref.append(ref["T"])
        t_femm.append(-weighted_torque(f))
        f.mi_close()

    psi_ref = np.asarray(psi_ref)
    psi_femm = np.asarray(psi_femm)
    t_ref = np.asarray(t_ref)
    t_femm = np.asarray(t_femm)
    mdl = load_model("mlp")
    psi_pred, t_pred = model_psi_T(mdl, theta, ic)

    alpha_flux = float(np.sum(psi_pred * psi_femm) /
                       np.sum(psi_pred * psi_pred))
    alpha_torque = float(np.sum(t_pred * t_femm) /
                         np.sum(t_pred * t_pred))
    return {
        "tap50_theta": theta.tolist(),
        "tap50_psi_ref": psi_ref.tolist(),
        "tap50_psi_femm": psi_femm.tolist(),
        "tap50_psi_mlp": psi_pred.tolist(),
        "tap50_torque_ref": t_ref.tolist(),
        "tap50_torque_femm": t_femm.tolist(),
        "tap50_torque_mlp": t_pred.tolist(),
        "tap50_flux_corr": float(np.corrcoef(psi_pred.ravel(),
                                             psi_femm.ravel())[0, 1]),
        "tap50_flux_shape_reldiff": float(
            np.linalg.norm(alpha_flux * psi_pred - psi_femm) /
            np.linalg.norm(psi_femm)),
        "tap50_torque_corr": float(np.corrcoef(t_pred, t_femm)[0, 1]),
        "tap50_torque_shape_reldiff": float(
            np.linalg.norm(alpha_torque * t_pred - t_femm) /
            np.linalg.norm(t_femm)),
    }


def main():
    f = Femm()
    res = {}

    # ---- V1: linear mutual inductance, PM off ----
    print("V1: linear inductance matrix (12 solves)", flush=True)
    L = np.zeros((12, 12))
    e = np.zeros(12)
    for k in range(12):
        build_geometry(f, 0.0, pm_on=False, linear_iron=True)
        ic = e.copy(); ic[k] = 1.0
        coil_currents(f, ic)
        L[:, k] = solve_psi(f, f'L{k}')
        f.mi_close()
        print("  coil %d diag %.4f" % (k, L[k, k]), flush=True)
    ref = json.loads((ROOT / "coil_linear_reciprocity.json").read_text())
    L0 = np.array(ref["L"])
    res["L_relnorm_diff"] = float(np.linalg.norm(L - L0)
                                  / np.linalg.norm(L0))
    alpha = float(np.sum(L * L0) / np.sum(L0 * L0))
    res["L_scale_alpha"] = alpha
    res["L_relerr_after_scale"] = float(
        np.linalg.norm(L - alpha * L0) / np.linalg.norm(L0))
    d = np.abs(np.diag(L) - np.diag(L0)) / np.abs(np.diag(L0))
    res["L_diag_relerr_mean"] = float(d.mean())
    res["L_corr"] = float(np.corrcoef(L.ravel(), L0.ravel())[0, 1])
    res["L_femm"] = L.tolist()
    res["L_femm_diag"] = np.diag(L).tolist()

    # ---- V2: PM-only flux linkage vs theta ----
    print("V2: PM-only flux linkage (12 angles)", flush=True)
    ths = np.arange(0, 180, 15)
    P = np.zeros((len(ths), 12))
    for j, t in enumerate(ths):
        build_geometry(f, float(t), pm_on=True, linear_iron=False)
        coil_currents(f, np.zeros(12))
        P[j] = solve_psi(f, f'pm{j}')
        f.mi_close()
        print("  theta %d psi0 %.4f" % (t, P[j, 0]), flush=True)
    z = np.load(ROOT / "coil_zero.npz")
    P0 = z["Y"][:, :12]
    res["PM_psi_reldiff_mean"] = float(
        np.mean(np.abs(P - P0)) / np.mean(np.abs(P0)))
    tt = np.deg2rad(ths)
    fund = lambda x: 2 * np.abs((x * np.exp(-2j * tt)).mean())
    f0 = np.array([fund(P0[:, k]) for k in range(12)])
    f1 = np.array([fund(P[:, k]) for k in range(12)])
    res["PM_psi_corr"] = float(np.corrcoef(P.ravel(), P0.ravel())[0, 1])
    res["PM_psi_shape_reldiff"] = float(
        np.linalg.norm(P / np.linalg.norm(P)
                       - P0 / np.linalg.norm(P0)))
    res["PM_fund_relerr_mean"] = float(np.mean(np.abs(f1 - f0) / f0))
    res["PM_fund_relerr_max"] = float(np.max(np.abs(f1 - f0) / f0))

    # ---- V3: one nonlinear saturated case ----
    print("V3: nonlinear case (4S at 60 A, theta=30)", flush=True)
    sys.path.insert(0, str(ROOT))
    from coil_fea import solve_coils
    from gen_coil_dataset import connection_currents
    ip = 60.0 * np.array([np.cos(np.pi / 4), np.cos(np.pi / 4 - 2 * np.pi / 3),
                          np.cos(np.pi / 4 + 2 * np.pi / 3)])
    ic = connection_currents("4S", ip)
    r = solve_coils(30.0, ic, nr=40, nt=96, pm_on=True,
                    linear_iron=False, tol=2e-4)
    build_geometry(f, 30.0, pm_on=True, linear_iron=False)
    coil_currents(f, ic)
    psi_f = solve_psi(f, 'nl')
    T_f = lorentz_torque(f)
    T_w = weighted_torque(f)
    f.mi_close()
    psi_o = r["psi"]
    res["NL_psi_reldiff_mean"] = float(
        np.mean(np.abs(psi_f - psi_o)) / np.mean(np.abs(psi_o)))
    res["NL_psi_reldiff_max"] = float(
        np.max(np.abs(psi_f - psi_o) / np.maximum(np.abs(psi_o), 1e-9)))
    res["NL_psi_corr"] = float(np.corrcoef(psi_f, psi_o)[0, 1])
    res["NL_T_ours"] = float(r["T"])
    res["NL_T_femm_lorentz"] = T_f
    res["NL_T_femm_weighted"] = -T_w
    res["NL_T_reldiff"] = float(abs(-T_w - r["T"]) / max(abs(r["T"]), 1e-12))

    # ---- V4: zero-shot tap50 along a synchronous-current trajectory ----
    print("V4: zero-shot tap50 synchronous trajectory", flush=True)
    res.update(validate_tap50(f))

    with open(ROOT / "exp" / "results_femm.json", "w") as fp:
        json.dump(res, fp, indent=1)
    print(json.dumps(res, indent=1)[:900])


if __name__ == "__main__":
    main()
