# -*- coding: utf-8 -*-
"""Second machine for cross-machine replication: 6-slot/4-pole SPM.

Same solver family and conventions as coil_fea.py (12s4p) but a different
slot/pole combination, so the coverage principles can be re-tested on a
machine with different winding factor, inductance matrix and harmonics.

Concentrated double-layer-style grouping, one independent current per slot;
phase axes are computed at import and asserted to sit at 0/120/240 deg.
"""

import exp.wmi_fix  # noqa: F401
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

MU0 = 4 * np.pi * 1e-7
RIN, ROUT = 0.005, 0.065
R_SHAFT, R_ROTOR, R_BORE = 0.012, 0.0295, 0.0310
R_SLOT_IN, R_SLOT_OUT = 0.0355, 0.0475
N_SLOT, N_POLE = 6, 4
PPAIR = N_POLE // 2
SLOT_W = 2 * np.pi / N_SLOT * 0.52
MU_R0, B_SAT, H0 = 4000.0, 1.6, 6000.0
N_TURN = 40
L_STACK = 0.1

# slot grouping for 6s4p: A={1,4}, B={2,5}, C={0,3}, all same sign
# (slot electrical angles: 60/180/300 deg -> phase axes 120 deg apart)
PHASE_COILS = {
    "A": ([1, 4], np.array([1.0, 1.0])),
    "B": ([2, 5], np.array([1.0, 1.0])),
    "C": ([0, 3], np.array([1.0, 1.0])),
}


def _axis(coils, signs):
    return sum(s * np.exp(1j * PPAIR * 2 * np.pi * (c + 0.5) / N_SLOT)
               for c, s in zip(coils, signs))


_AX = {k: _axis(*PHASE_COILS[k]) for k in "ABC"}
_ANG = {k: np.degrees(np.angle(_AX[k])) % 360 for k in "ABC"}
assert all(abs(abs(_ANG["A"] - _ANG[b]) - 120) < 1e-6 for b in "BC"), _ANG


def connection_currents(kind, ip):
    ic = np.zeros(N_SLOT)
    ia, ib, icp = ip
    if kind == "2S":
        for p, key in enumerate("ABC"):
            coils, signs = PHASE_COILS[key]
            ic[coils] = signs * ip[p]
    elif kind == "2P":
        for p, key in enumerate("ABC"):
            coils, signs = PHASE_COILS[key]
            ic[coils] = 0.5 * signs * ip[p]
    elif kind == "tap50":
        for p, key in enumerate("ABC"):
            coils, signs = PHASE_COILS[key]
            ic[coils[0]] = signs[0] * ip[p]
    elif kind == "openB":
        coilsA, sA = PHASE_COILS["A"]
        coilsC, sC = PHASE_COILS["C"]
        ic[coilsA] = sA * ia
        ic[coilsC] = sC * (-ia)
    else:
        raise ValueError(kind)
    return ic


def mu_r_of_B(B):
    B = np.maximum(B, 1e-9)
    H = B / (MU0 * MU_R0) + (B / B_SAT) ** 6 * H0
    return B / (MU0 * H)


def build(nr, nt):
    r_edges = np.linspace(RIN, ROUT, nr + 1)
    dr = (ROUT - RIN) / nr
    dth = 2 * np.pi / nt
    r_c = 0.5 * (r_edges[:-1] + r_edges[1:])
    th_c = (np.arange(nt) + 0.5) * dth
    r2 = r_c[:, None]
    th2 = th_c[None, :]
    is_air = np.zeros((nr, nt), bool)
    is_air |= (r2 >= R_ROTOR) & (r2 <= R_BORE)
    slot_mask = (r2 >= R_SLOT_IN) & (r2 <= R_SLOT_OUT) & (
        np.abs(np.mod(th2 + np.pi / (2 * N_SLOT), 2 * np.pi / N_SLOT)
               - np.pi / N_SLOT) < SLOT_W / 2)
    is_air |= slot_mask
    is_air |= (r2 < R_SHAFT)
    radial_pm = (r_c >= R_ROTOR - 0.004) & (r_c < R_ROTOR)
    in_pm_ring = np.broadcast_to(radial_pm[:, None], (nr, nt))
    is_air &= ~in_pm_ring
    is_iron = ~is_air & ~in_pm_ring
    slot_index = np.round(th_c / (2 * np.pi / N_SLOT)).astype(int) % N_SLOT
    slot_id = -np.ones((nr, nt), dtype=int)
    slot_id[slot_mask] = np.broadcast_to(slot_index[None, :], (nr, nt))[slot_mask]
    slot_area = (R_SLOT_OUT - R_SLOT_IN) * SLOT_W * (R_SLOT_IN + R_SLOT_OUT) / 2
    return dict(r_c=r_c, th_c=th_c, th2=th2, dr=dr, dth=dth, r2=r2,
                slot_mask=slot_mask, slot_id=slot_id, slot_area=slot_area,
                in_pm_ring=in_pm_ring, is_iron=is_iron, is_air=is_air,
                r_edges=r_edges)


def magnetization(g, theta_deg, pm_on=True):
    """Rotating 80%-arc radial PM source and its support mask."""
    if not pm_on:
        zero = np.zeros_like(g["in_pm_ring"], dtype=float)
        return zero, zero.astype(bool)
    phase = np.mod(g["th2"] - np.deg2rad(theta_deg), 2 * np.pi / N_POLE)
    distance = np.abs(phase - np.pi / N_POLE)
    half_arc = np.pi / N_POLE * 0.80
    mask = g["in_pm_ring"] & (distance < half_arc)
    pole = np.sign(np.sin(PPAIR * (g["th2"] - np.deg2rad(theta_deg))))
    return np.where(mask, (1.2 / MU0) * pole, 0.0), mask


def solve_coils(theta_deg, coil_currents, nr=40, nt=48, pm_on=True,
                linear_iron=False, tol=2e-4, maxit=80, return_fields=False):
    g = build(nr, nt)
    r_c, th_c = g["r_c"], g["th_c"]
    r2, th2 = g["r2"], g["th_c"][None, :]
    dr, dth = g["dr"], g["dth"]
    slot_id, slot_area = g["slot_id"], g["slot_area"]
    th_r = np.deg2rad(theta_deg)
    M_r, is_mag = magnetization(g, theta_deg, pm_on)
    src_pm = -np.gradient(M_r, dth, axis=1) / r_c[:, None]
    nonmagnetic = g["is_air"] | g["in_pm_ring"]
    j_slot = np.asarray(coil_currents, dtype=float) * N_TURN / slot_area
    J = np.zeros((nr, nt))
    for k in range(N_SLOT):
        J[slot_id == k] = j_slot[k]
    SRC = J + src_pm
    idx = np.arange(nr * nt).reshape(nr, nt)

    def assemble(nu_cell):
        nu_r = 0.5 * (nu_cell[:-1, :] + nu_cell[1:, :])
        nu_t = 0.5 * (nu_cell + np.roll(nu_cell, -1, axis=1))
        rf = g["r_edges"][1:-1, None]
        cr_p = nu_r * rf / (r_c[:-1, None] * dr * dr)
        cr_m = nu_r * rf / (r_c[1:, None] * dr * dr)
        ct = nu_t / (r_c[:, None] ** 2 * dth * dth)
        Ar_p = np.zeros((nr, nt)); Ar_p[:-1, :] = cr_p
        Ar_m = np.zeros((nr, nt)); Ar_m[1:, :] = cr_m
        At_p = ct; At_m = np.roll(ct, 1, axis=1)
        diag = Ar_p + Ar_m + At_p + At_m
        rows = [idx.ravel()]; cols = [idx.ravel()]; vals = [diag.ravel()]
        for off, d in ((np.roll(idx, -1, axis=0), Ar_p),
                       (np.roll(idx, 1, axis=0), Ar_m),
                       (np.roll(idx, -1, axis=1), At_p),
                       (np.roll(idx, 1, axis=1), At_m)):
            m = d != 0
            rows.append(idx[m]); cols.append(off[m]); vals.append((-d)[m])
        A = sp.coo_matrix((np.concatenate(vals),
                           (np.concatenate(rows), np.concatenate(cols))),
                          shape=(nr * nt, nr * nt)).tocsc()
        b = SRC.ravel().copy()
        A = A.tolil()
        for rr in (idx[-1, :], idx[0, :]):
            A[rr, :] = 0
            for k in rr:
                A[k, k] = 1.0
            b[rr] = 0.0
        return A.tocsc(), b

    if linear_iron:
        nu = np.ones((nr, nt)) / MU0
        nu[g["is_iron"]] = 1.0 / (MU0 * MU_R0)
        A, b = assemble(nu)
        Az = spl.spsolve(A, b).reshape(nr, nt)
        iters, rel = 1, 0.0
    else:
        nu = np.ones((nr, nt)) / MU0
        for it in range(maxit):
            A, b = assemble(nu)
            Az = spl.spsolve(A, b).reshape(nr, nt)
            Br = np.gradient(Az, dth, axis=1) / r_c[:, None]
            Bth = -np.gradient(Az, dr, axis=0)
            Bmag = np.hypot(Br, Bth)
            nu_new = 1.0 / (MU0 * mu_r_of_B(Bmag))
            nu_new[nonmagnetic] = 1.0 / MU0
            rel = np.linalg.norm(nu_new - nu) / max(np.linalg.norm(nu_new), 1e-30)
            nu = 0.7 * nu + 0.3 * nu_new
            if rel < tol and it > 20:
                break
        iters = it + 1

    Br = np.gradient(Az, dth, axis=1) / r_c[:, None]
    Bth = -np.gradient(Az, dr, axis=0)
    Bmag = np.hypot(Br, Bth)
    psi = np.zeros(N_SLOT)
    for k in range(N_SLOT):
        m = slot_id == k
        psi[k] = N_TURN * L_STACK / slot_area * np.sum(
            Az[m] * np.broadcast_to(g["r2"], (nr, nt))[m] * dr * dth)
    T = float(np.sum(
        (r_c[:, None] * J * Br) * (r_c[:, None] * dr * dth) * g["slot_mask"]
    ))
    out = {"theta": theta_deg, "psi": psi, "T": T, "iters": iters,
           "rel": float(rel)}
    if return_fields:
        out.update(Az=Az, Br=Br, Bth=Bth, Bmag=Bmag, J=J, M_r=M_r,
                   is_mag=is_mag, in_pm_ring=g["in_pm_ring"])
    return out
