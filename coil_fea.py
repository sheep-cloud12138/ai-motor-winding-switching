# -*- coding: utf-8 -*-
"""Coil-level 2D nonlinear magnetostatic FEA for winding-topology experiments.

This is a research prototype adapted from the existing motor_demo solver.
It exposes one independent current per slot/coil and computes coil flux
linkages from the magnetic vector potential.

Usage:
    python coil_fea.py --mode linear --nr 40 --nt 96
    python coil_fea.py --mode nonlinear --nr 40 --nt 96 --bias 20
"""

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

MU0 = 4 * np.pi * 1e-7
RIN, ROUT = 0.005, 0.065
R_SHAFT, R_ROTOR, R_BORE = 0.012, 0.0295, 0.0310
R_SLOT_IN, R_SLOT_OUT = 0.0355, 0.0475
N_SLOT, N_POLE = 12, 4
PPAIR = N_POLE // 2
SLOT_W = 2 * np.pi / N_SLOT * 0.52
MU_R0, B_SAT, H0 = 4000.0, 1.6, 6000.0
N_TURN = 40
L_STACK = 0.1


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
        np.abs(np.mod(th2 + np.pi / (2 * N_SLOT), 2 * np.pi / N_SLOT) - np.pi / N_SLOT)
        < SLOT_W / 2
    )
    is_air |= slot_mask
    is_air |= (r2 < R_SHAFT)
    in_mag = (r2 >= R_ROTOR - 0.004) & (r2 < R_ROTOR)
    is_mag = in_mag & (
        np.abs(np.mod(th2, 2 * np.pi / N_POLE) - np.pi / N_POLE) < np.pi / N_POLE * 0.80
    )
    is_air &= ~is_mag
    is_iron = ~is_air & ~is_mag

    slot_index = np.round(th_c / (2 * np.pi / N_SLOT)).astype(int) % N_SLOT
    slot_id = -np.ones((nr, nt), dtype=int)
    slot_id[slot_mask] = np.broadcast_to(slot_index[None, :], (nr, nt))[slot_mask]
    slot_area = (R_SLOT_OUT - R_SLOT_IN) * SLOT_W * (R_SLOT_IN + R_SLOT_OUT) / 2

    return {
        "r_edges": r_edges,
        "dr": dr,
        "dth": dth,
        "r_c": r_c,
        "th_c": th_c,
        "r2": r2,
        "th2": th2,
        "is_air": is_air,
        "slot_mask": slot_mask,
        "slot_id": slot_id,
        "slot_area": slot_area,
        "is_mag": is_mag,
        "is_iron": is_iron,
    }


def solve_coils(
    theta_deg,
    coil_currents,
    nr=40,
    nt=96,
    pm_on=True,
    linear_iron=False,
    tol=2e-4,
    maxit=80,
    return_fields=False,
):
    g = build(nr, nt)
    r_c, th_c = g["r_c"], g["th_c"]
    r2, th2 = g["r2"], g["th2"]
    dr, dth = g["dr"], g["dth"]
    slot_id, slot_area = g["slot_id"], g["slot_area"]
    th_r = np.deg2rad(theta_deg)

    # Permanent magnets: radial, rotated with rotor.
    pole_sign = np.tanh(np.sin(PPAIR * (th2 - th_r)) / 0.12)
    M_r = np.where(g["is_mag"], (1.2 / MU0 if pm_on else 0.0) * pole_sign, 0.0)
    src_pm = -np.gradient(M_r, dth, axis=1) / r_c[:, None]

    # One independent current per slot.
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
        Ar_p = np.zeros((nr, nt))
        Ar_p[:-1, :] = cr_p
        Ar_m = np.zeros((nr, nt))
        Ar_m[1:, :] = cr_m
        At_p = ct
        At_m = np.roll(ct, 1, axis=1)
        diag = Ar_p + Ar_m + At_p + At_m
        rows = [idx.ravel()]
        cols = [idx.ravel()]
        vals = [diag.ravel()]
        for off, d in (
            (np.roll(idx, -1, axis=0), Ar_p),
            (np.roll(idx, 1, axis=0), Ar_m),
            (np.roll(idx, -1, axis=1), At_p),
            (np.roll(idx, 1, axis=1), At_m),
        ):
            m = d != 0
            rows.append(idx[m])
            cols.append(off[m])
            vals.append((-d)[m])
        A = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(nr * nt, nr * nt),
        ).tocsc()
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
            nu_new[g["is_air"] | g["is_mag"]] = 1.0 / MU0
            rel = np.linalg.norm(nu_new - nu) / max(np.linalg.norm(nu_new), 1e-30)
            nu = 0.7 * nu + 0.3 * nu_new
            if rel < tol and it > 20:
                break
        iters = it + 1

    Br = np.gradient(Az, dth, axis=1) / r_c[:, None]
    Bth = -np.gradient(Az, dr, axis=0)
    Bmag = np.hypot(Br, Bth)

    # Coil flux linkage: psi_k = N * L / S * integral(A_z dS).
    psi = np.zeros(N_SLOT)
    for k in range(N_SLOT):
        m = slot_id == k
        psi[k] = N_TURN * L_STACK / slot_area * np.sum(
            Az[m] * np.broadcast_to(g["r2"], (nr, nt))[m] * dr * dth
        )

    # Lorentz torque on the slot conductors.
    # The cross-sectional torque integral already yields N m for this
    # current-density convention; the co-energy Maxwell check is the
    # authoritative dimensional test.
    T = float(np.sum(
        (r_c[:, None] * J * Br) * (r_c[:, None] * dr * dth) * g["slot_mask"]
    ))

    out = {
        "theta": theta_deg,
        "psi": psi,
        "T": T,
        "iters": iters,
        "rel": float(rel),
        "Bgap": float(Bmag[(r_c > R_ROTOR) & (r_c < R_BORE)].mean()),
    }
    if return_fields:
        out.update(
            Az=Az,
            Br=Br,
            Bth=Bth,
            Bmag=Bmag,
            J=J,
            slot_id=slot_id,
            is_iron=g["is_iron"],
            is_mag=g["is_mag"],
        )
    return out


def linear_reciprocity(nr, nt, i_test=1.0):
    n = N_SLOT
    L = np.zeros((n, n))
    psi_all = np.zeros((n, n))
    t0 = time.time()
    for k in range(n):
        I = np.zeros(n)
        I[k] = i_test
        r = solve_coils(0.0, I, nr=nr, nt=nt, pm_on=False, linear_iron=True)
        psi_all[:, k] = r["psi"]
        L[:, k] = r["psi"] / i_test
    asym = np.linalg.norm(L - L.T) / max(np.linalg.norm(L), 1e-30)
    eig = np.linalg.eigvalsh(0.5 * (L + L.T))
    return {
        "mode": "linear_reciprocity",
        "nr": nr,
        "nt": nt,
        "seconds": time.time() - t0,
        "L": L.tolist(),
        "psi": psi_all.tolist(),
        "asymmetry": float(asym),
        "eigenvalues": eig.tolist(),
        "min_eigenvalue": float(eig.min()),
        "max_eigenvalue": float(eig.max()),
    }


def nonlinear_reciprocity(nr, nt, bias=20.0, delta=1.0, theta=0.0):
    n = N_SLOT
    I0 = bias * np.sin(PPAIR * np.deg2rad(np.arange(n) * 30.0))
    L = np.zeros((n, n))
    t0 = time.time()
    for k in range(n):
        Ip = I0.copy()
        Im = I0.copy()
        Ip[k] += delta
        Im[k] -= delta
        rp = solve_coils(theta, Ip, nr=nr, nt=nt, pm_on=True, linear_iron=False, tol=1e-5)
        rm = solve_coils(theta, Im, nr=nr, nt=nt, pm_on=True, linear_iron=False, tol=1e-5)
        L[:, k] = (rp["psi"] - rm["psi"]) / (2 * delta)
    asym = np.linalg.norm(L - L.T) / max(np.linalg.norm(L), 1e-30)
    eig = np.linalg.eigvalsh(0.5 * (L + L.T))
    return {
        "mode": "nonlinear_reciprocity",
        "nr": nr,
        "nt": nt,
        "bias": bias,
        "delta": delta,
        "theta": theta,
        "seconds": time.time() - t0,
        "L": L.tolist(),
        "asymmetry": float(asym),
        "eigenvalues": eig.tolist(),
        "min_eigenvalue": float(eig.min()),
        "max_eigenvalue": float(eig.max()),
    }


def energy_path(nr, nt, theta=0.0, steps=4, target=None, tol=1e-6):
    """Check path independence of co-energy W = integral psi . di.

    Compares a radial path (all currents scaled together) with a sequential
    path (coils energized one by one). A conservative magnetic field should
    give the same co-energy within solver tolerance.
    """
    n = N_SLOT
    if target is None:
        target = np.zeros(n)
        target[:3] = 10.0
    target = np.asarray(target, dtype=float)

    cache = {}

    def psi_at(I):
        key = tuple(np.round(I, 9))
        if key not in cache:
            r = solve_coils(
                theta, np.asarray(I), nr=nr, nt=nt, pm_on=True,
                linear_iron=False, tol=tol, maxit=120
            )
            cache[key] = r["psi"]
        return cache[key]

    # Radial path: I(t) = t * target.
    ts = np.linspace(0.0, 1.0, steps + 1)
    W_rad = 0.0
    psi_prev = psi_at(np.zeros(n))
    for t in ts[1:]:
        psi_now = psi_at(t * target)
        W_rad += 0.5 * (np.dot(psi_prev, target) + np.dot(psi_now, target)) * (1.0 / steps)
        psi_prev = psi_now

    # Sequential path: energize coil 0, then 1, ...
    I = np.zeros(n)
    W_seq = 0.0
    for k in range(n):
        if abs(target[k]) < 1e-12:
            continue
        psi_prev = psi_at(I)
        for j in range(1, steps + 1):
            I_new = I.copy()
            I_new[k] = target[k] * j / steps
            psi_now = psi_at(I_new)
            W_seq += 0.5 * (psi_prev[k] + psi_now[k]) * (I_new[k] - I[k])
            I = I_new
            psi_prev = psi_now

    return {
        "mode": "energy_path",
        "nr": nr,
        "nt": nt,
        "theta": theta,
        "steps": steps,
        "target": target.tolist(),
        "W_radial": float(W_rad),
        "W_sequential": float(W_seq),
        "relative_difference": float(abs(W_rad - W_seq) / max(abs(W_rad), 1e-30)),
        "solves": len(cache),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["linear", "nonlinear", "energy"], default="linear")
    ap.add_argument("--nr", type=int, default=40)
    ap.add_argument("--nt", type=int, default=96)
    ap.add_argument("--bias", type=float, default=20.0)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--theta", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.mode == "linear":
        out = linear_reciprocity(args.nr, args.nt)
    elif args.mode == "nonlinear":
        out = nonlinear_reciprocity(
            args.nr, args.nt, bias=args.bias, delta=args.delta, theta=args.theta
        )
    else:
        out = energy_path(args.nr, args.nt, theta=args.theta, steps=args.steps)

    print(json.dumps({k: v for k, v in out.items() if k not in ("L", "psi", "eigenvalues")}, indent=1))
    if "asymmetry" in out:
        print("asymmetry = %.3e" % out["asymmetry"])
        print("eigenvalue range = [%.3e, %.3e]" % (out["min_eigenvalue"], out["max_eigenvalue"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
