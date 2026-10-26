# -*- coding: utf-8 -*-
"""Compare coil flux patterns between connections at matched operating points."""

import numpy as np


def main():
    d = np.load("coil_dataset_full.npz", allow_pickle=True)
    X, Y, kind = d["X"], d["Y"], d["kind"]
    theta, amp, beta = d["theta"], d["amp"], d["beta"]

    for a in [10.0, 40.0, 60.0]:
        for b in [0.0, np.pi / 2]:
            print(f"\namp={a} beta={b:.2f}")
            for th in [0.0, 45.0, 90.0]:
                i4 = np.where((kind == "4S") & (theta == th) & (amp == a) & (beta == b))[0]
                ia = np.where((kind == "asym") & (theta == th) & (amp == a) & (beta == b))[0]
                if len(i4) == 0 or len(ia) == 0:
                    continue
                p4 = Y[i4[0], :12]
                pa = Y[ia[0], :12]
                ratio = pa / np.where(np.abs(p4) > 1e-9, p4, np.nan)
                cos = np.dot(p4, pa) / (np.linalg.norm(p4) * np.linalg.norm(pa) + 1e-12)
                print(
                    f"theta={th:5.1f} |4S|={np.linalg.norm(p4):.4f} "
                    f"|asym|={np.linalg.norm(pa):.4f} cos={cos:.4f} "
                    f"ratio_mean={np.nanmean(ratio):.3f} ratio_std={np.nanstd(ratio):.3f}"
                )
                print("  psi4S ", np.round(p4, 3))
                print("  psiasym", np.round(pa, 3))


if __name__ == "__main__":
    main()
