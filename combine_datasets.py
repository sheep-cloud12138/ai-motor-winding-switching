# -*- coding: utf-8 -*-
"""Combine connection, random-pattern, and zero-current datasets."""

import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default="coil_dataset_full.npz")
    ap.add_argument("--random", default="coil_dataset_random.npz")
    ap.add_argument("--zero", default="coil_zero.npz")
    ap.add_argument("--out", default="coil_dataset_mixed.npz")
    args = ap.parse_args()
    full = np.load(args.full, allow_pickle=True)
    rnd = np.load(args.random, allow_pickle=True)
    zero = np.load(args.zero)

    # Keep all full rows; the training script excludes the holdout by label.
    X = np.concatenate([full["X"], rnd["X"], zero["X"]], axis=0)
    Y = np.concatenate([full["Y"], rnd["Y"], zero["Y"]], axis=0)
    kind = np.concatenate([full["kind"], rnd["kind"], np.array(["zero"] * len(zero["X"]))])
    theta = X[:, 0]
    np.savez_compressed(
        args.out,
        X=X,
        Y=Y,
        kind=kind,
        theta=theta.astype(np.float32),
    )
    print("n", len(X), "kinds", {k: int((kind == k).sum()) for k in np.unique(kind)})


if __name__ == "__main__":
    main()
