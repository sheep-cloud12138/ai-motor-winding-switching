# -*- coding: utf-8 -*-
"""Dataset assembly and split utilities for the winding-topology benchmark."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exp.models import make_features, theta_feature_grad  # noqa: E402

STANDARD = ("4S", "2S2P", "delta", "asym")
EXT = ("4P", "tap50", "openB")


def _load(name):
    d = np.load(ROOT / name, allow_pickle=True)
    if "kind" in d.files:
        K = np.array([str(k) for k in d["kind"]])
    else:
        K = np.array([""] * len(d["X"]))
    return d["X"], d["Y"], K


def load_all():
    """Concatenate every generated dataset into one pool with kind labels."""
    Xs, Ys, Ks = [], [], []
    for name, tag in (
        ("coil_dataset_full.npz", None),
        ("coil_dataset_random48.npz", "random"),
        ("coil_dataset_random24z.npz", "randomz"),
        ("coil_zero.npz", "zero"),
        ("coil_dataset_ext.npz", None),
    ):
        if not (ROOT / name).exists():
            continue
        X, Y, K = _load(name)
        if tag is not None:
            K = np.array([tag] * len(X))
        Xs.append(X); Ys.append(Y); Ks.append(K)
    X = np.concatenate(Xs); Y = np.concatenate(Ys); K = np.concatenate(Ks)
    return X, Y, K


def make_split(X, Y, K, holdout, variant="base", val_frac=0.12, seed=0):
    """Return dict with features/tensors for train, val, and test sets.

    holdout: kind excluded from training (LOCO over STANDARD, or one of EXT).
    variant: base | norandom | withz  (which random-pattern pools are used).
    """
    rng = np.random.default_rng(seed)
    is_random = K == "random"
    is_randomz = K == "randomz"
    if variant == "norandom":
        use_random = np.zeros(len(K), bool)
    elif variant == "withz":
        use_random = is_random | is_randomz
    else:
        use_random = is_random

    in_train_kind = np.isin(K, STANDARD) & (K != holdout)
    in_train_kind |= use_random | (K == "zero")
    if holdout in EXT:
        # EXT protocol: all four standard connections are seen in training.
        in_train_kind = np.isin(K, STANDARD) | use_random | (K == "zero")
    tr_pool = np.where(in_train_kind)[0]

    # Stratified validation slice out of the training pool.
    val_mask = np.zeros(len(tr_pool), bool)
    for k in np.unique(K[tr_pool]):
        idx = np.where(K[tr_pool] == k)[0]
        n_val = max(4, int(round(val_frac * len(idx))))
        val_mask[rng.choice(idx, size=n_val, replace=False)] = True
    val_idx = tr_pool[val_mask]
    fit_idx = tr_pool[~val_mask]

    te_idx = np.where(K == holdout)[0]

    def pack(rows):
        F = make_features(X[rows, 0], X[rows, 1:])
        dphi = theta_feature_grad(X[rows, 0])
        return F.astype(np.float32), dphi.astype(np.float32), rows

    return {
        "fit": pack(fit_idx), "val": pack(val_idx), "test": pack(te_idx),
        "fit_idx": fit_idx, "val_idx": val_idx, "test_idx": te_idx,
    }
