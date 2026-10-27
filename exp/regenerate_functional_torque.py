"""Recompute flux and torque after the rotating-PM geometry correction."""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import exp.wmi_fix  # noqa: F401
import coil_fea
from exp.energy_torque import functional_torque

TAG = "functional_v5_pmrotor"


def one_case(args):
    theta, ic, nr, nt = args
    solved = coil_fea.solve_coils(
        float(theta), np.asarray(ic, dtype=float),
        nr=nr, nt=nt, pm_on=True, linear_iron=False, tol=2e-4,
    )
    torque = functional_torque(
        float(theta), np.asarray(ic, dtype=float),
        nr=nr, nt=nt, dtheta_deg=7.5,
    )
    return np.concatenate([solved["psi"], [torque]])


def update_dataset(path, pool, nr, nt):
    with np.load(path, allow_pickle=True) as data:
        fields = {name: data[name].copy() for name in data.files}
    if fields.get("torque_model", np.array("")) == TAG:
        print("skip", path.name, flush=True)
        return
    if "Y" in fields and fields["Y"].shape[1] == 13:
        theta, ic = fields["X"][:, 0], fields["X"][:, 1:]
        result = pool.map(
            one_case,
            [(float(t), row, nr, nt) for t, row in zip(theta, ic)],
            chunksize=4,
        )
        result = np.asarray(result, dtype=np.float32)
        fields["Y"] = fields["Y"].copy()
        fields["Y"][:, :12] = result[:, :12]
        fields["Y"][:, 12] = result[:, 12]
    elif path.name == "coil_fine_check.npz":
        pool_data = {}
        for name in ("full", "ext", "random48"):
            with np.load(
                ROOT / f"coil_dataset_{name}.npz", allow_pickle=True
            ) as data:
                pool_data[name] = {
                    key: data[key].copy() for key in data.files
                }
        cases = []
        for src, row in zip(fields["src"], fields["row"]):
            d = pool_data[str(src)]
            cases.append((float(d["X"][row, 0]), d["X"][row, 1:], 80, 192))
        result = np.asarray(pool.map(one_case, cases, chunksize=1))
        fields["psi"] = result[:, :12]
        fields["T"] = result[:, 12]
    else:
        return
    fields["torque_model"] = np.array(TAG)
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **fields)
    os.replace(tmp, path)
    print("updated", path.name, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    paths = [
        ROOT / name for name in (
            "coil_dataset_full.npz",
            "coil_zero.npz",
            "coil_dataset_random.npz",
            "coil_dataset_random8.npz",
            "coil_dataset_random24.npz",
            "coil_dataset_random48.npz",
            "coil_dataset_random24z.npz",
            "coil_dataset_random3.npz",
            "coil_dataset_ext.npz",
            "coil_dataset_densebeta.npz",
            "coil_dataset_sync.npz",
            "coil_fine_check.npz",
        )
    ]
    pool = mp.Pool(args.workers)
    started = time.time()
    for path in paths:
        t0 = time.time()
        update_dataset(path, pool, 40, 96)
        print(path.name, "%.1fs" % (time.time() - t0), flush=True)
    pool.close()
    pool.join()
    print(json.dumps({"seconds": time.time() - started}, indent=1))


if __name__ == "__main__":
    main()
