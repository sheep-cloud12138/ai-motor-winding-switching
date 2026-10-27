# Coverage-Controlled Connection-Invariant Surrogates

Code, data, and benchmark for the paper *"Coverage-controlled learning of
connection-invariant magnetic surrogates for reconfigurable electric
machines"*.

Source repository:
<https://github.com/sheep-cloud12138/ai-motor-winding-switching> (private
until the manuscript is released).

## One-command reproduction

```bash
# 1. FEA data generation (all datasets, ~1.5 h on 8 cores)
python exp/gen_data_round2.py --workers 8        # benchmark + ablation pools
python exp/gen_random3_dataset.py               # phase-random patterns
python exp/gen_dense_beta.py                    # dense current-angle sweep
python exp/gen_sync_dataset.py                  # synchronous dq trajectories

# 2. Zero-shot benchmark (Table 2-4, Jacobian probes)
python exp/run_benchmark.py

# 3. Coverage ablations and scaling (Table 3, Fig. scaling)
python exp/run_phase.py
python exp/run_scaling.py

# 4. Data-efficiency learning curves
python exp/run_curves.py

# 5. Drive-cycle co-design (Table 5, Fig. drive)
python exp/train_drive_model.py
python exp/drive_cycle.py
python exp/drive_r2.py

# 6. Independent FEMM cross-check (requires FEMM 4.2 + pywin32)
python exp/femm_validate.py
python exp/femm_unseen_withz.py --holdout tap50
python exp/femm_unseen_withz.py --holdout 4P
python exp/femm_drive_candidate.py --kind 3S1P

# 7. All paper figures
python exp/make_figures.py
```

## Layout

- `coil_fea.py` - 2-D nonlinear magnetostatic solver (one current per coil,
  Froehlich iron, Lorentz torque; reciprocity/path-independence verified).
- `gen_*.py` (root) - P0-era datasets used by the benchmark (standard
  connections, magnitude-random patterns 2/8/24).
- `exp/` - everything produced for the paper:
  - `gen_*.py` - dataset generators (round-2 patterns, hold-out
    connections, fine-mesh check, phase-random, dense-beta, synchronous).
  - `models.py` - Direct MLP, monotone vector field (negative result),
    structured co-energy network (ours), trainers, Jacobian probes.
  - `data.py` - benchmark splits (LOCO + EXT protocols).
  - `run_benchmark.py` - main benchmark; `run_phase.py` / `run_scaling.py` /
    `run_curves.py` - ablations.
  - `drive_cycle.py` - surrogate-in-the-loop loss maps, DP scheduler,
    baselines (fixed-4P / 2-mode rule / greedy), sensitivity sweeps.
  - `femm_validate.py` - independent FEMM 4.2 checks of the inductance
    matrix, PM waveform, saturated 4S solve, and the tap50 synchronous
    trajectory. It deliberately reports failures rather than calibrating
    them away.
  - `make_figures.py` - all figures from the JSON results.
  - `results_*.json`, `drive_arrays.npz`, `ckpt_*.pt`, `figs/`.
- `paper/` - LaTeX source, cover letter, highlights.
- `V_class1.txt` / `wltc_*.py` - official JRC WLTC class-1 trace
  (checksum-verified) and cycle module metadata.

## Notes

- Windows: `exp/wmi_fix.py` works around a hung WMI service that blocks
  `platform.win32_ver()` inside scipy imports; it is imported first
  everywhere it is needed.
- Torque uses the full Lorentz integral including stack length. The
  drive-cycle result is calibrated to a 540-kg light EV with four identical
  400-V SiC wheel drives; the original single-motor/low-voltage calibration
  was incompatible with the corrected torque scale and produced unrealistic
  inverter losses.
- The demo machine's original phase grouping is non-standard (phase-C axis
  at 60 deg electrical). This is irrelevant to the coil-level benchmark
  (all connections are just linear maps); for the synchronous drive-cycle
  study the port grouping is defined with standard 0/120/240-deg axes in
  `exp/drive_cycle.py:connection_matrix` - no retraining needed, which is
  rather the point of the method.

## Correction log (2026-10-27)

- The Lorentz torque integral had omitted `L_STACK`. Because the omission is
  a Cartesian factor, existing torque labels were corrected exactly by
  multiplying by `L_STACK = 0.1`; the solver now includes the factor directly.
- The original single-motor/low-voltage drive calibration became infeasible
  after that correction. The case study is now a 540-kg light EV with four
  identical wheel motors, a 400-V SiC inverter model, 50% slot fill and a
  matching reduction ratio; torque and energy are aggregated over all four
  drives.
- FEMM slot coordinates were aligned to the polar-grid slot sectors, and the
  torque post-processor now sums each slot-Lorentz integral explicitly.
- The fine-mesh inductance shape agrees with FEMM to 2.1%, but a 1.13x
  absolute normalisation difference remains. Re-solving with the actual
  discretised slot area removes about half of it, but that convention change
  is not applied to the released benchmark data; it is documented as an
  unresolved solver-alignment issue rather than hidden by rescaling.
- Consequently, the old drive percentages are obsolete. At 400 V, fixed 4P
  is the only fully feasible fixed mode and the direct-MLP DP saves about
  `3.1%`. At 700 V and above, fixed delta becomes feasible and the gain
  collapses to `0.4-0.6%`. Either way the claim is conditional.
- Independent FEMM checks now use the same turns, stack length, BH curve,
  slot positions and full-ring radial PM waveform. The 12-slot fine-mesh
  inductance shape agrees to `2.1%` after a remaining `1.13x` global scale;
  PM fundamental amplitudes still differ by `41.8%`.
- Over 36 synchronous points, the drive MLP transfers torque to unseen 3S1P
  (`r=0.79`, `R2=0.55`, MAE `0.124 N m`). Correct-axis tap50 tests reach
  `r=0.66-0.73`, but raw `R2` remains negative to `0.26`.
  No hardware or HIL measurement is available in this workspace.

## License / citation

Code and data: released for review; license and citation block to be added
on acceptance.
