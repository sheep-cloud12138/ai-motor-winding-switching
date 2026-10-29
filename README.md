# Coverage-Controlled Connection-Invariant Surrogates

Code, data, and benchmark for the paper *"Coverage-controlled learning of
connection-invariant magnetic surrogates for reconfigurable electric
machines"*.

Source repository:
<https://github.com/sheep-cloud12138/ai-motor-winding-switching> (private
until the manuscript is released).

## One-command reproduction

```bash
# 1. FEA data generation (all datasets, roughly 2 h on 22 cores)
python exp/gen_data_round2.py --workers 8        # benchmark + ablation pools
python exp/gen_random3_dataset.py               # phase-random patterns
python exp/gen_dense_beta.py                    # dense current-angle sweep
python exp/gen_sync_dataset.py                  # synchronous dq trajectories

# Recompute every torque label from the discrete magnetic co-energy
python exp/regenerate_functional_torque.py --workers 22
python exp/validate_energy_torque.py
python exp/validate_energy_torque2.py
python exp/validate_energy_torque_random.py

# 2. Zero-shot benchmark (main table and physical probes)
python exp/run_benchmark.py --force

# 3. Coverage ablations and scaling
python exp/run_phase.py
python exp/run_scaling.py

# 4. Data-efficiency learning curves
python exp/run_curves.py

# 5. Drive-cycle co-design (fixed strategies, DP, and regime grid)
python exp/train_drive_model.py
python exp/drive_cycle.py
python exp/drive_r2.py
python exp/sweep_drive_regimes.py

# 6. Independent FEMM cross-check (requires FEMM 4.2 + pywin32)
python exp/femm_validate.py
python exp/femm_unseen_withz.py --holdout tap50
python exp/femm_unseen_withz.py --holdout 4P
python exp/femm_drive_candidate.py --kind 3S1P
python exp/femm_drive_candidate.py --kind tap75

# 7. All paper figures
python exp/make_figures.py
```

## Layout

- `coil_fea.py` - 2-D nonlinear magnetostatic solver (one current per coil,
  Froehlich iron, rotating permanent magnets, and discrete co-energy torque).
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

## Current headline results

- Production-grid Maxwell check: 12-slot `r=0.99998`, residual `2.2%`; 6-slot `r=0.99927`,
  residual `3.9%`.
- FEMM: PM fundamental error `7.9%`, saturated torque error `1.6%`, and
  end-to-end torque `R2=0.86-0.94` for 4P/tap50/3S1P.
- The 64-point coarse/fine torque check gives `r=0.996`, `R2=0.984`; the
  remaining `0.90` norm ratio and `11.8%` median difference still limit
  absolute energies.
- Coverage: phase-random data reduce openB flux nRMSE to `0.066`; magnitude
  data reduce tap50 to `0.36`; both patterns reproduce on the 6-slot machine.
- Drive cycle: all four fixed connections complete; DP energy is within
  `0.006%` of fixed 4S. Across
  105 voltage/gear/cycle combinations, no reconfiguration-required case
  was found; the surrogate predicts at least `1%` gain in 28 of 87 feasible
  cases, with a maximum `2.64%`.

## Notes

- Windows: `exp/wmi_fix.py` works around a hung WMI service that blocks
  `platform.win32_ver()` inside scipy imports; it is imported first
  everywhere it is needed.
- Torque is differentiated from the discrete magnetic co-energy and satisfies
  the Maxwell relation. The drive-cycle result uses the original 540-kg,
  single-motor, 320-V platform with a modern SiC inverter model.
- The demo machine's original phase grouping is non-standard (phase-C axis
  at 60 deg electrical). This is irrelevant to the coil-level benchmark
  (all connections are just linear maps); for the synchronous drive-cycle
  study the port grouping is defined with standard 0/120/240-deg axes in
  `exp/drive_cycle.py:connection_matrix` - no retraining needed, which is
  rather the point of the method.

## Correction log (2026-10-29)

- The slot-local Lorentz torque did not satisfy the discrete energy relation.
  `exp/energy_torque.py` now differentiates the discrete magnetic co-energy;
  the 40x96 check gives correlation 0.99998 and 2.2% residual for
  `dT/di = dpsi/dtheta`. The 6-slot replication gives correlation 0.99927 and
  3.9% residual after the rotating-PM geometry correction.
- The same Maxwell stencil gives 20.9% residual on the 80x192 mesh. The
  64-point coarse/fine torque check gives correlation 0.996 and `R2=0.984`,
  with a 0.90 norm ratio and 11.8% median pointwise difference.
- Random-current checks pass on 18 cases with minimum correlation 0.9996
  and maximum residual 2.7%; this is stencil-specific, not mesh-converged.
- Formal datasets and checkpoints are regenerated with the corrected
  `functional_v6_theta15` labels. Potential-based models predict physical
  units and
  apply output normalisation only inside the loss; this preserves
  `dT/di = dpsi/dtheta` after de-standardisation.
- FEMM slot coordinates were aligned to the polar-grid slot sectors, and the
  torque post-processor now sums each slot-Lorentz integral explicitly.
- The fine-mesh inductance shape agrees with FEMM to 2.1%, but a 1.13x
  absolute normalisation difference remains. Re-solving with the actual
  discretised slot area removes about half of it, but that convention change
  is not applied to the released benchmark data; it is documented as an
  unresolved solver-alignment issue rather than hidden by rescaling.
- With the corrected labels, all four fixed connections complete the WLTC
  class-1 case and DP energy is within `0.006%` of fixed 4S. A 105-point
  voltage/gear sweep contains no reconfiguration-required case.
- Independent FEMM checks now use the same turns, stack length, BH curve,
  slot positions and rotating 80%-arc PM geometry. The 12-slot fine-mesh
  inductance shape agrees to `2.1%` after a remaining `1.13x` global scale.
  PM fundamental amplitudes differ by `7.9%` on average (`8.4%` maximum),
  and the nonlinear 4S torque differs by `1.6%` at the checked point.
- Over 36 synchronous points, end-to-end FEMM torque `R2` is `0.929`
  (4P), `0.935` (tap50), and `0.863` (3S1P). Correlations are
  `0.954-0.984`. Tap75's 282.1 Wh surrogate result is rejected by FEMM
  (`r=0.83`, `R2=0.54`). No hardware or HIL measurement is available in
  this workspace.

## License / citation

Code and data: released for review; license and citation block to be added
on acceptance.
