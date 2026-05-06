# Development Plan

This file is the working roadmap for `pinocchio_voids`. It tracks the current
stage, completed work, and the next development tasks.

## Current Stage

We are in **Milestone 3** with the paired-halo finder upgraded beyond the
geometry-only baseline into a first full scored-merge implementation. Active
void size-function calibration remains **n256-only**:

- Milestone 1 and Milestone 2 are complete.
- `ALGORITHM.md` remains the authoritative algorithm specification.
- Local `n032`, `n064`, and `n128` run products were removed because they do
  not provide a workable VSF for calibration.
- The only active local calibration target is the `n256` paired PINOCCHIO and
  VIDE catalogs under ignored `runs/` paths.
- VIDE `voidDesc` radii are corrected from normalized Voronoi volumes using
  `sample_info.txt` mean tracer separation.
- VSF-facing effective radii consistently use
  `R_eff = (3 V / 4 pi)^(1/3)`.
- The current `n256` MCMC maximum-likelihood row is:
  - linking factor `0.14605092780899798`
  - `radius_a0=6.14700029037185`
  - `radius_alpha=0.9313222316465706`
  - `adjacency_factor=0.5240470713979322`
- VSF-only calibration remains the baseline; the active calibration direction is
  now a joint VSF plus center-match posterior.
- Calibration can now select the VIDE target variant: `all`,
  `trimmed_nodencut`, `untrimmed`, or `untrimmed_dencut`.
- Geometry-only merging is retained as the default baseline, while weighted
  scored merging can now combine geometry, source bridge density, and
  compatibility scores.
- PINOCCHIO execution orchestration has not started.

## Milestone 3 Progress

Completed so far:

- Added paired PINOCCHIO catalog loading into canonical `HaloCatalog` objects.
- Added periodic geometry helpers, periodic FoF-like source clustering, and
  source-cluster summaries.
- Added spherical protovoid construction with
  `R_proto = a0 R_L(M)^alpha`.
- Added geometry-only protovoid adjacency and connected-component merging.
- Added source-cluster shape/compactness diagnostics and optional quality cuts.
- Added weighted protovoid edge scoring with geometry, source-catalog
  bridge-density, and compatibility terms.
- Added merge-threshold filtering so only accepted scored edges define final
  connected-component voids.
- Added final-void merge diagnostics, including mean/max merge score and source
  richness/compactness summaries.
- Added symmetric A-to-B and B-to-A paired execution.
- Added VIDE `voidDesc` reading and VSF metrics.
- Added finder-vs-VIDE comparison scripts with fixed paper-style bins.
- Added guarded deterministic calibration scores and Poisson VSF likelihood
  scoring.
- Added `emcee` posterior sampling for `n256` calibration, including chain
  products, trace plots, 68/95 percent contour plots, and best-fit plot commands.
- Added `scripts/optimize_n256_full_algorithm_mcmc.py` for the full
  eight-parameter scored-merge calibration.
- Updated the active plot refresh workflow to `scripts/plot_n256_void_size_function.py`.
- Added `scripts/plot_n256_void_slice.py` for object-level visual checks: a 2D
  periodic box slice with finder and VIDE voids overlaid as `R_eff` circles.
- Added `scripts/match_n256_void_centers.py` for finder-to-nearest-VIDE
  center diagnostics, including distance ratios against finder, VIDE, and
  minimum `R_eff`.
- Extended the n256 slice plot with optional VIDE `centers_all` vs
  `macrocenters_all` selection, nearest-neighbor match lines, and labels for
  the largest plotted objects.
- Added robust center-match scoring to calibration: symmetric nearest-neighbor
  center distances normalized by `min(R_eff)` with a Student-t likelihood.
- Added `scripts/optimize_n256_joint_calibration_mcmc.py` for a weighted
  `logL_vsf + logL_center` `emcee` posterior, with per-sample likelihood
  decomposition and best-fit VSF/center regeneration commands.
- Added `scripts/plot_n256_halo_void_slice.py` for halo-background slice plots:
  separate finder and VIDE void cross-section overlays on the same target-halo
  slab scatter, with CSV diagnostics for displayed radii.
- Added VIDE binary membership readers for input tracers, Voronoi volumes,
  void-to-zone membership, and zone-to-particle membership.
- Added `scripts/debug_n256_vide_region.py` to audit halo counts, nearest
  `R_eff` sphere margins, projected slice margins, and VIDE catalog variants
  around suspicious regions such as `(100, 100, 128)`.
- Extended halo-background plots with optional VIDE member-tracer overlays so
  object-level checks are not limited to spherical-equivalent `R_eff` circles.
- Added shared VIDE variant resolution and threaded `--vide-variant` through
  the n256 MCMC, VSF plotting, slice plotting, center matching, and region
  audit workflows.
- Updated generated MCMC best-fit commands and chain metadata so the chosen
  resolved VIDE target is preserved.
- Removed deprecated grid-search calibration scripts in favor of MCMC-only
  calibration workflows.
- Cleaned deprecated ignored products so `runs/void-statistics` keeps only
  current `n256` VSF and MCMC products.

## Next Tasks

1. Run the full scored-merge `n256` MCMC calibration.
   - Use `/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_full_algorithm_mcmc.py --vide-variant untrimmed --processes 8`.
   - Compare `n256_full_mcmc_untrimmed_summary.csv` against the geometry-only
     VSF and joint summaries.
   - Run `runs/void-statistics/n256_full_mcmc_untrimmed_best_fit_command.sh`.

2. Run the joint geometry-only `n256` MCMC calibration when a baseline refresh
   is needed.
   - Start with `--vsf-weight 1 --center-weight 1 --center-sigma 1 --center-nu 3`.
   - Use `--processes 8` for the production chain.
   - Compare `n256_joint_mcmc_summary.csv` against the VSF-only
     `n256_vsf_mcmc_summary.csv`.
   - Run `runs/void-statistics/n256_joint_mcmc_best_fit_command.sh`.

3. Run variant-target calibration comparisons.
   - Run VSF-only calibration with `--vide-variant untrimmed`.
   - Run joint calibration with `--vide-variant trimmed_nodencut`.
   - Run full calibration with `--vide-variant untrimmed`.
   - Compare variant products against the default `--vide-variant all` products.
   - Use variant-suffixed outputs to keep products separate.

4. Compare VSF-only, joint, and full best-fit products.
   - Compare paper-bin VSF residuals for VSF-only, joint, and full best fits.
   - Compare center-match summaries, especially median and p90
     `d / min(R_eff)`.
   - Refresh the annotated slice plot for the full best fit if center metrics
     improve without unacceptable VSF degradation.
   - Use halo-background slice plots to check whether finder and VIDE circles
     trace comparable halo underdensities in the same slab.
   - For halo-background plots, keep the default intersection/cross-section
     mode; use center-only full-`R_eff` overlays only as a diagnostic.

5. Inspect posterior stability.
   - Review `runs/void-statistics/n256_vsf_mcmc_contours.png`.
   - Review `runs/void-statistics/n256_joint_mcmc_contours.png`.
   - Review `runs/void-statistics/n256_full_mcmc_untrimmed_contours.png`.
   - Check trace stability in `runs/void-statistics/n256_vsf_mcmc_trace.png`.
   - Check trace stability in `runs/void-statistics/n256_joint_mcmc_trace.png`.
   - Check trace stability in `runs/void-statistics/n256_full_mcmc_untrimmed_trace.png`.
   - Decide whether to rerun a longer chain or narrow the prior region.

6. Use the VIDE region audit to resolve the empty-region question.
   - Run `scripts/debug_n256_vide_region.py` on `(100, 100, 128)` for target A.
   - Compare `default`, `trimmed_nodencut`, `untrimmed`, and
     `untrimmed_dencut` VIDE variants.
   - Regenerate `scripts/plot_n256_halo_void_slice.py --vide-overlay both` to
     check true VIDE member tracers against the spherical `R_eff` visualization.
   - If the region is absent from all variants and member overlays, treat it as
     a VIDE watershed/pruning behavior rather than a plotting bug.

7. Audit paper-comparison assumptions before interpreting amplitudes.
   - Match binning, volume normalization, halo mass cuts, tracer density, and
     VIDE settings as closely as local `n256` data allow.
   - Plan a paper-like local VIDE rerun only if post-processing cannot make the
     current references comparable.

## Test Plan

- Keep the full package test suite passing.
- Existing tests cover catalog validation, periodic geometry, source
  clustering, source-cluster shape diagnostics and quality cuts, protovoid
  construction, graph merging, weighted edge thresholding, bridge-density
  scoring, compatibility scoring, paired execution, VIDE parsing, VSF metrics,
  theory helpers, calibration scores, likelihood-grid ranking, robust
  center-match likelihoods, the short VSF, joint, and full MCMC optimizer smoke
  paths, VIDE center and membership parsing, region audits, and the n256
  slice-plot and center-matching smoke paths, including halo-background
  overlays and VIDE variant resolution.
- Use ignored local integration checks only for active `n256` paths.

## Local Reference Data

- Active ignored PINOCCHIO catalogs:
  - `runs/pinocchio-lowres/n256`
  - `runs/pinocchio-lowres/n256_paired`
- Active ignored VIDE references:
  - `runs/vide-lowres/n256`
  - `runs/vide-lowres/n256_paired`
- Active ignored VSF products live under `runs/void-statistics/` and should be
  limited to current `n256` finder/VIDE, paper-bin, MCMC best-fit, and MCMC
  diagnostic outputs.
- Generated `runs/` data must remain ignored. Commit only code, docs, tests,
  and tiny deterministic fixtures.

## Verification Commands

Use the named Miniforge environment:

```bash
/home/tcastro/miniforge3/envs/voidfinder/bin/python -m pytest
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid smoke-test
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid validate-config tests/fixtures/run_config_small.yaml
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid paired-prototype --help
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid paired-sweep --help
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_vsf_mcmc.py --help
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_joint_calibration_mcmc.py --help
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_full_algorithm_mcmc.py --help
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_vsf_mcmc.py --vide-variant untrimmed --walkers 8 --steps 8 --burn-in 2 --thin 1 --allow-degenerate --output-prefix /tmp/n256_vsf_mcmc_untrimmed_smoke
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_joint_calibration_mcmc.py --vide-variant trimmed_nodencut --walkers 8 --steps 8 --burn-in 2 --thin 1 --center-weight 0 --allow-degenerate --output-prefix /tmp/n256_joint_mcmc_trimmed_nodencut_smoke
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_full_algorithm_mcmc.py --vide-variant untrimmed --walkers 16 --steps 8 --burn-in 2 --thin 1 --center-weight 0 --allow-degenerate --output-prefix /tmp/n256_full_mcmc_untrimmed_smoke
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/plot_n256_void_size_function.py --paper-bins
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/plot_n256_void_size_function.py --vide-variant untrimmed --paper-bins
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/plot_n256_void_slice.py --slice-axis z --slice-center 128 --slice-thickness 20 --target both
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/plot_n256_halo_void_slice.py --slice-axis z --slice-center 128 --slice-thickness 20
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/plot_n256_halo_void_slice.py --slice-axis z --slice-center 128 --slice-thickness 20 --vide-variant untrimmed --vide-overlay both
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/debug_n256_vide_region.py --x 100 --y 100 --z 128 --slice-axis z --slice-center 128 --slice-thickness 20
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/match_n256_void_centers.py
```
