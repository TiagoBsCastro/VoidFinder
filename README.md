# PINOCCHIO Voids

`pinocchio_voids` is a scientific Python package under development for
identifying cosmic voids from PINOCCHIO-based structure formation products.

The package now contains the repository scaffold, PINOCCHIO catalog I/O, and a
Milestone 3 paired-halo void finder with geometry-only and weighted scored-merge
modes. Current VSF calibration work is focused on the local `n256` paired
catalogs; lower-resolution `n032`, `n064`, and `n128` runs were removed from
local active products because they do not provide a workable void size function.

This repository intentionally does not yet implement:

- PINOCCHIO execution;
- production workflow orchestration.

## Development Install

```bash
python -m pip install -e ".[dev]"
```

## Smoke Test

```bash
pinvoid smoke-test
```

## Milestone 3 Finder

Run the paired-halo finder on the active `n256` PINOCCHIO halo catalog pair.
The default remains the calibrated geometry-only baseline:

```bash
pinvoid paired-prototype \
  runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out \
  runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --linking-factor 0.14605092780899798 \
  --min-cluster-members 2
```

If VIDE `voidDesc_all_*` references are provided with `--vide-a` and `--vide-b`,
the command also reports reference void counts. Add `--size-bins N` to print a
shared-bin void size-function count difference.

The full scored merge path can be enabled without changing the seed and radius
model:

```bash
pinvoid paired-prototype \
  runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out \
  runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --linking-factor 0.14605092780899798 \
  --min-cluster-members 2 \
  --radius-a0 6.14700029037185 \
  --radius-alpha 0.9313222316465706 \
  --adjacency-factor 0.5240470713979322 \
  --merge-score-mode weighted \
  --merge-threshold 0.5 \
  --geom-weight 1 \
  --bridge-weight 1 \
  --compatibility-weight 0.25
```

Run a small geometry-only parameter sweep against paired VIDE references:

```bash
pinvoid paired-sweep \
  runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out \
  runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out \
  runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/sample_pinocchio_n256_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_ss1.0_z0.00_d00.out \
  runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/sample_pinocchio_n256_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --linking-factor 0.14 \
  --linking-factor 0.14605092780899798 \
  --radius-a0 6.14700029037185 \
  --radius-alpha 0.9313222316465706 \
  --adjacency-factor 0.5240470713979322
```

`--linking-length` tests fixed Mpc/h source-cluster links. `--linking-factor`
tests a source-catalog mean halo spacing factor and reports the resolved source
A/B linking lengths in the sweep table. The table reports both raw binned count
L1 and a guarded score that marks rows below `--min-predicted-fraction`.

Write per-bin `n256` finder-vs-VIDE void size-function statistics. Use fixed
paper-style bins for comparisons with PINOCCHIO/VIDE literature plots:

```bash
python scripts/compare_void_size_functions.py \
  runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out \
  runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out \
  runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/sample_pinocchio_n256_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_ss1.0_z0.00_d00.out \
  runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/sample_pinocchio_n256_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --linking-factor 0.14605092780899798 \
  --radius-a0 6.14700029037185 \
  --radius-alpha 0.9313222316465706 \
  --adjacency-factor 0.5240470713979322 \
  --bins 17 \
  --binning linear \
  --bin-min 10 \
  --bin-max 80 \
  --output-csv runs/void-statistics/n256_paper_bins_vsf.csv \
  --summary-csv runs/void-statistics/n256_paper_bins_vsf_summary.csv \
  --output-plot runs/void-statistics/n256_paper_bins_vsf.png
```

Run the geometry-only `n256` MCMC optimizer with `emcee`. This writes the posterior
chain, flattened samples, summary table, trace plot, 68/95 percent contour plot,
and a best-fit VSF regeneration command under `runs/void-statistics/`:

```bash
/home/tcastro/miniforge3/envs/voidfinder/bin/python -m pip install -e ".[dev]"

/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_vsf_mcmc.py \
  --walkers 32 \
  --steps 2000 \
  --burn-in 500 \
  --thin 10 \
  --processes 8 \
  --seed 12345 \
  --output-prefix runs/void-statistics/n256_vsf_mcmc

bash runs/void-statistics/n256_vsf_mcmc_best_fit_command.sh
```

Run the joint `n256` optimizer when calibrating both the VSF and object-level
center agreement. The scalar posterior is
`logP = log_prior + w_vsf logL_vsf + w_center logL_center`; the center term is
a robust Student-t likelihood of symmetric nearest-neighbor
`d / min(R_eff)` distances:

```bash
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_joint_calibration_mcmc.py \
  --walkers 32 \
  --steps 2000 \
  --burn-in 500 \
  --thin 10 \
  --processes 8 \
  --seed 12345 \
  --vsf-weight 1 \
  --center-weight 1 \
  --center-sigma 1 \
  --center-nu 3 \
  --output-prefix runs/void-statistics/n256_joint_mcmc

bash runs/void-statistics/n256_joint_mcmc_best_fit_command.sh
```

The joint samples CSV stores separate VSF and center likelihood components plus
center-match diagnostics for each sample. Use the VSF-only optimizer above as
the baseline when measuring the effect of the center term.

Run the full scored-merge optimizer when calibrating the active eight-parameter
model:

```bash
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_full_algorithm_mcmc.py \
  --vide-variant untrimmed \
  --walkers 64 \
  --steps 3000 \
  --burn-in 750 \
  --thin 10 \
  --processes 8 \
  --seed 12345 \
  --vsf-weight 1 \
  --center-weight 1 \
  --center-sigma 1 \
  --center-nu 3

bash runs/void-statistics/n256_full_mcmc_untrimmed_best_fit_command.sh
```

This samples `linking_factor`, `radius_a0`, `radius_alpha`,
`adjacency_factor`, `merge_threshold`, `bridge_radius_factor`,
`bridge_weight`, and `compatibility_weight` with `merge_score_mode=weighted`.
The generated best-fit command refreshes the VSF comparison, center-match CSV,
and halo-slice diagnostics with the full scored-merge parameters.

By default, calibration uses the conservative VIDE `voidDesc_all` outputs
(`--vide-variant all`). To test whether our finder better reproduces
no-density-cut VIDE catalogs, switch the reference variant directly:

```bash
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_vsf_mcmc.py \
  --vide-variant untrimmed \
  --walkers 32 \
  --steps 2000 \
  --burn-in 500 \
  --thin 10 \
  --processes 8 \
  --seed 12345

/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_joint_calibration_mcmc.py \
  --vide-variant trimmed_nodencut \
  --walkers 32 \
  --steps 2000 \
  --burn-in 500 \
  --thin 10 \
  --processes 8 \
  --seed 12345 \
  --vsf-weight 1 \
  --center-weight 1 \
  --center-sigma 1 \
  --center-nu 3
```

Non-default variants write variant-suffixed MCMC products by default, for
example `n256_vsf_mcmc_untrimmed_*` or
`n256_joint_mcmc_trimmed_nodencut_*`; the full optimizer writes
`n256_full_mcmc_untrimmed_*` for `--vide-variant untrimmed`. The generated
best-fit command preserves the selected resolved VIDE paths.

The current local `n256` plot defaults use the first MCMC maximum-likelihood
row: `--linking-factor 0.14605092780899798`,
`--radius-a0 6.14700029037185`, `--radius-alpha 0.9313222316465706`,
and `--adjacency-factor 0.5240470713979322`.

Regenerate the current local ignored `n256` finder-vs-VIDE plots:

```bash
python scripts/plot_n256_void_size_function.py
python scripts/plot_n256_void_size_function.py --paper-bins
python scripts/plot_n256_void_size_function.py --vide-variant untrimmed --paper-bins
python scripts/plot_n256_void_size_function.py --vide-variant trimmed_nodencut --paper-bins
```

Plot a 2D slice of the `n256` box with finder and VIDE voids overlaid as
circles with radius `R_eff`. The default slice is a `z=128 Mpc/h` slab with
`20 Mpc/h` thickness:

```bash
python scripts/plot_n256_void_slice.py \
  --slice-axis z \
  --slice-center 128 \
  --slice-thickness 20 \
  --target both
```

This writes `runs/void-statistics/n256_void_slice_comparison.png` and a matching
CSV listing the selected object centers and radii.

For the center-matching debug pass, first write the nearest-neighbor object
diagnostics:

```bash
python scripts/match_n256_void_centers.py
```

This writes `runs/void-statistics/n256_void_center_matches.csv` and
`runs/void-statistics/n256_void_center_match_summary.csv`. The default uses
VIDE `centers_all`; rerun with `--vide-center-kind macrocenter` to test whether
the mismatch is primarily a VIDE center convention issue. Macrocenter runs write
`n256_void_macrocenter_matches.csv` and
`n256_void_macrocenter_match_summary.csv` by default.

To annotate the slice plot with nearest finder-to-VIDE center links:

```bash
python scripts/plot_n256_void_slice.py \
  --slice-axis z \
  --slice-center 128 \
  --slice-thickness 20 \
  --target both \
  --show-nearest-matches \
  --max-match-lines 12 \
  --label-count 4
```

Plot the target halo positions as a slice scatter plot and overlay void
cross-sections in two separate images, one for the finder and one for VIDE. By
default this uses target A, the latest joint MCMC summary if it exists, and
writes `n256_halo_slice_finder.png` plus `n256_halo_slice_vide.png`:

```bash
python scripts/plot_n256_halo_void_slice.py \
  --slice-axis z \
  --slice-center 128 \
  --slice-thickness 20
```

Use `--target B` for the paired target, or `--target both` to write finder and
VIDE halo-slice overlays for both targets. The default void overlay includes
voids whose 3D spheres intersect the slab and draws
`sqrt(R_eff^2 - dz_outside_slab^2)` cross-section radii; use
`--void-selection centers --circle-radius-mode reff` for the older center-only
full-`R_eff` view. Matching CSV diagnostics are written next to each PNG with
the displayed radius and distance to the slice.

The VIDE circle overlay is only a spherical-equivalent visualization of a
watershed void. To highlight the actual VIDE tracer membership for selected
voids in the slab, use:

```bash
python scripts/plot_n256_halo_void_slice.py \
  --slice-axis z \
  --slice-center 128 \
  --slice-thickness 20 \
  --vide-variant untrimmed \
  --vide-overlay both
```

Audit the sparse region near `(x, y, z) = (100, 100, 128)` against finder
voids, VIDE `R_eff` spheres, and VIDE catalog variants:

```bash
python scripts/debug_n256_vide_region.py \
  --x 100 \
  --y 100 \
  --z 128 \
  --slice-axis z \
  --slice-center 128 \
  --slice-thickness 20
```

This writes `runs/void-statistics/n256_vide_region_debug.csv` with halo-density
counts, nearest 3D sphere margins, and projected slice-coverage margins for
`voidDesc_all`, `trimmed_nodencut`, `untrimmed`, and `untrimmed_dencut` VIDE
variants when those files are present.

The Vdn/SVdW reference is documented under `Papers/`.

## Current Development Scope

The package currently contains the repository scaffold plus Milestone 2 input
design primitives and the first Milestone 3 algorithm slice:

- validated YAML run configuration models;
- a lightweight reader for ASCII PINOCCHIO halo catalogs;
- paired PINOCCHIO halo catalog loading;
- canonical halo and tracer catalog data objects;
- periodic geometry helpers;
- periodic FoF-like source clustering;
- spherical protovoid construction;
- protovoid graph merging with geometry-only and weighted scored-edge modes;
- optional source-catalog bridge-density and compatibility scoring;
- symmetric A-to-B and B-to-A paired execution;
- VIDE `voidDesc` reading and void size-function metrics;
- conversion of normalized VIDE `VoidVol` values to physical radii using the
  local `sample_info.txt` mean tracer separation;
- lightweight readers for VIDE input tracers, `vol_*.dat`, `voidZone_*.dat`,
  and `voidPart_*.dat` membership files;
- selectable VIDE catalog variants for calibration and diagnostics:
  `all`, `trimmed_nodencut`, `untrimmed`, and `untrimmed_dencut`;
- a shared spherical-equivalent radius definition,
  `R_eff = (3 V / 4 pi)^(1/3)`, for VIDE volumes and modeled final finder
  volumes used in VSF comparisons;
- a local `pinvoid paired-prototype` integration command;
- a local `pinvoid paired-sweep` geometry-only calibration scaffold;
- guarded sweep scoring and source mean-spacing linking factors;
- `n256` finder-vs-VIDE per-bin void size-function comparison scripts;
- `n256` finder-vs-VIDE slice plots for object-level visual checks;
- fixed-bin `n256` VSF and radius-summary outputs for paper-style comparisons;
- `emcee` posterior sampling and contour diagnostics for geometry-only VSF,
  geometry-only joint VSF+center, and full scored-merge `n256` calibration;
- Vdn/SVdW theoretical void size-function overlays from PINOCCHIO cosmology
  tables;
- documentation of the expected PINOCCHIO catalog columns.

PINOCCHIO execution and production workflow orchestration remain intentionally
out of scope for this stage.
