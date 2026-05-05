# PINOCCHIO Voids

`pinocchio_voids` is a scientific Python package under development for
identifying cosmic voids from PINOCCHIO-based structure formation products.

The package now contains the repository scaffold, PINOCCHIO catalog I/O, and the
first geometry-only paired-halo void-finder prototype. The current prototype is
intended for Milestone 3 development. Current VSF calibration work is focused
on the local `n256` paired catalogs; lower-resolution `n032`, `n064`, and
`n128` runs were removed from local active products because they do not provide
a workable void size function.

This repository intentionally does not yet implement:

- PINOCCHIO execution;
- bridge-density scoring;
- production workflow orchestration.

## Development Install

```bash
python -m pip install -e ".[dev]"
```

## Smoke Test

```bash
pinvoid smoke-test
```

## Milestone 3 Prototype

Run the geometry-only paired-halo prototype on the active `n256` PINOCCHIO halo
catalog pair:

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

Run a cached likelihood-grid diagnostic only when a quick check around the
current MCMC region is useful:

```bash
python scripts/calibrate_vsf_likelihood.py \
  runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out \
  runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out \
  runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/sample_pinocchio_n256_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_ss1.0_z0.00_d00.out \
  runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/sample_pinocchio_n256_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --linking-factor 0.14 \
  --linking-factor 0.14605092780899798 \
  --min-cluster-members 2 \
  --min-cluster-mass 0 \
  --radius-a0 6.14700029037185 \
  --radius-alpha 0.9313222316465706 \
  --adjacency-factor 0.5240470713979322 \
  --output-csv runs/void-statistics/n256_vsf_likelihood_calibration.csv
```

Run the true `n256` MCMC optimizer with `emcee`. This writes the posterior
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

The current local `n256` plot defaults use the first MCMC maximum-likelihood
row: `--linking-factor 0.14605092780899798`,
`--radius-a0 6.14700029037185`, `--radius-alpha 0.9313222316465706`,
and `--adjacency-factor 0.5240470713979322`.

Regenerate the current local ignored `n256` finder-vs-VIDE plots:

```bash
python scripts/plot_n256_void_size_function.py
python scripts/plot_n256_void_size_function.py --paper-bins
```

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
- geometry-only protovoid graph merging;
- symmetric A-to-B and B-to-A paired execution;
- VIDE `voidDesc` reading and void size-function metrics;
- conversion of normalized VIDE `VoidVol` values to physical radii using the
  local `sample_info.txt` mean tracer separation;
- a shared spherical-equivalent radius definition,
  `R_eff = (3 V / 4 pi)^(1/3)`, for VIDE volumes and modeled final finder
  volumes used in VSF comparisons;
- a local `pinvoid paired-prototype` integration command;
- a local `pinvoid paired-sweep` geometry-only calibration scaffold;
- guarded sweep scoring and source mean-spacing linking factors;
- `n256` finder-vs-VIDE per-bin void size-function comparison scripts;
- fixed-bin `n256` VSF and radius-summary outputs for paper-style comparisons;
- cached Poisson likelihood searches for matching finder `n256` VSF counts to VIDE;
- `emcee` posterior sampling and contour diagnostics for `n256` VSF calibration;
- Vdn/SVdW theoretical void size-function overlays from PINOCCHIO cosmology
  tables;
- documentation of the expected PINOCCHIO catalog columns.

PINOCCHIO execution, bridge-density scoring, and production workflow
orchestration remain intentionally out of scope for this stage.
