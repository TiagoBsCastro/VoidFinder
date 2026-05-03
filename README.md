# PINOCCHIO Voids

`pinocchio_voids` is a scientific Python package under development for
identifying cosmic voids from PINOCCHIO-based structure formation products.

The package now contains the repository scaffold, PINOCCHIO catalog I/O, and the
first geometry-only paired-halo void-finder prototype. The current prototype is
intended for Milestone 3 development and is not yet calibrated against VIDE.
Current sweep scores are guarded against degenerate underprediction but remain
exploratory and should not be treated as final calibrated parameters.

This repository intentionally does not yet implement:

- PINOCCHIO execution;
- calibrated VIDE evaluation;
- parameter optimization.

## Development Install

```bash
python -m pip install -e ".[dev]"
```

## Smoke Test

```bash
pinvoid smoke-test
```

## Milestone 3 Prototype

Run the geometry-only paired-halo prototype on two existing PINOCCHIO halo
catalogs:

```bash
pinvoid paired-prototype \
  runs/pinocchio-lowres/n032/pinocchio.0.0000.lowres_n032.catalog.out \
  runs/pinocchio-lowres/n032_paired/pinocchio.0.0000.lowres_n032_paired.catalog.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --linking-length 8 \
  --min-cluster-members 2
```

If VIDE `voidDesc_all_*` references are provided with `--vide-a` and `--vide-b`,
the command also reports reference void counts. Add `--size-bins N` to print a
shared-bin void size-function count difference.

Run a small geometry-only parameter sweep against paired VIDE references:

```bash
pinvoid paired-sweep \
  runs/pinocchio-lowres/n032/pinocchio.0.0000.lowres_n032.catalog.out \
  runs/pinocchio-lowres/n032_paired/pinocchio.0.0000.lowres_n032_paired.catalog.out \
  runs/vide-lowres/n032/outputs/pinocchio_n032_ss1.0/sample_pinocchio_n032_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n032_ss1.0_z0.00_d00.out \
  runs/vide-lowres/n032_paired/outputs/pinocchio_n032_paired_ss1.0/sample_pinocchio_n032_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n032_paired_ss1.0_z0.00_d00.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --linking-factor 0.15 \
  --linking-factor 0.20 \
  --radius-a0 0.7 \
  --radius-a0 1.0 \
  --adjacency-factor 0.8 \
  --adjacency-factor 1.2
```

`--linking-length` tests fixed Mpc/h source-cluster links. `--linking-factor`
tests a source-catalog mean halo spacing factor and reports the resolved source
A/B linking lengths in the sweep table. The table reports both raw binned count
L1 and a guarded score that marks rows below `--min-predicted-fraction`.

Write per-bin finder-vs-VIDE void size-function statistics. The active
size-function comparisons should use `n128` and `n256`; lower resolutions are
kept for quick smoke checks only. Add `--summary-csv` to write finder/VIDE
radius percentiles next to the VSF table. The Vdn/SVdW theoretical curve is
optional; add `--theory vdn-svdw` and the matching PINOCCHIO cosmology file only
when an analytic overlay is desired:

```bash
python scripts/compare_void_size_functions.py \
  runs/pinocchio-lowres/n128/pinocchio.0.0000.lowres_n128.catalog.out \
  runs/pinocchio-lowres/n128_paired/pinocchio.0.0000.lowres_n128_paired.catalog.out \
  runs/vide-lowres/n128_pair/n128/outputs/pinocchio_n128_ss1.0/sample_pinocchio_n128_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n128_ss1.0_z0.00_d00.out \
  runs/vide-lowres/n128_pair/n128_paired/outputs/pinocchio_n128_paired_ss1.0/sample_pinocchio_n128_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n128_paired_ss1.0_z0.00_d00.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --linking-factor 0.15 \
  --bins 12 \
  --output-csv runs/void-statistics/n128_vsf.csv \
  --summary-csv runs/void-statistics/n128_vsf_summary.csv \
  --output-plot runs/void-statistics/n128_vsf.png
```

Use fixed paper-style bins for comparisons with PINOCCHIO/VIDE literature plots:

```bash
python scripts/compare_void_size_functions.py \
  runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out \
  runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out \
  runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/sample_pinocchio_n256_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_ss1.0_z0.00_d00.out \
  runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/sample_pinocchio_n256_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --linking-factor 0.13 \
  --radius-a0 6.5 \
  --radius-alpha 1 \
  --adjacency-factor 0.30 \
  --bins 17 \
  --binning linear \
  --bin-min 10 \
  --bin-max 80 \
  --output-csv runs/void-statistics/n256_paper_bins_vsf.csv \
  --summary-csv runs/void-statistics/n256_paper_bins_vsf_summary.csv \
  --output-plot runs/void-statistics/n256_paper_bins_vsf.png
```

Run a cached radius-scale calibration diagnostic when the paper-bin finder curve
is missing or visibly mis-scaled:

```bash
python scripts/calibrate_radius_scale.py \
  runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out \
  runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out \
  runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/sample_pinocchio_n256_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_ss1.0_z0.00_d00.out \
  runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/sample_pinocchio_n256_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --min-cluster-members 2 \
  --min-cluster-mass 0 \
  --output-csv runs/void-statistics/n256_radius_scale_calibration.csv
```

Search the same parameter surface with the VSF likelihood objective. Rows are
ranked by maximum Poisson log-likelihood of the fixed-bin VIDE counts, with
non-degenerate finder rows kept ahead of zero/near-zero underpredictions:

```bash
python scripts/calibrate_vsf_likelihood.py \
  runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out \
  runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out \
  runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/sample_pinocchio_n256_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_ss1.0_z0.00_d00.out \
  runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/sample_pinocchio_n256_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --linking-factor 0.12 \
  --linking-factor 0.13 \
  --linking-factor 0.14 \
  --min-cluster-members 2 \
  --min-cluster-mass 0 \
  --radius-a0 5.5 \
  --radius-a0 6.0 \
  --radius-a0 6.5 \
  --radius-alpha 1.0 \
  --radius-alpha 1.05 \
  --adjacency-factor 0.3 \
  --adjacency-factor 0.4 \
  --output-csv runs/void-statistics/n256_vsf_likelihood_calibration.csv
```

Regenerate the local ignored `n128` and `n256` finder-vs-VIDE plots. Add
`--include-theory` only when the Vdn/SVdW curve should be included:

```bash
python scripts/plot_n128_n256_void_size_functions.py
python scripts/plot_n128_n256_void_size_functions.py --paper-bins
python scripts/plot_n128_n256_void_size_functions.py --paper-bins --include-theory
```

Debug the theory amplitude against an existing comparison CSV:

```bash
python scripts/debug_theory_vsf.py \
  runs/void-statistics/n128_finder_vide_theory_vsf.csv \
  --box-size 256 \
  --cosmology-file runs/pinocchio-lowres/n128/pinocchio.lowres_n128.cosmology.out
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
- finder-vs-VIDE per-bin void size-function comparison scripts;
- fixed-bin VSF and radius-summary outputs for paper-style comparisons;
- cached radius-scale calibration diagnostics for `n128`/`n256`;
- cached Poisson likelihood searches for matching finder VSF counts to VIDE;
- Vdn/SVdW theoretical void size-function overlays from PINOCCHIO cosmology
  tables;
- documentation of the expected PINOCCHIO catalog columns.

PINOCCHIO execution, calibrated VIDE comparison, bridge-density scoring, and
optimization remain intentionally out of scope for this stage.
