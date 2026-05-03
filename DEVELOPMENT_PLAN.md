# Development Plan

This file is the working roadmap for `pinocchio_voids`. It tracks the current
stage, what has already been completed, and the next implementation tasks.

## Current Stage

We have started **Milestone 3** with real low-resolution paired PINOCCHIO
catalogs:

- Milestone 1 is complete.
- Milestone 2 is complete.
- `ALGORITHM.md` is the authoritative algorithm specification for the next
  implementation phase.
- Low-resolution paired PINOCCHIO catalogs exist locally for `n032`, `n064`,
  `n128`, and `n256` under `runs/pinocchio-lowres/`.
- Matching VIDE reference outputs exist locally under `runs/vide-lowres/` and
  should be used as calibration and evaluation targets.
- Void size-function development is now focused on `n128` and `n256`; `n032`
  and `n064` remain useful only for quick smoke tests and historical checks.
- The VIDE radius scale has been corrected: local `voidDesc` `VoidVol` values
  are normalized Voronoi volumes, and the reader now converts them with the
  `sample_info.txt` mean tracer separation. The corrected `n128`/`n256` VIDE
  radii are on the expected `~10-70 Mpc/h` scale.
- The effective-radius audit found that VSF-facing radii are consistently
  spherical-equivalent radii, `R_eff = (3 V / 4 pi)^(1/3)`. VIDE uses corrected
  void volumes; the finder uses modeled final-void volumes from merged
  protovoids. Source-cluster `effective_radius_mpc_h` is only an internal RMS
  cluster-size diagnostic and is not used in VSF comparisons.
- The initial geometry-only paired-halo void-finding prototype is implemented.
- Geometry-only calibration scaffolding now includes guarded sweep scoring and
  source mean-spacing linking factors.
- Bridge-density scoring, PINOCCHIO execution orchestration, and optimization
  have not started.

## Milestone 1: Repository Foundation - Done

Completed:

- Initialized Python packaging with a `src/` layout.
- Added a minimal `pinvoid` command-line interface.
- Added baseline tests and documentation.
- Created and verified a Miniforge environment named `voidfinder`.
- Kept scientific implementation out of the first scaffold.

## Milestone 2: Data Model and I/O Design - Done

Completed:

- Added validated YAML run configuration models.
- Documented the expected ASCII PINOCCHIO halo catalog format.
- Added a lightweight reader for PINOCCHIO `*.catalog.out` halo catalogs.
- Added small synthetic catalog and configuration fixtures.
- Added tests for package import, configuration loading, and PINOCCHIO catalog reading.
- Added `pinvoid validate-config` for validating run configuration files.
- Added canonical internal halo and tracer catalog representations.
- Added conversion utilities from PINOCCHIO halo catalogs to canonical catalogs.
- Documented coordinate, unit, and periodic-box conventions.
- Added tests for shape validation, unit metadata, read-only arrays, and periodic wrapping.

## Milestone 3: Paired-Halo Void-Finder Prototype - In Progress

Completed so far:

- Added paired PINOCCHIO catalog loading into canonical `HaloCatalog` objects.
- Added minimum-image distance, displacement, and periodic center-of-mass
  helpers.
- Added periodic FoF-like source clustering with `linking_length`,
  `min_cluster_members`, and `min_cluster_mass` controls.
- Added source-cluster summaries with member indices, total mass,
  mass-weighted center, richness, and effective radius.
- Added spherical protovoid construction with
  `R_proto = a0 R_L(M)^alpha`.
- Added geometry-only protovoid adjacency and connected-component merging.
- Added final void catalog fields for center, effective radius, member
  protovoids, source-cluster IDs, and total source mass.
- Added symmetric A-to-B and B-to-A execution.
- Added a VIDE `voidDesc` reader and first void size-function metric.
- Added `pinvoid paired-prototype` for local ignored paired-run checks.
- Added `pinvoid paired-sweep` for deterministic geometry-only parameter
  inspection against paired VIDE references.
- Added a non-degenerate sweep score guard so rows with too few predicted voids
  do not outrank viable calibration regions solely through low raw L1.
- Added `--linking-factor` support for linking lengths expressed as a factor of
  source halo mean spacing, with separate resolved source-A/source-B lengths.
- Added `scripts/compare_void_size_functions.py` to write per-bin finder-vs-VIDE
  void size-function CSVs and optional plots for paired runs.
- Added a Vdn/SVdW theoretical void size-function curve from local PINOCCHIO
  cosmology tables, with configurable linear barriers and non-linear void
  density mapping.
- Added `scripts/plot_n128_n256_void_size_functions.py` to regenerate the
  ignored `n128` and `n256` finder-vs-VIDE comparison plots.
- Added `scripts/calibrate_radius_scale.py` for cached paper-bin radius-scale
  calibration against VIDE references.
- Added a `Papers/` reference note for Jennings, Li & Hu 2013, the source of
  the Vdn/SVdW theoretical size-function baseline.
- Fixed VIDE `voidDesc` radius handling: `VoidVol` is a normalized Voronoi
  volume in the local VIDE outputs, so the reader now uses the mean tracer
  separation from `sample_info.txt` to convert volumes to `(Mpc/h)^3`.
- Added fixed radius-bin support for finder-vs-VIDE VSF comparisons, including
  paper-like `17` linear bins from `10` to `80 Mpc/h`.
- Made theoretical VSF overlays optional in the n128/n256 plot driver; default
  plots now show only finder and VIDE, and `--include-theory` adds Vdn/SVdW.
- Added radius-summary CSV output for finder and VIDE catalogs, including
  min/percentile/max radii and counts in the `10-80 Mpc/h` range.
- Centralized the spherical-equivalent radius conversion so VIDE radii, finder
  merged radii, and tests use the same `R_eff` definition.
- Regenerated local ignored `n128` and `n256` finder-vs-VIDE diagnostics in
  both shared automatic bins and paper-like `10-80 Mpc/h` bins.
- Refreshed the canonical ignored VSF plot products in default no-theory mode:
  `n128_finder_vide_vsf`, `n256_finder_vide_vsf`, `n128_paper_bins_vsf`, and
  `n256_paper_bins_vsf`. These CSVs contain only `finder` and `vide` rows; the
  separate `*_theory_vsf` products are generated only with `--include-theory`.
- The first corrected paper-bin diagnostics showed that the old parameters put
  finder voids below the `10-80 Mpc/h` paper range; this was a calibration
  issue, not a plotting issue.
- A focused cached radius-scale calibration found non-degenerate `n128` and
  `n256` rows with visible paper-bin finder VSFs:
  - `n128`: linking factor `0.15`, `radius_a0=5`, `radius_alpha=1`,
    `adjacency_factor=0.5`; counts `45/50` and `48/65`, medians
    `29.9/30.1` and `27.8/34.2 Mpc/h`.
  - `n256`: linking factor `0.12`, `radius_a0=6`, `radius_alpha=1`,
    `adjacency_factor=0.35`; counts `78/121` and `95/124`, medians
    `21.9/23.5` and `22.0/23.7 Mpc/h`.
- Added unit tests and tiny paired PINOCCHIO fixtures.
- Verified a local ignored `n032` paired run can execute without writing
  generated outputs.
- Verified the same geometry-only prototype executes on local `n064`, `n128`,
  and `n256` paired catalogs.
- Initial geometry-only counts are substantially above VIDE counts on several
  low-resolution pairs, so parameter inspection and calibration should happen
  before bridge-density scoring is added.
- A small local `n032` geometry sweep runs successfully and shows that lowering
  the linking length from `8` to `6` reduces the raw predicted-count excess in
  the tested parameter grid.
- A broader `n032` sweep over `linking_length`, `radius_a0`, `radius_alpha`, and
  `adjacency_factor` exposed a metric limitation: pure binned count L1 can favor
  degenerate underprediction, including zero predicted voids in one direction.
- Cross-checking low-link parameter regions showed resolution dependence:
  `n064` count totals can be close to VIDE, while the same style of fixed-Mpc
  linking overpredicts strongly on `n128` and `n256`.
- Focused local mean-spacing-factor sweeps were run on `n032`, `n064`, `n128`,
  and `n256`. The top guarded rows avoided the earlier zero-void degeneracy.
  The sampled grids favored factor `0.15` for `n032`, `n064`, and `n128`, while
  `n256` favored a smaller factor near `0.10`. These rows remain useful as
  count-oriented historical diagnostics, but their binned VSF residuals must be
  regenerated after the VIDE radius-unit correction.
  - `n032`: factor `0.15`, resolved links `5.51/5.51`, counts `3/9` and `4/4`.
  - `n064`: factor `0.15`, resolved links `3.33/3.35`, counts `6/18` and `13/22`.
  - `n128`: factor `0.15`, resolved links `2.13/2.14`, counts `58/50` and `68/65`.
  - `n256`: factor `0.10`, resolved links `0.99/1.00`, counts `40/121` and `60/124`.

Planned:

- Refine the corrected radius-scale calibration. The current rows now put the
  finder into the correct `10-80 Mpc/h` range, but count and shape residuals
  remain, especially for `n256`.
- Audit the comparison with Lepinzan et al. 2025 before interpreting amplitude
  differences: match binning, volume normalization, halo mass cuts, tracer
  number density, and VIDE settings as closely as the local catalogs allow.
- Continue debugging theory and finder amplitudes after the empirical radius
  scale is stable.
- Use `voidDesc_all_*` outputs under `runs/vide-lowres/` as reference catalogs
  for evaluation while keeping generated run products ignored.
- Keep pure synthetic catalogs and `n032`/`n064` runs as test/smoke support,
  not the main development path.

Entry criteria:

- Milestone 2 canonical data objects are stable.
- PINOCCHIO halo-to-`HaloCatalog` conversion is tested.
- `ALGORITHM.md` has been reviewed as the source of algorithm assumptions,
  phases, parameters, units, and coordinate conventions.
- Local generated `runs/` data remains ignored; committed tests use tiny
  fixtures only.
- No execution orchestration is mixed into the scientific algorithm.

Remaining Phase 1 scope:

- Inspect geometry-only output quality before adding bridge-density scoring.
- Refine source mean-spacing factor grids and decide whether one factor can span
  all tested low-resolution catalogs.
- Inspect whether an additional resolution-aware scale is needed for `n256`.
- Keep calibration as small deterministic grids, not a broad optimizer.

## Milestone 4: VIDE Evaluation and Workflow Integration - Not Started

Planned:

- Add interfaces for consuming existing PINOCCHIO and VIDE outputs.
- Keep PINOCCHIO execution orchestration separate from scientific analysis.
- Add reproducible local workflow examples.
- Treat generated low-resolution run products as ignored development artifacts.
- Add evaluation hooks against VIDE reference catalogs.
  - Read `voidDesc_all_*` files as reference void catalogs.
  - Start with void size function comparison.
  - Keep calibration sweeps and optimization deferred until the full Phase 1
    pipeline runs.

## Milestone 5: Validation and Optimization - Not Started

Planned:

- Compare package outputs against reference cases, including VIDE where appropriate.
- Add validation plots and summary statistics.
- Add performance profiling after the baseline method is correct.
- Introduce optimization only after correctness and validation are established.

## Next Tasks

1. Refine the radius-scale calibration around the current best rows.
   - `n128`: densify around linking factor `0.15`, `radius_a0=5`,
     `adjacency_factor=0.35-0.5`.
   - `n256`: densify around linking factor `0.12`, `radius_a0=6`,
     `adjacency_factor=0.2-0.5`.
   - Include `radius_alpha` values near `1.0` and inspect whether separate
     `n128`/`n256` parameters are required.
   - Compare count totals, paper-bin VSF residuals, and radius percentiles in
     default no-theory plots before adding optional theory overlays.

2. Build a paper-comparison audit for the Lepinzan et al. PINOCCHIO/VIDE VSF.
   - Use the same `10-80 Mpc/h` linear binning for local plots.
   - Quantify the volume scaling between the local `256 Mpc/h` box and the
     paper's larger box before comparing counts or densities.
   - Check whether the local VIDE catalogs used all halos while the paper uses
     a `10^13 Msun/h` halo mass cut and number-density matching.
   - Plan a local VIDE rerun with paper-like tracer cuts if the current
     reference catalogs cannot be made comparable by post-processing.

3. Debug the Vdn/SVdW amplitude mismatch.
   - Compare theory/VIDE and theory/finder ratios per radius bin using corrected
     VIDE physical radii.
   - Keep theory out of default VSF plots so empirical calibration plots remain
     readable; use `--include-theory` for dedicated theory diagnostics.
   - Convert theory densities back to implied per-bin counts using
     `box_volume * dlnR`.
   - Print intermediate theory factors: Eulerian radius, Lagrangian radius,
     sigma, `dln sigma^-1 / dln R`, first-crossing fraction, denominator
     volume, and final `dn/dlnR`.
   - Test unit-conversion and normalization toggles before changing defaults.

4. Add a repeatable plot-refresh step to every new calibration candidate.
   - Run both default and paper-bin `scripts/plot_n128_n256_void_size_functions.py`
     outputs after changing finder parameters.
   - Use no-theory plots as the default empirical comparison and reserve
     `--include-theory` outputs for dedicated theory diagnostics.
   - Preserve generated products under ignored `runs/` paths.
   - Record only stable conclusions and commands in committed docs.

5. Add bridge-density scoring only after the `n128`/`n256` geometry-only
   baseline has a defensible parameter region.
   - Use source-catalog halo density between adjacent source clusters.
   - Keep the score modular so it can be disabled for baseline comparisons.

## Test Plan

- Keep existing package tests passing.
- Added unit tests for periodic distance and displacement.
- Added unit tests for periodic FoF clustering across box boundaries.
- Added unit tests for cluster mass, richness, and center summaries.
- Added unit tests for protovoid radius mapping.
- Added unit tests for symmetric A-to-B and B-to-A execution on tiny paired
  fixtures.
- Added unit tests for graph connected-component merging.
- Added unit tests for VIDE `voidDesc` parsing and void size-function metrics.
- Added CLI test coverage for `pinvoid paired-prototype`.
- Added unit and CLI test coverage for geometry-only parameter sweeps.
- Added unit and CLI test coverage for guarded calibration scores and
  mean-spacing linking factors.
- Added fixture-backed tests for finder-vs-VIDE void size-function CSV output.
- Added unit tests for PINOCCHIO cosmology parsing and Vdn/SVdW theory curves.
- Added fixture-backed tests for theory rows in finder-vs-VIDE VSF CSV output.
- Added tests ensuring the n128/n256 plot driver omits theory by default and
  includes it only with `--include-theory`.
- Added tests for Vdn/SVdW debug decomposition and theory-implied per-bin counts.
- Added tests for converting normalized VIDE `VoidVol` values into physical
  radii using `sample_info.txt`.
- Added tests for fixed linear VSF bins and finder/VIDE radius-summary CSV
  output.
- Added tests for paper-bin radius-scale scoring, zero-in-bin degeneracy,
  cached calibration parity, and the calibration diagnostic script.
- Add ignored local integration checks using:
  - `runs/pinocchio-lowres/n128`
  - `runs/pinocchio-lowres/n128_paired`
  - `runs/pinocchio-lowres/n256`
  - `runs/pinocchio-lowres/n256_paired`
  - matching `runs/vide-lowres/` references and cosmology files

## Local Reference Data

- PINOCCHIO paired halo catalogs are available under
  `runs/pinocchio-lowres/{n032,n064,n128,n256}` and the matching
  `_paired` directories.
- Active size-function plots and calibration should use `n128` and `n256`.
- VIDE reference outputs for `n032`, `n064`, and `n256` are available under
  `runs/vide-lowres/` with matching `_paired` directories.
- VIDE reference outputs for `n128` are currently nested under
  `runs/vide-lowres/n128_pair/{n128,n128_paired}`.
- The Vdn/SVdW theory overlay uses the matching
  `pinocchio.lowres_n128.cosmology.out` and
  `pinocchio.lowres_n256.cosmology.out` files and is opt-in for generated VSF
  plots.
- The theoretical VSF reference is documented under `Papers/`.
- Paper-style comparison plots should use `17` linear radius bins from `10` to
  `80 Mpc/h` and should be read alongside the generated radius-summary CSVs.
- Generated `runs/` data must remain ignored. Commit only tiny deterministic
  fixtures needed by tests.

## Verification Commands

Use the named Miniforge environment:

```bash
/home/tcastro/miniforge3/envs/voidfinder/bin/python -m pytest
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid smoke-test
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid validate-config tests/fixtures/run_config_small.yaml
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid paired-sweep --help
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/calibrate_radius_scale.py --help
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/plot_n128_n256_void_size_functions.py --only n128
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/plot_n128_n256_void_size_functions.py --paper-bins --only n256
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/plot_n128_n256_void_size_functions.py --paper-bins --include-theory --only n256
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/debug_theory_vsf.py runs/void-statistics/n128_finder_vide_theory_vsf.csv --box-size 256 --cosmology-file runs/pinocchio-lowres/n128/pinocchio.lowres_n128.cosmology.out
```
