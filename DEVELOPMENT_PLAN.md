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
  `n256` favored a smaller factor near `0.10`.
  - `n032`: factor `0.15`, resolved links `5.51/5.51`, counts `3/9` and `4/4`.
  - `n064`: factor `0.15`, resolved links `3.33/3.35`, counts `6/18` and `13/22`.
  - `n128`: factor `0.15`, resolved links `2.13/2.14`, counts `58/50` and `68/65`.
  - `n256`: factor `0.10`, resolved links `0.99/1.00`, counts `40/121` and `60/124`.

Planned:

- Refine mean-spacing-factor calibration with focused grids across `n032`,
  `n064`, `n128`, and `n256`.
- Add lightweight per-bin size-function inspection for top sweep rows.
- Use `voidDesc_all_*` outputs under `runs/vide-lowres/` as reference catalogs
  for evaluation while keeping generated run products ignored.
- Keep pure synthetic catalogs as unit-test fixtures, not the main development
  path.

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

1. Refine the resolution-aware factor grid.
   - Densify factors around `0.10` through `0.18`, especially for `n256`.
   - Compare count totals and binned size-function residuals separately so the
     ranking is not driven by one diagnostic.
   - Keep broad sweeps and optimization for a later milestone.

2. Add lightweight evaluation output for inspection.
   - Export or print per-bin predicted/reference size-function counts for top
     rows.
   - Add a small helper for comparing top-row centers/radii against VIDE
     summaries without committing generated run products.

3. Add bridge-density scoring only after the geometry-only baseline has a
   defensible parameter region.
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
- Add ignored local integration checks using:
  - `runs/pinocchio-lowres/n032`
  - `runs/pinocchio-lowres/n032_paired`
  - `runs/vide-lowres/n032`
  - `runs/vide-lowres/n032_paired`

## Local Reference Data

- PINOCCHIO paired halo catalogs are available under
  `runs/pinocchio-lowres/{n032,n064,n128,n256}` and the matching
  `_paired` directories.
- VIDE reference outputs for `n032`, `n064`, and `n256` are available under
  `runs/vide-lowres/` with matching `_paired` directories.
- VIDE reference outputs for `n128` are currently nested under
  `runs/vide-lowres/n128_pair/{n128,n128_paired}`.
- Generated `runs/` data must remain ignored. Commit only tiny deterministic
  fixtures needed by tests.

## Verification Commands

Use the named Miniforge environment:

```bash
/home/tcastro/miniforge3/envs/voidfinder/bin/python -m pytest
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid smoke-test
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid validate-config tests/fixtures/run_config_small.yaml
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid paired-sweep --help
```
