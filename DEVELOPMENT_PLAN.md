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
- Geometry-only calibration scaffolding has started.
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

Planned:

- Inspect and refine the geometry-only prototype using
  `runs/pinocchio-lowres/n032` and `runs/pinocchio-lowres/n032_paired`.
- Validate the same pipeline on the `n064`, `n128`, and `n256` paired catalogs.
- Use `voidDesc_all_*` outputs under `runs/vide-lowres/` as reference catalogs
  for evaluation once the geometry-only prototype is inspectable.
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
- Add a non-degenerate geometry-only score guard before trusting ranked sweeps.
- Inspect whether `linking_length` should be parameterized relative to source
  halo mean spacing or another resolution-aware scale.
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

1. Fix the geometry-only sweep score so degenerate underprediction does not rank
   as the best calibration region.
   - Penalize or filter directions where predicted void counts fall below a
     minimum fraction of the VIDE count.
   - Keep reporting raw binned count L1 for transparency.

2. Inspect resolution-aware clustering parameters.
   - Compare fixed-Mpc `linking_length` against a linking length expressed as a
     factor of the source halo mean separation.
   - Re-run focused sweeps on `n032`, `n064`, `n128`, and `n256`.
   - Keep broad sweeps and optimization for a later milestone.

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
```
