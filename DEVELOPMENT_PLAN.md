# Development Plan

This file is the working roadmap for `pinocchio_voids`. It tracks the current
stage, what has already been completed, and the next implementation tasks.

## Current Stage

We are ready to start **Milestone 3** with real low-resolution paired
PINOCCHIO catalogs:

- Milestone 1 is complete.
- Milestone 2 is complete.
- `ALGORITHM.md` is the authoritative algorithm specification for the next
  implementation phase.
- Low-resolution paired PINOCCHIO catalogs exist locally for `n032`, `n064`,
  `n128`, and `n256` under `runs/pinocchio-lowres/`.
- Matching VIDE reference outputs exist locally under `runs/vide-lowres/` and
  should be used as calibration and evaluation targets.
- Void finding, calibration, PINOCCHIO execution orchestration, and optimization
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

## Milestone 3: Paired-Halo Void-Finder Prototype - Ready to Begin

Planned:

- Implement the first clear, testable prototype of the paired-halo algorithm in
  `ALGORITHM.md`.
- Use paired canonical `HaloCatalog` inputs built directly from PINOCCHIO final
  halo positions and masses.
- Start with `runs/pinocchio-lowres/n032` and
  `runs/pinocchio-lowres/n032_paired` for fast iteration.
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

Phase 1 implementation scope:

- Add paired-run data loading.
  - Take two PINOCCHIO catalog paths plus box size.
  - Return canonical `HaloCatalog` objects for catalogs A and B.
  - Use final halo positions and masses.
- Add periodic geometry utilities.
  - Minimum-image displacement and distance.
  - Periodic center-of-mass helper for cluster centers.
- Implement periodic FoF-like halo clustering.
  - Configure with `linking_length`, `min_cluster_members`, and
    `min_cluster_mass`.
  - Return source-cluster catalogs with member indices, total mass,
    mass-weighted center, richness, and effective radius.
- Implement spherical protovoid construction.
  - Map source clusters in A into protovoids for B.
  - Map source clusters in B into protovoids for A.
  - Use `R_proto = a0 R_L(M)^alpha` with `radius_a0`, `radius_alpha`, and
    `reference_rho_bar`.
- Implement the first graph merge prototype.
  - Build adjacency from `d_ij < adjacency_factor * (R_i + R_j)`.
  - Start with a geometric score only.
  - Merge connected components into final voids.
  - Return final void fields: center, effective radius, member protovoids,
    source-cluster IDs, and total source mass.
  - Defer bridge-density scoring until the geometry-only output is inspectable.
- Run the same code path symmetrically for A-to-B and B-to-A.

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

1. Add paired-run data loading.
   - Read two PINOCCHIO halo catalog paths and a box size.
   - Return canonical `HaloCatalog` objects for A and B.
   - Add tiny committed paired fixtures for tests.

2. Add periodic geometry utilities.
   - Minimum-image displacement and distance.
   - Periodic center-of-mass helper.
   - Tests for boundary-crossing pairs.

3. Implement Phase 1 source clustering.
   - Periodic FoF-like clustering with the initial free parameters from
     `ALGORITHM.md`.
   - Source-cluster summaries: member indices, total mass, mass-weighted center,
     richness, and effective radius.

4. Implement spherical protovoids.
   - Convert source clusters in A into protovoids for B.
   - Convert source clusters in B into protovoids for A.
   - Test mass-to-radius mapping and A/B symmetry.

5. Implement geometry-only graph merging.
   - Build adjacency with `d_ij < adjacency_factor * (R_i + R_j)`.
   - Use connected components for final voids.
   - Defer bridge-density score until geometry-only output can be inspected.

6. Add first VIDE evaluation hooks.
   - Read `voidDesc_all_*` reference catalogs.
   - Compare predicted and reference void size functions.
   - Keep calibration sweeps and optimization for a later milestone.

## Test Plan

- Keep existing package tests passing.
- Add unit tests for periodic distance and displacement.
- Add unit tests for periodic FoF clustering across box boundaries.
- Add unit tests for cluster mass, richness, and center summaries.
- Add unit tests for protovoid radius mapping.
- Add unit tests for symmetric A-to-B and B-to-A execution on tiny paired
  fixtures.
- Add unit tests for graph connected-component merging.
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
