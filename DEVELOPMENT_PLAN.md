# Development Plan

This file is the working roadmap for `pinocchio_voids`. It tracks the current
stage, what has already been completed, and the next implementation tasks.

## Current Stage

We are between **Milestone 1** and **Milestone 2**:

- Milestone 1 is complete.
- Milestone 2 is in progress.
- Void finding, PINOCCHIO execution, and optimization have not started.

## Milestone 1: Repository Foundation - Done

Completed:

- Initialized Python packaging with a `src/` layout.
- Added a minimal `pinvoid` command-line interface.
- Added baseline tests and documentation.
- Created and verified a Miniforge environment named `voidfinder`.
- Kept scientific implementation out of the first scaffold.

## Milestone 2: Data Model and I/O Design - In Progress

Completed:

- Added validated YAML run configuration models.
- Documented the expected ASCII PINOCCHIO halo catalog format.
- Added a lightweight reader for PINOCCHIO `*.catalog.out` halo catalogs.
- Added small synthetic catalog and configuration fixtures.
- Added tests for package import, configuration loading, and PINOCCHIO catalog reading.
- Added `pinvoid validate-config` for validating run configuration files.

Remaining before Milestone 3:

- Define a canonical internal catalog representation for halos and tracers.
- Add conversion utilities from PINOCCHIO halo catalogs to that internal representation.
- Document coordinate, unit, and periodic-box conventions in one place.
- Add tests for shape validation, unit metadata, and periodic wrapping behavior.
- Optionally add non-committed local checks against low-resolution PINOCCHIO outputs under `runs/`.

## Milestone 3: Void-Finder Prototype - Not Started

Planned:

- Implement the first clear, testable void-finding prototype.
- Start with controlled synthetic catalogs, not full simulation outputs.
- Add numerical tests for simple known geometries.
- Document algorithm assumptions, units, and coordinate conventions.

Entry criteria:

- Milestone 2 canonical data objects are stable.
- PINOCCHIO halo-to-tracer conversion is tested.
- No execution orchestration is mixed into the scientific algorithm.

## Milestone 4: PINOCCHIO Workflow Integration - Not Started

Planned:

- Add interfaces for consuming existing PINOCCHIO outputs.
- Keep PINOCCHIO execution orchestration separate from scientific analysis.
- Add reproducible local workflow examples.
- Treat generated low-resolution run products as ignored development artifacts.

## Milestone 5: Validation and Optimization - Not Started

Planned:

- Compare package outputs against reference cases, including VIDE where appropriate.
- Add validation plots and summary statistics.
- Add performance profiling after the baseline method is correct.
- Introduce optimization only after correctness and validation are established.

## Next Tasks

1. Commit the current Milestone 2 foundation.
   - Include package, docs, fixtures, tests, README, and `.gitignore` updates.
   - Keep external dependency trees and generated run products untracked.
   - Put validation plotting utilities in a separate commit if they are kept.

2. Finish the Milestone 2 canonical data layer.
   - Add immutable `HaloCatalog` and/or `TracerCatalog` data objects.
   - Store IDs, masses, positions, velocities, particle counts, box size, and unit metadata.
   - Validate array shapes and consistent row counts.

3. Add PINOCCHIO-to-canonical conversion utilities.
   - Convert the current ASCII reader output into the canonical data representation.
   - Use final positions as the default analysis positions.
   - Keep conversion separate from VIDE formatting and void finding.

4. Expand documentation of scientific conventions.
   - Positions: comoving `Mpc/h`.
   - Masses: `Msun/h`.
   - Velocities: `km/s`.
   - Periodic boxes: wrap final positions into `[0, box_size)`.

5. Start Milestone 3 only after the above tasks pass tests.
   - Implement the first prototype against synthetic data.
   - Defer PINOCCHIO execution, VIDE comparison, and optimization.

## Verification Commands

Use the named Miniforge environment:

```bash
/home/tcastro/miniforge3/envs/voidfinder/bin/python -m pytest
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid smoke-test
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid validate-config tests/fixtures/run_config_small.yaml
```
