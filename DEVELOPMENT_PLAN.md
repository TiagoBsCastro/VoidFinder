# Development Plan

This file is the working roadmap for `pinocchio_voids`. It tracks the current
stage, what has already been completed, and the next implementation tasks.

## Current Stage

We are ready to start **Milestone 3**:

- Milestone 1 is complete.
- Milestone 2 is complete.
- Void finding, PINOCCHIO execution, and optimization have not started.

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

1. Start Milestone 3 with a simple synthetic-catalog void-finder prototype.
   - Use the canonical `TracerCatalog` as input.
   - Keep the first algorithm small, deterministic, and heavily tested.
   - Do not add PINOCCHIO execution, VIDE comparison, or optimization.

2. Define synthetic validation cases for Milestone 3.
   - Uniform tracer field with no obvious central void.
   - Controlled spherical underdensity.
   - Periodic-boundary case.

3. Keep validation tooling separate.
   - Commit plotting and VIDE-comparison utilities separately from core science code.
   - Keep generated files under `runs/` ignored.

## Verification Commands

Use the named Miniforge environment:

```bash
/home/tcastro/miniforge3/envs/voidfinder/bin/python -m pytest
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid smoke-test
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid validate-config tests/fixtures/run_config_small.yaml
```
