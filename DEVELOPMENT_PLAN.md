# Development Plan

## Milestone 1: Repository Foundation

- Initialize Python packaging with a `src/` layout.
- Add a minimal command-line interface.
- Add baseline tests and documentation.
- Keep scientific implementation out of the first scaffold.

## Milestone 2: Data Model and I/O Design

- Define validated configuration schemas.
- Specify expected PINOCCHIO catalog inputs.
- Add lightweight readers for supported file formats.
- Add fixtures with small synthetic data only.

## Milestone 3: Void-Finder Prototype

- Implement the first clear, testable void-finding algorithm.
- Add numerical tests using controlled synthetic catalogs.
- Document assumptions, units, and coordinate conventions.

## Milestone 4: PINOCCHIO Workflow Integration

- Add interfaces for consuming PINOCCHIO outputs.
- Keep execution orchestration separate from scientific analysis.
- Add reproducible workflow examples.

## Milestone 5: Validation and Optimization

- Compare results against reference cases.
- Add performance profiling.
- Introduce optimization only after the baseline method is correct and tested.
