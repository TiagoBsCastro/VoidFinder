# PINOCCHIO Voids

`pinocchio_voids` is a scientific Python package under development for identifying cosmic voids from PINOCCHIO-based structure formation products.

The initial goal is to provide a clean, tested package scaffold before adding the scientific void-finding implementation. Future work will define data interfaces, catalog validation, void-finding algorithms, and reproducible command-line workflows.

This repository intentionally does not yet implement:

- void finding;
- PINOCCHIO execution;
- parameter optimization.

## Development Install

```bash
python -m pip install -e ".[dev]"
```

## Smoke Test

```bash
pinvoid smoke-test
```

## Current Development Scope

The package currently contains the repository scaffold plus Milestone 2 input
design primitives:

- validated YAML run configuration models;
- a lightweight reader for ASCII PINOCCHIO halo catalogs;
- documentation of the expected PINOCCHIO catalog columns.

Void finding, PINOCCHIO execution, and optimization remain intentionally out of
scope for this stage.
