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
