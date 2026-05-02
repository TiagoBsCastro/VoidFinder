# PINOCCHIO Voids

`pinocchio_voids` is a scientific Python package under development for
identifying cosmic voids from PINOCCHIO-based structure formation products.

The package now contains the repository scaffold, PINOCCHIO catalog I/O, and the
first geometry-only paired-halo void-finder prototype. The current prototype is
intended for Milestone 3 development and is not yet calibrated against VIDE.

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
- a local `pinvoid paired-prototype` integration command;
- documentation of the expected PINOCCHIO catalog columns.

PINOCCHIO execution, calibrated VIDE comparison, bridge-density scoring, and
optimization remain intentionally out of scope for this stage.
