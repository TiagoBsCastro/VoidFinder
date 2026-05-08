# PINOCCHIO Voids

This repository is a compact research reference for an exploratory
PINOCCHIO-based paired-halo void-finder prototype. It is no longer being
developed toward production use.

## Outcome

The main result is negative but useful:

- the prototype can be tuned to reproduce a VIDE-like void size function (VSF);
- that tuning does not give convincing object-by-object agreement with VIDE
  void centers;
- using PINOCCHIO initial positions makes the current model strongly
  degenerate, producing too few voids;
- VAST/VIDE comparison experiments showed that different finder definitions can
  produce substantially different catalogs even from matched tracers.

The retained code is meant to document the tested algorithm, file conventions,
and reference diagnostics. Git history contains the removed calibration and
debugging experiments.

## Retained Workflow

Install in editable mode:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

The reference commands use ignored local `n256` products under `runs/`.
See [docs/reference_workflow.md](docs/reference_workflow.md) for the current
VSF, center-match, and halo-slice diagnostic commands.

## Retained Scripts

- `scripts/compare_void_size_functions.py`: compare the paired-halo finder VSF
  against VIDE `voidDesc` catalogs.
- `scripts/match_n256_void_centers.py`: nearest-neighbor center diagnostics
  between finder and VIDE catalogs.
- `scripts/plot_n256_void_slice.py`: 2D slice plot of finder and VIDE void
  circles.
- `scripts/plot_n256_halo_void_slice.py`: halo-background slice plots with
  finder or VIDE circles.

## Package Contents

The package keeps only the reusable reference pieces:

- PINOCCHIO and VIDE readers;
- periodic geometry helpers;
- canonical halo/tracer catalog data structures;
- the paired-halo finder prototype;
- VSF, center-match, and theory helpers used by the retained diagnostics.

The command-line app, YAML workflow config, MCMC optimizers, VAST workflow, and
one-off debug scripts were removed to keep the repository readable as a
reference artifact.

## Scientific Notes

- `ALGORITHM.md` records the tested approximate paired-halo algorithm.
- `docs/scientific_conventions.md` records units, periodic boundaries, and
  effective-radius conventions.
- `docs/pinocchio_catalog_format.md` records the supported PINOCCHIO ASCII halo
  catalog layout.
- `PROJECT_STATUS.md` summarizes the final status and cleanup rationale.
