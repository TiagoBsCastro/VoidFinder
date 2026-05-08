# Project Status

This repository is now a reference artifact for a sandboxed void-finder
investigation. It is not a production roadmap.

## Final Assessment

The paired-halo algorithm implemented here can be calibrated to reproduce a
VIDE-like void size function, but the resulting void centers do not reliably
match VIDE objects. Joint VSF-plus-center calibration exposed the tension
directly: improving one diagnostic did not produce a convincing solution for
the other.

Initial-position experiments were also not encouraging. With the current model
and parameter ranges, initial-position calibration collapsed into degenerate
catalogs with too few predicted voids.

External VAST.VoidFinder comparisons were useful as a cross-check, but they
reinforced that finder definitions and pruning choices dominate object-level
catalog differences. Those scripts were removed from the active tree; the
conclusion is recorded here and in Git history.

## What Remains

The retained code supports:

- reading PINOCCHIO halo catalogs and VIDE output catalogs;
- periodic geometry and catalog validation;
- the approximate paired-halo finder prototype;
- VSF comparison against VIDE;
- finder-to-VIDE center diagnostics;
- 2D slice plots over halo backgrounds.

The retained scripts are intentionally small and diagnostic-focused:

- `scripts/compare_void_size_functions.py`
- `scripts/match_n256_void_centers.py`
- `scripts/plot_n256_void_slice.py`
- `scripts/plot_n256_halo_void_slice.py`

## Removed From The Active Tree

The cleanup removed:

- the Typer CLI and YAML workflow config;
- MCMC calibration scripts;
- VAST workflow and mismatch audit scripts;
- one-off region and initial-position debug scripts;
- tests that only covered removed experimental workflows.

Git history preserves these experiments if they are needed later.

## Recommended Use

Use this repository to inspect the prototype implementation and reproduce the
final reference diagnostics. Do not treat the finder as a validated replacement
for VIDE or as a production component.
