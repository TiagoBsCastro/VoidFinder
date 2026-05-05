# Development Plan

This file is the working roadmap for `pinocchio_voids`. It tracks the current
stage, completed work, and the next development tasks.

## Current Stage

We are in **Milestone 3** with the geometry-only paired-halo prototype. Active
void size-function calibration is now **n256-only**:

- Milestone 1 and Milestone 2 are complete.
- `ALGORITHM.md` remains the authoritative algorithm specification.
- Local `n032`, `n064`, and `n128` run products were removed because they do
  not provide a workable VSF for calibration.
- The only active local calibration target is the `n256` paired PINOCCHIO and
  VIDE catalogs under ignored `runs/` paths.
- VIDE `voidDesc` radii are corrected from normalized Voronoi volumes using
  `sample_info.txt` mean tracer separation.
- VSF-facing effective radii consistently use
  `R_eff = (3 V / 4 pi)^(1/3)`.
- The current `n256` MCMC maximum-likelihood row is:
  - linking factor `0.14605092780899798`
  - `radius_a0=6.14700029037185`
  - `radius_alpha=0.9313222316465706`
  - `adjacency_factor=0.5240470713979322`
- Bridge-density scoring and PINOCCHIO execution orchestration have not started.

## Milestone 3 Progress

Completed so far:

- Added paired PINOCCHIO catalog loading into canonical `HaloCatalog` objects.
- Added periodic geometry helpers, periodic FoF-like source clustering, and
  source-cluster summaries.
- Added spherical protovoid construction with
  `R_proto = a0 R_L(M)^alpha`.
- Added geometry-only protovoid adjacency and connected-component merging.
- Added symmetric A-to-B and B-to-A paired execution.
- Added VIDE `voidDesc` reading and VSF metrics.
- Added finder-vs-VIDE comparison scripts with fixed paper-style bins.
- Added guarded deterministic calibration scores, Poisson VSF likelihood
  scoring, and cached likelihood-grid diagnostics.
- Added `emcee` posterior sampling for `n256` calibration, including chain
  products, trace plots, 68/95 percent contour plots, and best-fit plot commands.
- Updated the active plot refresh workflow to `scripts/plot_n256_void_size_function.py`.
- Cleaned deprecated ignored products so `runs/void-statistics` keeps only
  current `n256` VSF and MCMC products.

## Next Tasks

1. Inspect the current `n256` MCMC posterior.
   - Review `runs/void-statistics/n256_vsf_mcmc_contours.png`.
   - Check trace stability in `runs/void-statistics/n256_vsf_mcmc_trace.png`.
   - Decide whether to rerun a longer chain or narrow the prior region.

2. Validate the `n256` best-fit VSF.
   - Compare `n256_paper_bins_vsf` and `n256_mcmc_best_fit_paper_bins_vsf`.
   - Inspect finder/VIDE count residuals, median radii, and bin-by-bin shape.
   - Keep theory overlays out of default calibration plots.

3. Audit paper-comparison assumptions before interpreting amplitudes.
   - Match binning, volume normalization, halo mass cuts, tracer density, and
     VIDE settings as closely as local `n256` data allow.
   - Plan a paper-like local VIDE rerun only if post-processing cannot make the
     current references comparable.

4. Add bridge-density scoring after the `n256` geometry-only baseline is stable.
   - Keep bridge scoring modular and optional.
   - Compare geometry-only and bridge-scored VSFs with the same MCMC-calibrated
     plotting workflow.

## Test Plan

- Keep the full package test suite passing.
- Existing tests cover catalog validation, periodic geometry, source
  clustering, protovoid construction, graph merging, paired execution, VIDE
  parsing, VSF metrics, theory helpers, calibration scores, likelihood-grid
  ranking, and the short MCMC optimizer smoke path.
- Use ignored local integration checks only for active `n256` paths.

## Local Reference Data

- Active ignored PINOCCHIO catalogs:
  - `runs/pinocchio-lowres/n256`
  - `runs/pinocchio-lowres/n256_paired`
- Active ignored VIDE references:
  - `runs/vide-lowres/n256`
  - `runs/vide-lowres/n256_paired`
- Active ignored VSF products live under `runs/void-statistics/` and should be
  limited to current `n256` finder/VIDE, paper-bin, MCMC best-fit, and MCMC
  diagnostic outputs.
- Generated `runs/` data must remain ignored. Commit only code, docs, tests,
  and tiny deterministic fixtures.

## Verification Commands

Use the named Miniforge environment:

```bash
/home/tcastro/miniforge3/envs/voidfinder/bin/python -m pytest
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid smoke-test
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid validate-config tests/fixtures/run_config_small.yaml
/home/tcastro/miniforge3/envs/voidfinder/bin/pinvoid paired-sweep --help
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/calibrate_vsf_likelihood.py --help
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/optimize_n256_vsf_mcmc.py --help
/home/tcastro/miniforge3/envs/voidfinder/bin/python scripts/plot_n256_void_size_function.py --paper-bins
```
