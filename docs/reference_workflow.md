# Reference Workflow

These commands reproduce the retained `n256` diagnostics. They assume the local
ignored PINOCCHIO and VIDE products exist under `runs/`.

## Void Size Function

```bash
python scripts/compare_void_size_functions.py \
  runs/pinocchio-lowres/n256/pinocchio.0.0000.lowres_n256.catalog.out \
  runs/pinocchio-lowres/n256_paired/pinocchio.0.0000.lowres_n256_paired.catalog.out \
  runs/vide-lowres/n256/outputs/pinocchio_n256_ss1.0/sample_pinocchio_n256_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_ss1.0_z0.00_d00.out \
  runs/vide-lowres/n256_paired/outputs/pinocchio_n256_paired_ss1.0/sample_pinocchio_n256_paired_ss1.0_z0.00_d00/voidDesc_all_pinocchio_n256_paired_ss1.0_z0.00_d00.out \
  --box-size 256 \
  --rho-bar 8.63025e10 \
  --linking-factor 0.14605092780899798 \
  --radius-a0 6.14700029037185 \
  --radius-alpha 0.9313222316465706 \
  --adjacency-factor 0.5240470713979322 \
  --bins 17 \
  --binning linear \
  --bin-min 10 \
  --bin-max 80 \
  --output-csv runs/void-statistics/n256_reference_vsf.csv \
  --summary-csv runs/void-statistics/n256_reference_vsf_summary.csv \
  --output-plot runs/void-statistics/n256_reference_vsf.png
```

## Center Matching

```bash
python scripts/match_n256_void_centers.py \
  --linking-factor 0.14605092780899798 \
  --radius-a0 6.14700029037185 \
  --radius-alpha 0.9313222316465706 \
  --adjacency-factor 0.5240470713979322 \
  --output-csv runs/void-statistics/n256_reference_center_matches.csv \
  --summary-csv runs/void-statistics/n256_reference_center_match_summary.csv
```

## Halo Slice Plots

```bash
python scripts/plot_n256_halo_void_slice.py \
  --linking-factor 0.14605092780899798 \
  --radius-a0 6.14700029037185 \
  --radius-alpha 0.9313222316465706 \
  --adjacency-factor 0.5240470713979322 \
  --target both \
  --slice-axis z \
  --slice-center 128 \
  --slice-thickness 20 \
  --vide-overlay both \
  --output-prefix n256_reference_halo_slice
```

## Notes

The numbers above are retained as the reference VSF-oriented calibration point.
They are not a production best fit. In particular, this calibration should be
read together with the center-match diagnostics, which show that VSF agreement
does not imply object-level VIDE agreement.
