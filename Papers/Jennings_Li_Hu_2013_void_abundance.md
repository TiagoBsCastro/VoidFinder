# Jennings, Li & Hu 2013: Vdn/SVdW Void Abundance

This paper is the reference for the built-in theoretical void size-function
overlay currently exposed as `--theory vdn-svdw`.

## Citation

Elise Jennings, Yin Li, and Wayne Hu, "The abundance of voids and the excursion
set formalism", Monthly Notices of the Royal Astronomical Society, Volume 434,
Issue 3, 21 September 2013, Pages 2167-2181.

- DOI: `10.1093/mnras/stt1169`
- arXiv: `1304.6087`
- MNRAS page: <https://academic.oup.com/mnras/article/434/3/2167/1036592>
- arXiv page: <https://arxiv.org/abs/1304.6087>

## Implementation Relevance

The paper compares excursion-set void abundance models against spherical voids
identified in dark-matter simulations. The repository's first theory overlay is
the Vdn/SVdW-style curve, with defaults:

- `delta_v_linear = -2.7`
- `delta_c_linear = 1.686`
- `delta_v_nonlinear = -0.8`

The comparison against our current VIDE catalogs is diagnostic, not a final
physical calibration. The paper explicitly warns that halo or galaxy void
abundances are related to the underlying dark-matter void abundance in a
complicated way, so a direct amplitude mismatch is expected to require careful
debugging before changing the algorithm.
