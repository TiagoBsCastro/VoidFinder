# Scientific Conventions

These conventions define the data contract used by the retained reference
prototype and diagnostics.

## Units

- Positions are comoving coordinates in `Mpc/h`.
- Masses are in `Msun/h`.
- Velocities are in `km/s`.
- Particle counts and catalog IDs are dimensionless integers.

## Coordinates

- The default analysis coordinates are the PINOCCHIO final halo positions.
- Initial positions remain available through the PINOCCHIO reader and paired
  loader for diagnostics, but final positions are the default reference mode.
- Periodic boxes are represented by a positive `box_size_mpc_h`.
- Wrapped positions are mapped into `[0, box_size)`.

## Catalog Roles

- `HaloCatalog` stores halo-specific quantities: IDs, masses, final positions,
  velocities, particle counts, box size, and unit metadata.
- `TracerCatalog` stores generic point tracers.
- Converting halos to tracers does not format data for VIDE.

## Void Radii

- Finder and VIDE size-function diagnostics use
  `R_eff = (3 V / 4 pi)^(1/3)`.
- VIDE `voidDesc` radii are corrected from normalized Voronoi volumes using the
  `sample_info.txt` mean tracer separation.
