# Scientific Conventions

These conventions define the Milestone 2 data contract. They are intentionally
limited to catalog representation and do not define a void-finding algorithm.

## Units

- Positions are comoving coordinates in `Mpc/h`.
- Masses are in `Msun/h`.
- Velocities are in `km/s`.
- Particle counts and catalog IDs are dimensionless integers.

## Coordinates

- The default analysis coordinates are the PINOCCHIO final halo positions.
- Initial positions remain available in the PINOCCHIO-specific reader but are
  not used for the canonical halo/tracer positions.
- Periodic boxes are represented by a positive `box_size_mpc_h`.
- Wrapped positions are mapped into `[0, box_size)`.

## Catalog Roles

- `HaloCatalog` stores halo-specific quantities: IDs, masses, final positions,
  velocities, particle counts, box size, and unit metadata.
- `TracerCatalog` stores point tracers for future void-finding algorithms.
- Converting halos to tracers does not run a void finder and does not format
  data for VIDE.
