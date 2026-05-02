# PINOCCHIO Catalog Format

Milestone 2 supports one lightweight input format: the ASCII PINOCCHIO halo
catalog produced as `*.catalog.out`.

The reader expects header lines beginning with `#`, followed by 12 numeric
columns:

| Column | Name | Unit |
| --- | --- | --- |
| 1 | group ID | dimensionless |
| 2 | group mass | Msun/h |
| 3-5 | initial position x, y, z | Mpc/h |
| 6-8 | final position x, y, z | Mpc/h |
| 9-11 | velocity x, y, z | km/s |
| 12 | number of particles | dimensionless |

The current package only reads and validates this catalog layout. It does not
run PINOCCHIO, does not find voids, and does not optimize parameters.

For canonical package data objects, final positions are used as the default
analysis positions. See `docs/scientific_conventions.md` for the unit and
periodic-box conventions.
