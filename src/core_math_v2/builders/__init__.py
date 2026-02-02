"""
Geometry builders - pure geometry construction, no operators dependency.

EXPORTS:
- Contract wrappers: build_*_mesh, build_*_periodic (return mesh dicts)
- Raw geometry: build_*_cell, build_*_supercell (return V, E, F tuples)
- Polyhedra: build_cube, build_octahedron, build_tetrahedron, build_truncated_cube

NOT EXPORTED (use direct import if needed):
- compute_* functions: use operators, belong in analysis layer

MOVED TO analysis/:
- random_foam.py: Voronoi foam statistics (use core_math_v2.analysis.random_foam)
"""

# === Contract wrappers (return mesh dicts) ===
from .kelvin import build_kelvin_cell_mesh
from .multicell_periodic import build_bcc_foam_periodic
from .solids_periodic import build_sc_solid_periodic, build_fcc_solid_periodic

# === Raw geometry (return V, E, F tuples) ===
from .kelvin import build_kelvin_cell
from .multicell_periodic import build_bcc_supercell_periodic
from .solids import build_sc_cell, build_fcc_cell
from .solids_periodic import build_sc_supercell_periodic, build_fcc_supercell_periodic
from .weaire_phelan_periodic import build_wp_supercell_periodic
from .c15_periodic import build_c15_supercell_periodic

# === Polyhedra (surface cells) ===
from .polyhedra import build_cube, build_octahedron, build_tetrahedron, build_truncated_cube
