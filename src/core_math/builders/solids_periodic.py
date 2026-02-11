"""
Periodic Solid Lattice Structures
=================================

N×N×N periodic supercells for SC and FCC lattices with periodic boundary conditions (3-torus T³).

STRUCTURES:
    - SC (Simple Cubic): N³ cube cells, MINIMUM N=3
    - FCC (Face-Centered Cubic): 4N³ rhombic dodecahedra, MINIMUM N=1

TOPOLOGY (faces per edge in 2-skeleton):
    - SC: 4 faces/edge → Tr(d₁ᵀd₁) = 4E
    - FCC: 3 faces/edge → Tr(d₁ᵀd₁) = 3E (same trace as Kelvin foam)
    - Kelvin: 3 faces/edge → Tr(d₁ᵀd₁) = 3E

    NOTE: FCC has same edge topology (k=3) as foam, but is a crystal tiling,
    not a soap film complex. The trace identity is topological.

VERIFIED IDENTITIES:
    - χ(2-skeleton) = V - E + F = C (number of 3-cells)
    - χ(3-complex) = V - E + F - C = 0 (correct for T³)
    - d₁ @ d₀ = 0 (exactness)

MINIMUM N REQUIREMENT:
    SC needs N≥3 because period L=2N must exceed cell diagonal.
    For N<3, periodic identifications collapse edge/face incidences,
    yielding degenerate structure (not correct SC honeycomb).

Date: Jan 2026
"""

import numpy as np
from typing import Tuple, List, Dict, Set

from .solids import build_sc_cell, build_fcc_cell
from ..spec.structures import canonical_face, create_mesh
from ..spec.constants import WRAP_DECIMALS, COMPLEX_SOLID, COMPLEX_TILING, EPS_CLOSE


def wrap_coord(x: float, L: float) -> float:
    """Wrap coordinate to [0, L) with tolerance for numerical precision."""
    result = x % L
    # Snap to 0 if very close to 0 (from rounding noise on exact multiples).
    # The abs(result - L) check is defensive: % should return [0, L), but
    # some edge cases with float precision can produce values very close to L.
    if abs(result) < EPS_CLOSE or abs(result - L) < EPS_CLOSE:
        result = 0.0
    return result


def wrap_position(pos: np.ndarray, L: float) -> tuple:
    """Wrap 3D position to canonical form in [0, L)³."""
    wrapped = np.array([wrap_coord(x, L) for x in pos])
    return tuple(np.round(wrapped, WRAP_DECIMALS))


# =============================================================================
# SC Periodic
# =============================================================================

def generate_sc_centers(N: int) -> List[Tuple[float, float, float]]:
    """
    Generate SC cube centers for N×N×N supercell.

    Centers at (2i, 2j, 2k) for i,j,k ∈ [0, N).
    Total: N³ cells.
    """
    from itertools import product
    centers = []
    for i, j, k in product(range(N), repeat=3):
        centers.append((2.0*i, 2.0*j, 2.0*k))
    return centers


def build_sc_supercell_periodic(N: int) -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], List[List[Tuple[int, int]]]]:
    """
    Build N×N×N SC supercell with periodic boundary conditions.

    Args:
        N: supercell size (N³ cubes total), MINIMUM N=3

    Returns:
        vertices: (V, 3) array of unique vertex positions
        edges: list of (i, j) tuples with i < j
        faces: list of vertex index lists (CCW orientation)
        cell_face_incidence: list of (face_idx, orientation) per cell

    TOPOLOGY:
        V = N³ vertices
        E = 3N³ edges
        F = 3N³ faces
        C = N³ cubes
        χ(2-skeleton) = V - E + F = N³ = C
        χ(3-complex) = V - E + F - C = 0 (T³)

    KEY: Each edge bounds 4 faces (NOT foam-like!)
         Tr(d₁ᵀd₁) = 4E

    Raises:
        ValueError: if N < 3 (structure is degenerate)
    """
    if N < 3:
        raise ValueError(f"SC periodic requires N >= 3, got N={N}. "
                        f"For N<3, periodic identifications collapse incidences (E,F).")

    L = 2.0 * N  # period

    # Get base SC cube
    base_v, base_e, base_f, _ = build_sc_cell()
    base_v = np.array(base_v)

    # Shift base cell so it's centered at origin with vertices at (±1, ±1, ±1)
    # The build_sc_cell() already does this

    # Step 1: Collect all vertices with periodic identification
    vertex_dict: Dict[tuple, int] = {}
    vertices: List[np.ndarray] = []

    centers = generate_sc_centers(N)

    # Map: (cell_idx, local_v_idx) -> global_v_idx
    cell_vertex_map: List[List[int]] = []

    for center in centers:
        center = np.array(center)
        cell_map = []

        for local_v in base_v:
            pos = center + local_v
            canonical_pos = wrap_position(pos, L)

            if canonical_pos not in vertex_dict:
                vertex_dict[canonical_pos] = len(vertices)
                vertices.append(np.array(canonical_pos))

            cell_map.append(vertex_dict[canonical_pos])

        cell_vertex_map.append(cell_map)

    # Step 2: Collect edges with deduplication
    edge_set: Set[Tuple[int, int]] = set()

    for cell_idx, cell_map in enumerate(cell_vertex_map):
        for i, j in base_e:
            gi, gj = cell_map[i], cell_map[j]
            edge = (min(gi, gj), max(gi, gj))
            edge_set.add(edge)

    edges = sorted(edge_set)

    # Step 3: Collect faces with deduplication using canonical ordering
    face_data: Dict[tuple, dict] = {}

    for cell_idx, cell_map in enumerate(cell_vertex_map):
        for local_face in base_f:
            global_face = [cell_map[v] for v in local_face]
            canonical, rel_orient = canonical_face(global_face)

            if canonical not in face_data:
                face_data[canonical] = {
                    'face': list(canonical),
                    'cells': {}
                }

            # Orientation from rel_orient:
            # Local faces have CCW orientation when viewed from outside the cell.
            # rel_orient = +1 → canonical normal points outward → orientation = +1
            # rel_orient = -1 → canonical normal points inward → orientation = -1
            face_data[canonical]['cells'][cell_idx] = rel_orient

    # Build face list and cell-face incidence
    faces = []
    canonical_to_face_idx = {}
    for canonical, data in face_data.items():
        canonical_to_face_idx[canonical] = len(faces)
        faces.append(data['face'])

    n_cells_total = len(centers)
    cell_face_incidence = [[] for _ in range(n_cells_total)]
    for canonical, data in face_data.items():
        face_idx = canonical_to_face_idx[canonical]
        for cell_idx, orient in data['cells'].items():
            cell_face_incidence[cell_idx].append((face_idx, orient))

    return np.array(vertices), edges, faces, cell_face_incidence


def get_sc_periodic_topology(N: int) -> Dict:
    """
    Compute topology numbers for periodic N×N×N SC supercell.

    Args:
        N: supercell size, MINIMUM N=3

    Returns:
        dict with V, E, F, C, chi_2skeleton, chi_3complex

    Raises:
        ValueError: if N < 3
    """
    vertices, edges, faces, _ = build_sc_supercell_periodic(N)  # raises if N < 3

    V = len(vertices)
    E = len(edges)
    F = len(faces)
    C = N**3  # Number of 3-cells (cubes)

    chi_2skeleton = V - E + F
    chi_3complex = V - E + F - C

    return {
        'N': N,
        'n_cells': C,
        'V': V,
        'E': E,
        'F': F,
        'C': C,
        'chi_2skeleton': chi_2skeleton,
        'chi_3complex': chi_3complex,
        'chi_2skeleton_equals_C': chi_2skeleton == C,
        'is_valid_T3': chi_3complex == 0
    }


# NOTE: verify_sc_solid_structure moved to analysis/verify_topology.py (layer separation)


# =============================================================================
# FCC Periodic
# =============================================================================

def generate_fcc_centers(N: int) -> List[Tuple[float, float, float]]:
    """
    Generate FCC lattice cell centers for N×N×N supercell.

    FCC lattice: 4 rhombic dodecahedra per conventional cubic cell.
    Centers at:
        (4i, 4j, 4k)           - corner positions
        (4i+2, 4j+2, 4k)       - face center xy
        (4i+2, 4j, 4k+2)       - face center xz
        (4i, 4j+2, 4k+2)       - face center yz

    Total: 4N³ cells.
    """
    from itertools import product
    centers = []
    for i, j, k in product(range(N), repeat=3):
        # 4 FCC positions per conventional cell
        centers.append((4.0*i, 4.0*j, 4.0*k))
        centers.append((4.0*i + 2.0, 4.0*j + 2.0, 4.0*k))
        centers.append((4.0*i + 2.0, 4.0*j, 4.0*k + 2.0))
        centers.append((4.0*i, 4.0*j + 2.0, 4.0*k + 2.0))
    return centers


def build_fcc_supercell_periodic(N: int) -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], List[List[Tuple[int, int]]]]:
    """
    Build N×N×N FCC supercell with periodic boundary conditions.

    Args:
        N: supercell size (4N³ rhombic dodecahedra total), N≥1

    Returns:
        vertices: (V, 3) array of unique vertex positions
        edges: list of (i, j) tuples with i < j
        faces: list of vertex index lists (canonical orientation)
        cell_face_incidence: list of (face_idx, orientation) per cell

    TOPOLOGY:
        Total cells C = 4N³
        Each edge bounds 3 faces (same k as foam)
        Tr(d₁ᵀd₁) = 3E

    NOTE: FCC works for N≥1 (verified by T2 histogram tests).
          Unlike SC, FCC doesn't need N≥3 guard because period L=4N
          is large enough relative to cell size even for N=1.
    """
    L = 4.0 * N  # period

    # Get base FCC rhombic dodecahedron
    base_v, base_e, base_f, _ = build_fcc_cell()
    base_v = np.array(base_v)

    # Step 1: Collect all vertices with periodic identification
    vertex_dict: Dict[tuple, int] = {}
    vertices: List[np.ndarray] = []

    centers = generate_fcc_centers(N)

    # Map: (cell_idx, local_v_idx) -> global_v_idx
    cell_vertex_map: List[List[int]] = []

    for center in centers:
        center = np.array(center)
        cell_map = []

        for local_v in base_v:
            pos = center + local_v
            canonical_pos = wrap_position(pos, L)

            if canonical_pos not in vertex_dict:
                vertex_dict[canonical_pos] = len(vertices)
                vertices.append(np.array(canonical_pos))

            cell_map.append(vertex_dict[canonical_pos])

        cell_vertex_map.append(cell_map)

    # Step 2: Collect edges with deduplication
    edge_set: Set[Tuple[int, int]] = set()

    for cell_idx, cell_map in enumerate(cell_vertex_map):
        for i, j in base_e:
            gi, gj = cell_map[i], cell_map[j]
            edge = (min(gi, gj), max(gi, gj))
            edge_set.add(edge)

    edges = sorted(edge_set)

    # Step 3: Collect faces with deduplication using canonical ordering
    face_data: Dict[tuple, dict] = {}

    for cell_idx, cell_map in enumerate(cell_vertex_map):
        for local_face in base_f:
            global_face = [cell_map[v] for v in local_face]
            canonical, rel_orient = canonical_face(global_face)

            if canonical not in face_data:
                face_data[canonical] = {
                    'face': list(canonical),
                    'cells': {}
                }

            # Orientation from rel_orient:
            # Local faces have CCW orientation when viewed from outside the cell.
            # rel_orient = +1 → canonical normal points outward → orientation = +1
            # rel_orient = -1 → canonical normal points inward → orientation = -1
            face_data[canonical]['cells'][cell_idx] = rel_orient

    # Build face list and cell-face incidence
    faces = []
    canonical_to_face_idx = {}
    for canonical, data in face_data.items():
        canonical_to_face_idx[canonical] = len(faces)
        faces.append(data['face'])

    n_cells_total = len(centers)
    cell_face_incidence = [[] for _ in range(n_cells_total)]
    for canonical, data in face_data.items():
        face_idx = canonical_to_face_idx[canonical]
        for cell_idx, orient in data['cells'].items():
            cell_face_incidence[cell_idx].append((face_idx, orient))

    return np.array(vertices), edges, faces, cell_face_incidence


def get_fcc_periodic_topology(N: int) -> Dict:
    """
    Compute topology numbers for periodic N×N×N FCC supercell.
    """
    vertices, edges, faces, _ = build_fcc_supercell_periodic(N)

    V = len(vertices)
    E = len(edges)
    F = len(faces)
    C = 4 * N**3  # Number of 3-cells

    chi_2skeleton = V - E + F
    chi_3complex = V - E + F - C

    return {
        'N': N,
        'n_cells': C,
        'V': V,
        'E': E,
        'F': F,
        'C': C,
        'chi_2skeleton': chi_2skeleton,
        'chi_3complex': chi_3complex,
        'chi_2skeleton_equals_C': chi_2skeleton == C,
        'is_valid_T3': chi_3complex == 0
    }


# NOTE: verify_fcc_structure moved to analysis/verify_topology.py (layer separation)


# =============================================================================
# CONTRACT-COMPLIANT WRAPPERS
# =============================================================================

def build_sc_solid_periodic(N: int, name: str = None) -> dict:
    """
    Build N×N×N periodic SC solid as contract-compliant mesh dict.

    This is a SOLID with 4 faces per edge (k=4).

    Args:
        N: supercell size (N³ cubes total), MINIMUM N=3
        name: mesh name (auto-generated if None)

    Returns:
        Contract-compliant mesh dict with:
        - complex_type = "solid"
        - faces_per_edge = 4
        - period_L = 2N
        - cell_face_incidence for d₂ support
    """
    if name is None:
        name = f"sc_solid_{N}x{N}x{N}"

    V, E, F, cell_face_inc = build_sc_supercell_periodic(N)
    n_cells = N**3
    L = 2.0 * N  # Period length: each SC cube has side 2, N cells per dimension

    return create_mesh(
        V=V,
        E=E,
        F=F,
        complex_type=COMPLEX_SOLID,
        name=name,
        n_cells=n_cells,
        periodic=True,
        cell_face_incidence=cell_face_inc,
        period_L=L
    )


def build_fcc_solid_periodic(N: int, name: str = None) -> dict:
    """
    Build N×N×N periodic FCC structure as contract-compliant mesh dict.

    FCC has k=3 faces per edge (same as foam), but is a crystal tiling.

    Args:
        N: supercell size (4N³ rhombic dodecahedra total), N≥1
        name: mesh name (auto-generated if None)

    Returns:
        Contract-compliant mesh dict with:
        - complex_type = "tiling" (k=3, crystal packing, NOT foam physics)
        - faces_per_edge = 3
        - period_L = 4N
        - cell_face_incidence for d₂ support

    NOTE: TILING vs FOAM distinction:
        - Math: identical (k=3 → Tr(d₁ᵀd₁) = 3E)
        - Physics: FOAM has Plateau borders, surface tension; TILING is crystal packing
    """
    if name is None:
        name = f"fcc_tiling_{N}x{N}x{N}"

    V, E, F, cell_face_inc = build_fcc_supercell_periodic(N)
    n_cells = 4 * N**3
    L = 4.0 * N  # Period length

    return create_mesh(
        V=V,
        E=E,
        F=F,
        complex_type=COMPLEX_TILING,  # k=3 crystal tiling, NOT foam
        name=name,
        n_cells=n_cells,
        periodic=True,
        cell_face_incidence=cell_face_inc,
        period_L=L
    )


# =============================================================================
# Demo / Manual Verification (imports operators only when run directly)
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PERIODIC SOLID STRUCTURES - VERIFICATION")
    print("=" * 60)

    # Import operators and analysis functions ONLY when run as script
    # This doesn't affect layer separation for normal module usage
    from ..operators.incidence import build_incidence_matrices
    from ..analysis.verify_topology import verify_sc_solid_structure, verify_fcc_structure

    # Test SC
    print("\n" + "-" * 60)
    print("SC PERIODIC (4 faces/edge = NOT foam)")
    print("-" * 60)

    for N in [3, 4]:  # N >= 3 required
        print(f"\n--- N = {N} ({N**3} cubes) ---")

        vertices, edges, faces, _ = build_sc_supercell_periodic(N)
        topo = get_sc_periodic_topology(N)

        print(f"V = {topo['V']}, E = {topo['E']}, F = {topo['F']}, C = {topo['C']}")
        print(f"χ(2-skeleton) = {topo['chi_2skeleton']} (expect C = {topo['C']}): {'✓' if topo['chi_2skeleton_equals_C'] else '✗'}")
        print(f"χ(3-complex) = {topo['chi_3complex']} (expect 0 for T³): {'✓' if topo['is_valid_T3'] else '✗'}")

        # Build d1 and verify trace
        try:
            d0, d1 = build_incidence_matrices(vertices, edges, faces)
            verify = verify_sc_solid_structure(d1, topo['E'])
            print(f"Faces/edge: min={verify['min_faces_per_edge']}, max={verify['max_faces_per_edge']} (expect 4)")
            print(f"Tr(d₁ᵀd₁) = {verify['trace_d1td1']} (expect 4E = {verify['expected_trace']}): {'✓' if verify['sc_trace_theorem_holds'] else '✗'}")
        except Exception as e:
            print(f"Error building operators: {e}")

    # Test FCC
    print("\n" + "-" * 60)
    print("FCC PERIODIC (3 faces/edge, same k as foam)")
    print("-" * 60)

    for N in [1, 2]:
        print(f"\n--- N = {N} ({4*N**3} rhombic dodecahedra) ---")

        try:
            vertices, edges, faces, _ = build_fcc_supercell_periodic(N)
            topo = get_fcc_periodic_topology(N)

            print(f"V = {topo['V']}, E = {topo['E']}, F = {topo['F']}, C = {topo['C']}")
            print(f"χ(2-skeleton) = {topo['chi_2skeleton']} (expect C = {topo['C']}): {'✓' if topo['chi_2skeleton_equals_C'] else '✗'}")
            print(f"χ(3-complex) = {topo['chi_3complex']} (expect 0 for T³): {'✓' if topo['is_valid_T3'] else '✗'}")

            d0, d1 = build_incidence_matrices(vertices, edges, faces)
            verify = verify_fcc_structure(d1, topo['E'])
            print(f"Faces/edge: min={verify['min_faces_per_edge']}, max={verify['max_faces_per_edge']} (expect 3)")
            print(f"Tr(d₁ᵀd₁) = {verify['trace_d1td1']} (expect 3E = {verify['expected_trace']}): {'✓' if verify['fcc_trace_theorem_holds'] else '✗'}")
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("COMPARISON: SOLID vs FOAM")
    print("=" * 60)
    print()
    print("| Structure | Faces/edge | Tr(d₁ᵀd₁) | Type     |")
    print("|-----------|------------|-----------|----------|")
    print("| SC cube   | 4          | 4E        | SOLID    |")
    print("| FCC rhomb | 3          | 3E        | k=3      |")
    print("| BCC Kelvin| 3          | 3E        | FOAM     |")
    print()
    print("KEY INSIGHT: FCC has same edge topology as foam!")
    print("             But κ-locking still distinguishes them.")
