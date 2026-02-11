"""
Periodic BCC Multicell
======================

N×N×N BCC supercell with PERIODIC boundary conditions (3-torus T³).

GEOMETRY:
    Period: L = 4N in each direction
    All coordinates wrapped: x → x mod L

TOPOLOGY (verified Jan 2026):
    This is a FOAM (Plateau structure), NOT a 2-manifold!

    - χ(2-skeleton) = V - E + F = C (number of 3-cells)
    - χ(3-complex) = V - E + F - C = 0 (correct for T³)
    - Every edge bounds exactly 3 faces (Plateau border!)
    - Tr(d₁ᵀd₁) = 3E (foam trace theorem)
    - d₁ @ d₀ = 0 (orientations consistent)

    Key insight: BCC Kelvin is space-filling foam where each edge
    is a Plateau border (junction of 3 soap films), not 2.

COMPARISON with non-periodic (multicell.py):
    - multicell.py: open boundary, boundary effects
    - This module: periodic, proper k-vector, no boundary

PARITY (verified Jan 2026):
    - Parity P: x → (L - x) mod L is FREE action (no fixed vertices/edges)
    - Tr(P|_H) = 1 (Lefschetz holds)
    - BUT: P anti-commutes with d₁ᵀ in single cell, NOT in periodic
    - Result: antipodal currents → bridge (single cell) vs ring (periodic)

Date: Jan 2026
"""

import numpy as np
from typing import Tuple, List, Dict, Set
from itertools import product

from .kelvin import build_kelvin_cell
from ..spec.structures import create_mesh, canonical_face
from ..spec.constants import COMPLEX_FOAM, WRAP_DECIMALS, EPS_CLOSE


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


def generate_bcc_centers(N: int) -> List[Tuple[float, float, float]]:
    """
    Generate BCC cell centers for N×N×N supercell.

    SET A: (4i, 4j, 4k) for i,j,k ∈ [0, N)
    SET B: (4i+2, 4j+2, 4k+2) for i,j,k ∈ [0, N)

    Total: 2N³ cells
    """
    centers = []

    for i, j, k in product(range(N), repeat=3):
        centers.append((4.0*i, 4.0*j, 4.0*k))
        centers.append((4.0*i + 2.0, 4.0*j + 2.0, 4.0*k + 2.0))

    return centers


def build_bcc_supercell_periodic(N: int) -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], List[List[Tuple[int, int]]]]:
    """
    Build N×N×N BCC supercell with periodic boundary conditions.

    Args:
        N: supercell size (2N³ cells total)

    Returns:
        vertices: (V, 3) array of unique vertex positions
        edges: list of (i, j) tuples with i < j
        faces: list of vertex index lists (CCW orientation)
        cell_face_incidence: list of (face_idx, orientation) per cell

    TOPOLOGY:
        χ(2-skeleton) = V - E + F = C (number of cells)
        χ(3-complex) = V - E + F - C = 0 (3-torus)
        Every edge bounds exactly 3 faces (Plateau border)
    """
    L = 4.0 * N  # period

    # Get base Kelvin cell
    base_v, base_e, base_f, _ = build_kelvin_cell()
    base_v = np.array(base_v)

    # Step 1: Collect all vertices with periodic identification
    vertex_dict: Dict[tuple, int] = {}
    vertices: List[np.ndarray] = []

    centers = generate_bcc_centers(N)

    # Map: (cell_idx, local_v_idx) -> global_v_idx
    cell_vertex_map: List[List[int]] = []

    for center in centers:
        center = np.array(center)
        cell_map = []

        for local_v in base_v:
            pos = center + local_v
            canonical = wrap_position(pos, L)

            if canonical not in vertex_dict:
                vertex_dict[canonical] = len(vertices)
                vertices.append(np.array(canonical))

            cell_map.append(vertex_dict[canonical])

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
    # This preserves orientation information for d₂
    #
    # face_data[canonical] = {
    #     'face': list (canonical vertex order),
    #     'cells': {cell_idx: orientation}
    # }
    # where orientation = +1 if cell sees canonical direction as outward, -1 if inward
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
            # - Kelvin cell faces have CCW orientation when viewed from outside the cell
            # - rel_orient = +1 means global_face has same winding as canonical
            #   → canonical normal points outward from this cell → orientation = +1
            # - rel_orient = -1 means global_face has opposite winding
            #   → canonical normal points inward to this cell → orientation = -1
            #
            # This works because two cells sharing a face see it from opposite sides,
            # so their local face windings are reversed, giving opposite rel_orient values.
            face_data[canonical]['cells'][cell_idx] = rel_orient

    # Build face list and cell-face incidence
    faces = []
    canonical_to_face_idx = {}
    for canonical, data in face_data.items():
        canonical_to_face_idx[canonical] = len(faces)
        faces.append(data['face'])

    # Build cell_face_incidence: for each cell, list of (face_idx, orientation)
    n_cells_total = len(centers)
    cell_face_incidence = [[] for _ in range(n_cells_total)]
    for canonical, data in face_data.items():
        face_idx = canonical_to_face_idx[canonical]
        for cell_idx, orientation in data['cells'].items():
            cell_face_incidence[cell_idx].append((face_idx, orientation))

    return np.array(vertices), edges, faces, cell_face_incidence


def get_periodic_topology(N: int) -> Dict:
    """
    Compute topology numbers for periodic N×N×N BCC supercell.

    Returns:
        dict with V, E, F, C, chi_2skeleton, chi_3complex

    KEY INSIGHT (Jan 2026):
        - χ(2-skeleton) = V - E + F = C (number of cells), NOT 0
        - χ(3-complex) = V - E + F - C = 0 (correct for T³)
        - This is a FOAM: every edge bounds 3 faces (Plateau)
    """
    vertices, edges, faces, _ = build_bcc_supercell_periodic(N)

    V = len(vertices)
    E = len(edges)
    F = len(faces)
    C = 2 * N**3  # Number of 3-cells (Kelvin cells)

    chi_2skeleton = V - E + F  # This equals C, not 0!
    chi_3complex = V - E + F - C  # This is 0 for T³

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


def verify_foam_structure(d1: np.ndarray, E: int) -> Dict:
    """
    Verify foam structure (Plateau: every edge bounds 3 faces).

    For BCC Kelvin foam, edges are Plateau borders where 3 soap films meet.
    This is different from a 2-manifold where each edge bounds 2 faces.

    Args:
        d1: (F, E) incidence matrix
        E: number of edges

    Returns:
        dict with verification results
    """
    # Count how many faces each edge bounds
    faces_per_edge = np.sum(np.abs(d1), axis=0)

    all_bound_3 = np.all(faces_per_edge == 3)
    min_bound = np.min(faces_per_edge)
    max_bound = np.max(faces_per_edge)

    # Trace theorem for foam: Tr(d₁ᵀd₁) = 3E
    d1td1 = d1.T @ d1
    trace = np.trace(d1td1)
    expected_trace_foam = 3 * E
    trace_ok = abs(trace - expected_trace_foam) < EPS_CLOSE

    return {
        'is_plateau_foam': all_bound_3,
        'min_faces_per_edge': int(min_bound),
        'max_faces_per_edge': int(max_bound),
        'trace_d1td1': int(trace),
        'expected_trace_foam': expected_trace_foam,
        'foam_trace_theorem_holds': trace_ok
    }


# =============================================================================
# CONTRACT-COMPLIANT WRAPPER
# =============================================================================

def build_bcc_foam_periodic(N: int, name: str = None) -> dict:
    """
    Build N×N×N periodic BCC foam as contract-compliant mesh dict.

    This is a TRUE FOAM with 3 faces per edge (Plateau structure).

    Args:
        N: supercell size (2N³ cells total)
        name: mesh name (auto-generated if None)

    Returns:
        Contract-compliant mesh dict with:
        - complex_type = "foam"
        - faces_per_edge = 3
        - cell_face_incidence for d₂ support

    NOTE: N≥2 required. At N=1, some faces are "self-glued" (identified with
    themselves under periodic wrap), which breaks d₂ verification (each face
    must appear in exactly 2 distinct cells with opposite orientations).
    """
    if N < 2:
        raise ValueError(
            f"BCC foam periodic requires N≥2, got N={N}. "
            f"At N=1, faces are self-glued under periodic identification, "
            f"which breaks d₂ (cell-face incidence) structure."
        )

    if name is None:
        name = f"bcc_foam_{N}x{N}x{N}"

    V, E, F, cell_face_inc = build_bcc_supercell_periodic(N)
    n_cells = 2 * N**3
    L = 4.0 * N  # Period length: each Kelvin cell has side 4, N cells per dimension

    return create_mesh(
        V=V,
        E=E,
        F=F,
        complex_type=COMPLEX_FOAM,
        name=name,
        n_cells=n_cells,
        periodic=True,
        cell_face_incidence=cell_face_inc,
        period_L=L
    )


# Test when run directly
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 2)[0])
    from operators import build_operators_from_mesh

    print("=" * 60)
    print("PERIODIC BCC FOAM - CONTRACT VERIFICATION")
    print("=" * 60)

    for N in [1, 2]:
        print(f"\n--- N = {N} ({2*N**3} cells) ---")

        mesh = build_bcc_foam_periodic(N)
        print(f"Mesh: {mesh['name']}")
        print(f"complex_type: {mesh['complex_type']}")
        print(f"faces_per_edge: {mesh['faces_per_edge']}")
        print(f"V={mesh['n_V']}, E={mesh['n_E']}, F={mesh['n_F']}")

        topo = get_periodic_topology(N)
        print(f"χ(2-skeleton) = {topo['chi_2skeleton']} (expected C = {topo['C']})")
        print(f"χ(3-complex) = {topo['chi_3complex']} (expected 0 for T³)")

        # Build operators with contract
        ops = build_operators_from_mesh(mesh)
        E = mesh['n_E']
        k = mesh['faces_per_edge']

        print(f"\nTrace identities (k={k}):")
        print(f"  Tr(d₀d₀ᵀ) = {ops['traces']['Tr_d0d0t']:.0f} (expected 2E = {2*E})")
        print(f"  Tr(d₁ᵀd₁) = {ops['traces']['Tr_d1td1']:.0f} (expected {k}E = {k*E})")
        print(f"  Tr(L₁)    = {ops['traces']['Tr_L1']:.0f} (expected {2+k}E = {(2+k)*E})")

    print("\n" + "=" * 60)
    print("Contract verification complete.")
    print("=" * 60)
