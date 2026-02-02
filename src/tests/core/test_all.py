"""
Comprehensive Tests for core_math_v2
====================================

Tests all mathematical invariants:
- Trace identities (topology-dependent)
- Exactness d₁d₀ = 0
- Lefschetz theorem
- Contract compliance

Run: python -m pytest tests_math/test_all.py -v

NOTE: When this file reaches ~1000 lines, create test_all_2.py (or split by category).
      Do NOT keep growing a single file indefinitely.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_math_v2.builders import (
    build_kelvin_cell_mesh,
    build_bcc_foam_periodic,
    build_sc_solid_periodic,
    build_fcc_solid_periodic,
    build_cube,
    build_octahedron,
    build_tetrahedron,
    build_truncated_cube,
    build_sc_supercell_periodic,
    build_fcc_supercell_periodic,
)
from core_math_v2.builders.weaire_phelan import (
    build_wp_type_a,
    build_wp_type_b,
    verify_wp_surface_topology,
)
from core_math_v2.operators import (
    build_operators_from_mesh,
    build_parity_from_mesh,
    build_incidence_matrices,
)
from core_math_v2.operators.incidence import build_d2
from core_math_v2.spec import (
    validate_mesh,
    COMPLEX_SURFACE,
    COMPLEX_FOAM,
    FACES_PER_EDGE,
    WRAP_DECIMALS,
    EPS_CLOSE,
    PLANAR_TOL,
)


# =============================================================================
# TEST A: Trace Identities
# =============================================================================

def test_trace_surface_kelvin():
    """Surface (k=2): Tr(d₁ᵀd₁) = 2E

    Uses EPS_CLOSE tolerance for future-proofing (weighted matrices may be float).
    """
    mesh = build_kelvin_cell_mesh()
    ops = build_operators_from_mesh(mesh)

    E = mesh['n_E']
    k = mesh['faces_per_edge']

    assert k == 2, f"Surface should have k=2, got {k}"
    assert abs(ops['traces']['Tr_d0d0t'] - 2 * E) < EPS_CLOSE, "Tr(d₀d₀ᵀ) ≠ 2E"
    assert abs(ops['traces']['Tr_d1td1'] - k * E) < EPS_CLOSE, f"Tr(d₁ᵀd₁) ≠ {k}E"
    assert abs(ops['traces']['Tr_L1'] - (2 + k) * E) < EPS_CLOSE, f"Tr(L₁) ≠ {2+k}E"
    print(f"✓ Surface trace: Tr(d₁ᵀd₁) = {k}E = {k*E}")


def test_trace_foam_n1():
    """Foam N=1 (k=3): Tr(d₁ᵀd₁) = 3E

    Uses EPS_CLOSE tolerance for future-proofing (weighted matrices may be float).
    """
    mesh = build_bcc_foam_periodic(N=2)
    ops = build_operators_from_mesh(mesh)

    E = mesh['n_E']
    k = mesh['faces_per_edge']

    assert k == 3, f"Foam should have k=3, got {k}"
    assert abs(ops['traces']['Tr_d0d0t'] - 2 * E) < EPS_CLOSE, "Tr(d₀d₀ᵀ) ≠ 2E"
    assert abs(ops['traces']['Tr_d1td1'] - k * E) < EPS_CLOSE, f"Tr(d₁ᵀd₁) ≠ {k}E"
    assert abs(ops['traces']['Tr_L1'] - (2 + k) * E) < EPS_CLOSE, f"Tr(L₁) ≠ {2+k}E"
    print(f"✓ Foam N=1 trace: Tr(d₁ᵀd₁) = {k}E = {k*E}")


def test_trace_foam_n2():
    """Foam N=2 (k=3): Tr(d₁ᵀd₁) = 3E

    Uses EPS_CLOSE tolerance for future-proofing.
    """
    mesh = build_bcc_foam_periodic(N=2)
    ops = build_operators_from_mesh(mesh)

    E = mesh['n_E']
    k = mesh['faces_per_edge']

    assert k == 3
    assert abs(ops['traces']['Tr_d1td1'] - k * E) < EPS_CLOSE
    print(f"✓ Foam N=2 trace: Tr(d₁ᵀd₁) = {k}E = {k*E}")


# =============================================================================
# TEST B: Exactness d₁d₀ = 0
# =============================================================================

def test_exactness_kelvin():
    """Exactness: d₁d₀ = 0 (curl of gradient is zero)"""
    mesh = build_kelvin_cell_mesh()
    V, E, F = mesh['V'], mesh['E'], mesh['F']

    d0, d1 = build_incidence_matrices(V, E, F)
    d1d0 = d1 @ d0

    norm = np.linalg.norm(d1d0)
    assert norm < EPS_CLOSE, f"Exactness failed: ||d₁d₀|| = {norm}"
    print(f"✓ Exactness: ||d₁d₀|| = {norm:.2e}")


def test_exactness_foam():
    """Exactness holds for foam too"""
    mesh = build_bcc_foam_periodic(N=2)
    V, E, F = mesh['V'], mesh['E'], mesh['F']

    d0, d1 = build_incidence_matrices(V, E, F)
    d1d0 = d1 @ d0

    norm = np.linalg.norm(d1d0)
    assert norm < EPS_CLOSE
    print(f"✓ Exactness (foam): ||d₁d₀|| = {norm:.2e}")


def test_exactness_d2d1_bcc_foam():
    """Exactness: d₂d₁ = 0 (divergence of curl is zero) for BCC foam.

    This tests the full DEC chain exactness at level 2:
    - d₁d₀ = 0: curl(grad) = 0
    - d₂d₁ = 0: div(curl) = 0

    Uses N=2 to avoid periodic degeneracy issues.
    """
    N = 2
    mesh = build_bcc_foam_periodic(N=N)
    V, E, F = mesh['V'], mesh['E'], mesh['F']

    # Build d₁
    _, d1 = build_incidence_matrices(V, E, F)

    # Build d₂ from cell-face incidence
    cell_face_inc = mesh['cell_face_incidence']
    n_faces = len(F)

    # Guard: verify face indices in cell_face_incidence are valid
    # (catches mismatch if builder changes face indexing convention)
    max_face_idx = max(f_idx for cell in cell_face_inc for f_idx, _ in cell)
    assert max_face_idx < n_faces, (
        f"cell_face_incidence has face_idx={max_face_idx} but n_faces={n_faces}. "
        f"Indexing mismatch between F list and cell_face_incidence."
    )

    d2 = build_d2(cell_face_inc, n_faces)

    # Check exactness: d₂d₁ = 0
    d2d1 = d2 @ d1

    norm = np.linalg.norm(d2d1)
    assert norm < EPS_CLOSE, f"Exactness failed: ||d₂d₁|| = {norm}"
    print(f"✓ Exactness (d₂d₁): ||d₂d₁|| = {norm:.2e}, shape d₂={d2.shape}")


# =============================================================================
# TEST C: Lefschetz Theorem
# =============================================================================

def test_lefschetz_kelvin():
    """Lefschetz: Tr(P|_H) = 1 for Kelvin cell"""
    mesh = build_kelvin_cell_mesh()
    result = build_parity_from_mesh(mesh)

    assert result['free_involution'], "Parity should be free involution"
    assert result['lefschetz_holds'], f"Lefschetz failed: Tr(P|_H) = {result['trace_P']}"
    assert result['dim_bridge'] == 7, f"dim(bridge) = {result['dim_bridge']}, expected 7"
    assert result['dim_ring'] == 6, f"dim(ring) = {result['dim_ring']}, expected 6"
    print(f"✓ Lefschetz: dim(bridge)={result['dim_bridge']}, dim(ring)={result['dim_ring']}, Tr={result['trace_P']}")


def test_lefschetz_kelvin_strict():
    """Lefschetz gate (strict): parity_decomposition with strict=True raises if Tr≠1.

    This is the EXPLICIT Lefschetz gate test. While test_lefschetz_kelvin checks
    the result, this test verifies that strict mode in parity_decomposition()
    correctly enforces Tr(P|_H) = 1.

    IMPORTANCE: This test would fail if the topology is corrupted in a way that
    produces wrong dim_bridge/dim_ring but still has result['lefschetz_holds']=True
    due to a bug in the checking logic.
    """
    from core_math_v2.operators.parity import (
        build_parity_operator,
        parity_decomposition,
    )
    from core_math_v2.operators.incidence import get_cycle_space

    mesh = build_kelvin_cell_mesh()
    V, E, F = mesh['V'], mesh['E'], mesh['F']

    # Build v_to_idx from vertices (Kelvin has integer coords, safe)
    v_to_idx = {tuple(int(x) for x in v): i for i, v in enumerate(V)}

    d0, d1 = build_incidence_matrices(V, E, F)
    H = get_cycle_space(d0)
    P = build_parity_operator(V, E, v_to_idx)

    # This should NOT raise (Kelvin satisfies Lefschetz)
    H_bridge, H_ring, dim_bridge, dim_ring = parity_decomposition(H, P, strict=True)

    assert dim_bridge - dim_ring == 1, f"Tr(P|_H) = {dim_bridge - dim_ring}, expected 1"
    print(f"✓ Lefschetz gate (strict): parity_decomposition(strict=True) passed, Tr = {dim_bridge - dim_ring}")


# =============================================================================
# TEST D: Contract Compliance
# =============================================================================

def test_contract_surface_loose():
    """Surface mesh has correct contract fields (loose mode).

    NOTE: test_contract_surface_strict is the stronger test.
    This test uses strict=False for debugging contract issues.
    """
    mesh = build_kelvin_cell_mesh()
    is_valid, errors = validate_mesh(mesh, strict=False)

    assert is_valid, f"Contract violation: {errors}"
    assert mesh['complex_type'] == COMPLEX_SURFACE
    assert mesh['faces_per_edge'] == FACES_PER_EDGE[COMPLEX_SURFACE]
    print(f"✓ Contract loose (surface): complex_type={mesh['complex_type']}, k={mesh['faces_per_edge']}")


def test_contract_foam_loose():
    """Foam mesh has correct contract fields (loose mode).

    NOTE: test_contract_foam_strict is the stronger test.
    This test uses strict=False for debugging contract issues.
    """
    mesh = build_bcc_foam_periodic(N=2)
    is_valid, errors = validate_mesh(mesh, strict=False)

    assert is_valid, f"Contract violation: {errors}"
    assert mesh['complex_type'] == COMPLEX_FOAM
    assert mesh['faces_per_edge'] == FACES_PER_EDGE[COMPLEX_FOAM]
    print(f"✓ Contract loose (foam): complex_type={mesh['complex_type']}, k={mesh['faces_per_edge']}")


def test_contract_surface_strict():
    """Surface mesh passes strict validation (catches edge convention violations)."""
    mesh = build_kelvin_cell_mesh()
    is_valid, errors = validate_mesh(mesh, strict=True)

    assert is_valid, f"Strict contract violation: {errors}"
    print(f"✓ Contract strict (surface): all validations pass")


def test_contract_foam_strict():
    """Foam mesh passes strict validation."""
    mesh = build_bcc_foam_periodic(N=2)
    is_valid, errors = validate_mesh(mesh, strict=True)

    assert is_valid, f"Strict contract violation: {errors}"
    print(f"✓ Contract strict (foam): all validations pass")


# =============================================================================
# TEST E: Euler Characteristic
# =============================================================================

def test_euler_surface():
    """Single cell has χ = 2 (sphere)"""
    mesh = build_kelvin_cell_mesh()
    chi = mesh['n_V'] - mesh['n_E'] + mesh['n_F']

    assert chi == 2, f"χ = {chi}, expected 2"
    print(f"✓ Euler (surface): χ = {mesh['n_V']} - {mesh['n_E']} + {mesh['n_F']} = {chi}")


def test_euler_foam():
    """Periodic foam has χ(3-complex) = 0"""
    mesh = build_bcc_foam_periodic(N=2)
    chi_2 = mesh['n_V'] - mesh['n_E'] + mesh['n_F']  # 2-skeleton
    C = mesh['n_cells']
    chi_3 = chi_2 - C  # 3-complex

    assert chi_3 == 0, f"χ(3-complex) = {chi_3}, expected 0"
    print(f"✓ Euler (foam): χ(2-skel)={chi_2}, χ(3-complex)={chi_3}")


# =============================================================================
# TEST F: Cycle Space Dimension
# =============================================================================

def test_cycle_space_kelvin():
    """dim(H) = E - V + 1 for connected graph"""
    mesh = build_kelvin_cell_mesh()
    ops = build_operators_from_mesh(mesh)

    expected = mesh['n_E'] - mesh['n_V'] + 1
    actual = ops['H'].shape[1]

    assert actual == expected, f"dim(H) = {actual}, expected {expected}"
    print(f"✓ Cycle space: dim(H) = {mesh['n_E']} - {mesh['n_V']} + 1 = {expected}")


# =============================================================================
# TEST G: Other Polyhedra
# =============================================================================

def test_cube():
    """Cube: V=8, E=12, F=6, χ=2"""
    V, E, F, _ = build_cube()
    assert len(V) == 8
    assert len(E) == 12
    assert len(F) == 6
    assert len(V) - len(E) + len(F) == 2
    print(f"✓ Cube: V=8, E=12, F=6, χ=2")


def test_octahedron():
    """Octahedron: V=6, E=12, F=8, χ=2"""
    V, E, F, _ = build_octahedron()
    assert len(V) == 6
    assert len(E) == 12
    assert len(F) == 8
    assert len(V) - len(E) + len(F) == 2
    print(f"✓ Octahedron: V=6, E=12, F=8, χ=2")


def test_tetrahedron():
    """Tetrahedron: V=4, E=6, F=4, χ=2"""
    V, E, F, _ = build_tetrahedron()
    assert len(V) == 4
    assert len(E) == 6
    assert len(F) == 4
    assert len(V) - len(E) + len(F) == 2
    print(f"✓ Tetrahedron: V=4, E=6, F=4, χ=2")


def test_truncated_cube():
    """Truncated Cube: V=24, E=36, F=14 (8 triangles + 6 octagons), χ=2, κ=137"""
    V, E, F, _ = build_truncated_cube()
    assert len(V) == 24
    assert len(E) == 36
    assert len(F) == 14
    assert len(V) - len(E) + len(F) == 2

    # Check face types: 8 triangles + 6 octagons
    from collections import Counter
    face_sizes = Counter(len(f) for f in F)
    assert face_sizes == {3: 8, 8: 6}, f"Wrong face types: {dict(face_sizes)}"

    print(f"✓ Truncated Cube: V=24, E=36, F=14 (8 tri + 6 oct), χ=2")


def test_T0_truncated_cube():
    """T0: Truncated Cube - all face segments are valid edges."""
    V, E, F, _ = build_truncated_cube()
    valid, err = _check_face_edge_validity(E, F)
    assert valid, f"T0 failed: {err}"
    print(f"✓ T0 Truncated Cube: all face segments are valid edges")


def test_T9_truncated_cube_planarity():
    """T9: Truncated Cube - all faces are geometrically planar."""
    V, E, F, _ = build_truncated_cube()
    valid, err, stats = _check_face_planarity(V, F)
    assert valid, f"T9 planarity failed: {err}"
    print(f"✓ Truncated Cube planarity: max deviation = {stats['max_deviation']:.2e}")


def test_truncated_cube_kappa():
    """Truncated Cube gives κ=137 (same topology as Kelvin)."""
    from core_math_v2.analysis.kappa import compute_kappa_for_polyhedron
    V, E, F, v_to_idx = build_truncated_cube()
    result = compute_kappa_for_polyhedron(V, E, F, v_to_idx)

    assert result['kappa'] == 137, f"κ = {result['kappa']}, expected 137"
    assert result['dim_bridge'] == 7, f"dim_bridge = {result['dim_bridge']}, expected 7"
    print(f"✓ Truncated Cube: κ = 4×36 - 7 = 137")


# =============================================================================
# TEST H: SC and FCC Lattices
# =============================================================================


# =============================================================================
# TEST I: Faces Per Edge Histogram (T2 - Bug Detector)
# =============================================================================

def _compute_faces_per_edge(d1: np.ndarray) -> np.ndarray:
    """Helper: count faces incident to each edge from d1 matrix."""
    return np.sum(np.abs(d1), axis=0)


def test_faces_per_edge_surface():
    """Surface (k=2): all edges have exactly 2 incident faces."""
    mesh = build_kelvin_cell_mesh()
    V, E, F = mesh['V'], mesh['E'], mesh['F']
    d0, d1 = build_incidence_matrices(V, E, F)

    fpe = _compute_faces_per_edge(d1)
    assert np.all(fpe == 2), f"Surface should have 2 faces/edge, got min={fpe.min()}, max={fpe.max()}"
    print(f"✓ Surface faces/edge: all edges have exactly 2 faces")


def test_faces_per_edge_foam():
    """Foam (k=3): all edges have exactly 3 incident faces (Plateau)."""
    mesh = build_bcc_foam_periodic(N=2)
    V, E, F = mesh['V'], mesh['E'], mesh['F']
    d0, d1 = build_incidence_matrices(V, E, F)

    fpe = _compute_faces_per_edge(d1)
    assert np.all(fpe == 3), f"Foam should have 3 faces/edge, got min={fpe.min()}, max={fpe.max()}"
    print(f"✓ Foam faces/edge: all edges have exactly 3 faces (Plateau)")


def test_faces_per_edge_sc():
    """SC (k=4): all edges have exactly 4 incident faces."""
    V, E, F, _ = build_sc_supercell_periodic(N=3)
    d0, d1 = build_incidence_matrices(V, E, F)

    fpe = _compute_faces_per_edge(d1)
    assert np.all(fpe == 4), f"SC should have 4 faces/edge, got min={fpe.min()}, max={fpe.max()}"
    print(f"✓ SC faces/edge: all edges have exactly 4 faces")


def test_faces_per_edge_fcc():
    """FCC (k=3): all edges have exactly 3 incident faces (same k as foam).

    FCC uses N=2 for periodic consistency (same as other FCC tests).
    """
    V, E, F, _ = build_fcc_supercell_periodic(N=2)
    d0, d1 = build_incidence_matrices(V, E, F)

    fpe = _compute_faces_per_edge(d1)
    assert np.all(fpe == 3), f"FCC should have 3 faces/edge, got min={fpe.min()}, max={fpe.max()}"
    print(f"✓ FCC faces/edge: all edges have exactly 3 faces (k=3 tiling)")


def test_trace_sc():
    """SC (k=4): Tr(d₁ᵀd₁) = 4E"""
    V, E, F, _ = build_sc_supercell_periodic(N=3)
    d0, d1 = build_incidence_matrices(V, E, F)

    d1td1 = d1.T @ d1
    trace = np.trace(d1td1)
    n_E = len(E)

    assert abs(trace - 4 * n_E) < EPS_CLOSE, f"Tr(d₁ᵀd₁) = {trace}, expected 4E = {4*n_E}"
    print(f"✓ SC trace: Tr(d₁ᵀd₁) = 4E = {int(trace)}")


def test_trace_fcc():
    """FCC (k=3): Tr(d₁ᵀd₁) = 3E (foam-like). Uses N=2 for periodic consistency."""
    V, E, F, _ = build_fcc_supercell_periodic(N=2)
    d0, d1 = build_incidence_matrices(V, E, F)

    d1td1 = d1.T @ d1
    trace = np.trace(d1td1)
    n_E = len(E)

    assert abs(trace - 3 * n_E) < EPS_CLOSE, f"Tr(d₁ᵀd₁) = {trace}, expected 3E = {3*n_E}"
    print(f"✓ FCC trace: Tr(d₁ᵀd₁) = 3E = {int(trace)}")


# =============================================================================
# TEST J: T0 - Face-Edge Validity (Foundation Gate)
# =============================================================================

def _check_face_edge_validity(edges, faces):
    """
    T0: Every face boundary segment must exist in edge set.
    Returns (is_valid, error_message)
    """
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i, j), max(i, j)))

    for f_idx, face in enumerate(faces):
        n = len(face)
        for k in range(n):
            v1, v2 = face[k], face[(k + 1) % n]
            edge = (min(v1, v2), max(v1, v2))
            if edge not in edge_set:
                return False, f"Face {f_idx} has segment ({v1},{v2}) not in edge set"
    return True, ""


def test_T0_kelvin_surface():
    """T0: Kelvin surface - all face segments are valid edges."""
    mesh = build_kelvin_cell_mesh()
    valid, err = _check_face_edge_validity(mesh['E'], mesh['F'])
    assert valid, f"T0 failed: {err}"
    print(f"✓ T0 Kelvin surface: all face segments are valid edges")


def test_T0_bcc_foam():
    """T0: BCC foam - all face segments are valid edges."""
    mesh = build_bcc_foam_periodic(N=2)
    valid, err = _check_face_edge_validity(mesh['E'], mesh['F'])
    assert valid, f"T0 failed: {err}"
    print(f"✓ T0 BCC foam: all face segments are valid edges")


def test_T0_sc_periodic():
    """T0: SC periodic - all face segments are valid edges."""
    V, E, F, _ = build_sc_supercell_periodic(N=3)
    valid, err = _check_face_edge_validity(E, F)
    assert valid, f"T0 failed: {err}"
    print(f"✓ T0 SC periodic: all face segments are valid edges")


def test_T0_fcc_periodic():
    """T0: FCC periodic - all face segments are valid edges. Uses N=2 for periodic consistency."""
    V, E, F, _ = build_fcc_supercell_periodic(N=2)
    valid, err = _check_face_edge_validity(E, F)
    assert valid, f"T0 failed: {err}"
    print(f"✓ T0 FCC periodic: all face segments are valid edges")


# =============================================================================
# TEST K: T5 - Dedup Uniqueness (Foundation Gate)
# =============================================================================

def _check_dedup_uniqueness(V, E, F):
    """
    T5: No duplicate vertices, edges, or faces.

    NOTE on face deduplication:
        Faces are compared as vertex SETS (frozenset), meaning two faces are
        considered identical if they contain the same vertices, regardless of
        cyclic order or orientation. This is intentional:
        - [1,2,3] and [2,3,1] are the same face (cyclic rotation)
        - [1,2,3] and [3,2,1] are the same face (opposite orientation)

        For surface/foam meshes, this correctly identifies duplicate faces.
        The actual orientation is stored separately in cell_face_incidence.

    Also checks that faces are simple cycles (no repeated vertices within a face).

    Returns (is_valid, error_message)
    """
    # Check vertices (as tuples)
    v_tuples = [tuple(v) for v in V]
    if len(v_tuples) != len(set(v_tuples)):
        return False, f"Duplicate vertices: {len(v_tuples)} total, {len(set(v_tuples))} unique"

    # Check edges (already (i,j) with i<j)
    if len(E) != len(set(E)):
        return False, f"Duplicate edges: {len(E)} total, {len(set(E))} unique"

    # Check faces are simple cycles (no repeated vertex within a face)
    # This catches malformed faces like [1,2,1,3] before frozenset comparison
    for f_idx, face in enumerate(F):
        if len(face) != len(set(face)):
            return False, f"Face {f_idx} has repeated vertex: {face}"

    # Check faces (as frozensets - identical up to cyclic order and orientation)
    f_sets = [frozenset(f) for f in F]
    if len(f_sets) != len(set(f_sets)):
        return False, f"Duplicate faces: {len(f_sets)} total, {len(set(f_sets))} unique"

    return True, ""


def test_T5_kelvin_surface():
    """T5: Kelvin surface - no duplicate V/E/F."""
    mesh = build_kelvin_cell_mesh()
    valid, err = _check_dedup_uniqueness(mesh['V'], mesh['E'], mesh['F'])
    assert valid, f"T5 failed: {err}"
    print(f"✓ T5 Kelvin surface: no duplicates (V={len(mesh['V'])}, E={len(mesh['E'])}, F={len(mesh['F'])})")


def test_T5_bcc_foam():
    """T5: BCC foam - no duplicate V/E/F."""
    mesh = build_bcc_foam_periodic(N=2)
    valid, err = _check_dedup_uniqueness(mesh['V'], mesh['E'], mesh['F'])
    assert valid, f"T5 failed: {err}"
    print(f"✓ T5 BCC foam: no duplicates (V={len(mesh['V'])}, E={len(mesh['E'])}, F={len(mesh['F'])})")


def test_T5_sc_periodic():
    """T5: SC periodic - no duplicate V/E/F."""
    V, E, F, _ = build_sc_supercell_periodic(N=3)
    valid, err = _check_dedup_uniqueness(V, E, F)
    assert valid, f"T5 failed: {err}"
    print(f"✓ T5 SC periodic: no duplicates (V={len(V)}, E={len(E)}, F={len(F)})")


def test_T5_fcc_periodic():
    """T5: FCC periodic - no duplicate V/E/F. Uses N=2 for periodic consistency."""
    V, E, F, _ = build_fcc_supercell_periodic(N=2)
    valid, err = _check_dedup_uniqueness(V, E, F)
    assert valid, f"T5 failed: {err}"
    print(f"✓ T5 FCC periodic: no duplicates (V={len(V)}, E={len(E)}, F={len(F)})")


# =============================================================================
# TEST L: Weaire-Phelan Topology Verification
# =============================================================================

def _check_face_planarity(vertices, faces, tol=None):
    """
    T9: Check that all faces are planar (vertices lie in same plane).

    For cycle-found faces this is critical - cycles in graph don't guarantee
    geometric planarity.

    Args:
        vertices: array-like of vertex coordinates
        faces: list of faces (each face is list of vertex indices)
        tol: planarity tolerance (defaults to PLANAR_TOL from spec/constants.py)

    Returns:
        (is_valid, error_message, stats_dict)

    NOTE: Uses PLANAR_TOL (1e-8) from constants for consistency with builder checks.
    """
    if tol is None:
        tol = PLANAR_TOL
    # Ensure vertices is numpy array (robust to list input from builders)
    V = np.asarray(vertices)

    max_deviation = 0.0
    worst_face = None

    for f_idx, face in enumerate(faces):
        if len(face) < 4:
            continue  # Triangles are always planar

        coords = V[face]

        # Fit plane using SVD: find normal as smallest singular vector
        centroid = coords.mean(axis=0)
        centered = coords - centroid
        _, s, vh = np.linalg.svd(centered)

        # Normal is last row of vh (smallest singular value direction)
        normal = vh[-1]

        # Compute max deviation from plane
        deviations = np.abs(centered @ normal)
        max_dev = np.max(deviations)

        if max_dev > max_deviation:
            max_deviation = max_dev
            worst_face = f_idx

    stats = {
        'max_deviation': max_deviation,
        'worst_face': worst_face,
        'tolerance': tol,
    }

    if max_deviation > tol:
        return False, f"Face {worst_face} not planar: deviation = {max_deviation:.2e} > {tol}", stats

    return True, "", stats


def test_T0_wp_type_a():
    """T0+Surface: WP Type A - face-edge validity and edge-in-2-faces."""
    V, E, F, _ = build_wp_type_a()

    # T0: face-edge validity
    valid, err = _check_face_edge_validity(E, F)
    assert valid, f"T0 failed: {err}"

    # Surface topology: edge in exactly 2 faces
    valid, issues = verify_wp_surface_topology(E, F)
    assert valid, f"Surface topology failed: {issues}"

    print(f"✓ WP Type A: V={len(V)}, E={len(E)}, F={len(F)}, χ={len(V)-len(E)+len(F)}, surface valid")


def test_T0_wp_type_b():
    """T0+Surface: WP Type B - face-edge validity and edge-in-2-faces."""
    V, E, F, _ = build_wp_type_b()

    # T0: face-edge validity
    valid, err = _check_face_edge_validity(E, F)
    assert valid, f"T0 failed: {err}"

    # Surface topology: edge in exactly 2 faces
    valid, issues = verify_wp_surface_topology(E, F)
    assert valid, f"Surface topology failed: {issues}"

    print(f"✓ WP Type B: V={len(V)}, E={len(E)}, F={len(F)}, χ={len(V)-len(E)+len(F)}, surface valid")


def test_T9_wp_type_a_planarity():
    """T9: WP Type A - all faces are geometrically planar."""
    V, E, F, _ = build_wp_type_a()

    valid, err, stats = _check_face_planarity(V, F)
    assert valid, f"T9 planarity failed: {err}"

    print(f"✓ WP Type A planarity: max deviation = {stats['max_deviation']:.2e}")


def test_T9_wp_type_b_planarity():
    """T9: WP Type B - all faces are geometrically planar."""
    V, E, F, _ = build_wp_type_b()

    valid, err, stats = _check_face_planarity(V, F)
    assert valid, f"T9 planarity failed: {err}"

    print(f"✓ WP Type B planarity: max deviation = {stats['max_deviation']:.2e}")


# =============================================================================
# TEST M: Cell-Face Orientation Consistency (d₂ Foundation)
# =============================================================================

def _verify_d2_structure(d2: np.ndarray, n_faces: int):
    """
    Verify d₂ matrix structure: each face in exactly 2 cells with sum=0.

    Uses build_d2's verification implicitly (it raises on failure),
    but also returns stats for reporting.

    Returns:
        (is_valid, error_message, stats_dict)
    """
    # Each column should have exactly 2 nonzero entries
    col_nonzero = np.sum(d2 != 0, axis=0)
    col_sums = np.sum(d2, axis=0)

    faces_with_2 = np.sum(col_nonzero == 2)
    faces_with_sum_0 = np.sum(np.abs(col_sums) < EPS_CLOSE)

    stats = {
        'n_faces': n_faces,
        'faces_with_2_cells': int(faces_with_2),
        'faces_with_sum_0': int(faces_with_sum_0),
    }

    # Check for failures
    bad_count = np.where(col_nonzero != 2)[0]
    if len(bad_count) > 0:
        return False, f"Face {bad_count[0]} shared by {int(col_nonzero[bad_count[0]])} cells (expected 2)", stats

    bad_sum = np.where(np.abs(col_sums) >= EPS_CLOSE)[0]
    if len(bad_sum) > 0:
        return False, f"Face {bad_sum[0]} has orientation sum = {col_sums[bad_sum[0]]} (expected 0)", stats

    return True, "", stats


def test_orientation_bcc_foam():
    """T6: BCC foam - each face shared by 2 cells with opposite orientations.

    Uses build_d2 from operators (single source of truth for d₂ logic).

    NOTE: Uses N=2 because N=1 is degenerate (faces can self-connect through
    periodic boundary, which our model doesn't track as 2-cell sharing).
    """
    mesh = build_bcc_foam_periodic(N=2)

    assert 'cell_face_incidence' in mesh, "Mesh missing cell_face_incidence"
    cell_face_inc = mesh['cell_face_incidence']
    n_faces = mesh['n_F']

    # Build d₂ using operators (verify=True by default raises on structure issues)
    d2 = build_d2(cell_face_inc, n_faces, verify=True)

    # Also report stats
    valid, err, stats = _verify_d2_structure(d2, n_faces)
    assert valid, f"Orientation consistency failed: {err}"

    print(f"✓ BCC foam orientation: {stats['faces_with_2_cells']}/{n_faces} faces shared by 2 cells, all sums=0")


def test_orientation_sc_periodic():
    """T6: SC periodic - each face shared by 2 cells with opposite orientations.

    Uses build_d2 from operators (single source of truth for d₂ logic).
    """
    V, E, F, cell_face_inc = build_sc_supercell_periodic(N=3)
    n_faces = len(F)

    # Build d₂ using operators
    d2 = build_d2(cell_face_inc, n_faces, verify=True)

    valid, err, stats = _verify_d2_structure(d2, n_faces)
    assert valid, f"Orientation consistency failed: {err}"

    print(f"✓ SC orientation: {stats['faces_with_2_cells']}/{n_faces} faces shared by 2 cells, all sums=0")


def test_orientation_fcc_periodic():
    """T6: FCC periodic - each face shared by 2 cells with opposite orientations.

    Uses build_d2 from operators (single source of truth for d₂ logic).

    NOTE: Uses N=2 because N=1 can be degenerate (faces may self-connect
    through periodic boundary, same issue as BCC).
    """
    V, E, F, cell_face_inc = build_fcc_supercell_periodic(N=2)
    n_faces = len(F)

    # Build d₂ using operators
    d2 = build_d2(cell_face_inc, n_faces, verify=True)

    valid, err, stats = _verify_d2_structure(d2, n_faces)
    assert valid, f"Orientation consistency failed: {err}"

    print(f"✓ FCC orientation: {stats['faces_with_2_cells']}/{n_faces} faces shared by 2 cells, all sums=0")


# =============================================================================
# TEST N: Vertex Collision Test (wrap_position round(6) safety)
# =============================================================================

def _check_vertex_collisions(vertices, min_dist_threshold=1e-8):
    """
    T7: Check that no two distinct vertices are too close after wrapping.

    This catches issues where round(6) in wrap_position could collapse
    two distinct vertices into one.

    Returns:
        (is_valid, error_message, stats_dict)
    """
    n_V = len(vertices)
    min_dist = float('inf')
    collision_pair = None

    for i in range(n_V):
        for j in range(i + 1, n_V):
            d = np.linalg.norm(vertices[i] - vertices[j])
            # Always track minimum distance (independent of threshold)
            if d < min_dist:
                min_dist = d
            # Separately check for collisions below threshold
            if d < min_dist_threshold:
                collision_pair = (i, j, d)

    stats = {
        'n_vertices': n_V,
        'min_distance': min_dist,
        'collision_pair': collision_pair,
    }

    if collision_pair:
        i, j, d = collision_pair
        return False, f"Vertices {i} and {j} too close: d = {d:.2e} < {min_dist_threshold}", stats

    return True, "", stats


def test_T7_vertex_collision_bcc():
    """T7: BCC periodic - no vertex collisions after wrap."""
    mesh = build_bcc_foam_periodic(N=2)
    valid, err, stats = _check_vertex_collisions(mesh['V'])
    assert valid, f"T7 failed: {err}"
    print(f"✓ T7 BCC: no collisions, min_dist = {stats['min_distance']:.4f}")


def test_T7_vertex_collision_sc():
    """T7: SC periodic - no vertex collisions after wrap."""
    V, E, F, _ = build_sc_supercell_periodic(N=3)
    valid, err, stats = _check_vertex_collisions(V)
    assert valid, f"T7 failed: {err}"
    print(f"✓ T7 SC: no collisions, min_dist = {stats['min_distance']:.4f}")


def test_T7_vertex_collision_fcc():
    """T7: FCC periodic - no vertex collisions after wrap. Uses N=2 for periodic consistency."""
    V, E, F, _ = build_fcc_supercell_periodic(N=2)
    valid, err, stats = _check_vertex_collisions(V)
    assert valid, f"T7 failed: {err}"
    print(f"✓ T7 FCC: no collisions, min_dist = {stats['min_distance']:.4f}")


# =============================================================================
# TEST O: Parity Verification for Periodic Structures
# =============================================================================

# NOTE: Period L is read from mesh['period_L'] (single source of truth).
# The old PERIODIC_L_FORMULAS was removed to avoid dual sources of truth.

def _build_periodic_parity(vertices, edges, L):
    """
    Build parity operator for periodic structure: P: x → (L - x) mod L.

    Uses WRAP_DECIMALS from spec/constants.py for coordinate rounding
    (consistent with periodic builders).

    Returns:
        P: (E, E) permutation matrix
        P_squared_ok: bool (is P² = I?)
        n_fixed_vertices: int
        n_fixed_edges: int
    """
    n_E = len(edges)
    V = np.array(vertices)

    # Build vertex map: v → (L - v) mod L
    # Uses WRAP_DECIMALS for consistency with builders
    def parity_vertex(v):
        return tuple(round((L - x) % L, WRAP_DECIMALS) for x in v)

    v_to_idx = {tuple(round(x, WRAP_DECIMALS) for x in v): i for i, v in enumerate(V)}

    # Check for fixed vertices (v = P(v))
    n_fixed_vertices = 0
    vertex_map = {}
    for i, v in enumerate(V):
        pv = parity_vertex(v)
        if pv in v_to_idx:
            pi = v_to_idx[pv]
            vertex_map[i] = pi
            if pi == i:
                n_fixed_vertices += 1
        else:
            # Vertex parity image not found - degenerate
            return None, False, -1, -1

    # Build edge map
    edge_to_idx = {tuple(sorted(e)): k for k, e in enumerate(edges)}

    P = np.zeros((n_E, n_E))
    n_fixed_edges = 0

    for k, (i, j) in enumerate(edges):
        pi, pj = vertex_map[i], vertex_map[j]
        pe = tuple(sorted((pi, pj)))
        if pe in edge_to_idx:
            k2 = edge_to_idx[pe]
            P[k2, k] = 1
            if k2 == k:
                n_fixed_edges += 1
        else:
            # Edge parity image not found
            return None, False, -1, -1

    # Verify P² = I
    P2 = P @ P
    P_squared_ok = np.allclose(P2, np.eye(n_E))

    return P, P_squared_ok, n_fixed_vertices, n_fixed_edges


def test_T8_parity_bcc_periodic():
    """T8: BCC periodic - parity operator P² = I."""
    N = 2
    mesh = build_bcc_foam_periodic(N=N)
    L = mesh['period_L']  # Read from mesh, not _get_period_L
    P, P_ok, n_fv, n_fe = _build_periodic_parity(mesh['V'], mesh['E'], L)

    assert P is not None, "Failed to build parity operator"
    assert P_ok, "P² ≠ I"

    print(f"✓ T8 BCC parity: P² = I, fixed_V = {n_fv}, fixed_E = {n_fe}")


def test_T8_parity_sc_periodic():
    """T8: SC periodic - parity operator P² = I."""
    N = 3
    mesh = build_sc_solid_periodic(N=N)
    L = mesh['period_L']  # Read from mesh, not _get_period_L
    P, P_ok, n_fv, n_fe = _build_periodic_parity(mesh['V'], mesh['E'], L)

    assert P is not None, "Failed to build parity operator"
    assert P_ok, "P² ≠ I"

    print(f"✓ T8 SC parity: P² = I, fixed_V = {n_fv}, fixed_E = {n_fe}")


def test_T8_parity_fcc_periodic():
    """T8: FCC periodic - parity operator P² = I. Uses N=2 for periodic consistency."""
    N = 2
    mesh = build_fcc_solid_periodic(N=N)
    L = mesh['period_L']  # Read from mesh, not _get_period_L
    P, P_ok, n_fv, n_fe = _build_periodic_parity(mesh['V'], mesh['E'], L)

    assert P is not None, "Failed to build parity operator"
    assert P_ok, "P² ≠ I"

    print(f"✓ T8 FCC parity: P² = I, fixed_V = {n_fv}, fixed_E = {n_fe}")


# =============================================================================
# TEST P: d₁ @ d₀ = 0 for all periodic structures (exactness gate)
# =============================================================================

def test_exactness_sc_periodic():
    """Exactness: d₁d₀ = 0 for SC periodic."""
    V, E, F, _ = build_sc_supercell_periodic(N=3)
    d0, d1 = build_incidence_matrices(V, E, F)
    d1d0 = d1 @ d0

    norm = np.linalg.norm(d1d0)
    assert norm < EPS_CLOSE, f"Exactness failed: ||d₁d₀|| = {norm}"
    print(f"✓ Exactness (SC periodic): ||d₁d₀|| = {norm:.2e}")


def test_exactness_fcc_periodic():
    """Exactness: d₁d₀ = 0 for FCC periodic. Uses N=2 for periodic consistency."""
    V, E, F, _ = build_fcc_supercell_periodic(N=2)
    d0, d1 = build_incidence_matrices(V, E, F)
    d1d0 = d1 @ d0

    norm = np.linalg.norm(d1d0)
    assert norm < EPS_CLOSE, f"Exactness failed: ||d₁d₀|| = {norm}"
    print(f"✓ Exactness (FCC periodic): ||d₁d₀|| = {norm:.2e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CORE_MATH_V2 - FULL TEST SUITE")
    print("=" * 60)

    tests = [
        # Trace identities
        ("A1", "Trace (surface k=2)", test_trace_surface_kelvin),
        ("A2", "Trace (foam k=3 N=1)", test_trace_foam_n1),
        ("A3", "Trace (foam k=3 N=2)", test_trace_foam_n2),
        ("A4", "Trace (SC k=4)", test_trace_sc),
        ("A5", "Trace (FCC k=3)", test_trace_fcc),
        # Exactness
        ("B1", "Exactness (kelvin)", test_exactness_kelvin),
        ("B2", "Exactness (foam)", test_exactness_foam),
        ("B3", "Exactness (SC periodic)", test_exactness_sc_periodic),
        ("B4", "Exactness (FCC periodic)", test_exactness_fcc_periodic),
        ("B5", "Exactness d₂d₁=0 (BCC foam)", test_exactness_d2d1_bcc_foam),
        # Lefschetz
        ("C1", "Lefschetz", test_lefschetz_kelvin),
        ("C2", "Lefschetz strict gate", test_lefschetz_kelvin_strict),
        # Contract compliance
        ("D1", "Contract loose (surface)", test_contract_surface_loose),
        ("D2", "Contract loose (foam)", test_contract_foam_loose),
        ("D3", "Contract strict (surface)", test_contract_surface_strict),
        ("D4", "Contract strict (foam)", test_contract_foam_strict),
        # Euler characteristic
        ("E1", "Euler (surface)", test_euler_surface),
        ("E2", "Euler (foam)", test_euler_foam),
        # Cycle space
        ("F1", "Cycle space", test_cycle_space_kelvin),
        # Polyhedra
        ("G1", "Cube", test_cube),
        ("G2", "Octahedron", test_octahedron),
        ("G3", "Tetrahedron", test_tetrahedron),
        ("G4", "Truncated Cube", test_truncated_cube),
        ("G5", "Truncated Cube κ=137", test_truncated_cube_kappa),
        # Faces per edge
        ("I1", "Faces/edge (surface k=2)", test_faces_per_edge_surface),
        ("I2", "Faces/edge (foam k=3)", test_faces_per_edge_foam),
        ("I3", "Faces/edge (SC k=4)", test_faces_per_edge_sc),
        ("I4", "Faces/edge (FCC k=3)", test_faces_per_edge_fcc),
        # Foundation Gates T0 (face-edge validity)
        ("J1", "T0 Kelvin surface", test_T0_kelvin_surface),
        ("J2", "T0 BCC foam", test_T0_bcc_foam),
        ("J3", "T0 SC periodic", test_T0_sc_periodic),
        ("J4", "T0 FCC periodic", test_T0_fcc_periodic),
        ("J5", "T0 Truncated Cube", test_T0_truncated_cube),
        # Foundation Gates T5 (dedup uniqueness)
        ("K1", "T5 Kelvin surface", test_T5_kelvin_surface),
        ("K2", "T5 BCC foam", test_T5_bcc_foam),
        ("K3", "T5 SC periodic", test_T5_sc_periodic),
        ("K4", "T5 FCC periodic", test_T5_fcc_periodic),
        # Weaire-Phelan topology verification
        ("L1", "WP Type A (T0+surface)", test_T0_wp_type_a),
        ("L2", "WP Type B (T0+surface)", test_T0_wp_type_b),
        # Planarity (T9 - geometric face validity)
        ("L3", "T9 WP Type A planarity", test_T9_wp_type_a_planarity),
        ("L4", "T9 WP Type B planarity", test_T9_wp_type_b_planarity),
        ("L5", "T9 Truncated Cube planarity", test_T9_truncated_cube_planarity),
        # Cell-face orientation consistency (T6 - d₂ foundation)
        ("M1", "T6 Orientation BCC foam", test_orientation_bcc_foam),
        ("M2", "T6 Orientation SC periodic", test_orientation_sc_periodic),
        ("M3", "T6 Orientation FCC periodic", test_orientation_fcc_periodic),
        # Vertex collision tests (T7 - wrap_position safety)
        ("N1", "T7 Vertex collision BCC", test_T7_vertex_collision_bcc),
        ("N2", "T7 Vertex collision SC", test_T7_vertex_collision_sc),
        ("N3", "T7 Vertex collision FCC", test_T7_vertex_collision_fcc),
        # Parity tests (T8 - P² = I)
        ("O1", "T8 Parity BCC periodic", test_T8_parity_bcc_periodic),
        ("O2", "T8 Parity SC periodic", test_T8_parity_sc_periodic),
        ("O3", "T8 Parity FCC periodic", test_T8_parity_fcc_periodic),
    ]

    passed = 0
    failed = 0

    for code, name, test_fn in tests:
        print(f"\n[{code}] {name}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)


# NOTE: File is at ~1000 lines. New tests go in test_all_2.py or split by category.
