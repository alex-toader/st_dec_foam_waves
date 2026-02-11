"""
Guard and Edge Case Tests for core_math
==========================================

Tests for boundary conditions, guards, and edge cases.
Separated from test_all.py to keep main suite focused on invariants.

Run: python -m pytest tests_math/test_guards.py -v

NOTE: When this file reaches ~500 lines, split by category.
"""

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_math.builders import build_bcc_foam_periodic
from core_math.builders.solids_periodic import build_sc_supercell_periodic
from core_math.spec.structures import canonical_face


# =============================================================================
# P1: CRITICAL - Guards for degenerate cases
# =============================================================================

def test_bcc_foam_n1_raises():
    """P1.1: BCC foam N=1 should raise (self-glued faces break d₂)."""
    with pytest.raises(ValueError, match=r"N\s*(?:≥|>=)\s*2"):
        build_bcc_foam_periodic(N=1)


def test_sc_periodic_n1_raises():
    """P1.2: SC periodic N=1 should raise (degenerate under periodic ID)."""
    with pytest.raises(ValueError, match=r"N\s*(?:≥|>=)\s*3"):
        build_sc_supercell_periodic(N=1)


def test_sc_periodic_n2_raises():
    """P1.3: SC periodic N=2 should also raise (need N≥3)."""
    with pytest.raises(ValueError, match=r"N\s*(?:≥|>=)\s*3"):
        build_sc_supercell_periodic(N=2)


# =============================================================================
# P2: IMPORTANT - canonical_face correctness
# =============================================================================

def test_canonical_face_rotation_invariant():
    """P2.1: Rotations of same face → same canonical key."""
    # All rotations should give same canonical
    rotations = [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ]

    canonicals = [canonical_face(r) for r in rotations]
    keys = [c[0] for c in canonicals]

    # All keys should be identical
    assert all(k == keys[0] for k in keys), f"Rotations gave different keys: {keys}"


def test_canonical_face_reverse_orientation():
    """P2.2: Reversed face → same key, opposite orientation."""
    face = [0, 1, 2, 3]
    face_rev = [0, 3, 2, 1]  # Same vertices, opposite winding

    canon_fwd = canonical_face(face)
    canon_rev = canonical_face(face_rev)

    # Same key (same face geometrically)
    assert canon_fwd[0] == canon_rev[0], "Forward and reverse should have same key"

    # Opposite orientation
    assert canon_fwd[1] == -canon_rev[1], "Forward and reverse should have opposite orientation"


def test_canonical_face_different_faces():
    """P2.3: Different faces → different keys."""
    face1 = [0, 1, 2, 3]
    face2 = [0, 1, 2, 4]  # Different vertex

    canon1 = canonical_face(face1)
    canon2 = canonical_face(face2)

    assert canon1[0] != canon2[0], "Different faces should have different keys"


def test_canonical_face_triangle():
    """P2.4: canonical_face works for triangles too."""
    tri = [5, 7, 9]
    tri_rot = [7, 9, 5]
    tri_rev = [5, 9, 7]

    canon = canonical_face(tri)
    canon_rot = canonical_face(tri_rot)
    canon_rev = canonical_face(tri_rev)

    # Rotation: same key, same orientation
    assert canon[0] == canon_rot[0]
    assert canon[1] == canon_rot[1]

    # Reverse: same key, opposite orientation
    assert canon[0] == canon_rev[0]
    assert canon[1] == -canon_rev[1]


# =============================================================================
# P3: faces_per_edge contract vs histogram
# =============================================================================

def test_faces_per_edge_matches_contract_kelvin():
    """P3.1: Kelvin mesh['faces_per_edge'] matches actual histogram."""
    from core_math.builders import build_kelvin_cell_mesh
    from core_math.operators.incidence import build_d1

    mesh = build_kelvin_cell_mesh()
    d1 = build_d1(mesh['V'], mesh['E'], mesh['F'])

    # Histogram from d₁
    faces_per_edge = np.sum(np.abs(d1), axis=0).astype(int)

    # All edges should have exactly k faces
    k = mesh['faces_per_edge']
    assert np.all(faces_per_edge == k), \
        f"Expected all edges to have {k} faces, got min={faces_per_edge.min()}, max={faces_per_edge.max()}"


def test_faces_per_edge_matches_contract_bcc_foam():
    """P3.2: BCC foam mesh['faces_per_edge'] matches actual histogram."""
    from core_math.operators.incidence import build_d1

    mesh = build_bcc_foam_periodic(N=2)
    d1 = build_d1(mesh['V'], mesh['E'], mesh['F'])

    faces_per_edge = np.sum(np.abs(d1), axis=0).astype(int)
    k = mesh['faces_per_edge']

    assert np.all(faces_per_edge == k), \
        f"Expected all edges to have {k} faces, got min={faces_per_edge.min()}, max={faces_per_edge.max()}"


# =============================================================================
# P3: NICE TO HAVE - WP contract validation
# =============================================================================

def test_wp_type_a_contract_validation():
    """P3.1: WP Type A passes validate_mesh."""
    from core_math.builders.weaire_phelan import build_wp_type_a
    from core_math.spec.structures import validate_mesh, create_mesh
    from core_math.spec.constants import COMPLEX_SURFACE

    V, E, F, v_to_idx = build_wp_type_a()
    mesh = create_mesh(V, E, F, COMPLEX_SURFACE, name="wp_type_a")

    is_valid, errors = validate_mesh(mesh, strict=False)
    assert is_valid, f"WP Type A failed validation: {errors}"


def test_wp_type_b_contract_validation():
    """P3.1: WP Type B passes validate_mesh."""
    from core_math.builders.weaire_phelan import build_wp_type_b
    from core_math.spec.structures import validate_mesh, create_mesh
    from core_math.spec.constants import COMPLEX_SURFACE

    V, E, F, v_to_idx = build_wp_type_b()
    mesh = create_mesh(V, E, F, COMPLEX_SURFACE, name="wp_type_b")

    is_valid, errors = validate_mesh(mesh, strict=False)
    assert is_valid, f"WP Type B failed validation: {errors}"


# =============================================================================
# P1/P2: κ WP diagnostic - verify parity doesn't invent dimensions
# =============================================================================

def test_kappa_wp_type_a_diagnostic():
    """κ WP Type A: parity succeeds, doesn't invent dimensions."""
    from core_math.analysis.weaire_phelan_kappa import compute_wp_kappa

    result = compute_wp_kappa('A')

    # Parity should succeed (Type A is centrosymmetric)
    assert result['is_centrosymmetric'], "Type A should be centrosymmetric"
    assert not result['parity_failed'], f"Parity failed: {result['parity_error']}"

    # trace_P should be computed (not None)
    assert result['trace_P'] is not None, "trace_P should be computed"

    # Dimensions should be consistent
    assert result['dim_H'] == result['dim_bridge'] + result['dim_ring'], \
        f"dim_H={result['dim_H']} != bridge({result['dim_bridge']}) + ring({result['dim_ring']})"

    # κ should be computed from actual bridge, not default 0
    # Type A: E=30, and if parity works, bridge should be > 0
    assert result['kappa'] == 4 * result['E'] - result['dim_bridge']


def test_kappa_wp_type_b_diagnostic():
    """κ WP Type B: vertices centrosymmetric, but parity may or may not be graph automorphism."""
    from core_math.analysis.weaire_phelan_kappa import compute_wp_kappa

    result = compute_wp_kappa('B')

    # Type B: vertices are centrosymmetric
    assert result['is_centrosymmetric'], "Type B vertices are centrosymmetric"

    # κ formula must be consistent regardless of parity outcome
    if result['parity_failed']:
        # Parity failed → fallback: dim_bridge=0, dim_ring=dim_H
        assert result['dim_bridge'] == 0, "Fallback: dim_bridge should be 0"
        assert result['dim_ring'] == result['dim_H'], "Fallback: dim_ring should equal dim_H"
        assert result['kappa'] == 4 * result['E'], f"κ should be 4E={4*result['E']}, got {result['kappa']}"
    else:
        # Parity succeeded → formula κ = 4E - dim_bridge holds
        assert result['kappa'] == 4 * result['E'] - result['dim_bridge']
        assert result['dim_H'] == result['dim_bridge'] + result['dim_ring']


def test_kappa_wp_reports_fixed_points():
    """κ WP: reports fixed vertices/edges for Lefschetz analysis."""
    from core_math.analysis.weaire_phelan_kappa import compute_wp_kappa

    for cell_type in ['A', 'B']:
        result = compute_wp_kappa(cell_type)

        # Fixed point counts should be reported
        assert 'n_fixed_vertices' in result
        assert 'n_fixed_edges' in result
        assert 'free_involution' in result

        # free_involution should be consistent with fixed counts
        if result['n_fixed_vertices'] == 0 and result['n_fixed_edges'] == 0:
            assert result['free_involution'], f"Type {cell_type}: should be free involution"


# =============================================================================
# P4: Import smoke tests
# =============================================================================

def test_import_smoke_builders():
    """P4.1: Import builders module works."""
    from core_math import builders
    assert hasattr(builders, 'build_kelvin_cell_mesh')
    assert hasattr(builders, 'build_bcc_foam_periodic')


def test_import_smoke_operators():
    """P4.2: Import operators module works."""
    from core_math import operators
    assert hasattr(operators, 'build_incidence_matrices')
    assert hasattr(operators, 'build_parity_operator')


def test_import_smoke_analysis():
    """P4.3: Import analysis module works."""
    from core_math import analysis
    assert hasattr(analysis, 'compute_kappa_for_polyhedron')


def test_smoke_pipeline_kelvin():
    """P4.4: Full pipeline: builder → validate → operators."""
    from core_math.builders import build_kelvin_cell_mesh
    from core_math.spec.structures import validate_mesh
    from core_math.operators.incidence import build_operators_from_mesh

    # Build
    mesh = build_kelvin_cell_mesh()

    # Validate
    is_valid, errors = validate_mesh(mesh, strict=True)
    assert is_valid, f"Validation failed: {errors}"

    # Operators
    ops = build_operators_from_mesh(mesh)
    assert 'd0' in ops
    assert 'd1' in ops
    assert 'L1' in ops


# =============================================================================
# Self-test when run directly
# =============================================================================

if __name__ == "__main__":
    import traceback

    tests = [
        # P1: Guards for degenerate cases
        test_bcc_foam_n1_raises,
        test_sc_periodic_n1_raises,
        test_sc_periodic_n2_raises,
        # P2: canonical_face correctness
        test_canonical_face_rotation_invariant,
        test_canonical_face_reverse_orientation,
        test_canonical_face_different_faces,
        test_canonical_face_triangle,
        # P3: faces_per_edge contract
        test_faces_per_edge_matches_contract_kelvin,
        test_faces_per_edge_matches_contract_bcc_foam,
        # WP contract validation
        test_wp_type_a_contract_validation,
        test_wp_type_b_contract_validation,
        # κ WP diagnostics
        test_kappa_wp_type_a_diagnostic,
        test_kappa_wp_type_b_diagnostic,
        test_kappa_wp_reports_fixed_points,
        # P4: Import smoke tests
        test_import_smoke_builders,
        test_import_smoke_operators,
        test_import_smoke_analysis,
        test_smoke_pipeline_kelvin,
    ]

    print("=" * 60)
    print("GUARD AND EDGE CASE TESTS")
    print("=" * 60)

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: ERROR - {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
