"""
Test 19: Holonomy — Discrete Curvature and SU(2) Transport
============================================================

Verifies standard properties of discrete Gaussian curvature
(Descartes angular deficits), Gauss-Bonnet theorem, parallel
transport holonomy, and SU(2) lift on foam cell boundaries.

All tested properties are consequences of:
  - Descartes theorem: Sum(deficits) = 4*pi for genus-0 polyhedra
  - Gauss-Bonnet: holonomy = enclosed curvature
  - SU(2) double cover: exp(-i*pi*sigma_z) = -I

Tests use Kelvin (BCC N=2) and C15 (N=1) foam cells.

Feb 2026
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from core_math.analysis.holonomy import (
    compute_vertex_deficits,
    gauss_bonnet_total,
    compute_face_normals,
    circuit_gauss_map_holonomy,
    su2_from_holonomy,
    circuit_su2_holonomy,
)
from core_math.analysis.cell_topology import (
    get_cell_geometry,
    find_simple_cycles,
    circuit_holonomy,
    find_best_belt,
)
from core_math.builders.multicell_periodic import (
    build_bcc_supercell_periodic, generate_bcc_centers
)
from core_math.builders.c15_periodic import (
    build_c15_supercell_periodic, get_c15_points
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture(scope='module')
def kelvin_mesh():
    N = 2
    v, e, f, cfi = build_bcc_supercell_periodic(N)
    centers = np.array(generate_bcc_centers(N))
    L = 4 * N
    return v, e, f, cfi, centers, L


@pytest.fixture(scope='module')
def c15_mesh():
    N = 1
    L_cell = 4.0
    v, e, f, cfi = build_c15_supercell_periodic(N, L_cell)
    centers = get_c15_points(N, L_cell)
    L = N * L_cell
    return v, e, f, cfi, centers, L


@pytest.fixture(scope='module')
def kelvin_cell(kelvin_mesh):
    """Face data and adjacency for first Kelvin cell."""
    v, e, f, cfi, centers, L = kelvin_mesh
    face_data, adj = get_cell_geometry(0, centers[0], v, f, cfi, L)
    return face_data, adj, centers[0]


@pytest.fixture(scope='module')
def c15_z12_cell(c15_mesh):
    """Face data and adjacency for first Z12 cell."""
    v, e, f, cfi, centers, L = c15_mesh
    for ci in range(len(cfi)):
        if len(cfi[ci]) == 12:
            face_data, adj = get_cell_geometry(ci, centers[ci], v, f, cfi, L)
            return face_data, adj, centers[ci]


@pytest.fixture(scope='module')
def c15_z16_cell(c15_mesh):
    """Face data and adjacency for first Z16 cell."""
    v, e, f, cfi, centers, L = c15_mesh
    for ci in range(len(cfi)):
        if len(cfi[ci]) == 16:
            face_data, adj = get_cell_geometry(ci, centers[ci], v, f, cfi, L)
            return face_data, adj, centers[ci]


# ═════════════════════════════════════════════════════════════
# SECTION 1: VERTEX DEFICITS AND GAUSS-BONNET
# ═════════════════════════════════════════════════════════════

class TestVertexDeficits:
    """Angular deficit computation and Descartes theorem."""

    def test_kelvin_total_4pi(self, kelvin_cell):
        """Kelvin cell (14 faces): total deficit = 4*pi."""
        face_data, adj, cc = kelvin_cell
        total = gauss_bonnet_total(face_data)
        assert abs(total - 4 * np.pi) < 1e-10

    def test_c15_z12_total_4pi(self, c15_z12_cell):
        """C15 Z12 (dodecahedron): total deficit = 4*pi."""
        face_data, adj, cc = c15_z12_cell
        total = gauss_bonnet_total(face_data)
        assert abs(total - 4 * np.pi) < 1e-8

    def test_c15_z16_total_4pi(self, c15_z16_cell):
        """C15 Z16 (16 faces): total deficit = 4*pi."""
        face_data, adj, cc = c15_z16_cell
        total = gauss_bonnet_total(face_data)
        assert abs(total - 4 * np.pi) < 1e-8

    def test_all_deficits_positive_kelvin(self, kelvin_cell):
        """Convex cell: all vertex deficits are positive."""
        face_data, adj, cc = kelvin_cell
        deficits = compute_vertex_deficits(face_data)
        for vid, d in deficits.items():
            assert d > 0, f"vertex {vid} has non-positive deficit {d}"

    def test_deficit_count_matches_vertices(self, kelvin_cell):
        """One deficit per unique vertex."""
        face_data, adj, cc = kelvin_cell
        deficits = compute_vertex_deficits(face_data)
        # Kelvin N=2: 24 vertices per cell
        assert len(deficits) == 24

    def test_all_c15_cells_4pi(self, c15_mesh):
        """Every cell in C15 N=1 has total deficit = 4*pi."""
        v, e, f, cfi, centers, L = c15_mesh
        for ci in range(len(cfi)):
            face_data, adj = get_cell_geometry(ci, centers[ci], v, f, cfi, L)
            total = gauss_bonnet_total(face_data)
            assert abs(total - 4 * np.pi) < 1e-8, \
                f"cell {ci} ({len(cfi[ci])} faces): total={total/np.pi:.6f}*pi"


# ═════════════════════════════════════════════════════════════
# SECTION 2: FACE NORMALS
# ═════════════════════════════════════════════════════════════

class TestFaceNormals:
    """Outward-pointing face normal computation."""

    def test_normals_unit_length(self, kelvin_cell):
        """All face normals have unit length."""
        face_data, adj, cc = kelvin_cell
        normals = compute_face_normals(face_data, cc)
        for i, n in enumerate(normals):
            assert abs(np.linalg.norm(n) - 1.0) < 1e-10, \
                f"face {i}: |n| = {np.linalg.norm(n)}"

    def test_normals_outward(self, kelvin_cell):
        """All normals point away from cell center."""
        face_data, adj, cc = kelvin_cell
        normals = compute_face_normals(face_data, cc)
        for i, n in enumerate(normals):
            d = face_data[i]['center'] - cc
            assert np.dot(n, d) > 0, \
                f"face {i}: normal points inward"

    def test_normals_count(self, kelvin_cell):
        """One normal per face."""
        face_data, adj, cc = kelvin_cell
        normals = compute_face_normals(face_data, cc)
        assert len(normals) == len(face_data)


# ═════════════════════════════════════════════════════════════
# SECTION 3: SU(2) ALGEBRA
# ═════════════════════════════════════════════════════════════

class TestSU2Algebra:
    """SU(2) lift: algebraic properties."""

    def test_su2_identity(self):
        """omega = 0 -> U = +I."""
        U = su2_from_holonomy(0.0)
        assert np.allclose(U, np.eye(2))

    def test_su2_2pi_minus_I(self):
        """omega = 2*pi -> U = -I (spinor sign flip)."""
        U = su2_from_holonomy(2 * np.pi)
        assert np.allclose(U, -np.eye(2))

    def test_su2_4pi_identity(self):
        """omega = 4*pi -> U = +I (full cycle)."""
        U = su2_from_holonomy(4 * np.pi)
        assert np.allclose(U, np.eye(2), atol=1e-14)

    def test_su2_pi(self):
        """omega = pi -> U = diag(-i, i)."""
        U = su2_from_holonomy(np.pi)
        expected = np.array([[-1j, 0], [0, 1j]])
        assert np.allclose(U, expected, atol=1e-14)

    def test_trace_formula(self):
        """tr(U) = 2*cos(omega/2) for various omega."""
        for omega in [0, 0.5, 1.0, np.pi, 2.0, 2*np.pi, 3*np.pi]:
            U = su2_from_holonomy(omega)
            expected_trace = 2 * np.cos(omega / 2)
            actual_trace = np.real(np.trace(U))
            assert abs(actual_trace - expected_trace) < 1e-14, \
                f"omega={omega}: tr={actual_trace}, expected={expected_trace}"

    def test_su2_unitary(self):
        """U is unitary: U^dag U = I."""
        for omega in [0.3, 1.7, np.pi, 2*np.pi]:
            U = su2_from_holonomy(omega)
            product = U.conj().T @ U
            assert np.allclose(product, np.eye(2), atol=1e-14)

    def test_su2_determinant_one(self):
        """det(U) = 1 (special unitary)."""
        for omega in [0.3, 1.7, np.pi, 2*np.pi]:
            U = su2_from_holonomy(omega)
            assert abs(np.linalg.det(U) - 1.0) < 1e-14


# ═════════════════════════════════════════════════════════════
# SECTION 4: SU(2) HOLONOMY ON FOAM CELLS
# ═════════════════════════════════════════════════════════════

class TestSU2Holonomy:
    """SU(2) holonomy from Gauss-Bonnet on real foam circuits."""

    def test_kelvin_best_belt_minus_I(self, kelvin_cell):
        """Kelvin best belt: Omega=2*pi -> U = -I."""
        face_data, adj, cc = kelvin_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, hol = result
        su2 = circuit_su2_holonomy(circuit, face_data, adj)
        assert su2 is not None
        assert su2['is_minus_I']
        assert abs(su2['trace'] + 2.0) < 0.02

    def test_c15_z16_best_belt_minus_I(self, c15_z16_cell):
        """C15 Z16 best belt: Omega=2*pi -> U = -I."""
        face_data, adj, cc = c15_z16_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, hol = result
        su2 = circuit_su2_holonomy(circuit, face_data, adj)
        assert su2 is not None
        assert su2['is_minus_I']
        assert abs(su2['trace'] + 2.0) < 0.02

    def test_c15_z12_best_belt_minus_I(self, c15_z12_cell):
        """C15 Z12 (dodecahedron, all pentagons): Omega=2*pi -> U = -I."""
        face_data, adj, cc = c15_z12_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, hol = result
        su2 = circuit_su2_holonomy(circuit, face_data, adj)
        assert su2 is not None
        assert su2['is_minus_I']
        assert abs(su2['trace'] + 2.0) < 0.02

    def test_trace_matches_omega(self, kelvin_cell):
        """tr(U) = 2*cos(Omega/2) for all separating circuits."""
        face_data, adj, cc = kelvin_cell
        n_f = len(face_data)
        cycles = find_simple_cycles(adj, n_f, max_length=8)
        n_checked = 0
        for cyc in cycles:
            su2 = circuit_su2_holonomy(cyc, face_data, adj)
            if su2 is None:
                continue
            expected_trace = 2 * np.cos(su2['omega'] / 2)
            assert abs(su2['trace'] - expected_trace) < 1e-12
            n_checked += 1
        assert n_checked > 0


# ═════════════════════════════════════════════════════════════
# SECTION 5: GAUSS MAP vs GAUSS-BONNET
# ═════════════════════════════════════════════════════════════

class TestGaussMapHolonomy:
    """Gauss map holonomy (spherical excess) agrees with Gauss-Bonnet."""

    def test_kelvin_transport_equals_gb(self, kelvin_cell):
        """On Kelvin cell: theta_transport = Omega_GB (mod 2*pi)."""
        face_data, adj, cc = kelvin_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, hol = result

        transport = circuit_gauss_map_holonomy(circuit, face_data, cc)
        assert transport is not None

        omega_gb = circuit_holonomy(circuit, face_data, adj)

        # Compare mod 2*pi
        diff = (transport['theta_raw'] - omega_gb) % (2 * np.pi)
        if diff > np.pi:
            diff -= 2 * np.pi
        assert abs(diff) < 0.1, \
            f"theta={transport['theta_raw']/np.pi:.4f}*pi, " \
            f"omega={omega_gb/np.pi:.4f}*pi, diff={diff/np.pi:.4f}*pi"

    def test_c15_z16_transport_equals_gb(self, c15_z16_cell):
        """On C15 Z16: theta_transport = Omega_GB (mod 2*pi)."""
        face_data, adj, cc = c15_z16_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, hol = result

        transport = circuit_gauss_map_holonomy(circuit, face_data, cc)
        assert transport is not None

        omega_gb = circuit_holonomy(circuit, face_data, adj)

        diff = (transport['theta_raw'] - omega_gb) % (2 * np.pi)
        if diff > np.pi:
            diff -= 2 * np.pi
        assert abs(diff) < 0.1, \
            f"theta={transport['theta_raw']/np.pi:.4f}*pi, " \
            f"omega={omega_gb/np.pi:.4f}*pi, diff={diff/np.pi:.4f}*pi"

    def test_transport_n_angles(self, kelvin_cell):
        """Transport returns N spherical angles for N-face circuit."""
        face_data, adj, cc = kelvin_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, hol = result

        transport = circuit_gauss_map_holonomy(circuit, face_data, cc)
        assert transport is not None
        assert len(transport['spherical_angles']) == len(circuit)

    def test_transport_returns_none_for_short_circuit(self, kelvin_cell):
        """Circuit with < 3 faces returns None."""
        face_data, adj, cc = kelvin_cell
        result = circuit_gauss_map_holonomy((0, 1), face_data, cc)
        assert result is None

    def test_transport_angles_real_valued(self, kelvin_cell):
        """Spherical angles are real-valued and finite."""
        face_data, adj, cc = kelvin_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, hol = result
        transport = circuit_gauss_map_holonomy(circuit, face_data, cc)
        assert transport is not None
        for a in transport['spherical_angles']:
            assert np.isfinite(a)

    def test_gauss_map_all_kelvin_cells(self, kelvin_mesh):
        """Gauss map holonomy = GB for best belt of every Kelvin cell."""
        v, e, f, cfi, centers, L = kelvin_mesh
        n_checked = 0
        for ci in range(len(cfi)):
            fd, adj = get_cell_geometry(ci, centers[ci], v, f, cfi, L)
            result = find_best_belt(fd, adj, centers[ci])
            if result is None:
                continue
            circuit, hol = result
            transport = circuit_gauss_map_holonomy(
                circuit, fd, centers[ci])
            if transport is None:
                continue
            diff = (transport['theta_raw'] - hol) % (2 * np.pi)
            if diff > np.pi:
                diff -= 2 * np.pi
            assert abs(diff) < 0.15, \
                f"Kelvin cell {ci}: theta={transport['theta_raw']/np.pi:.4f}*pi, "\
                f"omega={hol/np.pi:.4f}*pi, diff={diff/np.pi:.4f}*pi"
            n_checked += 1
        assert n_checked >= 4, f"Only {n_checked} Kelvin cells checked"

    def test_gauss_map_all_c15_cells(self, c15_mesh):
        """Gauss map holonomy = GB for best belt of every C15 cell."""
        v, e, f, cfi, centers, L = c15_mesh
        n_checked = 0
        for ci in range(len(cfi)):
            fd, adj = get_cell_geometry(ci, centers[ci], v, f, cfi, L)
            result = find_best_belt(fd, adj, centers[ci])
            if result is None:
                continue
            circuit, hol = result
            transport = circuit_gauss_map_holonomy(
                circuit, fd, centers[ci])
            if transport is None:
                continue
            diff = (transport['theta_raw'] - hol) % (2 * np.pi)
            if diff > np.pi:
                diff -= 2 * np.pi
            assert abs(diff) < 0.15, \
                f"C15 cell {ci}: theta={transport['theta_raw']/np.pi:.4f}*pi, "\
                f"omega={hol/np.pi:.4f}*pi, diff={diff/np.pi:.4f}*pi"
            n_checked += 1
        assert n_checked >= 5, f"Only {n_checked} C15 cells checked"

    def test_gauss_map_mod2pi_tighter(self, kelvin_mesh):
        """theta_mod_2pi agrees with GB mod 2*pi, tighter tolerance."""
        v, e, f, cfi, centers, L = kelvin_mesh
        n_checked = 0
        for ci in range(len(cfi)):
            fd, adj = get_cell_geometry(ci, centers[ci], v, f, cfi, L)
            result = find_best_belt(fd, adj, centers[ci])
            if result is None:
                continue
            circuit, hol = result
            gm = circuit_gauss_map_holonomy(circuit, fd, centers[ci])
            if gm is None:
                continue
            # wrap GB to [-pi, pi]
            gb_mod = hol % (2 * np.pi)
            if gb_mod > np.pi:
                gb_mod -= 2 * np.pi
            diff = abs(gm['theta_mod_2pi'] - gb_mod)
            assert diff < 0.05, \
                f"cell {ci}: theta_mod={gm['theta_mod_2pi']/np.pi:.4f}*pi, " \
                f"gb_mod={gb_mod/np.pi:.4f}*pi"
            n_checked += 1
        assert n_checked >= 4

    def test_gauss_map_cosine_crosscheck(self, kelvin_cell):
        """cos(theta_gauss_map) = cos(omega_GB): 2pi-periodic cross-check."""
        face_data, adj, cc = kelvin_cell
        result = find_best_belt(face_data, adj, cc)
        assert result is not None
        circuit, hol = result
        gm = circuit_gauss_map_holonomy(circuit, face_data, cc)
        assert gm is not None
        omega_gb = circuit_holonomy(circuit, face_data, adj)
        # cos is 2pi-periodic, so this is the valid comparison
        # (Gauss map returns holonomy mod 2pi; half-angle cos is 4pi-periodic)
        assert abs(np.cos(gm['theta_raw']) - np.cos(omega_gb)) < 0.05, \
            f"cos(theta)={np.cos(gm['theta_raw']):.4f}, " \
            f"cos(omega)={np.cos(omega_gb):.4f}"

    def test_gauss_map_multi_circuit(self, kelvin_mesh):
        """Gauss map vs GB on best belt of multiple cells (not just cell 0)."""
        v, e, f, cfi, centers, L = kelvin_mesh
        n_checked = 0
        for ci in range(min(len(cfi), 8)):
            fd, adj = get_cell_geometry(ci, centers[ci], v, f, cfi, L)
            # Try all equatorial circuits, not just best belt
            n_f = len(fd)
            cycles = find_simple_cycles(adj, n_f, max_length=8)
            for cyc in cycles:
                if len(cyc) < 5:
                    continue
                omega_gb = circuit_holonomy(cyc, fd, adj)
                if omega_gb is None or abs(omega_gb - 2 * np.pi) > 0.1:
                    continue
                gm = circuit_gauss_map_holonomy(cyc, fd, centers[ci])
                if gm is None:
                    continue
                assert abs(np.cos(gm['theta_raw']) - np.cos(omega_gb)) < 0.15, \
                    f"cell {ci} circuit {cyc}: " \
                    f"cos(theta)={np.cos(gm['theta_raw']):.4f}, " \
                    f"cos(omega)={np.cos(omega_gb):.4f}"
                n_checked += 1
        assert n_checked >= 4, f"Only {n_checked} circuits checked"
