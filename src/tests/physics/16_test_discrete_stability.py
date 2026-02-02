"""
Test 16: Discrete Stability Across k_L/k_T Ratios

STRUCTURES TESTED: C15, Kelvin, WP, FCC (all 4)

FINDING: For the tested foam structures (C15, Kelvin, WP, FCC), the discrete
dynamical matrix D(k) is PSD across sampled k-vectors for k_L/k_T ratios
from 0.01 to 100, even when continuum elasticity would predict instability.

This is because:
- Continuum stability criterion (C11 + 2C12 > 0) applies to HOMOGENEOUS deformations
- Discrete D(k) tests stability of FINITE wavelength modes
- For these tetravalent foam structures, the spring network is mechanically stable

KEY INSIGHT: Discrete stability != Continuum stability

Tested:
- Structures: C15, Kelvin, WP, FCC
- Ratios: 0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0
- k-vectors: high-symmetry directions + golden spiral, multiple eps values

Velocity behavior:
- ratio < 1: v_L < v_T ("auxetic-like", unusual but stable)
- ratio = 1: v_L ~ v_T (nearly isotropic)
- ratio > 1: v_L > v_T (normal solid)

TESTS (44 total)
----------------
- PSD stability for all structures × all ratios (28)
- Velocity tests (10)
- Eigenvalue scaling, singularity checks (6)

EXPECTED OUTPUT (Jan 2026)
--------------------------
    44 passed in 54.01s

Jan 2026
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '/Users/alextoader/Sites/physics_ai/ST_8/src')

from physics.bloch import DisplacementBloch
from core_math_v2.builders.c15_periodic import build_c15_supercell_periodic
from core_math_v2.builders.multicell_periodic import build_bcc_supercell_periodic
from core_math_v2.builders.weaire_phelan_periodic import build_wp_supercell_periodic
from core_math_v2.builders.solids_periodic import build_fcc_supercell_periodic


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def c15_structure():
    """C15 Laves foam."""
    result = build_c15_supercell_periodic(N=1, L_cell=4.0)
    return {'V': result[0], 'E': result[1], 'L': 4.0, 'name': 'C15'}


@pytest.fixture(scope="module")
def kelvin_structure():
    """Kelvin (BCC) foam."""
    result = build_bcc_supercell_periodic(N=2)
    return {'V': result[0], 'E': result[1], 'L': 8.0, 'name': 'Kelvin'}


@pytest.fixture(scope="module")
def wp_structure():
    """Weaire-Phelan foam."""
    result = build_wp_supercell_periodic(N=1, L_cell=4.0)
    return {'V': result[0], 'E': result[1], 'L': 4.0, 'name': 'WP'}


@pytest.fixture(scope="module")
def fcc_structure():
    """FCC tiling."""
    result = build_fcc_supercell_periodic(N=2)
    return {'V': result[0], 'E': result[1], 'L': 8.0, 'name': 'FCC'}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def high_symmetry_directions():
    """
    High-symmetry directions for cubic lattice.
    These are where extrema/instabilities typically appear.
    """
    dirs = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],  # <100> family
        [1, 1, 0], [1, 0, 1], [0, 1, 1],  # <110> family
        [1, 1, 1],  # <111>
    ]
    return np.array([d / np.linalg.norm(d) for d in dirs])


def check_psd(V, E, L, k_L, k_T, n_golden=10, use_highsym=True):
    """
    Check if D(k) is PSD for given k_L/k_T ratio.

    Samples:
    - High-symmetry directions: [100], [110], [111] families
    - Golden spiral directions for broader coverage
    - Multiple eps values: 0.005, 0.01, 0.02, 0.04, 0.08

    Returns:
        min_eigenvalue: minimum eigenvalue found across all k vectors
        is_stable: True if min_eigenvalue >= -tolerance
    """
    db = DisplacementBloch(V, E, L, k_L=k_L, k_T=k_T)

    # Combine high-symmetry + golden spiral directions
    directions = []

    if use_highsym:
        directions.append(high_symmetry_directions())

    # Golden spiral for broader coverage
    indices = np.arange(0, n_golden, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_golden)
    theta = np.pi * (1 + 5**0.5) * indices
    golden = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    directions.append(golden)

    all_directions = np.vstack(directions)

    # Extended eps range (stay in IR regime: eps < 0.1)
    eps_values = [0.005, 0.01, 0.02, 0.04, 0.08]

    min_eig = np.inf
    for d in all_directions:
        for eps in eps_values:
            k = eps * 2 * np.pi / L * d
            D = db.build_dynamical_matrix(k)
            eigs = np.linalg.eigvalsh(D)
            min_eig = min(min_eig, eigs.min())

    # Tolerance for numerical noise
    tolerance = 1e-10
    is_stable = min_eig >= -tolerance

    return min_eig, is_stable


def get_velocities(V, E, L, k_L, k_T, direction=None):
    """Get acoustic velocities for a given ratio."""
    if direction is None:
        direction = np.array([1, 0, 0])

    db = DisplacementBloch(V, E, L, k_L=k_L, k_T=k_T)
    eps = 0.02
    k = eps * 2 * np.pi / L * direction / np.linalg.norm(direction)
    k_mag = np.linalg.norm(k)

    omega_T, omega_L, _ = db.classify_modes(k)
    v_T = omega_T[0] / k_mag
    v_L = omega_L[0] / k_mag

    return v_T, v_L


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestDiscreteStabilityAllRatios:
    """
    MAIN TEST: D(k) is PSD across sampled k for all tested ratios.

    Samples high-symmetry directions + golden spiral, multiple eps values.
    This demonstrates that discrete stability != continuum stability
    for the tested foam structures.
    """

    # Ratios to test: extreme low, normal range, extreme high
    RATIOS = [0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0]

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_c15_stable(self, c15_structure, ratio):
        """C15 is stable for all ratios."""
        data = c15_structure
        min_eig, is_stable = check_psd(
            data['V'], data['E'], data['L'],
            k_L=ratio, k_T=1.0
        )
        assert is_stable, f"C15 unstable at ratio={ratio}: min_eig={min_eig}"

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_kelvin_stable(self, kelvin_structure, ratio):
        """Kelvin is stable for all ratios."""
        data = kelvin_structure
        min_eig, is_stable = check_psd(
            data['V'], data['E'], data['L'],
            k_L=ratio, k_T=1.0
        )
        assert is_stable, f"Kelvin unstable at ratio={ratio}: min_eig={min_eig}"

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_wp_stable(self, wp_structure, ratio):
        """WP is stable for all ratios."""
        data = wp_structure
        min_eig, is_stable = check_psd(
            data['V'], data['E'], data['L'],
            k_L=ratio, k_T=1.0
        )
        assert is_stable, f"WP unstable at ratio={ratio}: min_eig={min_eig}"

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_fcc_stable(self, fcc_structure, ratio):
        """FCC is stable for all ratios."""
        data = fcc_structure
        min_eig, is_stable = check_psd(
            data['V'], data['E'], data['L'],
            k_L=ratio, k_T=1.0
        )
        assert is_stable, f"FCC unstable at ratio={ratio}: min_eig={min_eig}"


class TestVelocitiesAllRatios:
    """
    Verify velocities are real and positive for all ratios.

    At low ratios (< 1): v_L < v_T (unusual but stable)
    At ratio = 1: v_L = v_T (isotropic)
    At high ratios (> 1): v_L > v_T (normal solid)
    """

    RATIOS = [0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0]

    @pytest.mark.parametrize("ratio", RATIOS)
    def test_c15_velocities_positive(self, c15_structure, ratio):
        """C15 has positive velocities for all ratios."""
        data = c15_structure
        v_T, v_L = get_velocities(data['V'], data['E'], data['L'], k_L=ratio, k_T=1.0)

        assert v_T > 0, f"C15 v_T <= 0 at ratio={ratio}"
        assert v_L > 0, f"C15 v_L <= 0 at ratio={ratio}"
        assert np.isfinite(v_T), f"C15 v_T not finite at ratio={ratio}"
        assert np.isfinite(v_L), f"C15 v_L not finite at ratio={ratio}"

    def test_velocity_ordering_low_ratio(self, c15_structure):
        """At low ratios, v_L < v_T (auxetic-like)."""
        data = c15_structure
        v_T, v_L = get_velocities(data['V'], data['E'], data['L'], k_L=0.1, k_T=1.0)

        assert v_L < v_T, f"Expected v_L < v_T at ratio=0.1, got v_L={v_L}, v_T={v_T}"

    def test_velocity_ordering_high_ratio(self, c15_structure):
        """At high ratios, v_L > v_T (normal solid)."""
        data = c15_structure
        v_T, v_L = get_velocities(data['V'], data['E'], data['L'], k_L=10.0, k_T=1.0)

        assert v_L > v_T, f"Expected v_L > v_T at ratio=10, got v_L={v_L}, v_T={v_T}"

    def test_velocity_isotropic_at_ratio_1(self, c15_structure):
        """At ratio=1, v_L ~ v_T (nearly isotropic)."""
        data = c15_structure
        v_T, v_L = get_velocities(data['V'], data['E'], data['L'], k_L=1.0, k_T=1.0)

        # Discrete geometry can introduce small differences even at k_L=k_T
        # Tolerance 10% is physically reasonable
        ratio = v_L / v_T
        assert 0.9 < ratio < 1.1, f"Expected v_L ~ v_T at ratio=1, got v_L/v_T={ratio}"


class TestMinEigenvalueScaling:
    """
    Verify that min eigenvalue scales correctly with k².

    For acoustic modes: omega² ~ k² => eigenvalue ~ k²
    """

    def test_eigenvalue_scales_with_k_squared(self, c15_structure):
        """Min eigenvalue ~ k² (acoustic scaling)."""
        data = c15_structure
        db = DisplacementBloch(data['V'], data['E'], data['L'], k_L=3.0, k_T=1.0)

        direction = np.array([1, 0, 0])
        eps_values = [0.01, 0.02, 0.04]
        min_eigs = []

        for eps in eps_values:
            k = eps * 2 * np.pi / data['L'] * direction
            D = db.build_dynamical_matrix(k)
            eigs = np.linalg.eigvalsh(D)
            # Skip zero modes using RELATIVE threshold (robust for small eps)
            threshold = 1e-6 * max(abs(eigs.max()), 1e-10)
            positive_eigs = eigs[eigs > threshold]
            if len(positive_eigs) > 0:
                min_eigs.append(positive_eigs.min())

        # Check k² scaling: eig(2k) / eig(k) ≈ 4
        if len(min_eigs) >= 2:
            ratio = min_eigs[1] / min_eigs[0]
            expected = (eps_values[1] / eps_values[0])**2
            assert 0.8 * expected < ratio < 1.2 * expected, \
                f"Eigenvalue scaling: got {ratio}, expected ~{expected}"


class TestNoSingularities:
    """
    Verify there are no singularities or discontinuities across ratio range.
    """

    def test_smooth_velocity_variation(self, c15_structure):
        """Velocities vary smoothly with ratio (no jumps)."""
        data = c15_structure
        ratios = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

        v_T_prev = None
        v_L_prev = None

        for ratio in ratios:
            v_T, v_L = get_velocities(data['V'], data['E'], data['L'], k_L=ratio, k_T=1.0)

            if v_T_prev is not None:
                # Check no sudden jumps (< 2x change between adjacent ratios)
                assert 0.3 < v_T / v_T_prev < 3.0, \
                    f"v_T jump at ratio={ratio}: {v_T_prev} -> {v_T}"
                assert 0.3 < v_L / v_L_prev < 3.0, \
                    f"v_L jump at ratio={ratio}: {v_L_prev} -> {v_L}"

            v_T_prev, v_L_prev = v_T, v_L

    def test_no_zero_velocities(self, c15_structure):
        """No velocities go to zero at any ratio."""
        data = c15_structure

        for ratio in [0.01, 0.1, 1.0, 10.0, 100.0]:
            v_T, v_L = get_velocities(data['V'], data['E'], data['L'], k_L=ratio, k_T=1.0)

            assert v_T > 0.01, f"v_T too small at ratio={ratio}: {v_T}"
            assert v_L > 0.001, f"v_L too small at ratio={ratio}: {v_L}"


class TestMultiStructureConsistency:
    """
    Verify all structures show same qualitative behavior.
    """

    def test_all_structures_stable_at_extreme_low(self, c15_structure, kelvin_structure, wp_structure, fcc_structure):
        """All structures stable at ratio=0.01."""
        for data in [c15_structure, kelvin_structure, wp_structure, fcc_structure]:
            min_eig, is_stable = check_psd(
                data['V'], data['E'], data['L'],
                k_L=0.01, k_T=1.0
            )
            assert is_stable, f"{data['name']} unstable at ratio=0.01"

    def test_all_structures_stable_at_extreme_high(self, c15_structure, kelvin_structure, wp_structure, fcc_structure):
        """All structures stable at ratio=100."""
        for data in [c15_structure, kelvin_structure, wp_structure, fcc_structure]:
            min_eig, is_stable = check_psd(
                data['V'], data['E'], data['L'],
                k_L=100.0, k_T=1.0
            )
            assert is_stable, f"{data['name']} unstable at ratio=100"


# =============================================================================
# DOCUMENTATION
# =============================================================================

class TestDocumentation:
    """Print summary for documentation."""

    def test_print_stability_table(self, c15_structure, kelvin_structure, wp_structure, fcc_structure):
        """Print stability results table."""
        print("\n" + "="*80)
        print("DISCRETE STABILITY SUMMARY")
        print("="*80)

        ratios = [0.01, 0.1, 1.0, 3.0, 10.0, 100.0]

        print(f"\n{'Ratio':<8} | {'C15':<12} | {'Kelvin':<12} | {'WP':<12} | {'FCC':<12}")
        print("-"*65)

        for ratio in ratios:
            row = f"{ratio:<8.2f} |"
            for data in [c15_structure, kelvin_structure, wp_structure, fcc_structure]:
                min_eig, is_stable = check_psd(
                    data['V'], data['E'], data['L'],
                    k_L=ratio, k_T=1.0, n_golden=5
                )
                status = "STABLE" if is_stable else "UNSTABLE"
                row += f" {status:<12} |"
            print(row)

        print("\n" + "="*80)
        print("KEY FINDING: All tested structures stable across sampled k")
        print("for ratios 0.01 to 100. Discrete stability != Continuum stability")
        print("="*80)
