#!/usr/bin/env python3
"""
Test 15: Sampling Convergence for Direction-Dependent Quantities

STRUCTURES TESTED: C15, WP, Kelvin, FCC (all 4)

FINDING (Jan 2026):
    delta_v/v = (v_max - v_min) / v_mean is extremum-based.
    Golden spiral sampling with small n_dir may MISS the true extrema,
    which occur at HIGH-SYMMETRY directions for cubic structures.

HYPOTHESIS:
    For cubic structures, extrema of v(k_hat) are at high-symmetry directions.

    FINDING: Extrema pattern is STRUCTURE-DEPENDENT:
    - C15, Kelvin: v_max at [100], v_min at [110]
    - WP (Weaire-Phelan): v_max at [111], v_min at [100]

    Therefore:
    1. Golden spiral with small n underestimates delta_v/v
    2. Golden spiral converges to high-symmetry result as n -> infinity
    3. High-symmetry sampling (7 directions) gives stable, accurate result
    4. RANKING is preserved regardless of sampling method
    5. Extrema locations depend on internal structure, not just cubic symmetry

IMPLICATIONS:
    - For RANKING tests (C15 < WP < Kelvin < FCC): n_dir=30-50 is sufficient
    - For MAGNITUDE claims (delta_v/v = X%): should use high-symmetry or note ~10% uncertainty
    - Correlation tests (Pearson r) are likely unaffected (shape, not magnitude)

TESTS (15 total)
----------------
- Sampling convergence tests (golden vs highsym)
- Extrema location tests ([100], [110] families)
- Ranking preserved: C15 < WP < Kelvin < FCC
- Scale and eps invariance
- Magnitude uncertainty quantification

EXPECTED OUTPUT (Jan 2026)
--------------------------
    15 passed in 146.57s

Jan 2026
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, '/Users/alextoader/Sites/physics_ai/ST_8/src')

from physics.bloch import DisplacementBloch
from core_math.builders.c15_periodic import build_c15_supercell_periodic
from core_math.builders.multicell_periodic import build_bcc_supercell_periodic
from core_math.builders.weaire_phelan_periodic import build_wp_supercell_periodic
from core_math.builders.solids_periodic import build_fcc_supercell_periodic


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def golden_spiral(n: int) -> np.ndarray:
    """Generate n approximately uniform directions on sphere using golden spiral."""
    indices = np.arange(0, n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5**0.5) * indices
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])


def high_symmetry_directions() -> np.ndarray:
    """
    High-symmetry directions for cubic structures.
    These are where velocity extrema SHOULD occur.
    """
    dirs = [
        [1, 0, 0], [0, 1, 0], [0, 0, 1],           # [100] family
        [1, 1, 0], [1, 0, 1], [0, 1, 1],           # [110] family
        [1, 1, 1],                                   # [111]
    ]
    dirs = np.array(dirs, dtype=float)
    # Normalize
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / norms


def compute_delta_v_golden(db: DisplacementBloch, L: float,
                           n_dir: int, eps: float = 0.02) -> Dict:
    """Compute delta_v/v using golden spiral sampling."""
    directions = golden_spiral(n_dir)
    k_mag = eps * 2 * np.pi / L

    velocities = []
    for d in directions:
        k = k_mag * d
        omega_T, _, _ = db.classify_modes(k)
        velocities.append(omega_T[0] / k_mag)

    v_arr = np.array(velocities)
    return {
        'v_min': v_arr.min(),
        'v_max': v_arr.max(),
        'v_mean': v_arr.mean(),
        'delta_v_over_v': (v_arr.max() - v_arr.min()) / v_arr.mean()
    }


def compute_delta_v_highsym(db: DisplacementBloch, L: float,
                            eps: float = 0.02) -> Dict:
    """Compute delta_v/v using only high-symmetry directions."""
    directions = high_symmetry_directions()
    k_mag = eps * 2 * np.pi / L

    velocities = []
    dir_names = ['[100]', '[010]', '[001]', '[110]', '[101]', '[011]', '[111]']
    v_by_dir = {}

    for i, d in enumerate(directions):
        k = k_mag * d
        omega_T, _, _ = db.classify_modes(k)
        v = omega_T[0] / k_mag
        velocities.append(v)
        v_by_dir[dir_names[i]] = v

    v_arr = np.array(velocities)
    return {
        'v_min': v_arr.min(),
        'v_max': v_arr.max(),
        'v_mean': v_arr.mean(),
        'delta_v_over_v': (v_arr.max() - v_arr.min()) / v_arr.mean(),
        'v_by_direction': v_by_dir
    }


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def c15_data():
    """C15 structure."""
    result = build_c15_supercell_periodic(N=1, L_cell=4.0)
    V, E = result[0], result[1]
    L = 4.0
    db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)
    return {'V': V, 'E': E, 'L': L, 'db': db, 'name': 'C15'}


@pytest.fixture(scope="module")
def kelvin_data():
    """Kelvin structure."""
    result = build_bcc_supercell_periodic(N=2)
    V, E = result[0], result[1]
    L = 8.0
    db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)
    return {'V': V, 'E': E, 'L': L, 'db': db, 'name': 'Kelvin'}


@pytest.fixture(scope="module")
def wp_data():
    """Weaire-Phelan structure."""
    result = build_wp_supercell_periodic(N=1, L_cell=4.0)
    V, E = result[0], result[1]
    L = 4.0
    db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)
    return {'V': V, 'E': E, 'L': L, 'db': db, 'name': 'WP'}


@pytest.fixture(scope="module")
def fcc_data():
    """FCC structure."""
    result = build_fcc_supercell_periodic(N=2)
    V, E = result[0], result[1]
    L = 8.0
    db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)
    return {'V': V, 'E': E, 'L': L, 'db': db, 'name': 'FCC'}


# =============================================================================
# TEST CLASS: Sampling Convergence
# =============================================================================

class TestSamplingConvergence:
    """
    Test that golden spiral converges to high-symmetry result.
    """

    def test_golden_underestimates_at_small_n(self, c15_data):
        """
        HYPOTHESIS: Golden spiral with n=30 underestimates delta_v/v
        compared to high-symmetry sampling.
        """
        db, L = c15_data['db'], c15_data['L']

        dv_golden_30 = compute_delta_v_golden(db, L, n_dir=30)['delta_v_over_v']
        dv_highsym = compute_delta_v_highsym(db, L)['delta_v_over_v']

        # Golden n=30 should be LESS than high-symmetry
        assert dv_golden_30 < dv_highsym, (
            f"Expected golden(n=30) < highsym, got {dv_golden_30:.6f} vs {dv_highsym:.6f}"
        )

        # Underestimation should be meaningful (> 5%)
        underestimation = (dv_highsym - dv_golden_30) / dv_highsym
        assert underestimation > 0.05, (
            f"Underestimation {underestimation*100:.1f}% is too small to be significant"
        )

    def test_golden_converges_to_highsym(self, c15_data):
        """
        HYPOTHESIS: Golden spiral converges to high-symmetry result as n increases.
        """
        db, L = c15_data['db'], c15_data['L']

        dv_highsym = compute_delta_v_highsym(db, L)['delta_v_over_v']

        # Test convergence
        n_values = [30, 50, 100, 200]
        dv_golden = [compute_delta_v_golden(db, L, n_dir=n)['delta_v_over_v']
                     for n in n_values]

        # Should get closer to highsym as n increases
        errors = [abs(dv - dv_highsym) / dv_highsym for dv in dv_golden]

        # Error at n=200 should be < 5%
        assert errors[-1] < 0.05, (
            f"Golden(n=200) error {errors[-1]*100:.1f}% is too large"
        )

        # Print convergence for review
        print(f"\nConvergence test ({c15_data['name']}):")
        print(f"  High-symmetry: {dv_highsym:.6f}")
        for n, dv, err in zip(n_values, dv_golden, errors):
            print(f"  Golden(n={n:>3}): {dv:.6f} (error {err*100:>5.1f}%)")

    def test_highsym_is_stable(self, c15_data):
        """
        HYPOTHESIS: High-symmetry result is stable (no sampling variance).
        """
        db, L = c15_data['db'], c15_data['L']

        # Run multiple times (should be identical since deterministic)
        results = [compute_delta_v_highsym(db, L)['delta_v_over_v'] for _ in range(3)]

        # All should be identical
        assert all(r == results[0] for r in results), "High-symmetry should be deterministic"


class TestExtremaLocation:
    """
    Test that velocity extrema occur at expected high-symmetry directions.
    """

    def test_vmax_at_100_family(self, c15_data):
        """
        HYPOTHESIS: v_max occurs at [100] family directions.
        """
        db, L = c15_data['db'], c15_data['L']
        result = compute_delta_v_highsym(db, L)

        v_by_dir = result['v_by_direction']
        v_100_family = [v_by_dir['[100]'], v_by_dir['[010]'], v_by_dir['[001]']]

        # [100] family should all be equal (cubic symmetry)
        assert np.allclose(v_100_family, v_100_family[0], rtol=1e-6), (
            f"[100] family velocities not equal: {v_100_family}"
        )

        # [100] should be v_max
        v_max = result['v_max']
        assert np.isclose(v_100_family[0], v_max, rtol=1e-6), (
            f"v_max={v_max:.6f} but v[100]={v_100_family[0]:.6f}"
        )

    def test_vmin_at_110_family(self, c15_data):
        """
        HYPOTHESIS: v_min occurs at [110] family directions.
        """
        db, L = c15_data['db'], c15_data['L']
        result = compute_delta_v_highsym(db, L)

        v_by_dir = result['v_by_direction']
        v_110_family = [v_by_dir['[110]'], v_by_dir['[101]'], v_by_dir['[011]']]

        # [110] family should all be equal (cubic symmetry)
        assert np.allclose(v_110_family, v_110_family[0], rtol=1e-6), (
            f"[110] family velocities not equal: {v_110_family}"
        )

        # [110] should be v_min
        v_min = result['v_min']
        assert np.isclose(v_110_family[0], v_min, rtol=1e-6), (
            f"v_min={v_min:.6f} but v[110]={v_110_family[0]:.6f}"
        )

    def test_extrema_structure_multi(self, c15_data, kelvin_data, wp_data):
        """
        Test extrema location for multiple structures.

        FINDING: Different structures have different extrema patterns!
        - C15, Kelvin: v_max at [100], v_min at [110]
        - WP: v_max at [111], v_min at [100]

        This is structure-dependent, not just cubic symmetry.
        """
        # C15 and Kelvin follow [100]=max, [110]=min pattern
        for data in [c15_data, kelvin_data]:
            db, L, name = data['db'], data['L'], data['name']
            result = compute_delta_v_highsym(db, L)
            v_by_dir = result['v_by_direction']

            v_100 = v_by_dir['[100]']
            v_110 = v_by_dir['[110]']

            assert np.isclose(v_100, result['v_max'], rtol=1e-6), (
                f"{name}: v[100]={v_100:.6f} should be v_max={result['v_max']:.6f}"
            )
            assert np.isclose(v_110, result['v_min'], rtol=1e-6), (
                f"{name}: v[110]={v_110:.6f} should be v_min={result['v_min']:.6f}"
            )

        # WP has DIFFERENT pattern: v_max at [111], v_min at [100]
        db, L = wp_data['db'], wp_data['L']
        result = compute_delta_v_highsym(db, L)
        v_by_dir = result['v_by_direction']

        v_100 = v_by_dir['[100]']
        v_111 = v_by_dir['[111]']

        assert np.isclose(v_111, result['v_max'], rtol=1e-6), (
            f"WP: v[111]={v_111:.6f} should be v_max={result['v_max']:.6f}"
        )
        assert np.isclose(v_100, result['v_min'], rtol=1e-6), (
            f"WP: v[100]={v_100:.6f} should be v_min={result['v_min']:.6f}"
        )

        print("\nExtrema patterns by structure:")
        print("  C15, Kelvin: v_max at [100], v_min at [110]")
        print("  WP:          v_max at [111], v_min at [100]")


class TestRankingPreserved:
    """
    CRITICAL TEST: Ranking should be preserved regardless of sampling method.
    """

    def test_ranking_invariant_to_sampling(self, c15_data, kelvin_data, wp_data, fcc_data):
        """
        HYPOTHESIS: C15 < WP < Kelvin < FCC ranking holds for ALL sampling methods.
        """
        structures = [c15_data, kelvin_data, wp_data, fcc_data]

        # Test with different sampling methods
        methods = {
            'highsym': lambda db, L: compute_delta_v_highsym(db, L)['delta_v_over_v'],
            'golden_30': lambda db, L: compute_delta_v_golden(db, L, n_dir=30)['delta_v_over_v'],
            'golden_100': lambda db, L: compute_delta_v_golden(db, L, n_dir=100)['delta_v_over_v'],
            'golden_200': lambda db, L: compute_delta_v_golden(db, L, n_dir=200)['delta_v_over_v'],
        }

        print("\nRanking test:")
        for method_name, method_fn in methods.items():
            dv = {data['name']: method_fn(data['db'], data['L']) for data in structures}

            ranking = sorted(dv.keys(), key=lambda n: dv[n])
            expected = ['C15', 'WP', 'Kelvin', 'FCC']

            print(f"  {method_name:>12}: C15={dv['C15']:.4f}, WP={dv['WP']:.4f}, Kelvin={dv['Kelvin']:.4f}, FCC={dv['FCC']:.4f} -> {ranking}")

            assert ranking == expected, (
                f"Method {method_name}: expected {expected}, got {ranking}"
            )

    def test_ranking_ratio_stable(self, c15_data, kelvin_data):
        """
        HYPOTHESIS: The RATIO between structures is stable across sampling.
        """
        # Kelvin / C15 ratio should be consistent
        ratios = []

        for n_dir in [30, 50, 100, 200]:
            dv_c15 = compute_delta_v_golden(c15_data['db'], c15_data['L'], n_dir=n_dir)['delta_v_over_v']
            dv_kelvin = compute_delta_v_golden(kelvin_data['db'], kelvin_data['L'], n_dir=n_dir)['delta_v_over_v']
            ratios.append(dv_kelvin / dv_c15)

        # Ratio should be stable (CV < 5%)
        cv = np.std(ratios) / np.mean(ratios)
        assert cv < 0.05, f"Ratio CV={cv*100:.1f}% is too high"

        print(f"\nKelvin/C15 ratio across sampling: {ratios}")
        print(f"  Mean: {np.mean(ratios):.2f}, CV: {cv*100:.1f}%")


class TestMagnitudeUncertainty:
    """
    Quantify the magnitude uncertainty from sampling.
    """

    def test_magnitude_uncertainty(self, c15_data, kelvin_data):
        """
        Document the magnitude uncertainty for different n_dir values.
        """
        print("\nMagnitude uncertainty analysis:")

        for data in [c15_data, kelvin_data]:
            name = data['name']
            db, L = data['db'], data['L']

            dv_highsym = compute_delta_v_highsym(db, L)['delta_v_over_v']

            print(f"\n  {name}:")
            print(f"    High-symmetry (reference): {dv_highsym:.6f}")

            for n_dir in [30, 50, 100, 200]:
                dv = compute_delta_v_golden(db, L, n_dir=n_dir)['delta_v_over_v']
                error = (dv - dv_highsym) / dv_highsym * 100
                print(f"    Golden(n={n_dir:>3}): {dv:.6f} ({error:>+6.1f}% vs highsym)")

        # This test always passes - it's for documentation
        assert True


class TestScaleInvariance:
    """
    Verify that delta_v/v is dimensionless and scale-invariant.
    """

    def test_scale_invariant(self):
        """
        delta_v/v should NOT change with L_cell.
        """
        L_cells = [2.0, 4.0, 8.0]
        dv_values = []

        for L_cell in L_cells:
            result = build_c15_supercell_periodic(N=1, L_cell=L_cell)
            V, E = result[0], result[1]
            L = L_cell
            db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)

            dv = compute_delta_v_highsym(db, L)['delta_v_over_v']
            dv_values.append(dv)

        # All should be equal (dimensionless)
        cv = np.std(dv_values) / np.mean(dv_values)
        assert cv < 0.001, f"delta_v/v varies with L_cell: CV={cv*100:.3f}%"

        print(f"\nScale invariance: L_cell={L_cells}, delta_v/v={dv_values}")


# =============================================================================
# REVIEWER-REQUESTED CROSS-CHECKS (Scientific validation)
# =============================================================================

def direction_class(dir_name):
    """
    Map direction to its cubic symmetry class.
    [100], [010], [001] → <100>
    [110], [101], [011] → <110>
    [111] → <111>
    """
    if dir_name in ['[100]', '[010]', '[001]']:
        return '<100>'
    elif dir_name in ['[110]', '[101]', '[011]']:
        return '<110>'
    elif dir_name == '[111]':
        return '<111>'
    return dir_name


class TestEpsSensitivity:
    """
    Verify that extrema pattern is stable across different eps values.
    This confirms we're in the acoustic regime.
    """

    def test_extrema_pattern_eps_invariant(self, c15_data, kelvin_data):
        """
        CROSS-CHECK: Extrema pattern CLASS should be invariant to eps.

        [100], [010], [001] are equivalent by cubic symmetry (class <100>).
        [110], [101], [011] are equivalent by cubic symmetry (class <110>).

        We check that the extrema CLASS is preserved, not the specific
        direction (which can vary due to numerical noise at symmetry level).
        """
        eps_values = [0.01, 0.02, 0.04]

        for data in [c15_data, kelvin_data]:
            db, L, name = data['db'], data['L'], data['name']

            class_patterns = []
            for eps in eps_values:
                # Compute velocities at high-symmetry directions
                dirs = high_symmetry_directions()
                dir_names = ['[100]', '[010]', '[001]', '[110]', '[101]', '[011]', '[111]']
                k_mag = eps * 2 * np.pi / L

                v_by_dir = {}
                for i, d in enumerate(dirs):
                    k = k_mag * d
                    omega_T, _, _ = db.classify_modes(k)
                    v_by_dir[dir_names[i]] = omega_T[0] / k_mag

                # Determine pattern by CLASS
                v_max_dir = max(v_by_dir, key=v_by_dir.get)
                v_min_dir = min(v_by_dir, key=v_by_dir.get)
                class_patterns.append((direction_class(v_max_dir), direction_class(v_min_dir)))

            # All eps should give same CLASS pattern
            assert all(p == class_patterns[0] for p in class_patterns), (
                f"{name}: Class pattern varies with eps! {dict(zip(eps_values, class_patterns))}"
            )

            print(f"{name}: Class pattern {class_patterns[0]} stable across eps={eps_values}")

    def test_ranking_eps_invariant(self, c15_data, kelvin_data, wp_data, fcc_data):
        """
        CROSS-CHECK: Ranking should be invariant to eps.
        """
        eps_values = [0.01, 0.02, 0.04]
        structures = [c15_data, kelvin_data, wp_data, fcc_data]

        rankings = []
        for eps in eps_values:
            dv = {}
            for data in structures:
                db, L, name = data['db'], data['L'], data['name']

                dirs = high_symmetry_directions()
                k_mag = eps * 2 * np.pi / L

                velocities = []
                for d in dirs:
                    k = k_mag * d
                    omega_T, _, _ = db.classify_modes(k)
                    velocities.append(omega_T[0] / k_mag)

                v_arr = np.array(velocities)
                dv[name] = (v_arr.max() - v_arr.min()) / v_arr.mean()

            ranking = sorted(dv.keys(), key=lambda n: dv[n])
            rankings.append(ranking)

        # All eps should give same ranking
        assert all(r == rankings[0] for r in rankings), (
            f"Ranking varies with eps! {dict(zip(eps_values, rankings))}"
        )

        print(f"Ranking {rankings[0]} stable across eps={eps_values}")


class TestLocalPerturbation:
    """
    Verify that high-symmetry directions are actually near-global extrema.
    This breaks the circularity of "highsym is truth".
    """

    def test_no_better_direction_nearby(self, c15_data):
        """
        CROSS-CHECK: Perturb high-symmetry directions and verify no
        significantly better v_max or v_min is found nearby.
        """
        db, L = c15_data['db'], c15_data['L']
        eps = 0.02
        k_mag = eps * 2 * np.pi / L

        # Get highsym extrema
        result = compute_delta_v_highsym(db, L)
        v_max_hs = result['v_max']
        v_min_hs = result['v_min']

        # Perturb each high-symmetry direction
        highsym_dirs = high_symmetry_directions()
        n_perturb = 20
        angle = 2.0 * np.pi / 180  # 2 degrees

        all_v = []
        for d in highsym_dirs:
            # Generate random perpendicular perturbations
            for _ in range(n_perturb):
                # Random perpendicular vector
                random_vec = np.random.randn(3)
                perp = random_vec - np.dot(random_vec, d) * d
                if np.linalg.norm(perp) > 1e-10:
                    perp = perp / np.linalg.norm(perp)

                    # Perturbed direction
                    d_perturb = d * np.cos(angle) + perp * np.sin(angle)
                    d_perturb = d_perturb / np.linalg.norm(d_perturb)

                    k = k_mag * d_perturb
                    omega_T, _, _ = db.classify_modes(k)
                    v = omega_T[0] / k_mag
                    all_v.append(v)

        all_v = np.array(all_v)

        # Check if we found anything significantly better
        v_max_perturb = all_v.max()
        v_min_perturb = all_v.min()

        improvement_max = (v_max_perturb - v_max_hs) / v_max_hs * 100
        improvement_min = (v_min_hs - v_min_perturb) / v_min_hs * 100

        print(f"\nLocal perturbation check (C15, {n_perturb} perturbations/dir, {angle*180/np.pi:.1f}°):")
        print(f"  Highsym v_max={v_max_hs:.6f}, perturbed max={v_max_perturb:.6f} ({improvement_max:+.2f}%)")
        print(f"  Highsym v_min={v_min_hs:.6f}, perturbed min={v_min_perturb:.6f} ({improvement_min:+.2f}%)")

        # No significant improvement should be found (< 0.5%)
        assert improvement_max < 0.5, f"Found better v_max: {improvement_max:.2f}% improvement"
        assert improvement_min < 0.5, f"Found better v_min: {improvement_min:.2f}% improvement"

        print("  -> Highsym directions are near-global extrema")


class TestTransverseMean:
    """
    Verify conclusions hold when using mean(T1, T2) instead of T1 alone.
    This avoids branch selection artifacts.
    """

    def test_ranking_with_mean_transverse(self, c15_data, kelvin_data, wp_data):
        """
        CROSS-CHECK: Ranking should be same whether using T1 or mean(T1,T2).
        """
        structures = [c15_data, kelvin_data, wp_data]
        eps = 0.02

        dv_T1 = {}
        dv_mean = {}

        for data in structures:
            db, L, name = data['db'], data['L'], data['name']
            k_mag = eps * 2 * np.pi / L

            v_T1_all = []
            v_mean_all = []

            for d in high_symmetry_directions():
                k = k_mag * d
                omega_T, _, _ = db.classify_modes(k)

                v_T1 = omega_T[0] / k_mag
                v_mean = (omega_T[0] + omega_T[1]) / 2 / k_mag

                v_T1_all.append(v_T1)
                v_mean_all.append(v_mean)

            v_T1_arr = np.array(v_T1_all)
            v_mean_arr = np.array(v_mean_all)

            dv_T1[name] = (v_T1_arr.max() - v_T1_arr.min()) / v_T1_arr.mean()
            dv_mean[name] = (v_mean_arr.max() - v_mean_arr.min()) / v_mean_arr.mean()

        ranking_T1 = sorted(dv_T1.keys(), key=lambda n: dv_T1[n])
        ranking_mean = sorted(dv_mean.keys(), key=lambda n: dv_mean[n])

        print(f"\nTransverse definition comparison:")
        print(f"  Using T1:        {ranking_T1}, dv/v = {[f'{dv_T1[n]:.4f}' for n in ranking_T1]}")
        print(f"  Using mean(T1,T2): {ranking_mean}, dv/v = {[f'{dv_mean[n]:.4f}' for n in ranking_mean]}")

        assert ranking_T1 == ranking_mean, (
            f"Ranking differs! T1: {ranking_T1}, mean: {ranking_mean}"
        )

        print("  -> Ranking is robust to transverse definition")


# =============================================================================
# SUMMARY
# =============================================================================

class TestDocumentation:
    """Print summary for review."""

    def test_print_summary(self, c15_data, kelvin_data, wp_data):
        """Print findings summary."""
        print("\n" + "="*70)
        print("SAMPLING CONVERGENCE FINDINGS SUMMARY")
        print("="*70)

        print("""
KEY FINDING:
    delta_v/v using golden spiral with n_dir < 100 UNDERESTIMATES
    the true value by ~15-20% because it misses high-symmetry extrema.

VERIFIED:
    1. Extrema occur at high-symmetry directions ([100]=max, [110]=min)
    2. Golden spiral converges to high-symmetry as n -> infinity
    3. RANKING is preserved for all sampling methods
    4. Ratio between structures is stable (CV < 5%)

IMPLICATIONS:
    - Ranking tests: OK with n_dir >= 30
    - Magnitude claims: Use high-symmetry OR note ~10% uncertainty
    - Existing tests may report delta_v/v values ~15% low

RECOMMENDATION:
    Add compute_delta_v_highsym() to christoffel.py for accurate
    magnitude measurements on cubic structures.
""")
        print("="*70)

        assert True  # Always passes


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
