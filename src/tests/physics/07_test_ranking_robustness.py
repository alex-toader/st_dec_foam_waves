"""
Tests for ranking robustness across k_L/k_T parameter space
===========================================================

STRUCTURES TESTED: C15, WP, Kelvin, FCC (all 4)

Validates that the isotropy ranking C15 < WP < Kelvin < FCC is preserved
across different spring constant combinations.

TESTS (6 total)
---------------
T1: Isotropic limit (k_L = k_T gives δv/v ≈ 0)
T2: Sampling convergence (n=50 vs n=100 directions)
T3: Extreme k_L/k_T ratios (100:1 and 1:100)
T4: Reproducibility (deterministic results)
T5: Grid coverage (4 corner points)
T6: Physical bounds (0 < δv/v < 1)

EXPECTED OUTPUT (Jan 2026)
--------------------------
    6 passed in 44.28s

RANKING VERIFIED:
    C15 < WP < Kelvin < FCC across all tested (k_L, k_T) combinations

Jan 2026
"""

import sys
from pathlib import Path

# Find src directory robustly (works from any location)
def _find_src():
    """Find src/ by looking for physics/ subdirectory."""
    current = Path(__file__).resolve().parent
    for _ in range(10):  # max 10 levels up
        candidate = current / 'src'
        if (candidate / 'physics').is_dir():
            return candidate
        current = current.parent
    raise RuntimeError("Cannot find src/physics directory")

sys.path.insert(0, str(_find_src()))

import numpy as np
from physics.christoffel import compute_delta_v_direct
from physics.bloch import DisplacementBloch
from core_math.builders import build_fcc_supercell_periodic, build_wp_supercell_periodic
from core_math.builders.multicell_periodic import build_bcc_supercell_periodic
from core_math.builders.c15_periodic import build_c15_supercell_periodic


def build_c15():
    """Build C15 supercell, return (V, E, L)."""
    V, E, F, _ = build_c15_supercell_periodic(N=1, L_cell=4.0)
    return V, E, 4.0

def build_wp():
    """Build WP supercell, return (V, E, L)."""
    V, E, F = build_wp_supercell_periodic(1, L_cell=4.0)
    return V, E, 4.0

def build_kelvin():
    """Build Kelvin supercell, return (V, E, L)."""
    V, E, F, _ = build_bcc_supercell_periodic(2)
    return V, E, 8.0

def build_fcc():
    """Build FCC supercell, return (V, E, L)."""
    result = build_fcc_supercell_periodic(2)
    V, E, F = result[0], result[1], result[2]
    return V, E, 8.0


def test_T1_isotropic_limit():
    """
    T1: k_L = k_T gives isotropic limit (δv/v negligible vs anisotropic).

    When longitudinal and transverse springs are equal,
    the elastic medium has no preferred direction.

    Test with ABSOLUTE thresholds (more robust than ratio):
    - delta_iso < 1e-5 (numerical noise, empirically < 1e-6)
    - delta_aniso > 1e-3 (real signal, empirically > 2%)
    """
    print("T1: Isotropic limit (k_L = k_T)")

    V, E, L = build_kelvin()

    # Isotropic case: k_L = k_T = 1.0
    db_iso = DisplacementBloch(V, E, L, k_L=1.0, k_T=1.0)
    result_iso = compute_delta_v_direct(db_iso, L, n_directions=50)
    delta_iso = result_iso['delta_v_over_v']

    # Anisotropic case: k_L = 3.0, k_T = 1.0 (standard)
    db_aniso = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)
    result_aniso = compute_delta_v_direct(db_aniso, L, n_directions=50)
    delta_aniso = result_aniso['delta_v_over_v']

    print(f"  Isotropic (k_L=k_T=1.0):   δv/v = {delta_iso:.2e}")
    print(f"  Anisotropic (k_L=3,k_T=1): δv/v = {delta_aniso:.2e}")

    # Physics checks with absolute thresholds (more CI-robust than ratio)
    # Isotropic: should be numerical noise (< 1e-5, empirically < 1e-6)
    assert delta_iso < 1e-5, f"Isotropic δv/v too large: {delta_iso:.2e} >= 1e-5"
    # Anisotropic: should be real signal (> 1e-3, empirically > 2%)
    assert delta_aniso > 1e-3, f"Anisotropic δv/v too small: {delta_aniso:.2e} <= 1e-3"

    print(f"  delta_iso < 1e-5: PASS")
    print(f"  delta_aniso > 1e-3: PASS")

    # Additional check: perturbing slightly from isotropic gives measurable δv/v
    db_perturb = DisplacementBloch(V, E, L, k_L=1.0, k_T=0.99)
    result_perturb = compute_delta_v_direct(db_perturb, L, n_directions=50)
    delta_perturb = result_perturb['delta_v_over_v']

    print(f"  Perturbed (k_L=1,k_T=0.99): δv/v = {delta_perturb:.2e}")
    # Perturbing away from isotropic should give measurable signal (> 1e-4, empirically ~6e-4)
    assert delta_perturb > 1e-4, f"Perturbed δv/v too small: {delta_perturb:.2e} <= 1e-4"
    print(f"  delta_perturb > 1e-4: PASS")

    print("  → PASS")
    return True


def test_T2_sampling_convergence():
    """
    T2: n=50 directions is sufficient (< 2% diff from n=100).

    Empirically shows ~0.3% difference, so 2% threshold gives margin.
    Uses n=100 (not 200) to keep test fast.
    """
    print("T2: Sampling convergence (n=50 vs n=100)")

    # Test only on Kelvin (fastest structure) - if it converges, others will too
    V, E, L = build_kelvin()
    db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)

    r50 = compute_delta_v_direct(db, L, n_directions=50)
    r100 = compute_delta_v_direct(db, L, n_directions=100)

    # Guard against div/0 (unlikely but defensive)
    den = max(r100['delta_v_over_v'], 1e-12)
    diff = abs(r50['delta_v_over_v'] - r100['delta_v_over_v']) / den

    print(f"  Kelvin: n=50: {r50['delta_v_over_v']*100:.2f}%, n=100: {r100['delta_v_over_v']*100:.2f}%, diff: {diff*100:.1f}%")

    assert diff < 0.02, f"Diff {diff*100:.1f}% > 2%"
    print("  → PASS")
    return True


def test_T3_extreme_ratios():
    """
    T3: Ranking preserved at extreme k_L/k_T ratios.

    Tests ratio 100:1 in both directions. More extreme cases (1000:1)
    could have numerical conditioning issues on some BLAS implementations.
    """
    print("T3: Extreme k_L/k_T ratios")

    V_c15, E_c15, L_c15 = build_c15()
    V_wp, E_wp, L_wp = build_wp()
    V_k, E_k, L_k = build_kelvin()
    V_f, E_f, L_f = build_fcc()

    test_cases = [
        (10.0, 0.1),   # k_L >> k_T (ratio 100:1)
        (0.1, 10.0),   # k_T >> k_L (ratio 1:100)
    ]

    all_pass = True
    for k_L, k_T in test_cases:
        db_c15 = DisplacementBloch(V_c15, E_c15, L_c15, k_L=k_L, k_T=k_T)
        db_wp = DisplacementBloch(V_wp, E_wp, L_wp, k_L=k_L, k_T=k_T)
        db_k = DisplacementBloch(V_k, E_k, L_k, k_L=k_L, k_T=k_T)
        db_f = DisplacementBloch(V_f, E_f, L_f, k_L=k_L, k_T=k_T)

        c15 = compute_delta_v_direct(db_c15, L_c15, n_directions=50)['delta_v_over_v']
        wp = compute_delta_v_direct(db_wp, L_wp, n_directions=50)['delta_v_over_v']
        kelvin = compute_delta_v_direct(db_k, L_k, n_directions=50)['delta_v_over_v']
        fcc = compute_delta_v_direct(db_f, L_f, n_directions=50)['delta_v_over_v']

        # Guard: check for nan/inf before comparison
        assert np.isfinite(c15) and np.isfinite(wp) and np.isfinite(kelvin) and np.isfinite(fcc), \
            f"nan/inf at k_L={k_L}, k_T={k_T}: C15={c15}, WP={wp}, K={kelvin}, FCC={fcc}"

        preserved = c15 < wp < kelvin < fcc
        status = "✓" if preserved else "✗"
        print(f"  k_L={k_L:5.1f}, k_T={k_T:5.2f}: C15={c15*100:5.2f}% < WP={wp*100:5.2f}% < K={kelvin*100:5.2f}% < FCC={fcc*100:5.2f}% {status}")

        if not preserved:
            all_pass = False

    assert all_pass, "Ranking violated at some extreme ratio"
    print("  → PASS")
    return True


def test_T4_reproducibility():
    """
    T4: Results are reproducible (deterministic).

    No RNG involved - pure linear algebra. Should be perfectly
    deterministic, but use 1e-12 tolerance for CI platform differences.
    """
    print("T4: Reproducibility")

    V, E, L = build_kelvin()
    db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)

    r1 = compute_delta_v_direct(db, L, n_directions=50)
    r2 = compute_delta_v_direct(db, L, n_directions=50)

    diff = abs(r1['delta_v_over_v'] - r2['delta_v_over_v'])
    print(f"  Run 1: {r1['delta_v_over_v']*100:.4f}%")
    print(f"  Run 2: {r2['delta_v_over_v']*100:.4f}%")
    print(f"  Diff: {diff:.2e}")

    # Computation is deterministic (no RNG), but allow 1e-12 for platform differences
    assert diff < 1e-12, f"Results not reproducible: diff = {diff}"
    print("  → PASS")
    return True


def test_T5_grid_coverage():
    """
    T5: Grid corners preserve ranking (4/4 points).

    Tests corners of parameter space for speed. Full grid (8 points)
    tested in main script.
    """
    print("T5: Grid coverage (corners)")

    k_L_values = [1.0, 3.0]
    k_T_values = [0.1, 1.5]  # Just corners, not full grid

    structures = {
        'C15': build_c15(),
        'WP': build_wp(),
        'Kelvin': build_kelvin(),
        'FCC': build_fcc(),
    }

    n_pass = 0
    n_total = 0

    for k_L in k_L_values:
        for k_T in k_T_values:
            n_total += 1

            delta_v = {}
            for name, (V, E, L) in structures.items():
                db = DisplacementBloch(V, E, L, k_L=k_L, k_T=k_T)
                delta_v[name] = compute_delta_v_direct(db, L, n_directions=50)['delta_v_over_v']

            if delta_v['C15'] < delta_v['WP'] < delta_v['Kelvin'] < delta_v['FCC']:
                n_pass += 1

    print(f"  {n_pass}/{n_total} points preserve ranking")
    assert n_pass == n_total, f"Only {n_pass}/{n_total} passed"
    print("  → PASS")
    return True


def test_T6_physical_bounds():
    """
    T6: δv/v values are physically reasonable (0 < δv/v < 1).

    Quick sanity check on standard parameters.
    """
    print("T6: Physical bounds")

    # Test only on Kelvin with standard params (fast sanity check)
    V, E, L = build_kelvin()
    db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)
    result = compute_delta_v_direct(db, L, n_directions=50)
    delta = result['delta_v_over_v']

    assert 0 < delta < 1, f"δv/v={delta} out of bounds"
    assert result['v_min'] > 0, f"v_min <= 0"
    assert result['v_max'] > result['v_min'], f"v_max <= v_min"

    print(f"  δv/v = {delta*100:.2f}% (in (0, 100%))")
    print("  → PASS")
    return True


if __name__ == "__main__":
    print("="*60)
    print("RANKING ROBUSTNESS TESTS")
    print("="*60)
    print()

    tests = [
        test_T1_isotropic_limit,
        test_T2_sampling_convergence,
        test_T3_extreme_ratios,
        test_T4_reproducibility,
        test_T5_grid_coverage,
        test_T6_physical_bounds,
    ]

    n_pass = 0
    for test in tests:
        try:
            test()
            n_pass += 1
        except AssertionError as e:
            print(f"  → FAIL: {e}")
        print()

    print("="*60)
    print(f"SUMMARY: {n_pass}/{len(tests)} tests passed")
    print("="*60)
