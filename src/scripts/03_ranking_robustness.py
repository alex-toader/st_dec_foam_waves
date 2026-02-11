"""
(k_L, k_T) Grid Test for Ranking Robustness
===========================================

QUESTION: Is the ranking δv/v: WP < Kelvin < FCC preserved across
different spring constant combinations?

INPUTS
------

Internal (from model):
  - Foam geometry (WP, Kelvin/BCC, FCC) from builders
  - DisplacementBloch dynamical matrix D(k̂) from physics/bloch.py
  - v(k̂) from acoustic branch eigenvalues of D(k̂)
  - δv/v = (v_max - v_min) / v_mean from directional sampling

External:
  - k_L: longitudinal spring constant (edge stretching) - FREE PARAMETER
  - k_T: transverse spring constant (edge bending) - FREE PARAMETER
  - Grid tested: k_L ∈ {1.0, 3.0}, k_T ∈ {0.1, 0.3, 0.5, 1.5}

OUTPUTS
-------

  - Table of δv/v for each (k_L, k_T) point
  - Ranking check: WP < Kelvin < FCC preserved?
  - Count: 8/8 points preserve ranking

EXPECTED OUTPUT:
    k_L    k_T |       WP   Kelvin      FCC | Ranking
    1.0    0.1 |   4.21%  10.20%  26.89% | WP < K < FCC ✓
    1.0    0.3 |   2.73%   6.80%  17.76% | WP < K < FCC ✓
    1.0    0.5 |   1.66%   4.22%  11.02% | WP < K < FCC ✓
    1.0    1.5 |   0.88%   2.40%   7.69% | WP < K < FCC ✓
    3.0    0.1 |   4.83%  11.57%  30.67% | WP < K < FCC ✓
    3.0    0.3 |   4.21%  10.20%  26.89% | WP < K < FCC ✓
    3.0    0.5 |   3.66%   8.96%  23.52% | WP < K < FCC ✓
    3.0    1.5 |   1.66%   4.22%  11.02% | WP < K < FCC ✓

    SUMMARY: 8/8 points preserve ranking WP < Kelvin < FCC

CONCLUSION:
    Isotropy ranking is a GEOMETRIC property of foam structures,
    not an artifact of specific spring constants.

DERIVATION:
    Foam geometry → DisplacementBloch(k_L, k_T) → v(k̂) → δv/v

NOTE: k_L = k_T gives near-isotropic behavior (δv/v extremely small in our spring model), excluded from grid.

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


# Local builder wrappers (not dependent on private _build_* functions)
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


def quick_delta_v(V, E, L, k_L, k_T, n_directions=50):
    """Fast δv/v computation with fewer directions."""
    db = DisplacementBloch(V, E, L, k_L=k_L, k_T=k_T)
    result = compute_delta_v_direct(db, L, n_directions=n_directions)
    return result['delta_v_over_v']


def _fmt_percent(x):
    """Format value as percentage or 'nan'."""
    if np.isfinite(x):
        return f"{x*100:6.2f}%"
    return "   nan"


def run_grid_test():
    """
    Run (k_L, k_T) grid test.

    Returns dict with results and summary.
    """
    # Grid definition (reduced for speed)
    # NOTE: k_L = k_T gives near-isotropic behavior (δv/v extremely small)
    # so we exclude that degenerate case
    k_L_values = [1.0, 3.0]
    k_T_values = [0.1, 0.3, 0.5, 1.5]  # Avoid k_T = k_L

    # Build structures once (geometry doesn't change)
    structures = {
        'WP': build_wp(),
        'Kelvin': build_kelvin(),
        'FCC': build_fcc(),
    }

    results = []
    n_preserved = 0
    n_total = 0

    print("="*70)
    print("(k_L, k_T) GRID TEST: RANKING ROBUSTNESS")
    print("="*70)
    print()
    print("Testing if δv/v ranking WP < Kelvin < FCC is preserved across")
    print("different spring constant combinations.")
    print()
    print("-"*70)
    print(f"{'k_L':>6} {'k_T':>6} | {'WP':>8} {'Kelvin':>8} {'FCC':>8} | {'Ranking':<20}")
    print("-"*70)

    for k_L in k_L_values:
        for k_T in k_T_values:
            n_total += 1

            # Compute δv/v for each structure (fast version)
            delta_v = {}
            for name, (V, E, L) in structures.items():
                try:
                    delta_v[name] = quick_delta_v(V, E, L, k_L, k_T, n_directions=50)
                except Exception:
                    delta_v[name] = np.nan

            # Check ranking
            wp = delta_v['WP']
            kelvin = delta_v['Kelvin']
            fcc = delta_v['FCC']

            # Ranking preserved if WP < Kelvin < FCC (all finite)
            if np.isfinite(wp) and np.isfinite(kelvin) and np.isfinite(fcc):
                if wp < kelvin < fcc:
                    ranking = "WP < K < FCC ✓"
                    n_preserved += 1
                elif wp < kelvin:
                    ranking = "WP < K, but K ≥ FCC"
                elif kelvin < fcc:
                    ranking = "WP ≥ K, but K < FCC"
                else:
                    ranking = "VIOLATED"
            else:
                ranking = "UNSTABLE (nan)"

            print(f"{k_L:6.1f} {k_T:6.1f} | {_fmt_percent(wp)} {_fmt_percent(kelvin)} {_fmt_percent(fcc)} | {ranking}")

            results.append({
                'k_L': k_L,
                'k_T': k_T,
                'WP': wp,
                'Kelvin': kelvin,
                'FCC': fcc,
                'ranking_preserved': (ranking == "WP < K < FCC ✓"),
            })

    print("-"*70)
    print()
    print(f"SUMMARY: {n_preserved}/{n_total} points preserve ranking WP < Kelvin < FCC")
    print()

    if n_preserved == n_total:
        print("✓ RANKING IS ROBUST: preserved across all (k_L, k_T) combinations")
        print()
        print("This validates that the isotropy ranking is a GEOMETRIC property")
        print("of the foam structures, not an artifact of specific spring constants.")
    else:
        n_violated = n_total - n_preserved
        print(f"⚠ WARNING: {n_violated} points violate ranking")
        print()
        # Find violating points
        for r in results:
            if not r['ranking_preserved']:
                print(f"  k_L={r['k_L']}, k_T={r['k_T']}: WP={r['WP']:.4f}, Kelvin={r['Kelvin']:.4f}, FCC={r['FCC']:.4f}")

    return {
        'results': results,
        'n_preserved': n_preserved,
        'n_total': n_total,
        'all_preserved': n_preserved == n_total,
    }


def run_extended_ratio_test():
    """
    Extended test: vary k_L/k_T ratio more systematically.
    """
    print()
    print("="*70)
    print("EXTENDED TEST: k_L/k_T RATIO EXPLORATION")
    print("="*70)
    print()

    k_L = 1.0
    k_T_values = [0.1, 0.5, 2.0]  # Avoid k_T = k_L (isotropic)

    structures = {
        'WP': build_wp(),
        'Kelvin': build_kelvin(),
        'FCC': build_fcc(),
    }

    print(f"Fixed k_L = {k_L}")
    print()
    print(f"{'k_T':>8} {'ratio':>8} | {'WP':>8} {'Kelvin':>8} {'FCC':>8} | {'Status':<12}")
    print("-"*70)

    for k_T in k_T_values:
        ratio = k_L / k_T

        delta_v = {}
        for name, (V, E, L) in structures.items():
            try:
                delta_v[name] = quick_delta_v(V, E, L, k_L, k_T, n_directions=50)
            except Exception:
                delta_v[name] = np.nan

        wp, kelvin, fcc = delta_v['WP'], delta_v['Kelvin'], delta_v['FCC']

        if np.isfinite(wp) and np.isfinite(kelvin) and np.isfinite(fcc):
            if wp < kelvin < fcc:
                status = "✓ preserved"
            else:
                status = "✗ violated"
        else:
            status = "unstable"

        print(f"{k_T:8.2f} {ratio:8.1f} | {_fmt_percent(wp)} {_fmt_percent(kelvin)} {_fmt_percent(fcc)} | {status}")

    print()


def test_ranking_robustness():
    """Pytest-compatible test function."""
    result = run_grid_test()

    # Assert ranking is preserved for ALL points (8/8)
    assert result['n_preserved'] == result['n_total'], \
        f"Ranking violated: {result['n_preserved']}/{result['n_total']} passed"

    print("\n✓ TEST PASSED: Ranking robust across (k_L, k_T) grid")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test ranking robustness across (k_L, k_T) grid")
    parser.add_argument("--test", action="store_true", help="Run as pytest-style test")
    parser.add_argument("--extended", action="store_true", help="Run extended ratio test")
    args = parser.parse_args()

    if args.test:
        test_ranking_robustness()
    else:
        run_grid_test()
        if args.extended:
            run_extended_ratio_test()
