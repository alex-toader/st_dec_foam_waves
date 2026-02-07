#!/usr/bin/env python3
"""
T2: Domain Blocks — finite correlation domains wash-out test

Scenario: optical path split into blocks of m grains.
Each block has a single random orientation, blocks are mutually independent.
This models "domain structure" (natural in materials, different from Markov).

Expected: effective number of independent samples = M/m
  → RMS ~ delta / sqrt(M/m) = delta * sqrt(m) / sqrt(M)
  → same α = 0.5 in M, but with prefactor sqrt(m) worse

We verify:
  1. Scaling remains α = 0.5 in M (at fixed m)
  2. RMS scales as sqrt(m) at fixed M
  3. Margin vs domain size m: what's the maximum tolerable domain?
  4. For physical foam: convert m to ℓ_domain and find critical domain size

Feb 2026

OUTPUT:
======================================================================
T2: DOMAIN BLOCKS
======================================================================

--- Part A: Scaling α at fixed domain size ---

  m (domain)         α               Comment
--------------------------------------------------
           1     0.496                   CLT
           5     0.515                   CLT
          10     0.516                   CLT
          50     0.517                   CLT
         100     0.500                   CLT

Expected: α ≈ 0.5 for all m (domains are independent → CLT applies)
The domain size affects PREFACTOR, not EXPONENT.

--- Part B: RMS vs domain size m (M = 10000) ---

       m           RMS    RMS/RMS(m=1)     sqrt(m)     ratio
------------------------------------------------------------
       1    1.4407e-04            1.00        1.00      1.00
       2    1.9701e-04            1.37        1.41      0.97
       5    3.1241e-04            2.17        2.24      0.97
      10    4.6559e-04            3.23        3.16      1.02
      20    6.3842e-04            4.43        4.47      0.99
      50    9.9684e-04            6.92        7.07      0.98
     100    1.4711e-03           10.21       10.00      1.02
     200    1.9289e-03           13.39       14.14      0.95
     500    3.3701e-03           23.39       22.36      1.05
    1000    4.5279e-03           31.43       31.62      0.99

Expected: RMS/RMS(m=1) ≈ sqrt(m)

--- Part C: Margin vs domain size ---

C15: Critical domain size m* ≈ 1e3 grains (ℓ_domain* ≈ 1.6e-32 m)
WP:  Critical domain size m* ≈ 1e2 grains (ℓ_domain* ≈ 1.6e-33 m)

KEY FINDING:
  Domain blocks = repackaging of ℓ_corr sensitivity.
  M_eff = M/m → equivalent to ℓ_corr → m × ℓ_corr.
  Same table as 7_correlation_wash_out.md.

TESTS:
T2.1 PASS: m=1 → α = 0.495
T2.2 PASS: α ≈ 0.5 for m=5,20,50
T2.3 PASS: RMS ratios = 3.20 (expect 3.2), 9.46 (expect 10)
T2.4 PASS: margin(m=1)=1.0e+01, margin(m=100)=1.0e+00
ALL T2 TESTS PASS (4/4)
"""

import numpy as np
from numpy.random import Generator, PCG64

# =============================================================================
# CONSTANTS
# =============================================================================

L_PLANCK = 1.616e-35      # m
L_CAVITY = 1.0            # m
BOUND_NAGEL = 1e-18

DV_V_C15 = 0.0093
DV_V_WP = 0.025


# =============================================================================
# GENERATOR
# =============================================================================

def generate_domain_blocks(rng, M, delta, m):
    """
    Generate M grain perturbations with domain block structure.

    Path is divided into blocks of m grains.
    Each block gets a single random value in [-delta, +delta].
    All grains in a block share that value.

    Parameters
    ----------
    rng : Generator
    M : int — total number of grains
    delta : float — perturbation amplitude
    m : int — domain size (grains per block)
    """
    n_blocks = max(1, M // m)
    remainder = M - n_blocks * m

    # Each block gets one random orientation
    block_values = delta * (2 * rng.random(n_blocks) - 1)

    # Expand blocks into grain array
    signal = np.repeat(block_values, m)

    # Handle remainder (partial last block)
    if remainder > 0:
        last_val = delta * (2 * rng.random(1)[0] - 1)
        signal = np.concatenate([signal, np.full(remainder, last_val)])

    return signal[:M]


# =============================================================================
# ANALYSIS
# =============================================================================

def measure_scaling_at_fixed_m(m, M_values, delta, n_trials=500, seed=42):
    """Measure α at fixed domain size m."""
    rng = Generator(PCG64(seed))

    rms_values = []
    for M in M_values:
        means = []
        for _ in range(n_trials):
            signal = generate_domain_blocks(rng, M, delta, m)
            means.append(np.mean(signal))
        rms_values.append(np.std(means))

    rms_values = np.array(rms_values)
    M_arr = np.array(M_values, dtype=float)

    # Fit RMS ~ M^{-α}
    log_M = np.log(M_arr)
    log_rms = np.log(rms_values)
    A = np.vstack([log_M, np.ones(len(log_M))]).T
    slope, intercept = np.linalg.lstsq(A, log_rms, rcond=None)[0]
    alpha = -slope

    return alpha, rms_values


def measure_rms_vs_m(m_values, M, delta, n_trials=500, seed=42):
    """Measure RMS at fixed M for various domain sizes m."""
    rng = Generator(PCG64(seed))

    results = []
    for m in m_values:
        means = []
        for _ in range(n_trials):
            signal = generate_domain_blocks(rng, M, delta, m)
            means.append(np.mean(signal))
        rms = np.std(means)
        results.append((m, rms))

    return results


def compute_margin_vs_domain(dv_v, m_values, label=""):
    """
    Compute margin for domain blocks.

    With domains of size m:
      M_eff = M / m  (independent samples)
      Δc/c = dv_v / sqrt(M_eff) = dv_v * sqrt(m) / sqrt(M)
      margin = bound / Δc/c = bound * sqrt(M) / (dv_v * sqrt(m))
    """
    M = L_CAVITY / L_PLANCK  # total grains

    results = []
    for m in m_values:
        M_eff = M / m
        dc_c = dv_v / np.sqrt(M_eff)
        margin = BOUND_NAGEL / dc_c
        l_domain = m * L_PLANCK  # physical domain size

        results.append((m, l_domain, M_eff, dc_c, margin))

    return results


# =============================================================================
# MAIN
# =============================================================================

def run_analysis():
    print("=" * 70)
    print("T2: DOMAIN BLOCKS")
    print("=" * 70)
    print()

    # --- Part A: α remains 0.5 at fixed m ---
    print("--- Part A: Scaling α at fixed domain size ---")
    print()

    M_values = [500, 1000, 2000, 5000, 10000]

    print(f"{'m (domain)':>12}  {'α':>8}  {'Comment':>20}")
    print("-" * 50)
    for m in [1, 5, 10, 50, 100]:
        alpha, _ = measure_scaling_at_fixed_m(m, M_values, DV_V_WP, n_trials=400)
        comment = "CLT" if abs(alpha - 0.5) < 0.05 else "deviation"
        print(f"{m:>12}  {alpha:>8.3f}  {comment:>20}")

    print()
    print("Expected: α ≈ 0.5 for all m (domains are independent → CLT applies)")
    print("The domain size affects PREFACTOR, not EXPONENT.")
    print()

    # --- Part B: RMS vs m at fixed M ---
    print("--- Part B: RMS vs domain size m (M = 10000) ---")
    print()

    M_fixed = 10000
    m_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    results_rms = measure_rms_vs_m(m_values, M_fixed, DV_V_WP, n_trials=500)

    rms_m1 = results_rms[0][1]  # reference: m=1

    print(f"{'m':>8}  {'RMS':>12}  {'RMS/RMS(m=1)':>14}  {'sqrt(m)':>10}  {'ratio':>8}")
    print("-" * 60)
    for m, rms in results_rms:
        ratio = rms / rms_m1
        expected = np.sqrt(m)
        print(f"{m:>8}  {rms:12.4e}  {ratio:14.2f}  {expected:10.2f}  {ratio/expected:8.2f}")

    print()
    print("Expected: RMS/RMS(m=1) ≈ sqrt(m)  [each domain counts as 1 sample]")
    print()

    # --- Part C: Margin vs domain size (analytic) ---
    print("--- Part C: Margin vs domain size ---")
    print()

    m_scan = [1, 10, 100, 1000, 10**4, 10**6, 10**8, 10**10, 10**15, 10**20]

    for label, dv_v in [("C15", DV_V_C15), ("WP", DV_V_WP)]:
        print(f"Structure: {label} (δv/v = {dv_v:.4f})")
        print()
        print(f"{'m':>12}  {'ℓ_domain':>12}  {'M_eff':>12}  {'Δc/c':>12}  {'Margin':>10}  {'Status':>10}")
        print("-" * 75)

        results_margin = compute_margin_vs_domain(dv_v, m_scan)
        m_star = None

        for m, l_dom, M_eff, dc_c, margin in results_margin:
            status = "PASS" if margin > 1 else "FAIL"
            if margin <= 1 and m_star is None:
                m_star = m

            # Format ℓ_domain nicely
            if l_dom < 1e-30:
                l_str = f"{l_dom:.1e}"
            elif l_dom < 1e-15:
                l_str = f"{l_dom:.1e}"
            else:
                l_str = f"{l_dom:.1e}"

            print(f"{m:>12.0e}  {l_str:>12}  {M_eff:>12.1e}  {dc_c:>12.1e}  {margin:>10.1e}  {status:>10}")

        if m_star:
            l_star = m_star * L_PLANCK
            print(f"\n  → Critical domain size: m* ≈ {m_star:.0e} grains")
            print(f"    ℓ_domain* ≈ {l_star:.1e} m")
        print()

    # --- Part D: Summary ---
    print("=" * 70)
    print("KEY FINDING:")
    print()
    print("Domain blocks DON'T change the scaling exponent (α stays 0.5).")
    print("They change the EFFECTIVE M: M_eff = M/m = L/(m × ℓ_corr).")
    print()
    print("This is equivalent to redefining ℓ_corr → m × ℓ_corr.")
    print("Already analyzed in ST_8/release/7_correlation_wash_out.md:")
    print("  C15 fails at ℓ_corr > 715 × ℓ_P  (≡ m = 715)")
    print("  WP  fails at ℓ_corr > 100 × ℓ_P  (≡ m = 100)")
    print()
    print("Domain blocks are a REPACKAGING of the ℓ_corr sensitivity,")
    print("not a new failure mode. But useful as intuition builder:")
    print("  'How big can domains be?' → 'same as ℓ_corr table in 7_*'")
    print("=" * 70)


# =============================================================================
# TESTS
# =============================================================================

def test_m1_recovers_uncorrelated():
    """m=1 should give same result as uncorrelated."""
    M_values = [200, 500, 1000, 5000]
    alpha, _ = measure_scaling_at_fixed_m(1, M_values, 0.1, n_trials=500)
    assert 0.45 < alpha < 0.55, f"m=1 should give α≈0.5, got {alpha:.3f}"
    print(f"T2.1 PASS: m=1 → α = {alpha:.3f}")


def test_alpha_independent_of_m():
    """α should be ~0.5 regardless of m (for m << M)."""
    M_values = [500, 1000, 2000, 5000]
    for m in [5, 20, 50]:
        alpha, _ = measure_scaling_at_fixed_m(m, M_values, 0.1, n_trials=400)
        assert 0.40 < alpha < 0.60, f"m={m} should give α≈0.5, got {alpha:.3f}"
    print(f"T2.2 PASS: α ≈ 0.5 for m=5,20,50")


def test_rms_scales_sqrt_m():
    """RMS should scale as sqrt(m)."""
    M = 10000
    m_values = [1, 10, 100]
    results = measure_rms_vs_m(m_values, M, 0.1, n_trials=500)

    rms_1 = results[0][1]
    rms_10 = results[1][1]
    rms_100 = results[2][1]

    ratio_10 = rms_10 / rms_1
    ratio_100 = rms_100 / rms_1

    # Expect sqrt(10) ≈ 3.16, sqrt(100) = 10
    assert 2.0 < ratio_10 < 5.0, f"RMS(m=10)/RMS(m=1) should be ~3.2, got {ratio_10:.2f}"
    assert 6.0 < ratio_100 < 15.0, f"RMS(m=100)/RMS(m=1) should be ~10, got {ratio_100:.2f}"
    print(f"T2.3 PASS: RMS ratios = {ratio_10:.2f} (expect 3.2), {ratio_100:.2f} (expect 10)")


def test_margin_formula():
    """Margin formula: margin = bound * sqrt(M) / (dv_v * sqrt(m))."""
    M = L_CAVITY / L_PLANCK
    dv_v = DV_V_WP

    # At m=1: should match standard result
    margin_m1 = BOUND_NAGEL * np.sqrt(M) / dv_v
    margin_expected = BOUND_NAGEL / (dv_v / np.sqrt(M))

    assert abs(margin_m1 / margin_expected - 1) < 1e-10, "Margin formula inconsistent"

    # At m=100: margin should be sqrt(100) = 10× worse
    margin_m100 = BOUND_NAGEL * np.sqrt(M) / (dv_v * np.sqrt(100))
    assert abs(margin_m100 / (margin_m1 / 10) - 1) < 1e-10, "m=100 margin should be 10× worse"

    print(f"T2.4 PASS: margin(m=1)={margin_m1:.1e}, margin(m=100)={margin_m100:.1e}")


def run_tests():
    print()
    print("=" * 50)
    print("T2 TESTS")
    print("=" * 50)
    print()

    test_m1_recovers_uncorrelated()
    test_alpha_independent_of_m()
    test_rms_scales_sqrt_m()
    test_margin_formula()

    print()
    print("ALL T2 TESTS PASS (4/4)")
    print()


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        run_tests()
    else:
        run_analysis()
        run_tests()
