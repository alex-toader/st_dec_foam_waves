#!/usr/bin/env python3
"""
T1: Texture Contamination — partial alignment wash-out test

Scenario: fraction p of grains take a preferred orientation R0,
fraction (1-p) are random. This is different from Markov correlation
(which is nearest-neighbor memory) — here the "impurity" is global.

We measure:
  - RMS anisotropy vs p (at fixed M)
  - Threshold p* where signal crosses bound (10^-18 for L=1m)
  - Scaling α(p) — does partial alignment change the exponent?

Key output: maximum tolerable impurity fraction p*.
If p* is very small → model is fragile even if CLT holds for the random part.

Feb 2026

OUTPUT:
======================================================================
T1: TEXTURE CONTAMINATION
======================================================================

--- Part A: Std(x̄) vs |E[x̄]| (M = 10000) ---

  Std(x̄)  = stochastic spread  ∝ δ/√M  (washes out)
  |E[x̄]|  = systematic bias    = p·δ    (does NOT wash out)

           p       Std(x̄)       |E[x̄]|     |E|/Std   p·δ (exact)
-----------------------------------------------------------------
    1.00e-06    1.4283e-04    1.0225e-05        0.07    2.5000e-08
    1.83e-06    1.4351e-04    2.5361e-06        0.02    4.5825e-08
    3.36e-06    1.3897e-04    5.0266e-06        0.04    8.3995e-08
    6.16e-06    1.4265e-04    3.3793e-07        0.00    1.5396e-07
    1.13e-05    1.4513e-04    7.5287e-06        0.05    2.8221e-07
    2.07e-05    1.4513e-04    2.1443e-05        0.15    5.1728e-07
    3.79e-05    1.3924e-04    7.1322e-07        0.01    9.4817e-07
    6.95e-05    1.4807e-04    1.9274e-06        0.01    1.7380e-06
    1.27e-04    1.4090e-04    2.2055e-06        0.02    3.1857e-06
    2.34e-04    1.4176e-04    7.6393e-07        0.01    5.8393e-06
    4.28e-04    1.4662e-04    1.0799e-05        0.07    1.0703e-05
    7.85e-04    1.4910e-04    1.8039e-05        0.12    1.9619e-05
    1.44e-03    1.5045e-04    3.3600e-05        0.22    3.5961e-05
    2.64e-03    1.4464e-04    7.8531e-05        0.54    6.5916e-05
    4.83e-03    1.4510e-04    1.1329e-04        0.78    1.2082e-04
    8.86e-03    1.4807e-04    2.2278e-04        1.50    2.2147e-04
    1.62e-02    1.4654e-04    4.0782e-04        2.78    4.0594e-04
    2.98e-02    1.5306e-04    7.4642e-04        4.88    7.4409e-04
    5.46e-02    1.4905e-04    1.3725e-03        9.21    1.3639e-03
    1.00e-01    1.6061e-04    2.5070e-03       15.61    2.5000e-03

When |E|/Std >> 1: bias dominates (texture problem)
When |E|/Std << 1: stochastic dominates (CLT regime)

--- Part B: Scaling exponent α vs p ---
  (α measured on Std(x̄) only — stochastic component)

           p    α(Std)   |E[x̄]| at M=5000
---------------------------------------------
      0.0000     0.500          1.8133e-05
      0.0001     0.474          8.3069e-06
      0.0010     0.506          2.1086e-05
      0.0100     0.493          2.4904e-04
      0.0500     0.477          1.2551e-03
      0.1000     0.495          2.5173e-03

  α ≈ 0.5 for ALL p: stochastic component always washes out as 1/√M
  But |E[x̄]| = p·δ is CONSTANT in M — separate constraint

--- Part C: Critical p* ---

C15: p* (analytic) = 1.08e-16
WP:  p* (analytic) = 4.00e-17

--- Part D: Physical interpretation ---

C15: p* = 1.08e-16, ~6.7e18 aligned grains tolerated out of ~6e34
WP:  p* = 4.00e-17, ~2.5e18 aligned grains tolerated out of ~6e34

KEY FINDING:
  Texture contamination is a BIAS problem, not a variance problem.
  Two independent constraints:
    1. α > 0.47 (stochastic part washes out)
    2. p < p*   (systematic bias must be negligible)

TESTS:
T1.1 PASS: p=0 → α = 0.513, max|bias| = 1.7e-04
T1.2 PASS: p=1 → mean=0.1000, std=0.000000
T1.3 PASS: bias ratios = 7.6, 10.3 (expect ~10)
T1.4 PASS: p*(C15)=1.08e-16, p*(WP)=4.00e-17
T1.5 PASS: Std α=0.508 (washes out), |E[x̄]| ≈ 0.0049 ≈ p·δ = 0.0050 (constant)
ALL T1 TESTS PASS (5/5)
"""

import numpy as np
from numpy.random import Generator, PCG64

# =============================================================================
# CONSTANTS
# =============================================================================

L_PLANCK = 1.616e-35      # m
L_CAVITY = 1.0            # m
BOUND_NAGEL = 1e-18

DV_V_C15 = 0.0093         # C15 intrinsic anisotropy
DV_V_WP = 0.025           # WP intrinsic anisotropy


# =============================================================================
# GENERATOR
# =============================================================================

def generate_textured(rng, M, delta, p_aligned, R0=None):
    """
    Generate M grain perturbations with texture contamination.

    With probability p_aligned: grain takes orientation R0 (fixed).
    With probability (1-p_aligned): grain is random in [-delta, +delta].

    Parameters
    ----------
    rng : Generator
    M : int — number of grains
    delta : float — perturbation amplitude (δv/v)
    p_aligned : float — fraction of aligned grains (0 to 1)
    R0 : float or None — preferred orientation value.
         If None, use +delta (worst case: maximum bias).
    """
    if R0 is None:
        R0 = delta  # worst case: aligned grains all at max

    signal = np.empty(M)
    mask = rng.random(M) < p_aligned

    n_aligned = np.sum(mask)
    signal[mask] = R0
    signal[~mask] = delta * (2 * rng.random(M - n_aligned) - 1)

    return signal


# =============================================================================
# ANALYSIS
# =============================================================================

def measure_rms_vs_p(p_values, M, delta, n_trials=500, seed=42):
    """
    Measure Std(x̄) and |E[x̄]| separately for each p value.

    Fresh RNG per p for statistical independence.
    Returns array of (p, std, bias).
    """
    results = []

    for ip, p in enumerate(p_values):
        rng = Generator(PCG64(seed + ip * 1000))
        means = []
        for _ in range(n_trials):
            signal = generate_textured(rng, M, delta, p)
            means.append(np.mean(signal))
        means = np.array(means)
        std = np.std(means)           # stochastic spread
        bias = abs(np.mean(means))    # systematic shift (absolute)
        results.append((p, std, bias))

    return results


def measure_alpha_vs_p(p_values, M_values, delta, n_trials=300, seed=42):
    """
    Measure scaling exponent α for the STOCHASTIC component (Std) at each p.

    Fresh RNG per (p, M) pair for statistical independence.
    Reports Std(x̄) and |E[x̄]| separately:
      - Std(x̄) ∝ M^{-1/2}  (stochastic, washes out)
      - |E[x̄]| = p·δ        (bias, independent of M)
    """
    results = []

    for ip, p in enumerate(p_values):
        std_at_M = []
        bias_at_M = []
        for iM, M in enumerate(M_values):
            # Fresh RNG per (p, M) — no cross-contamination
            rng = Generator(PCG64(seed + ip * 1000 + iM))
            means = []
            for _ in range(n_trials):
                signal = generate_textured(rng, M, delta, p)
                means.append(np.mean(signal))
            means = np.array(means)
            std_at_M.append(np.std(means))       # stochastic spread
            bias_at_M.append(abs(np.mean(means))) # systematic shift

        std_at_M = np.array(std_at_M)
        bias_at_M = np.array(bias_at_M)
        M_arr = np.array(M_values, dtype=float)

        # Fit log(Std) = -α log(M) + const  (stochastic component only)
        log_M = np.log(M_arr)
        log_std = np.log(std_at_M)
        A = np.vstack([log_M, np.ones(len(log_M))]).T
        slope, intercept = np.linalg.lstsq(A, log_std, rcond=None)[0]
        alpha = -slope

        results.append((p, alpha, std_at_M, bias_at_M))

    return results


def find_p_star(delta, M, bound, n_trials=1000, seed=42):
    """
    Find threshold p* where signal crosses bound.

    The mean perturbation for textured model:
      <x> = p * R0 + (1-p) * 0 = p * delta  (bias)
      Var(x) = (1-p) * delta^2/3 / M  (random part CLT)
      RMS ≈ sqrt(bias^2 + var) ≈ p*delta for p >> 1/sqrt(M)

    So p* ≈ bound / delta analytically.

    We verify numerically.
    """
    # Analytic estimate
    p_star_analytic = bound / delta

    # Numerical scan
    p_scan = np.logspace(-10, -1, 40)
    rng = Generator(PCG64(seed))

    p_star_numeric = None

    for p in p_scan:
        means = []
        for _ in range(n_trials):
            signal = generate_textured(rng, M, delta, p)
            means.append(np.mean(signal))

        # The signal has both bias and fluctuation
        # Observable = |<x>| (systematic shift of cavity)
        # Use RMS which includes both
        rms = np.sqrt(np.mean(np.array(means)**2))

        if rms > bound and p_star_numeric is None:
            p_star_numeric = p

    return p_star_analytic, p_star_numeric


# =============================================================================
# MAIN
# =============================================================================

def run_analysis():
    print("=" * 70)
    print("T1: TEXTURE CONTAMINATION")
    print("=" * 70)
    print()

    M = 10000  # grains for RMS measurement (small M for speed)
    M_planck = int(L_CAVITY / L_PLANCK)  # ~6e34, used for p* only

    # --- Part A: Std vs |E[x̄]| at fixed M ---
    print("--- Part A: Std(x̄) vs |E[x̄]| (M = %d) ---" % M)
    print()
    print("  Std(x̄)  = stochastic spread  ∝ δ/√M  (washes out)")
    print("  |E[x̄]|  = systematic bias    = p·δ    (does NOT wash out)")
    print()

    p_values = np.logspace(-6, -1, 20)
    results_rms = measure_rms_vs_p(p_values, M, DV_V_WP, n_trials=500)

    print(f"{'p':>12}  {'Std(x̄)':>12}  {'|E[x̄]|':>12}  {'|E|/Std':>10}  {'p·δ (exact)':>12}")
    print("-" * 65)
    for p, std, bias in results_rms:
        ratio = bias / std if std > 0 else float('inf')
        exact_bias = p * DV_V_WP
        print(f"{p:12.2e}  {std:12.4e}  {bias:12.4e}  {ratio:10.2f}  {exact_bias:12.4e}")

    print()
    print("When |E|/Std >> 1: bias dominates (texture problem)")
    print("When |E|/Std << 1: stochastic dominates (CLT regime)")
    print()

    # --- Part B: α vs p (stochastic component) ---
    print("--- Part B: Scaling exponent α vs p ---")
    print("  (α measured on Std(x̄) only — stochastic component)")
    print()

    M_values = [200, 500, 1000, 2000, 5000]
    p_for_alpha = [0, 1e-4, 1e-3, 1e-2, 0.05, 0.1]

    results_alpha = measure_alpha_vs_p(p_for_alpha, M_values, DV_V_WP, n_trials=300)

    print(f"{'p':>12}  {'α(Std)':>8}  {'|E[x̄]| at M=5000':>18}")
    print("-" * 45)
    for p, alpha, std_at_M, bias_at_M in results_alpha:
        print(f"{p:12.4f}  {alpha:8.3f}  {bias_at_M[-1]:18.4e}")

    print()
    print("  α ≈ 0.5 for ALL p: stochastic component always washes out as 1/√M")
    print("  But |E[x̄]| = p·δ is CONSTANT in M — separate constraint")
    print()

    # --- Part C: p* threshold ---
    print("--- Part C: Critical p* ---")
    print()

    for label, dv_v in [("C15", DV_V_C15), ("WP", DV_V_WP)]:
        p_star_a = BOUND_NAGEL / dv_v
        print(f"{label}: p* (analytic) = {p_star_a:.2e}")
        print(f"  Interpretation: if even {p_star_a:.1e} of grains are aligned,")
        print(f"  the bias alone exceeds the Nagel bound.")
        print()

    # --- Part D: What p* means physically ---
    print("--- Part D: Physical interpretation ---")
    print()

    for label, dv_v in [("C15", DV_V_C15), ("WP", DV_V_WP)]:
        p_star = BOUND_NAGEL / dv_v
        n_aligned_per_m = p_star * (L_CAVITY / L_PLANCK)
        print(f"{label}:")
        print(f"  p* = {p_star:.2e}")
        print(f"  At M ~ 6e34 grains per meter:")
        print(f"  Number of aligned grains tolerated: {n_aligned_per_m:.2e}")
        print(f"  Fraction: 1 in {1/p_star:.1e}")
        print()

    print("=" * 70)
    print("KEY FINDING:")
    print()
    print("Texture contamination is a BIAS problem, not a variance problem.")
    print("The bias = p * delta does NOT wash out with M (it's independent of M).")
    print("The only protection is p being small enough: p < bound/delta.")
    print()
    print("This is a STRONGER constraint than α > 0.47:")
    print("  - α > 0.47 says 'random part washes out fast enough'")
    print("  - p < p* says 'systematic part must be negligible'")
    print("  - Both are needed independently")
    print("=" * 70)


# =============================================================================
# TESTS
# =============================================================================

def test_p0_recovers_clt():
    """At p=0, should recover standard CLT (α ≈ 0.5), zero bias."""
    M_values = [200, 500, 1000, 5000]
    results = measure_alpha_vs_p([0.0], M_values, delta=0.1, n_trials=500)
    p, alpha, std_at_M, bias_at_M = results[0]
    assert 0.45 < alpha < 0.55, f"p=0 should give α≈0.5, got {alpha:.3f}"
    # At p=0, bias should be negligible (just noise)
    max_bias = np.max(bias_at_M)
    assert max_bias < 0.01, f"p=0 bias should be ~0, got {max_bias:.4f}"
    print(f"T1.1 PASS: p=0 → α = {alpha:.3f}, max|bias| = {max_bias:.4e}")


def test_p1_no_washout():
    """At p=1, all grains aligned → no wash-out (bias dominates)."""
    rng = Generator(PCG64(42))
    M = 1000
    delta = 0.1

    means = [np.mean(generate_textured(rng, M, delta, p_aligned=1.0))
             for _ in range(100)]

    # All means should be ~delta (no randomness)
    spread = np.std(means)
    avg = np.mean(means)
    assert avg > 0.09, f"p=1 mean should be ~delta, got {avg:.4f}"
    assert spread < 0.01, f"p=1 should have no variance, got {spread:.4f}"
    print(f"T1.2 PASS: p=1 → mean={avg:.4f}, std={spread:.6f}")


def test_bias_scales_with_p():
    """Bias should scale linearly with p."""
    rng = Generator(PCG64(42))
    M = 5000
    delta = 0.1
    n_trials = 500

    biases = []
    p_test = [0.001, 0.01, 0.1]
    for p in p_test:
        rng2 = Generator(PCG64(42))
        means = [np.mean(generate_textured(rng2, M, delta, p)) for _ in range(n_trials)]
        biases.append(abs(np.mean(means)))

    # Bias should scale ~linearly with p
    ratio_1 = biases[1] / biases[0]  # p=0.01 / p=0.001 → expect ~10
    ratio_2 = biases[2] / biases[1]  # p=0.1  / p=0.01  → expect ~10

    assert 5 < ratio_1 < 20, f"Bias ratio should be ~10, got {ratio_1:.1f}"
    assert 5 < ratio_2 < 20, f"Bias ratio should be ~10, got {ratio_2:.1f}"
    print(f"T1.3 PASS: bias ratios = {ratio_1:.1f}, {ratio_2:.1f} (expect ~10)")


def test_p_star_analytic():
    """p* = bound/delta is correct order of magnitude."""
    p_star_c15 = BOUND_NAGEL / DV_V_C15
    p_star_wp = BOUND_NAGEL / DV_V_WP

    assert 1e-17 < p_star_c15 < 1e-15, f"C15 p* out of range: {p_star_c15:.2e}"
    assert 1e-17 < p_star_wp < 1e-15, f"WP p* out of range: {p_star_wp:.2e}"
    print(f"T1.4 PASS: p*(C15)={p_star_c15:.2e}, p*(WP)={p_star_wp:.2e}")


def test_two_independent_constraints():
    """Std(x̄) ∝ 1/√M (washes out) while |E[x̄]| = p·δ (constant in M)."""
    M_values = [500, 1000, 2000, 5000]
    p = 0.05
    delta = 0.1
    results = measure_alpha_vs_p([p], M_values, delta=delta, n_trials=500)
    _, alpha, std_at_M, bias_at_M = results[0]

    # Std should wash out (α ≈ 0.5)
    assert 0.40 < alpha < 0.60, f"Std exponent should be ~0.5, got {alpha:.3f}"

    # Bias should be ~constant across M (= p·δ = 0.005)
    expected_bias = p * delta
    for i, M in enumerate(M_values):
        rel = abs(bias_at_M[i] - expected_bias) / expected_bias
        assert rel < 0.5, f"Bias at M={M} should be ~{expected_bias:.4f}, got {bias_at_M[i]:.4f}"

    print(f"T1.5 PASS: Std α={alpha:.3f} (washes out), "
          f"|E[x̄]| ≈ {np.mean(bias_at_M):.4f} ≈ p·δ = {expected_bias:.4f} (constant)")


def run_tests():
    print()
    print("=" * 50)
    print("T1 TESTS")
    print("=" * 50)
    print()

    test_p0_recovers_clt()
    test_p1_no_washout()
    test_bias_scales_with_p()
    test_p_star_analytic()
    test_two_independent_constraints()

    print()
    print("ALL T1 TESTS PASS (5/5)")
    print()


if __name__ == '__main__':
    import sys
    if '--test' in sys.argv:
        run_tests()
    else:
        run_analysis()
        run_tests()
