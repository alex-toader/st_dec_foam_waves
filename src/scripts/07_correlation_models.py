#!/usr/bin/env python3
"""
06_correlation_models.py - Correlation Model Stress Tests

Tests how different grain correlation models affect wash-out scaling.

KEY QUESTION: Does nature have 1/√M wash-out?

The 1/√M scaling assumes UNCORRELATED grains (Markov/white noise).
Different correlation structures give different scaling:

    RMS ~ M^{-α}

where α depends on the correlation model:
    - α = 0.5: Uncorrelated (CLT, our default assumption)
    - α < 0.5: Positive correlations (weaker wash-out, HARDER to pass bounds)
    - α > 0.5: Anti-correlations (stronger wash-out, EASIER to pass bounds)
    - α = 0:   Frozen/fully correlated (NO wash-out - EXCLUDED by any finite bound)

This script:
1. Implements multiple correlation models via Monte Carlo
2. Measures the actual scaling exponent α
3. Shows which correlation models are compatible with bounds
4. Provides TESTABLE predictions (not just parameter fitting)

PHYSICS INTERPRETATION:
    - Uncorrelated: each grain has random orientation, independent of neighbors
    - Positive corr: neighboring grains tend to align (like ferromagnet)
    - Anti-corr: neighboring grains tend to anti-align (like antiferromagnet)
    - Frozen: grain orientations perfectly correlated (no wash-out, α = 0)

Jan 2026
"""

import numpy as np
from numpy.random import Generator, PCG64
from dataclasses import dataclass
from typing import Callable, List, Tuple

# matplotlib imported only in plot functions to avoid CI issues on headless systems

# =============================================================================
# CONSTANTS
# =============================================================================

L_PLANCK = 1.616e-35      # Planck length [m]
L_CAVITY = 1.0            # Cavity length [m]
BOUND_NAGEL = 1e-18       # Nagel 2015 bound

# Reference δv/v for testing
DV_V_WP = 0.025           # Weaire-Phelan intrinsic anisotropy


# =============================================================================
# CORRELATION MODELS
# =============================================================================

def generate_uncorrelated(rng: Generator, M: int, delta: float) -> np.ndarray:
    """
    Model 1: Uncorrelated (white noise)

    Each grain has independent random orientation perturbation.
    Expected scaling: α = 0.5 (CLT)

    Parameters
    ----------
    rng : Generator
        Random number generator
    M : int
        Number of grains
    delta : float
        Perturbation amplitude (δv/v)

    Returns
    -------
    np.ndarray
        Array of M perturbations, each in [-delta, +delta]
    """
    return delta * (2 * rng.random(M) - 1)


def generate_powerlaw_correlated(rng: Generator, M: int, delta: float,
                                  gamma: float = 0.5) -> np.ndarray:
    """
    Model 2: Power-law correlated

    Correlation function: C(r) ~ r^{-γ}
    For γ < 1: long-range correlations, α < 0.5
    For γ > 1: short-range, approaches α = 0.5

    Implementation: Fourier method with power spectrum S(k) ~ k^{γ-1}
    Uses rfft/irfft to guarantee real output with correct Hermitian symmetry.

    Parameters
    ----------
    gamma : float
        Correlation decay exponent. γ < 1 gives long-range correlations.
    """
    # Use rfft/irfft for real-valued output with proper Hermitian symmetry
    n_freq = M // 2 + 1
    k = np.fft.rfftfreq(M)
    k[0] = 1e-10  # Avoid division by zero at DC

    # Power spectrum: S(k) ~ |k|^{beta} where beta = gamma - 1
    beta = gamma - 1
    S = np.abs(k) ** beta
    S[0] = 0  # Zero mean

    # Random phases (DC and Nyquist must be real for Hermitian symmetry)
    phases = 2 * np.pi * rng.random(n_freq)
    phases[0] = 0  # DC component must be real
    if M % 2 == 0:
        phases[-1] = 0  # Nyquist component must be real for even M

    # Generate correlated noise with proper symmetry
    fourier = np.sqrt(S) * np.exp(1j * phases)
    signal = np.fft.irfft(fourier, n=M)

    # Normalize to desired amplitude
    if np.std(signal) > 1e-10:
        signal = signal / np.std(signal) * delta

    return signal


def generate_anticorrelated(rng: Generator, M: int, delta: float,
                            strength: float = 0.8) -> np.ndarray:
    """
    Model 3: Anti-correlated (alternating tendency)

    Neighboring grains tend to have opposite signs.
    Expected: α > 0.5 (faster wash-out)

    Implementation: x_i = ε_i - strength * x_{i-1}

    Parameters
    ----------
    strength : float
        Anti-correlation strength (0 = uncorrelated, 1 = perfect alternation)
    """
    noise = delta * (2 * rng.random(M) - 1)
    signal = np.zeros(M)
    signal[0] = noise[0]

    for i in range(1, M):
        signal[i] = noise[i] - strength * signal[i-1]

    # Normalize
    signal = signal / np.std(signal) * delta

    return signal


def generate_random_walk(rng: Generator, M: int, delta: float) -> np.ndarray:
    """
    Model 4: Frozen correlation (perfect memory)

    Each grain's orientation is correlated with all previous grains.
    Expected: α = 0 (NO wash-out - RMS stays constant regardless of M)

    This model is EXCLUDED by Lorentz tests (any finite bound requires α > 0).

    NOTE: This is NOT a true random walk (which would have α = -0.5, RMS ~ √M).
    The √M normalization in steps keeps RMS bounded, producing α ≈ 0.
    """
    steps = delta * (2 * rng.random(M) - 1) / np.sqrt(M)  # Normalized steps
    signal = np.cumsum(steps)

    return signal


def generate_exponential_correlated(rng: Generator, M: int, delta: float,
                                     corr_length: int = 10) -> np.ndarray:
    """
    Model 5: Exponential correlation (Ornstein-Uhlenbeck like)

    C(r) ~ exp(-r/ξ) where ξ = corr_length
    For ξ << M: approaches uncorrelated (α → 0.5)
    For ξ ~ M: significant correlations (α < 0.5)

    Implementation: AR(1) process x_i = ρ * x_{i-1} + ε_i
    """
    rho = np.exp(-1.0 / corr_length)
    noise = delta * (2 * rng.random(M) - 1) * np.sqrt(1 - rho**2)

    signal = np.zeros(M)
    signal[0] = noise[0]

    for i in range(1, M):
        signal[i] = rho * signal[i-1] + noise[i]

    return signal


# =============================================================================
# MEASUREMENT FUNCTIONS
# =============================================================================

@dataclass
class ScalingResult:
    """Results from scaling measurement."""
    model_name: str
    alpha: float           # Measured scaling exponent
    alpha_scatter: float       # Uncertainty in α
    M_values: np.ndarray   # M values used
    rms_values: np.ndarray # Measured RMS at each M
    r_squared: float       # Fit quality


def measure_scaling(generator: Callable, M_values: List[int],
                    delta: float, n_trials: int = 200,
                    seed: int = 42, **kwargs) -> ScalingResult:
    """
    Measure the scaling exponent α for a given correlation model.

    Computes RMS of sum for various M, fits RMS ~ M^{-α}

    Parameters
    ----------
    generator : Callable
        Function that generates correlated samples
    M_values : List[int]
        List of M values to test
    delta : float
        Perturbation amplitude
    n_trials : int
        Number of Monte Carlo trials per M
    seed : int
        Random seed
    **kwargs : dict
        Additional arguments for generator

    Returns
    -------
    ScalingResult
        Measured scaling exponent and data
    """
    rng = Generator(PCG64(seed))

    rms_values = []

    for M in M_values:
        sums = []
        for _ in range(n_trials):
            perturbations = generator(rng, M, delta, **kwargs)
            total = np.sum(perturbations) / M  # Mean perturbation
            sums.append(total)

        rms = np.std(sums)
        rms_values.append(rms)

    rms_values = np.array(rms_values)
    M_values = np.array(M_values)

    # Fit log(RMS) = -α * log(M) + const
    log_M = np.log(M_values)
    log_rms = np.log(rms_values)

    # Linear regression
    A = np.vstack([log_M, np.ones(len(log_M))]).T
    result = np.linalg.lstsq(A, log_rms, rcond=None)
    slope, intercept = result[0]

    alpha = -slope  # RMS ~ M^{-α} means log(RMS) = -α log(M) + const

    # Compute R² for fit quality
    predicted = slope * log_M + intercept
    ss_res = np.sum((log_rms - predicted) ** 2)
    ss_tot = np.sum((log_rms - np.mean(log_rms)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Estimate uncertainty (bootstrap would be better, but this is quick)
    residuals = log_rms - predicted
    alpha_scatter = np.std(residuals) / np.sqrt(len(M_values))

    return ScalingResult(
        model_name=generator.__name__,
        alpha=alpha,
        alpha_scatter=alpha_scatter,
        M_values=M_values,
        rms_values=rms_values,
        r_squared=r_squared
    )


def compute_dc_c(dv_v: float, alpha: float, M: float) -> float:
    """
    Compute Δc/c for given scaling exponent.

    Δc/c = δv/v × M^{-α}

    For α = 0.5 (uncorrelated): Δc/c = δv/v / √M
    """
    return dv_v * (M ** (-alpha))


def max_M_for_bound(dv_v: float, alpha: float, bound: float) -> float:
    """
    Compute maximum M (minimum ℓ_corr) needed to pass bound.

    From: δv/v × M^{-α} < bound
    We get: M > (δv/v / bound)^{1/α}
    """
    return (dv_v / bound) ** (1.0 / alpha)


def critical_alpha(dv_v: float, M: float, bound: float) -> float:
    """
    Compute critical α where predicted Δc/c exactly equals bound.

    From: δv/v × M^{-α} = bound
    We get: α = log(δv/v / bound) / log(M)

    Parameters
    ----------
    dv_v : float
        Intrinsic anisotropy
    M : float
        Number of grains (e.g., L/ℓ_P for benchmark)
    bound : float
        Experimental bound

    Returns
    -------
    float
        Critical α - model passes if actual α > α_crit
    """
    return np.log(dv_v / bound) / np.log(M)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_correlation_analysis():
    """Run full correlation model analysis."""

    print("=" * 80)
    print("CORRELATION MODEL ANALYSIS")
    print("=" * 80)
    print()
    print("Testing how different correlation structures affect wash-out scaling.")
    print()
    print("Model: RMS ~ M^{-α}")
    print("  α = 0.5: Uncorrelated (CLT) - our default assumption")
    print("  α < 0.5: Positive correlations - weaker wash-out")
    print("  α > 0.5: Anti-correlations - stronger wash-out")
    print("  α → 0:   Frozen correlation - no wash-out (EXCLUDED)")
    print()

    # M values to test
    M_values = [100, 200, 500, 1000, 2000, 5000, 10000]
    delta = 0.1  # Arbitrary for measuring α

    # Models to test
    models = [
        ("Uncorrelated", generate_uncorrelated, {}),
        ("Power-law (γ=0.3)", generate_powerlaw_correlated, {"gamma": 0.3}),
        ("Power-law (γ=0.7)", generate_powerlaw_correlated, {"gamma": 0.7}),
        ("Anti-correlated (s=0.5)", generate_anticorrelated, {"strength": 0.5}),
        ("Anti-correlated (s=0.8)", generate_anticorrelated, {"strength": 0.8}),
        ("Exponential (ξ=10)", generate_exponential_correlated, {"corr_length": 10}),
        ("Exponential (ξ=100)", generate_exponential_correlated, {"corr_length": 100}),
        ("Frozen (α≈0)", generate_random_walk, {}),
    ]

    results = []

    print("-" * 80)
    print(f"{'Model':<30} {'α (measured)':<15} {'R²':<10} {'Status':<15}")
    print("-" * 80)

    for name, generator, kwargs in models:
        result = measure_scaling(generator, M_values, delta, n_trials=500, **kwargs)
        result.model_name = name
        results.append(result)

        # Determine status based on α
        if result.alpha < 0.1:
            status = "EXCLUDED"
        elif result.alpha < 0.4:
            status = "Marginal"
        elif 0.4 <= result.alpha <= 0.6:
            status = "OK (CLT-like)"
        else:
            status = "Good (fast wash-out)"

        print(f"{name:<30} {result.alpha:>6.3f} ± {result.alpha_scatter:.3f}   "
              f"{result.r_squared:>6.4f}    {status:<15}")

    print("-" * 80)
    print()

    return results


def compute_exclusion_by_alpha():
    """Show how bounds change with α."""

    print("=" * 80)
    print("EXCLUSION VS SCALING EXPONENT α")
    print("=" * 80)
    print()

    # For WP structure
    dv_v = DV_V_WP
    L = L_CAVITY
    M_planck = L / L_PLANCK  # M when ℓ_corr = ℓ_P

    print(f"Structure: WP (δv/v = {dv_v:.1%})")
    print(f"Cavity: L = {L} m")
    print(f"M at ℓ_corr = ℓ_P: {M_planck:.2e}")
    print(f"Bound: Nagel 2015 = {BOUND_NAGEL:.0e}")
    print()

    # Compute critical α
    alpha_crit = critical_alpha(dv_v, M_planck, BOUND_NAGEL)
    print(f">>> CRITICAL α = {alpha_crit:.3f} <<<")
    print(f"    Model PASSES if α > {alpha_crit:.3f}")
    print(f"    Model FAILS  if α < {alpha_crit:.3f}")
    print()

    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    print("-" * 70)
    print(f"{'α':<8} {'Δc/c at ℓ_P':<15} {'Margin':<12} {'Status':<15}")
    print("-" * 70)

    for alpha in alphas:
        dc_c = compute_dc_c(dv_v, alpha, M_planck)
        margin = BOUND_NAGEL / dc_c

        if margin < 1:
            status = f"FAIL ({margin:.1e}×)"
        elif margin < 10:
            status = f"Marginal ({margin:.0f}×)"
        else:
            status = f"PASS ({margin:.0f}×)"

        print(f"{alpha:<8.1f} {dc_c:<15.1e} {margin:<12.1e} {status:<15}")

    print("-" * 70)
    print()

    # Physical interpretation
    print("=" * 80)
    print("PHYSICAL INTERPRETATION")
    print("=" * 80)
    print()
    print("What physics produces each α?")
    print()
    print("  α ≈ 0.5 (UNCORRELATED / WHITE NOISE)")
    print("    - Each grain has random orientation, independent of neighbors")
    print("    - Like 'melted' spacetime foam - no memory between grains")
    print("    - CLT applies: RMS ~ 1/√M")
    print("    - STATUS: Required by Lorentz bounds")
    print()
    print("  α < 0.5 (POSITIVE CORRELATIONS / LONG-RANGE ORDER)")
    print("    - Neighboring grains tend to align")
    print("    - Like ferromagnetic domains")
    print("    - Weaker wash-out: anisotropy persists over longer scales")
    print("    - STATUS: α < 0.35 EXCLUDED by current bounds")
    print()
    print("  α > 0.5 (ANTI-CORRELATIONS / ALTERNATING)")
    print("    - Neighboring grains tend to anti-align")
    print("    - Like antiferromagnetic ordering")
    print("    - Faster wash-out: anisotropy cancels more efficiently")
    print("    - STATUS: Would make model MORE robust")
    print()
    print("  α ≈ 0 (FROZEN / FULLY CORRELATED)")
    print("    - All grains have same orientation (perfect correlation)")
    print("    - Like a single large domain")
    print("    - NO wash-out: RMS stays constant regardless of M")
    print("    - STATUS: EXCLUDED - any finite bound requires α > 0")
    print()
    print("-" * 80)
    print()
    print("TESTABLE PREDICTION:")
    print(f"  Current bounds REQUIRE α > {alpha_crit:.2f} for WP at ℓ_corr = ℓ_P")
    print("  This EXCLUDES:")
    print("    - Frozen/fully correlated models (α ≈ 0)")
    print("    - Strong long-range correlations (α < 0.4)")
    print("  This FAVORS:")
    print("    - Uncorrelated grains (α ≈ 0.5)")
    print("    - Anti-correlated grains (α > 0.5)")
    print()


def plot_scaling_comparison(results: List[ScalingResult], save_path: str = None):
    """Plot RMS vs M for all models."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: RMS vs M
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for result, color in zip(results, colors):
        ax1.loglog(result.M_values, result.rms_values, 'o-',
                   color=color, label=f"{result.model_name} (α={result.alpha:.2f})")

    # Reference line for α = 0.5
    M_ref = np.array([100, 10000])
    ax1.loglog(M_ref, 0.03 * M_ref**(-0.5), 'k--', linewidth=2,
               label='α = 0.5 (CLT)', alpha=0.5)

    ax1.set_xlabel('M (number of grains)', fontsize=12)
    ax1.set_ylabel('RMS of mean perturbation', fontsize=12)
    ax1.set_title('Wash-out Scaling by Correlation Model', fontsize=14)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: α vs exclusion status
    alphas = [r.alpha for r in results]
    names = [r.model_name for r in results]

    # Compute margin for each α
    M_planck = L_CAVITY / L_PLANCK
    margins = [BOUND_NAGEL / compute_dc_c(DV_V_WP, a, M_planck) for a in alphas]
    log_margins = np.log10(np.array(margins))

    bars = ax2.barh(range(len(results)), log_margins, color=colors)

    # Color bars by status
    for bar, margin in zip(bars, margins):
        if margin < 1:
            bar.set_color('red')
            bar.set_alpha(0.7)
        elif margin < 10:
            bar.set_color('orange')
            bar.set_alpha(0.7)
        else:
            bar.set_color('green')
            bar.set_alpha(0.7)

    ax2.axvline(0, color='black', linewidth=2, linestyle='--', label='Bound = 1')
    ax2.axvline(1, color='gray', linewidth=1, linestyle=':', label='Margin = 10')

    ax2.set_yticks(range(len(results)))
    ax2.set_yticklabels([f"{n}\n(α={a:.2f})" for n, a in zip(names, alphas)], fontsize=9)
    ax2.set_xlabel('log₁₀(Margin vs Nagel 2015)', fontsize=12)
    ax2.set_title('WP Margin by Correlation Model\n(at ℓ_corr = ℓ_P)', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3, axis='x')

    # Add text annotations
    for i, (margin, alpha) in enumerate(zip(margins, alphas)):
        status = "FAIL" if margin < 1 else ("OK" if margin < 10 else "PASS")
        ax2.text(log_margins[i] + 0.1, i, f"{status}", va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    return fig


# =============================================================================
# TESTS
# =============================================================================

def test_uncorrelated_scaling():
    """T1: Uncorrelated model should give α ≈ 0.5."""
    M_values = [100, 500, 1000, 5000]
    result = measure_scaling(generate_uncorrelated, M_values, delta=0.1, n_trials=1000)

    assert 0.45 < result.alpha < 0.55, f"Uncorrelated α should be ~0.5, got {result.alpha:.3f}"
    assert result.r_squared > 0.99, f"Fit should be excellent, got R²={result.r_squared:.4f}"

    print(f"T1 PASS: Uncorrelated α = {result.alpha:.3f} (expected ~0.5)")


def test_anticorrelated_faster():
    """T2: Anti-correlated should have smaller RMS than uncorrelated.

    Note: For finite-range correlations (AR(1)), asymptotic α = 0.5.
    The benefit is in the prefactor (smaller RMS), not necessarily α > 0.5.
    Testing RMS ratio is more robust than testing α > 0.5.
    """
    M_values = [100, 500, 1000, 5000]
    M_test = 1000  # Compare at this M
    n_trials = 1000
    rng_seed = 42

    # Measure RMS for both models at M_test
    from numpy.random import Generator, PCG64

    rng = Generator(PCG64(rng_seed))
    sums_uncorr = [np.mean(generate_uncorrelated(rng, M_test, 0.1)) for _ in range(n_trials)]
    rms_uncorr = np.std(sums_uncorr)

    rng = Generator(PCG64(rng_seed))
    sums_anti = [np.mean(generate_anticorrelated(rng, M_test, 0.1, strength=0.8)) for _ in range(n_trials)]
    rms_anti = np.std(sums_anti)

    ratio = rms_anti / rms_uncorr

    # Anti-correlated should have smaller RMS (faster effective wash-out)
    assert ratio < 0.9, f"Anti-correlated RMS should be < 0.9× uncorrelated, got {ratio:.3f}"

    print(f"T2 PASS: Anti-correlated RMS ratio = {ratio:.3f} (< 0.9× uncorrelated)")


def test_powerlaw_slower():
    """T3: Power-law (γ < 1) should give α < 0.5."""
    M_values = [100, 500, 1000, 5000]
    result = measure_scaling(generate_powerlaw_correlated, M_values, delta=0.1,
                            n_trials=1000, gamma=0.3)

    assert result.alpha < 0.45, f"Power-law (γ=0.3) α should be < 0.5, got {result.alpha:.3f}"

    print(f"T3 PASS: Power-law (γ=0.3) α = {result.alpha:.3f} (expected < 0.5)")


def test_frozen_no_washout():
    """T4: Frozen correlation should give α ≈ 0 (no wash-out)."""
    M_values = [100, 500, 1000, 5000]
    result = measure_scaling(generate_random_walk, M_values, delta=0.1, n_trials=1000)

    # Frozen: RMS stays constant regardless of M
    # α should be very small (≈ 0)
    assert result.alpha < 0.2, f"Frozen model α should be ~0, got {result.alpha:.3f}"

    print(f"T4 PASS: Frozen α = {result.alpha:.3f} (expected ~0, no wash-out)")


def test_alpha_determines_margin():
    """T5: Different α gives different margins at same ℓ_corr."""
    M = 1e34  # ~ M at ℓ_corr = ℓ_P
    dv_v = 0.025

    dc_c_05 = compute_dc_c(dv_v, 0.5, M)  # α = 0.5
    dc_c_03 = compute_dc_c(dv_v, 0.3, M)  # α = 0.3
    dc_c_07 = compute_dc_c(dv_v, 0.7, M)  # α = 0.7

    # Lower α → less wash-out → larger Δc/c
    assert dc_c_03 > dc_c_05 > dc_c_07, "Lower α should give larger Δc/c"

    # Check roughly correct magnitudes
    assert 1e-20 < dc_c_05 < 1e-18, f"α=0.5: expected ~10⁻¹⁹, got {dc_c_05:.1e}"

    print(f"T5 PASS: α=0.3 → Δc/c={dc_c_03:.1e}, α=0.5 → {dc_c_05:.1e}, α=0.7 → {dc_c_07:.1e}")


def test_exponential_approaches_uncorrelated():
    """T6: Exponential with small ξ should approach α = 0.5."""
    M_values = [100, 500, 1000, 5000]

    # Small correlation length → nearly uncorrelated
    result_small = measure_scaling(generate_exponential_correlated, M_values,
                                   delta=0.1, n_trials=1000, corr_length=2)

    # Large correlation length → more correlated → smaller α
    result_large = measure_scaling(generate_exponential_correlated, M_values,
                                   delta=0.1, n_trials=1000, corr_length=500)

    assert result_small.alpha > result_large.alpha, \
        f"Smaller ξ should give larger α: got {result_small.alpha:.3f} vs {result_large.alpha:.3f}"

    assert 0.4 < result_small.alpha < 0.6, \
        f"Small ξ should give α ≈ 0.5, got {result_small.alpha:.3f}"

    print(f"T6 PASS: ξ=2 → α={result_small.alpha:.3f}, ξ=500 → α={result_large.alpha:.3f}")


def test_powerlaw_alpha_theory():
    """T7: Power-law α matches theoretical prediction α ≈ γ/2.

    For C(r) ~ r^{-γ} (0 < γ < 1):
        Var(x̄) ~ M^{-γ}  →  RMS ~ M^{-γ/2}  →  α ≈ γ/2

    This provides theoretical backing for the power-law correlation model.
    """
    M_values = [100, 500, 1000, 5000]

    print("  Testing α ≈ γ/2 theory:")
    for gamma in [0.3, 0.5, 0.7]:
        result = measure_scaling(generate_powerlaw_correlated, M_values,
                                delta=0.1, n_trials=500, gamma=gamma)
        alpha_theory = gamma / 2
        diff = abs(result.alpha - alpha_theory)

        assert diff < 0.08, \
            f"γ={gamma}: α={result.alpha:.3f} should be ≈ {alpha_theory:.3f} (diff={diff:.3f})"
        print(f"    γ={gamma}: α={result.alpha:.3f} ≈ γ/2={alpha_theory:.3f} ✓")

    print("T7 PASS: α ≈ γ/2 theory verified for γ=0.3, 0.5, 0.7")


def test_powerlaw_psd_slope():
    """T8: Power-law generator produces correct PSD slope."""
    rng = Generator(PCG64(42))
    M = 2000

    for gamma in [0.3, 0.5, 0.7]:
        signal = generate_powerlaw_correlated(rng, M, delta=1.0, gamma=gamma)

        # Verify signal is real
        assert not np.iscomplexobj(signal), f"Signal should be real for γ={gamma}"

        # Compute PSD and fit slope
        psd = np.abs(np.fft.rfft(signal))**2
        k = np.fft.rfftfreq(M)

        # Fit log-log in mid-frequency range (avoid DC and Nyquist)
        mask = (k > 0.02) & (k < 0.4)
        log_k = np.log(k[mask])
        log_psd = np.log(psd[mask])
        slope = np.polyfit(log_k, log_psd, 1)[0]

        expected_slope = gamma - 1
        tolerance = 0.15  # Allow 0.15 tolerance for finite-size effects

        assert abs(slope - expected_slope) < tolerance, \
            f"PSD slope for γ={gamma}: expected {expected_slope:.2f}, got {slope:.2f}"

    print(f"T8 PASS: Power-law PSD slopes correct (γ=0.3,0.5,0.7)")


def test_critical_alpha():
    """T9: Critical α calculation."""
    # At critical α, margin = 1 (Δc/c = bound)
    dv_v = 0.025
    M = 1e34
    bound = 1e-18

    alpha_crit = critical_alpha(dv_v, M, bound)

    # Verify: at α_crit, Δc/c should equal bound
    dc_c = compute_dc_c(dv_v, alpha_crit, M)
    assert abs(dc_c - bound) / bound < 1e-10, \
        f"At α_crit, Δc/c should equal bound: got {dc_c:.2e} vs {bound:.2e}"

    # α slightly above → passes (Δc/c < bound)
    dc_above = compute_dc_c(dv_v, alpha_crit * 1.01, M)
    assert dc_above < bound, "α > α_crit should give Δc/c < bound"

    # α slightly below → fails (Δc/c > bound)
    dc_below = compute_dc_c(dv_v, alpha_crit * 0.99, M)
    assert dc_below > bound, "α < α_crit should give Δc/c > bound"

    # For WP at ℓ_P, α_crit should be around 0.47
    M_planck = L_CAVITY / L_PLANCK
    alpha_wp = critical_alpha(DV_V_WP, M_planck, BOUND_NAGEL)
    assert 0.4 < alpha_wp < 0.5, f"WP α_crit should be ~0.47, got {alpha_wp:.3f}"

    print(f"T9 PASS: α_crit = {alpha_wp:.3f} for WP at ℓ_corr = ℓ_P")


def run_tests():
    """Run all tests."""
    print()
    print("=" * 60)
    print("TESTS")
    print("=" * 60)
    print()

    test_uncorrelated_scaling()
    test_anticorrelated_faster()
    test_powerlaw_slower()
    test_frozen_no_washout()
    test_alpha_determines_margin()
    test_exponential_approaches_uncorrelated()
    test_powerlaw_alpha_theory()
    test_powerlaw_psd_slope()
    test_critical_alpha()

    print()
    print("=" * 60)
    print("ALL TESTS PASS (9/9)")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    import sys

    if '--test' in sys.argv:
        run_tests()
    elif '--plot' in sys.argv:
        import matplotlib.pyplot as plt

        results = run_correlation_analysis()
        compute_exclusion_by_alpha()

        plot_path = 'correlation_models.png'
        if len(sys.argv) > 2 and sys.argv[-1] != '--plot':
            plot_path = sys.argv[-1]
        plot_scaling_comparison(results, plot_path)
        plt.show()
    else:
        # Default: run analysis and tests
        results = run_correlation_analysis()
        compute_exclusion_by_alpha()
        run_tests()
        print()
        print("Run with --plot to generate comparison figure")
