#!/usr/bin/env python3
"""
10_test_r2_bracketing.py - R2-2 Bracketing Validation Tests

Tests for release/4_r2_bracketing.md arithmetic and envelope properties.

Test A: Arithmetic verification
- factor = η + (1-η) * S_mat
- Δ = factor * (δv/v) / √M
- margin = bound / Δ
- Monotonicity: margin decreases when S_mat increases (for η < 1)

Test B: Bracketing envelope
- For S_mat > 1: max Δ at η = 0 (matter-dominated)
- For S_mat < 1: max Δ at η = 1 (photon-only)
- Worst-case over S_mat ∈ [0.2, 2.5] is at S_mat = 2.5

Date: Jan 2026
"""

import numpy as np
import pytest

# =============================================================================
# Model constants (from release/4_r2_bracketing.md)
# =============================================================================

# Structure anisotropies
STRUCTURES = {
    'C15': 0.0093,    # δv/v = 0.93%
    'WP': 0.025,      # δv/v = 2.5%
    'Kelvin': 0.064,  # δv/v = 6.4%
    'FCC': 0.165,     # δv/v = 16.5%
}

# Physical parameters
L_PATH = 1.0                    # m (cavity path length)
L_PLANCK = 1.616e-35            # m
M = L_PATH / L_PLANCK           # ≈ 6.19e34
SQRT_M = np.sqrt(M)             # ≈ 2.49e17 (more stable than M**(-0.5))
BOUND = 1e-18                   # Nagel proxy

# Documented √M for cross-check
DOC_SQRT_M = 2.49e17            # From release doc

# S_mat anchors
S_MAT_ANCHORS = [0.2, 0.5, 2.5]
S_MAT_MIN = 0.2
S_MAT_MAX = 2.5

# η values
ETA_VALUES = [0.0, 0.5, 1.0]


def compute_factor(eta: float, s_mat: float) -> float:
    """Compute bracketing factor: η + (1-η) * S_mat."""
    return eta + (1 - eta) * s_mat


def compute_delta(dv_v: float, factor: float) -> float:
    """Compute predicted Δν/ν = factor * (δv/v) / √M."""
    return factor * dv_v / SQRT_M


def compute_margin(delta: float) -> float:
    """Compute margin = bound / Δ."""
    return BOUND / delta


# =============================================================================
# Test A: Arithmetic Verification
# =============================================================================

class TestArithmetic:
    """Verify arithmetic consistency of the bracketing model."""

    def test_A0_sqrt_m_matches_doc(self):
        """Sanity check: √M should match documented value."""
        assert SQRT_M == pytest.approx(DOC_SQRT_M, rel=0.01), \
            f"SQRT_M mismatch: {SQRT_M:.2e} vs doc {DOC_SQRT_M:.2e}"
        print(f"A0: √M = {SQRT_M:.2e} matches doc {DOC_SQRT_M:.2e} ✓")

    def test_A1_factor_formula(self):
        """Test factor = η + (1-η) * S_mat for all combinations."""
        for eta in ETA_VALUES:
            for s_mat in S_MAT_ANCHORS:
                factor = compute_factor(eta, s_mat)
                expected = eta + (1 - eta) * s_mat
                assert factor == pytest.approx(expected, rel=1e-10), \
                    f"Factor mismatch: η={eta}, S_mat={s_mat}"
        print("A1: Factor formula verified for all (η, S_mat) combinations")

    def test_A2_delta_formula(self):
        """Test Δ = factor * (δv/v) / √M for C15."""
        dv_v = STRUCTURES['C15']

        # Photon-only (η=1, factor=1)
        delta_photon = compute_delta(dv_v, 1.0)
        expected_photon = dv_v / SQRT_M
        assert delta_photon == pytest.approx(expected_photon, rel=1e-10)

        # Check against documented value
        # Tolerance reflects rounding in release tables (δv/v given to 2-3 digits)
        assert delta_photon == pytest.approx(3.74e-20, rel=0.02), \
            f"C15 photon-only Δ = {delta_photon:.2e}, expected ~3.74e-20"

        print(f"A2: C15 photon-only Δ = {delta_photon:.2e} (matches doc)")

    def test_A3_margin_formula(self):
        """Test margin = bound / Δ for documented values."""
        dv_v = STRUCTURES['C15']

        # Photon-only
        delta = compute_delta(dv_v, 1.0)
        margin = compute_margin(delta)
        assert margin == pytest.approx(27, rel=0.05), \
            f"C15 photon-only margin = {margin:.1f}×, expected ~27×"

        # Worst-case ionic (S_mat=2.5, η=0)
        factor_ionic = compute_factor(0.0, 2.5)
        delta_ionic = compute_delta(dv_v, factor_ionic)
        margin_ionic = compute_margin(delta_ionic)
        assert margin_ionic == pytest.approx(11, rel=0.05), \
            f"C15 worst-case margin = {margin_ionic:.1f}×, expected ~11×"

        print(f"A3: C15 margins verified (photon-only: {margin:.0f}×, worst-case: {margin_ionic:.0f}×)")

    def test_A4_monotonicity_s_mat(self):
        """Margin decreases when S_mat increases (for η < 1)."""
        dv_v = STRUCTURES['C15']

        for eta in [0.0, 0.5]:  # Not η=1 where S_mat doesn't matter
            margins = []
            for s_mat in np.linspace(S_MAT_MIN, S_MAT_MAX, 10):
                factor = compute_factor(eta, s_mat)
                delta = compute_delta(dv_v, factor)
                margins.append(compute_margin(delta))

            # Check strictly decreasing (deterministic, no tolerance needed)
            for i in range(len(margins) - 1):
                assert margins[i] > margins[i+1], \
                    f"Monotonicity violated at η={eta}: margin[{i}]={margins[i]:.1f} not > margin[{i+1}]={margins[i+1]:.1f}"

        print("A4: Monotonicity verified: margin decreases with S_mat for η < 1")

    def test_A5_all_table_values(self):
        """Verify all values in Section 5.1 table.

        Tolerances (rel=0.03 for Δ, rel=0.05 for margin) reflect rounding
        in release tables where δv/v and printed margins are given to 2-3 digits.
        """
        expected = {
            'C15': (0.0093, 3.74e-20, 27),
            'WP': (0.025, 1.0e-19, 10),
            'Kelvin': (0.064, 2.57e-19, 3.9),
            'FCC': (0.165, 6.63e-19, 1.5),
        }

        for name, (dv_v, exp_delta, exp_margin) in expected.items():
            delta = compute_delta(dv_v, 1.0)  # photon-only
            margin = compute_margin(delta)

            assert delta == pytest.approx(exp_delta, rel=0.03), \
                f"{name}: Δ = {delta:.2e}, expected {exp_delta:.2e}"
            assert margin == pytest.approx(exp_margin, rel=0.05), \
                f"{name}: margin = {margin:.1f}×, expected {exp_margin}×"

        print("A5: All Section 5.1 table values verified")


# =============================================================================
# Test B: Bracketing Envelope
# =============================================================================

class TestBracketingEnvelope:
    """Verify envelope properties of the bracketing model."""

    def test_B1_s_mat_gt_1_max_at_eta_0(self):
        """For S_mat > 1, max Δ is at η = 0 (matter-dominated)."""
        dv_v = STRUCTURES['C15']
        s_mat = 2.5  # > 1

        deltas = {}
        for eta in ETA_VALUES:
            factor = compute_factor(eta, s_mat)
            deltas[eta] = compute_delta(dv_v, factor)

        # Max should be at η = 0 (robust float comparison)
        assert deltas[0.0] == pytest.approx(max(deltas.values()), rel=1e-10), \
            f"For S_mat={s_mat} > 1, max Δ should be at η=0, got max at η={max(deltas, key=deltas.get)}"

        print(f"B1: For S_mat={s_mat} > 1, max Δ at η=0 (matter-dominated) ✓")

    def test_B2_s_mat_lt_1_max_at_eta_1(self):
        """For S_mat < 1, max Δ is at η = 1 (photon-only)."""
        dv_v = STRUCTURES['C15']
        s_mat = 0.2  # < 1

        deltas = {}
        for eta in ETA_VALUES:
            factor = compute_factor(eta, s_mat)
            deltas[eta] = compute_delta(dv_v, factor)

        # Max should be at η = 1 (robust float comparison)
        assert deltas[1.0] == pytest.approx(max(deltas.values()), rel=1e-10), \
            f"For S_mat={s_mat} < 1, max Δ should be at η=1, got max at η={max(deltas, key=deltas.get)}"

        print(f"B2: For S_mat={s_mat} < 1, max Δ at η=1 (photon-only) ✓")

    def test_B3_worst_case_at_s_mat_max(self):
        """Worst-case over S_mat ∈ [0.2, 2.5] is at S_mat = 2.5."""
        dv_v = STRUCTURES['C15']

        # Scan all (η, S_mat) combinations
        max_delta = 0
        max_params = None

        for eta in np.linspace(0, 1, 11):
            for s_mat in np.linspace(S_MAT_MIN, S_MAT_MAX, 21):
                factor = compute_factor(eta, s_mat)
                delta = compute_delta(dv_v, factor)
                if delta > max_delta:
                    max_delta = delta
                    max_params = (eta, s_mat)

        # Worst case should be at η=0, S_mat=2.5
        assert max_params[0] == pytest.approx(0.0, abs=0.1), \
            f"Worst-case η should be ~0, got {max_params[0]}"
        assert max_params[1] == pytest.approx(S_MAT_MAX, abs=0.1), \
            f"Worst-case S_mat should be {S_MAT_MAX}, got {max_params[1]}"

        print(f"B3: Worst-case at (η={max_params[0]:.1f}, S_mat={max_params[1]:.1f}) ✓")

    def test_B4_envelope_bounds_all_structures(self):
        """Verify envelope correctly identifies PASS/FAIL for all structures."""
        # From doc: C15, WP PASS; Kelvin marginal; FCC FAIL (at worst-case)

        for name, dv_v in STRUCTURES.items():
            # Compute worst-case (η=0, S_mat=2.5)
            factor = compute_factor(0.0, S_MAT_MAX)
            delta = compute_delta(dv_v, factor)
            margin = compute_margin(delta)

            if name in ['C15', 'WP']:
                assert margin > 1.0, f"{name} should PASS (margin > 1), got {margin:.1f}×"
            elif name == 'Kelvin':
                # Marginal: doc says 1.6×, allow range [1.0, 3.0]
                assert 1.0 < margin < 3.0, f"Kelvin should be marginal (1-3×), got {margin:.1f}×"
            elif name == 'FCC':
                assert margin < 1.0, f"FCC should FAIL (margin < 1), got {margin:.1f}×"

        print("B4: Envelope PASS/FAIL status verified for all structures")

    def test_B5_factor_interpolation(self):
        """Factor interpolates linearly between photon-only and matter-dominated."""
        s_mat = 1.5

        # At η=1: factor = 1 (photon-only)
        # At η=0: factor = S_mat (matter-dominated)
        # At η=0.5: factor = 0.5 + 0.5*S_mat

        f_1 = compute_factor(1.0, s_mat)
        f_0 = compute_factor(0.0, s_mat)
        f_mid = compute_factor(0.5, s_mat)

        assert f_1 == pytest.approx(1.0, rel=1e-10)
        assert f_0 == pytest.approx(s_mat, rel=1e-10)
        assert f_mid == pytest.approx((f_1 + f_0) / 2, rel=1e-10), \
            "Factor should interpolate linearly in η"

        print(f"B5: Linear interpolation verified (f(0)={f_0}, f(0.5)={f_mid}, f(1)={f_1})")


# =============================================================================
# Test C: Cross-checks with documented values
# =============================================================================

class TestDocumentedValues:
    """Cross-check specific values from Section 5.2 table."""

    def test_C1_c15_sapphire_matter(self):
        """C15 matter-dominated sapphire: margin = 134×."""
        dv_v = STRUCTURES['C15']
        factor = compute_factor(0.0, 0.2)  # η=0, S_mat=0.2
        delta = compute_delta(dv_v, factor)
        margin = compute_margin(delta)

        assert margin == pytest.approx(134, rel=0.05), \
            f"C15 sapphire matter-dom margin = {margin:.0f}×, expected ~134×"
        print(f"C1: C15 sapphire (η=0, S_mat=0.2) margin = {margin:.0f}× ✓")

    def test_C2_c15_equal_ionic(self):
        """C15 equal-weight ionic: margin = 15×."""
        dv_v = STRUCTURES['C15']
        factor = compute_factor(0.5, 2.5)  # η=0.5, S_mat=2.5
        delta = compute_delta(dv_v, factor)
        margin = compute_margin(delta)

        assert margin == pytest.approx(15, rel=0.05), \
            f"C15 equal ionic margin = {margin:.0f}×, expected ~15×"
        print(f"C2: C15 ionic (η=0.5, S_mat=2.5) margin = {margin:.0f}× ✓")

    def test_C3_wp_worst_case(self):
        """WP worst-case: margin = 4×."""
        dv_v = STRUCTURES['WP']
        factor = compute_factor(0.0, 2.5)
        delta = compute_delta(dv_v, factor)
        margin = compute_margin(delta)

        assert margin == pytest.approx(4, rel=0.05), \
            f"WP worst-case margin = {margin:.1f}×, expected ~4×"
        print(f"C3: WP worst-case margin = {margin:.1f}× ✓")

    def test_C4_fcc_fail(self):
        """FCC worst-case should FAIL (margin < 1)."""
        dv_v = STRUCTURES['FCC']
        factor = compute_factor(0.0, 2.5)
        delta = compute_delta(dv_v, factor)
        margin = compute_margin(delta)

        assert margin < 1.0, f"FCC should FAIL, got margin = {margin:.2f}×"
        assert margin == pytest.approx(0.6, rel=0.1), \
            f"FCC worst-case margin = {margin:.2f}×, expected ~0.6×"
        print(f"C4: FCC worst-case margin = {margin:.2f}× (FAIL) ✓")

    def test_C5_c15_matter_quartz(self):
        """C15 matter-dominated quartz: margin = 53×."""
        dv_v = STRUCTURES['C15']
        factor = compute_factor(0.0, 0.5)  # η=0, S_mat=0.5
        delta = compute_delta(dv_v, factor)
        margin = compute_margin(delta)

        assert margin == pytest.approx(53, rel=0.05), \
            f"C15 quartz matter-dom margin = {margin:.0f}×, expected ~53×"
        print(f"C5: C15 quartz (η=0, S_mat=0.5) margin = {margin:.0f}× ✓")

    def test_C6_c15_equal_sapphire(self):
        """C15 equal-weight sapphire: margin = 45×."""
        dv_v = STRUCTURES['C15']
        factor = compute_factor(0.5, 0.2)  # η=0.5, S_mat=0.2
        delta = compute_delta(dv_v, factor)
        margin = compute_margin(delta)

        assert margin == pytest.approx(45, rel=0.05), \
            f"C15 equal sapphire margin = {margin:.0f}×, expected ~45×"
        print(f"C6: C15 sapphire (η=0.5, S_mat=0.2) margin = {margin:.0f}× ✓")

    def test_C7_c15_equal_quartz(self):
        """C15 equal-weight quartz: margin = 36×."""
        dv_v = STRUCTURES['C15']
        factor = compute_factor(0.5, 0.5)  # η=0.5, S_mat=0.5
        delta = compute_delta(dv_v, factor)
        margin = compute_margin(delta)

        assert margin == pytest.approx(36, rel=0.05), \
            f"C15 equal quartz margin = {margin:.0f}×, expected ~36×"
        print(f"C7: C15 quartz (η=0.5, S_mat=0.5) margin = {margin:.0f}× ✓")

    def test_C8_section53_equal_weight(self):
        """Verify Section 5.3 equal-weight column values."""
        # From doc: C15=15×, WP=5.7×, Kelvin=2.2×, FCC=0.86×
        expected = {
            'C15': 15,
            'WP': 5.7,
            'Kelvin': 2.2,
            'FCC': 0.86,
        }

        for name, exp_margin in expected.items():
            dv_v = STRUCTURES[name]
            factor = compute_factor(0.5, 2.5)  # η=0.5, S_mat=2.5 (equal-weight ionic)
            delta = compute_delta(dv_v, factor)
            margin = compute_margin(delta)

            assert margin == pytest.approx(exp_margin, rel=0.1), \
                f"{name} equal-weight ionic margin = {margin:.1f}×, expected ~{exp_margin}×"

        print("C8: Section 5.3 equal-weight column verified ✓")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
