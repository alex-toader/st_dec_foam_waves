# ST_8 Code Review - Issues & Notes

**Date:** Jan 2026
**Purpose:** Track hardcodings, hidden assumptions, circularities

---

## physics/constants.py - OK

No issues. Clean threshold definitions.

---

## physics/bloch.py - MINOR

| Line | Issue | Severity | Note |
|------|-------|----------|------|
| 284 | `K + 4*G/3` hardcoded | LOW | Standard isotropic elasticity formula, OK if documented |
| 511 | `D /= self.mass` | LOW | Assumes uniform mass per vertex - undeclared assumption |
| 232-253 | Hodge stars `a³, a², a²` | LOW | Assumes cubic lattice - documented in function name |

**Hidden assumptions (not bugs):**
- Uniform mass per vertex
- Cubic periodic boundary (L scalar, not tensor)
- Edge springs only (no angular/bending terms)

---

## physics/bath.py - OK

No issues found. Well documented theory. Uses constants from constants.py.

---

## scripts/05_dispersion_grb.py - MINOR

Well-documented file with 18 embedded tests (T0-T14). No critical issues.

| Line | Issue | Severity | Note |
|------|-------|----------|------|
| 14, 223 | λ=2.0 bath coupling | LOW | "Chosen to match ST_7" - physical interpretation vague |
| 147 | L_CELL = 4.0 hardcoded | LOW | Key dimensional bridge (unit cell → ℓ_Planck), well-tested by T7 |
| 469+ | k_L=3.0, k_T=1.0 repeated | LOW | Hardcoded in many places - should centralize in constants.py |
| 258 | BZ safety threshold 25% | INFO | Arbitrary but reasonable |

**Hidden assumptions (documented):**
- L-mode excluded from GRB comparison (bath couples to compression, makes L optical-like)
- ε defined relative to L_CELL, assumes all structures have same microstructure scale
- λ=2.0 affects L-mode stiffening but T-modes are λ-insensitive (verified by T6)

**Strengths:**
- Clear INPUTS/OUTPUTS sections
- Extensive test coverage including scale invariance (T7), cubic symmetry (T9)
- Sources referenced for GRB bounds (Fermi, Abdo+ 2009, Vasileiou+ 2013)

---

## scripts/06_birefringence_grb.py - OK

Excellent documentation. Explicitly states what it proves vs assumes (lines 33-41).

**Strengths:**
- FRW cosmology derivation with (1+z) cancellation proof (T-P0)
- Literature comparison (B7: matches Götz+ 2014)
- Numerical convergence tests (C2)
- Clear WHAT IT DOES NOT PROVE section

| Line | Issue | Severity | Note |
|------|-------|----------|------|
| 1515 | k_L=3.0, k_T=1.0 hardcoded | LOW | In test_tfoam - same as 05_dispersion |
| 67-69 | Cosmology params | INFO | Planck 2018 values, documented |

**Documented assumptions (not bugs):**
- Line 235: Initial full polarization (Q₀, U₀) = (1, 0)
- Line 39-40: NO direct foam→EM birefringence bridge

---

## scripts/07_correlation_models.py - OK

Clean Monte Carlo implementation. Good theory verification.

**Strengths:**
- α ≈ γ/2 theory verified (T7)
- PSD slope verification (T8)
- Critical α calculation correct (T9)
- Clear physics interpretation section

| Line | Issue | Severity | Note |
|------|-------|----------|------|
| 1 | Docstring says "06_correlation_models.py" | INFO | Minor - filename mismatch |
| 51 | DV_V_WP = 0.025 hardcoded | LOW | Should match computed value from bloch analysis |

**Documented assumptions:**
- Uncorrelated grains (Markov/white noise) for 1/√M scaling
- α = 0.5 is default assumption, not derived

---

## CROSS-FILE ISSUE: k_L=3.0, k_T=1.0

Hardcoded in multiple places:
- physics/bloch.py
- scripts/05_dispersion_grb.py (lines 469, 474, 478, 577, 634, etc.)
- scripts/06_birefringence_grb.py (line 1515)

**Recommendation:** Move to physics/constants.py as DEFAULT_K_L, DEFAULT_K_T

---

## tests/physics/*.py - OK (sampled)

Reviewed files:
- `01_test_bath_internals.py` - Comprehensive, 30+ tests for bath mechanism
- `10_test_r2_bracketing.py` - Clean arithmetic verification (19 tests)

**Common pattern:**
- k_L=3.0, k_T=1.0 hardcoded (same cross-file issue)
- Otherwise well-structured with clear test purposes

**Strengths:**
- Good coverage of edge cases
- Multiple verification levels (A-E in bath tests)
- Cross-checks against documented values

---

## SUMMARY

| File                          | Status | Critical Issues                          |
|-------------------------------|--------|------------------------------------------|
| physics/constants.py          | OK     | None                                     |
| physics/bloch.py              | MINOR  | Uniform mass assumption (undeclared)     |
| physics/bath.py               | OK     | None                                     |
| scripts/05_dispersion_grb.py  | MINOR  | λ=2.0 weakly motivated, k_L/k_T repeated |
| scripts/06_birefringence_grb.py | OK   | Explicit about assumptions               |
| scripts/07_correlation_models.py | OK  | Filename mismatch in docstring           |
| tests/physics/*.py            | OK     | k_L/k_T hardcoded                        |

**CROSS-FILE ACTION ITEM:**
Move k_L=3.0, k_T=1.0 to physics/constants.py as DEFAULT_K_L, DEFAULT_K_T

**Overall verdict:** Codebase is clean. No circularities found. Main issues are:
1. Repeated hardcodings (k_L, k_T) - refactor opportunity
2. Undeclared uniform mass assumption in bloch.py - documentation issue

---
