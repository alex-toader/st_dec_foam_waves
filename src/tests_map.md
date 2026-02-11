<!-- SYNC: This file is synced from release/1_foam/1_tests_map.md. Edit there first, then copy here. -->

# ST_ Comprehensive Test Map

**Date:** Feb 2026
**Purpose:** Map ALL tests across model domains

---

## Overview

```
MEDIUM PROPERTIES: Is universe foam/solid/glass?
                   └── Tests if elastic medium model is viable
```

NOTE: Particle sector (toroidal defects, α=137 derivation) is NOT part of ST_8 scope.
ST_8 focuses on medium properties only.

**What tests prove vs assume:**
- Tests verify that **code implements model correctly** (software validation)
- Tests verify **mathematical properties** (PSD, exactness, hermiticity)
- Tests do NOT prove **physics assumptions** (uncorrelated grains, Bu=0 constraint, k_L/k_T values)
- Key assumptions are documented in `0_a_inputs.md` (Assumptions section)

---

## MATERIAL COVERAGE (Jan 2026)

**All 4 structures tested across all relevant test files:**

| File | C15 | WP | Kelvin | FCC | Note |
|------|:---:|:--:|:------:|:---:|------|
| 01_bath | ✓ | ✓ | ✓ | ✓ | Longitudinal selectivity |
| 03_cavity | ✓ | ✓ | ✓ | ✓ | DELTA_* values |
| 05a_eta | ✓ | ✓ | ✓ | ✓ | Mode overlap |
| 05b_spectral | ✓ | ✓ | ✓ | ✓ | G structure |
| 07_ranking | ✓ | ✓ | ✓ | ✓ | C15 < WP < K < FCC |
| 08_cavity_pert | ✓ | ✓ | ✓ | ✓ | End-to-end |
| 10_r2_bracket | ✓ | ✓ | ✓ | ✓ | Values only |
| 11_hodge | ✓ | ✓ | ✓ | ❌ | FCC ≠ Plateau |
| 12_bridge | ✓ | ✓ | ✓ | ❌ | FCC ≠ Plateau |
| 13_l_mode | ✓ | ✓ | ✓ | ✓ | All mechanisms |
| 14_i4_az | ✓ | ✓ | ✓ | ✓ | Cross-check |
| 15_sampling | ✓ | ✓ | ✓ | ✓ | Ranking |
| 16_stability | ✓ | ✓ | ✓ | ✓ | PSD all ratios |

**Note:** FCC excluded from 11, 12 because FCC is NOT a Plateau foam (different topology).

---

## TEST FILE INVENTORY

| #   | File                                 | Tests   | Category          |
|-----|--------------------------------------|---------|-------------------|
| 01  | 01_test_bath_internals.py            | 16      | Bath/L-mode       |
| 02  | 02_test_bloch_internals.py           | 36      | Infrastructure    |
| 03  | 03_test_cavity_lorentz.py            | 7       | Lorentz/Cavity    |
| 04  | 04_test_no_drag.py                   | 6       | No-drag           |
| 05a | 05_a_test_eta_overlap.py             | 15      | 2 Radiative Modes |
| 05b | 05_b_test_spectral_verification.py   | 12      | 2 Radiative Modes |
| 06  | 06_test_rotation_transformation.py   | 9       | Lorentz/Cavity    |
| 07  | 07_test_ranking_robustness.py        | 6       | Structure         |
| 08  | 08_test_cavity_perturbation.py       | 9       | Lorentz/Cavity    |
| 09  | 09_test_3d_field_correlation.py      | 18      | Correlation       |
| 10  | 10_test_r2_bracketing.py             | 19      | R2 Margins        |
| 11  | 11_test_hodge_voronoi.py             | 35      | Infrastructure    |
| 12  | 12_test_gauge_elastic_bridge.py      | 25      | Bridge            |
| 13  | 13_test_l_mode_mechanisms.py         | 25      | L-mode            |
| 14  | 14_test_i4_az_cross_check.py         | 44      | Cross-validation  |
| 15  | 15_test_sampling_convergence.py      | 15      | Methodology       |
| 16  | 16_test_discrete_stability.py        | 44      | Stability         |
| 17  | 17_test_cell_topology.py             | —       | Infrastructure    |
| 18  | 18_test_belt_spectrum.py             | —       | Infrastructure    |
| 19  | 19_test_holonomy.py                  | —       | Infrastructure    |
| 20  | 20_test_texture_contamination.py     | 5       | Robustness        |
| 21  | 21_test_domain_blocks.py             | 4       | Robustness        |
| 22  | 22_test_geometric_jitter.py          | 6       | Robustness        |
| 23  | 23_test_analytic_benchmark.py        | 7       | Robustness        |
| 24  | 24_test_linear_tripwire.py           | 7       | Robustness        |
|     | **TOTAL**                            | **370** |                   |

---

## MEDIUM CONSTRAINTS

### A. "2 Radiative Modes" - DONE (27 tests)

**Files:** `05_a_test_eta_overlap.py` (15), `05_b_test_spectral_verification.py` (12)

| Test | Description | Status | Result |
|------|-------------|--------|--------|
| Embedding QC | η_j overlap for modes | **DONE** | 2 high-η with bath |
| N-scaling | N=2,3 independence | **DONE** | Consistent |
| T1 Spectral wt | max_a for L plane-wave | **DONE** | 0.88 (stiffened) |
| T2 Sum rule | Ση_j conservation | **DONE** | 3.0000 exact |
| S action | ‖S u_L‖/‖S u_T‖ ratio | **DONE** | 935× (S kills T) |
| Dispersion | m²_T vs m²_L | **DONE** | T propagates, L non-radiative |
| Polycrystal | All orientations | **DONE** | 100% show 2+1 |

**Note:** "On-shell" means stability holds for physical branches (ω ≈ v·k) with χ < 1/v².

### B. No Drag - REQUIRES PARTICLE SECTOR (6 tests)

**File:** `04_test_no_drag.py`

**Status:** Tests exist but mechanism is outside ST_8 scope. Drag depends on particle model (topological defect, soliton, etc.). Without specifying particle structure, drag is undefined.

### C. Lorentz/Cavity - PAPER-GRADE (25 tests)

**Files:** `03_test_cavity_lorentz.py` (7), `06_test_rotation_transformation.py` (9), `08_test_cavity_perturbation.py` (9)

| Test | Description | Status | Result |
|------|-------------|--------|--------|
| MC1 1/√M scaling | RMS vs M | **DONE** | Slope = -0.499 |
| MC2 2ω modulation | √(A²+B²) format | **DONE** | Matches analytical |
| MC3 Christoffel | Real δ(n̂) kernel | **DONE** | 1/√M confirmed |
| MC4 Markov | Correlated grains | **DONE** | Slope = +0.46 |
| MC5 Two-way | Round-trip geometry | **DONE** | Ratio = 2.0 |
| MC6 Geometry audit | 0.2 factor origin | **DONE** | 3D→2D projection |
| Rotation R(2φ) | (A,B) transform as spin-2 | **DONE** | Phase correct |
| Amplitude invariance | √(A²+B²) under rotation | **DONE** | Preserved |
| Foam kernel | End-to-end with Bloch | **DONE** | Consistent |

### D. Birefringence - PASS

**File:** `src/scripts/06_birefringence_grb.py`

| Test | Description | Status | Result |
|------|-------------|--------|--------|
| n=1 bound | b_max for DoP>0.3 | **DONE** | n=1 EXCLUDED |
| n=2 margin | b_max for n=2 | **DONE** | Margin 10⁸ vs O(1) |
| T-FOAM | n=2 derived from foam Bloch (all 4 structures) | **DONE** | ratio ≈ 1.0 |
| Foam→EFT bridge | b = Δã/(4π²), margin ~10¹¹ | **DONE** | Real margin 10¹¹ |
| Band-averaged DoP | FRW integral + cosmology | **DONE** | 28 tests pass |

**File:** `release/6_birefringence_grb.md`

### E. Dispersion (ToF) - PASS

**File:** `src/scripts/05_dispersion_grb.py`

| Test | Description | Status | Result |
|------|-------------|--------|--------|
| GRB bound | Δv/c < 6×10⁻²¹ | **DONE** | Margin ~10¹⁹ (C15) |
| T14 | ℓ_cell < 10⁹ × ℓ_P passes | **DONE** | All structures |
| T-D3 | Δv/c ∝ E² and ∝ ℓ_cell² scaling | **DONE** | 19 tests pass |

**File:** `release/5_dispersion_grb.md`

### F. Structure Discrimination (6 tests)

**File:** `07_test_ranking_robustness.py`

| Test | Description | Status | Result |
|------|-------------|--------|--------|
| C15 vs WP | Isotropy ranking | **DONE** | C15 best (0.93%) |
| Ranking robustness | k_L, k_T invariant | **DONE** | 8/8 consistent |
| Kelvin vs WP | δv/v comparison | **DONE** | Kelvin > WP |

**File:** `release/2_c15_isotropy.md`

### G. Correlation Wash-out (18 tests)

**File:** `09_test_3d_field_correlation.py`

| Test | Description | Status | Result |
|------|-------------|--------|--------|
| 1/√M scaling | 3D synthetic fields | **DONE** | Slope = -0.50 |
| Orientation correlation | Random vs correlated | **DONE** | CLT validated |
| Large M extrapolation | M ~ 10³⁴ regime | **DONE** | Consistent |

**File:** `release/7_correlation_wash_out.md`

### H. R2 Bracketing (19 tests)

**File:** `10_test_r2_bracketing.py`

| Test | Description | Status | Result |
|------|-------------|--------|--------|
| Arithmetic | factor = η + (1-η) * S_mat | **DONE** | Correct |
| Bracketing envelope | Worst-case over S_mat | **DONE** | S_mat=2.5 worst |
| Monotonicity | margin vs S_mat | **DONE** | Decreasing |
| All structures | C15, WP, Kelvin, FCC | **DONE** | All pass |

**File:** `release/4_r2_bracketing.md`

---

## INFRASTRUCTURE TESTS

### I. Bloch Internals (36 tests)

**File:** `02_test_bloch_internals.py`

| Test | Description | Status |
|------|-------------|--------|
| Edge crossings | Periodic boundary handling | **DONE** |
| d0, d1 Bloch | Discrete operators with phase | **DONE** |
| Acoustic branches | 3 branches at small k | **DONE** |
| Eigenvalue sanity | ω² ≥ 0 | **DONE** |

### J. Hodge/Voronoi (35 tests)

**File:** `11_test_hodge_voronoi.py`

| Test | Description | Status |
|------|-------------|--------|
| *₁ diagonal | Voronoi dual metric | **DONE** |
| d₁d₀ = 0 | Exactness | **DONE** |
| Curl-curl symmetric | d₁†*₂d₁ | **DONE** |
| Multi-geometry | C15, Kelvin, WP | **DONE** |

### K. Bath Internals (16 tests)

**File:** `01_test_bath_internals.py`

| Test | Description | Status |
|------|-------------|--------|
| Continuum P_L | Schur = k⊗k/k² | **DONE** |
| Discrete Schur | B†A⁻¹B on lattice | **DONE** |
| Mode classification | P_L identifies L mode | **DONE** |
| Dynamic bath | ω-dependent kernel | **DONE** |

### L. Elastic→EM Bridge (25 tests)

**File:** `12_test_gauge_elastic_bridge.py`

| Test | Description | Status | Result |
|------|-------------|--------|--------|
| Pearson r | Angular pattern correlation | **DONE** | r > 0.99 (all foams) |
| Spearman ρ | Rank correlation robustness | **DONE** | ρ > 0.99 (all foams) |
| α_aniso | δc/c / δv/v ratio | **DONE** | 0.01–0.06 |
| RMS residual | Normalized pattern match | **DONE** | < 0.05 |
| Multi-geometry | C15, Kelvin, WP consistency | **DONE** | All pass |

**Results (30 directions, L_cell=4.0):**

| Geometry | r | α_aniso | δc/c | δv/v |
|----------|---|---------|------|------|
| C15 | 1.0000 | 0.0153 | 0.0093% | 0.6048% |
| Kelvin | 0.9990 | 0.0142 | 0.0564% | 3.9795% |
| WP | 0.9998 | 0.0642 | 0.1036% | 1.6148% |

### M. L-Mode Mechanisms (25 tests)

**File:** `13_test_l_mode_mechanisms.py`

Two mechanisms for longitudinal mode suppression, tested across 4 structures.

**Fir C (Constraint Bu = 0):**

| Test | Description | Status | Result |
|------|-------------|--------|--------|
| T-L01 | rank(B) = V, dim(ker(B)) = 2V | **DONE** | C15, Kelvin, WP |
| T-L02 | ||Bu||/||u|| < 1e-12 | **DONE** | All 4 structures |
| T-L03 | 2 acoustic branches (not 3) | **DONE** | C15 |
| T-L04 | k̂·u ≈ 0 (transverse proxy) | **DONE** | C15 |

**Fir B (Stiffening D + g²S):**

| Test | Description | Status | Result |
|------|-------------|--------|--------|
| T-L05 | S = B†A⁺B is PSD | **DONE** | All 4 structures |
| T-L06 | L stiffened, T preserved | **DONE** | C15 |
| T-L07 | v_L/v_T > 5 across k values | **DONE** | C15 |
| T-L08 | Near-field > far-field decay | **DONE** | C15 |
| T-L09 | ker(B) = ker(S) | **DONE** | All 4 structures |
| T-L10 | Penalty → Constraint (μ→∞) | **DONE** | C15 |

**Multi-structure validation:**

| Test | Description | Status | Structures |
|------|-------------|--------|------------|
| T-L11 | Kernel dimension | **DONE** | C15, Kelvin, WP |
| T-L12 | Constraint satisfied | **DONE** | C15, Kelvin, WP, FCC |
| T-L13 | S is PSD | **DONE** | C15, Kelvin, WP, FCC |
| T-L14 | Subspace equivalence | **DONE** | C15, Kelvin, WP, FCC |

### N. I4→A_Z Cross-Check (44 tests)

**File:** `14_test_i4_az_cross_check.py`

Independent validation: geometry (I4) predicts dynamics (A_Z).

**Two paths to same answer:**
```
Path A (dynamics): Foam → Bloch → v(k̂) → fit C11,C12,C44 → A_Z
Path B (geometry): Foam → edge directions → I4 → formula → A_Z
```

**Formula:** `A_Z = 1 - 1.46 × (I4 - 0.6)`

| Structure | I4    | A_Z(meas) | A_Z(pred) | Error  |
|-----------|-------|-----------|-----------|--------|
| C15       | 0.585 | 1.02      | 1.02      | -0.2%  |
| WP        | 0.660 | 0.95      | 0.91      | +3.9%  |
| Kelvin    | 0.500 | 1.14      | 1.15      | -0.5%  |
| FCC       | 0.333 | 1.40      | 1.39      | +0.8%  |

**Pearson r = -0.996, RMS = 2%**

| Test Class               | Tests | Description                                      |
|--------------------------|-------|--------------------------------------------------|
| TestFixtureHealth        | 4     | All structures load                              |
| TestI4Computation        | 1     | I4 invariant under 48 cubic symmetry ops         |
| TestDeterminism          | 2     | analyze_structure and I4 deterministic           |
| TestCrossCheck           | 6     | A_Z prediction <5%, correlation |r|>0.99         |
| TestLeaveOneOut          | 4     | Fit on 3, predict 4th (<10% error)               |
| TestPhysicalInterpretation | 2   | C15 most isotropic, Spearman ranking             |
| TestTripwires            | 6     | No NaN, I4 in [0.3, 1], A_Z > 0                  |
| TestFitAlphaGuardrails   | 1     | Degenerate data rejected                         |
| TestI4Weighted           | 1     | Unweighted vs length-weighted                    |
| TestQ4CubicClosure       | 2     | Q4 cubic symmetry + I4 = trace(Q4)               |
| TestBootstrapStability   | 1     | C15 remains #1 under bootstrap (>90%)            |
| TestNScalingIndependence | 1     | I4 intensive (N=1 vs N=2 identical)              |
| TestI4toDeltaV           | 3     | I4 → δv/v: Spearman ρ>0.95, monotonic            |
| TestDispersionRanking    | 3     | I4 → δv/v → ã chain (ã computed in runtime)     |
| TestRotationInvariance   | 2     | A_Z, δv/v invariant under permutation + flip     |
| TestDeltaVvsAZLinear     | 2     | δv/v ∝ |A_Z-1| correlation                       |
| TestBootstrapExtended    | 2     | FCC most aniso, C15 beats WP under bootstrap     |
| TestDocumentation        | 1     | Print results table                              |

**Key findings:**
- **Unweighted I4 better** than length-weighted → edge orientation dominates, not metric
- **Q4 tensor is cubic** → I4 captures relevant anisotropy information
- **Ranking stable** under bootstrap (>90%) → not sampling artifact
- **Q2 = I/3 exact** for all structures → 2nd moment doesn't differentiate, need 4th moment (I4)
- **Leave-one-out works** → formula generalizes, not just curve fitting
- **I4 → ã chain works** → geometry predicts dispersion through I4 → A_Z → δv/v → ã (r=0.99)

### O. Sampling Convergence (15 tests)

**File:** `15_test_sampling_convergence.py`

Documents how δv/v sampling method affects measured values.

| Test Class               | Tests | Description                                      |
|--------------------------|-------|--------------------------------------------------|
| TestDeltaVDefinition     | 3     | Definition: δv/v = (v_max - v_min) / v_mean      |
| TestHighSymExtrema       | 3     | Extrema at high-symmetry directions              |
| TestGoldenUnderestimates | 2     | Golden spiral n=30 underestimates by 15-18%      |
| TestScaleInvariance      | 1     | δv/v independent of foam scale (N=1 vs N=2)      |
| TestRankingPreserved     | 2     | Ranking C15 < WP < Kelvin preserved              |
| TestEpsSensitivity       | 2     | Pattern stable across eps ∈ {0.01, 0.02, 0.04}   |
| TestLocalPerturbation    | 1     | High-sym are local extrema (±2° check)           |
| TestTransverseMean       | 1     | Ranking holds with mean(T1,T2)                   |

**Key findings:**
- **Extrema at high-symmetry:** v_max at <100> class, v_min at <110> class (C15, Kelvin)
- **Golden underestimates:** n=30 misses extrema by 15-18% (never overestimates)
- **Ranking preserved:** C15 < WP < Kelvin regardless of sampling method
- **Pattern is geometric:** Cubic symmetry class determines extrema, not specific direction

**Methodology note:** δv/v values in this project use high-symmetry directions as ground truth (7 directions). Golden spiral sampling underestimates because it rarely hits exact high-symmetry directions.

### P. Discrete Stability (44 tests)

**File:** `16_test_discrete_stability.py`

Verifies that D(k) is PSD across sampled k-vectors for tested foam structures.

| Test Class                      | Tests | Description                                      |
|---------------------------------|-------|--------------------------------------------------|
| TestDiscreteStabilityAllRatios  | 28    | D(k) PSD for ratios 0.01 to 100 (4 structures)   |
| TestVelocitiesAllRatios         | 10    | v_T, v_L positive and real for all ratios        |
| TestMinEigenvalueScaling        | 1     | min(eig) ~ k² (acoustic scaling)                 |
| TestNoSingularities             | 2     | Smooth variation, no jumps                       |
| TestMultiStructureConsistency   | 2     | All structures stable at extremes                |
| TestDocumentation               | 1     | Print summary table                              |

**Key finding: Discrete stability ≠ Continuum stability**

| k_L/k_T | Continuum | Discrete | v_L/v_T |
|---------|-----------|----------|---------|
| 0.01    | "Unstable" (K<0) | **STABLE** | 0.2 (v_L < v_T) |
| 0.1     | "Unstable" | **STABLE** | 0.55 |
| 1.0     | Stable | **STABLE** | ~1.0 |
| 3.0     | Stable | **STABLE** | 1.26 |
| 100.0   | Stable | **STABLE** | ~10 |

**Sampling:** High-symmetry ([100], [110], [111]) + golden spiral, eps ∈ {0.005, 0.01, 0.02, 0.04, 0.08}

**Implication:** For these tetravalent foam structures, no lower bound on k_L/k_T from stability.

### Q. Robustness Tests — Reviewer Proposals (29 tests)

**Files:** `20_test_texture_contamination.py` (5), `21_test_domain_blocks.py` (4), `22_test_geometric_jitter.py` (6), `23_test_analytic_benchmark.py` (7), `24_test_linear_tripwire.py` (7)

Six test proposals to consolidate the cavity prediction. T1-T5 implemented, T6 (dual sensitivity, LOW priority) not started.

| #  | Test | Tests | Finding |
|----|------|-------|---------|
| T1 | Texture contamination | 5 | Bias problem (p·δ), two independent constraints: α > 0.47 AND p < p* ~ 10⁻¹⁶ |
| T2 | Domain blocks | 4 | Equivalent to ℓ_corr repackaging, M_eff = M/m, no new failure mode |
| T3 | Geometric jitter | 6 | δv/v stable to < 1% for ε/d_nn < 2%, C15 NOT fine-tuned |
| T4 | Analytic benchmark | 7 | SC Born-von Kármán exact match (err < 3×10⁻⁴), acoustic gap ω₃/ω₂ = 67 |
| T5 | Linear tripwire | 7 | a₁/v₀ < 10⁻⁶ for all structures, dispersion is ω = v₀|k| + O(|k|³) |

**T1 (Texture Contamination):** Partial alignment (fraction p aligned, rest random). Texture is a BIAS problem, not variance. Std(x̄) ∝ 1/√M always washes out (α ≈ 0.5 for all p). Bias |E[x̄]| = p·δ is independent of M. Two independent constraints needed: stochastic wash-out AND p < bound/δ. For C15: p* = 1.08×10⁻¹⁶ (trivially satisfied for amorphous foam).

**T2 (Domain Blocks):** Path split into correlated domains of m grains. α stays 0.5 (CLT on domains), RMS scales as √m. Equivalent to ℓ_corr → m×ℓ_corr (same table as `7_correlation_wash_out.md`). C15 tolerates m ≤ ~700.

**T3 (Geometric Jitter, v2):** Perturbs C15 Wyckoff positions. L/T separation via classify_modes (zero crossings even at ε=0.1). 210 directions (200 golden-spiral + 10 HS). Full 3-complex topology (V=136, E=272, F=160, C=24, χ=0, all deg-4) preserved. δv/v = 0.94% is a topological property of the Laves structure, not a fragile geometric coincidence.

**T4 (Analytic Benchmark):** SC lattice exact Born-von Kármán formulas for [100], [110], [111]. Match at 3×10⁻⁴. Degeneracies exact at 10⁻¹². Hermiticity ||D−D†|| = 0. Acoustic gap ω₃/ω₂ > 10. Validates DisplacementBloch implementation.

**T5 (Linear Tripwire):** Phase velocity v = v₀ + a₁|k| + a₂|k|², testing a₁ = 0. Uses classify_modes for T1/T2/L branch tracking. All structures (SC, FCC, WP, C15) give |a₁|/v₀ < 10⁻⁶. Fit window stability verified across 4 k-windows. Dispersion confirmed quadratic.

---

## FALSIFICATION TESTS

| Test           | Condition                  | Current      | Status   |
|----------------|----------------------------|--------------|----------|
| Lorentz margin | δv/v × M^(-0.5) > 10^-18   | 27× margin   | **SAFE** |
| v_GW ≠ c       | LIGO                       | < 3×10⁻¹⁵    | **SAFE** |
| Dispersion     | GRB Δv/c > 10^-21          | ~10^-38      | **SAFE** |
| Birefringence  | n=1 operator viable        | n=2 (foam)   | **SAFE** |

---

## SUMMARY STATISTICS

| Category                 | Tests   | Files  |
|--------------------------|---------|--------|
| Medium constraints (A-H) | 101     | 7      |
| Infrastructure (I-P)     | 240     | 8      |
| Robustness (Q)           | 29      | 5      |
| **TOTAL**                | **370** | **21** |

### By Domain

| Domain               | Tests |
|----------------------|-------|
| 2 Radiative Modes    | 27    |
| No-drag              | 6     |
| Lorentz/Cavity       | 25    |
| Structure            | 6     |
| Correlation          | 18    |
| R2 Bracketing        | 19    |
| Bloch                | 36    |
| Hodge                | 35    |
| Bath                 | 16    |
| Bridge               | 25    |
| L-mode               | 25    |
| I4→A_Z Cross-check   | 44    |
| Sampling Convergence | 15    |
| Discrete Stability   | 44    |
| Robustness           | 29    |

---

## OPEN QUESTIONS

1. **No-drag mechanism** - Requires particle sector (outside ST_8 scope)

---

## FILES REFERENCED

**Test files (src/tests/physics/):**
- `01_test_bath_internals.py` - Bath/Schur complement
- `02_test_bloch_internals.py` - Bloch operators
- `03_test_cavity_lorentz.py` - Cavity MC
- `04_test_no_drag.py` - No-drag (out of scope)
- `05_a_test_eta_overlap.py` - η embedding overlap
- `05_b_test_spectral_verification.py` - Spectral tests
- `06_test_rotation_transformation.py` - Spin-2 rotation
- `07_test_ranking_robustness.py` - Structure ranking
- `08_test_cavity_perturbation.py` - Foam kernel cavity
- `09_test_3d_field_correlation.py` - 3D correlation
- `10_test_r2_bracketing.py` - R2 margins
- `11_test_hodge_voronoi.py` - Hodge stars
- `12_test_gauge_elastic_bridge.py` - Gauge-elastic
- `13_test_l_mode_mechanisms.py` - L-mode
- `14_test_i4_az_cross_check.py` - I4→A_Z cross-validation
- `15_test_sampling_convergence.py` - Sampling methodology
- `16_test_discrete_stability.py` - Discrete stability for all k_L/k_T
- `17_test_cell_topology.py` - Cell topology
- `18_test_belt_spectrum.py` - Belt spectrum
- `19_test_holonomy.py` - Holonomy
- `20_test_texture_contamination.py` - Texture contamination (robustness T1)
- `21_test_domain_blocks.py` - Domain blocks (robustness T2)
- `22_test_geometric_jitter.py` - Geometric jitter (robustness T3)
- `23_test_analytic_benchmark.py` - SC analytic benchmark (robustness T4)
- `24_test_linear_tripwire.py` - Linear tripwire (robustness T5)

**Release docs:**
- `release/0_b_outputs.md` - All outputs
- `release/0_f_longitudinal_mode.md` - L-mode mechanisms
- `release/2_c15_isotropy.md` - C15 discovery
- `release/3_bridge_end_to_end.md` - Pipeline
- `release/4_r2_bracketing.md` - R2 margins
- `release/5_dispersion_grb.md` - Dispersion
- `release/6_birefringence_grb.md` - Birefringence
- `release/7_correlation_wash_out.md` - Correlation

**Source modules (src/physics/):**
- `hodge.py` - Voronoi dual Hodge stars
- `bath.py` - Bath coupling operators
- `gauge_bloch.py` - Gauge anisotropy
- `bloch.py` - Bloch operators

---
