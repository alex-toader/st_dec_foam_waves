# DEC Wave Propagation on Periodic Foam Complexes

**Author:** Alex Toader
**Date:** January 2026
**Purpose:** Technical summary for review

---

## What this code does

Computes elastic and electromagnetic wave propagation on periodic foam cell complexes using Discrete Exterior Calculus (DEC). Four structures are analyzed: C15 Laves phase, Weaire-Phelan (A15), Kelvin (BCC foam), and FCC (control lattice).

**Main question:** Can a foam-like elastic medium support shear-like radiative behavior, and can the elastic-to-gauge mapping be computed (not assumed) on the same discrete complex?

---

## DEC Operators

### Hodge Stars (Voronoi dual)

Standard DEC formula on the foam complex:

```
*1[e] = dual_face_area(e) / edge_length(e)
*2[f] = dual_edge_length(f) / face_area(f)
```

Where dual_face(e) is the polygon formed by cell centers around edge e, and dual_edge(f) is the segment connecting cell centers adjacent to face f. Verified properties:

- Voronoi property: vertices equidistant from adjacent cell sites (< 1e-8)
- Plateau structure: 4 edges/vertex, 3 faces/edge (C15, Kelvin, WP)
- Dual orthogonality: |n_hat . d_hat| > 0.99
- Exactness: d1 d0 = 0

**Code:** `src/physics/hodge.py` (1043 lines), tests in `src/tests/physics/11_test_hodge_voronoi.py` (35 tests)

### Bloch Operators

Elastic wave propagation via displacement Bloch dynamical matrix:

```
D(k) u = omega^2 u
```

Built directly from the spring network (not via d0/d1), incorporating longitudinal (k_L) and transverse (k_T) spring constants. Hermitian by construction, PSD for all tested k_L/k_T ratios (0.01 to 100).

**Code:** `src/physics/bloch.py` (780 lines), tests in `src/tests/physics/02_test_bloch_internals.py` (36 tests)

### Gauge (Maxwell-like) Operator

Electromagnetic wave equation on the same foam complex:

```
d1(k)^dagger *2 d1(k) a = omega^2 *1 a
```

Generalized eigenvalue problem. Physical modes: omega^2 > 0 (gauge modes with omega^2 ~ 0 are excluded).

**Code:** `src/physics/gauge_bloch.py` (273 lines), tests in `src/tests/physics/12_test_gauge_elastic_bridge.py` (25 tests)

---

## Foam Structures

| Structure     | Cells | V   | E   | F   | Type          |
|---------------|-------|-----|-----|-----|---------------|
| C15 Laves     | 24    | 136 | 272 | 160 | Plateau foam  |
| Weaire-Phelan | 8     | 46  | 92  | 54  | Plateau foam  |
| Kelvin (BCC)  | 16    | 96  | 192 | 112 | Plateau foam  |
| FCC           | 32    | 96  | 256 | 192 | Control       |

All satisfy Euler relation on 3-torus: V - E + F - C = 0.

**Builders:** `src/core_math_v2/builders/` (c15_periodic.py, kelvin.py, weaire_phelan_periodic.py)

---

## Key Results

### 1. Elastic Anisotropy

delta_v/v = (v_max - v_min) / v_mean over high-symmetry directions:

| Structure | delta_v/v | I4    | A_Z  |
|-----------|-----------|-------|------|
| C15       | 0.93%     | 0.585 | 1.02 |
| WP        | 2.53%     | 0.660 | 0.95 |
| Kelvin    | 6.34%     | 0.500 | 1.14 |
| FCC       | 16.55%    | 0.333 | 1.40 |

Ranking C15 < WP < Kelvin < FCC preserved across all tested k_L/k_T ratios.

### 2. I4 to A_Z Cross-Validation

Two independent computational paths yield Zener anisotropy:

```
Path A (dynamics): Foam -> Bloch -> v(k_hat) -> Christoffel fit -> A_Z
Path B (geometry): Foam -> edge directions -> I4 -> formula -> A_Z
```

Discovered linear relation: `A_Z = 1 - 1.46 * (I4 - 0.6)`, Pearson r = -0.996.

Leave-one-out validation: fit on 3 structures, predict 4th, all < 10% error.

**Code:** `src/tests/physics/14_test_i4_az_cross_check.py` (44 tests)

### 3. Gauge-Elastic Bridge

Elastic and gauge anisotropy computed on the same foam complex:

| Geometry | r (correlation) | alpha (amplitude ratio) | delta_c/c | delta_v/v |
|----------|-----------------|-------------------------|-----------|-----------|
| C15      | 1.0000          | 0.0153                  | 0.0093%   | 0.6048%   |
| Kelvin   | 0.9990          | 0.0142                  | 0.0564%   | 3.9795%   |
| WP       | 0.9998          | 0.0642                  | 0.1036%   | 1.6148%   |

Identical angular patterns (r ~ 1.0), gauge sector ~60x more isotropic (alpha ~ 0.01-0.06). The gauge operator d1^dagger *2 d1 averages over more geometric elements than the elastic Christoffel operator.

---

## Test Suite

341 tests across 16 files. All pass on macOS / Python 3.9.6 / SciPy.

**To run all tests** (one file at a time):
```bash
cd src
OPENBLAS_NUM_THREADS=1 python3 -m pytest tests/physics/02_test_bloch_internals.py -v
OPENBLAS_NUM_THREADS=1 python3 -m pytest tests/physics/11_test_hodge_voronoi.py -v
OPENBLAS_NUM_THREADS=1 python3 -m pytest tests/physics/12_test_gauge_elastic_bridge.py -v
OPENBLAS_NUM_THREADS=1 python3 -m pytest tests/physics/14_test_i4_az_cross_check.py -v
```

**Note:** OPENBLAS_NUM_THREADS=1 prevents threading deadlock on macOS ARM64 (see CLAUDE.md Rule 77b).

### Test Categories

| Category              | Tests | Files                          |
|-----------------------|-------|--------------------------------|
| Bloch internals       | 36    | 02_test_bloch_internals.py     |
| Hodge / Voronoi       | 35    | 11_test_hodge_voronoi.py       |
| Gauge-elastic bridge  | 25    | 12_test_gauge_elastic_bridge.py|
| I4 -> A_Z cross-check | 44    | 14_test_i4_az_cross_check.py   |
| Christoffel / ranking | 6     | 07_test_ranking_robustness.py  |
| Discrete stability    | 44    | 16_test_discrete_stability.py  |
| Sampling convergence  | 15    | 15_test_sampling_convergence.py|
| Other (cavity, bath)  | 136   | remaining test files           |

---

## File Map

### Core physics modules (`src/physics/`)

| File             | Lines | What it does                              |
|------------------|-------|-------------------------------------------|
| hodge.py         | 1043  | Voronoi dual Hodge stars (*1, *2)         |
| bloch.py         | 780   | DisplacementBloch dynamical matrix        |
| gauge_bloch.py   | 273   | Discrete Maxwell operator                 |
| christoffel.py   | 912   | Christoffel fit, delta_v/v computation    |
| bath.py          | 529   | Schur complement (longitudinal coupling)  |

### Geometry builders (`src/core_math_v2/builders/`)

| File                       | Lines | Structure           |
|----------------------------|-------|---------------------|
| c15_periodic.py            | 480   | C15 Laves phase     |
| kelvin.py                  | 314   | Kelvin / BCC foam   |
| weaire_phelan_periodic.py  | 437   | Weaire-Phelan / A15 |
| solids_periodic.py         | 542   | FCC, BCC supercells |

### Incidence operators (`src/core_math_v2/operators/`)

| File          | Lines | What it does                     |
|---------------|-------|----------------------------------|
| incidence.py  | 609   | d0, d1, face-cell tracking       |


