"""
Gauge Bloch - Discrete Maxwell on Foam Complex
===============================================

Computes electromagnetic wave speeds on periodic foam using DEC.

Key result (Jan 2026):
    Gauge and elastic anisotropy patterns are identical (r ≈ 1.0).
    α_aniso = (δc/c)_gauge / (δv/v)_elastic ≈ 0.016

The elastic→EM bridge is now grounded in calculation, not assumption.

Operator:
    d₁(k)† *₂ d₁(k) a = ω² *₁ a

Where:
    d₀(k): gradient with Bloch phases
    d₁(k): curl with Bloch phases (exactness-preserving)
    *₁, *₂: Hodge stars from Voronoi dual

Physical modes: ω² > 0 eigenvalues (skip gauge modes with ω² ≈ 0).

Jan 2026
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, List, Dict

from .hodge import wrap_delta


def compute_edge_shifts(V: np.ndarray, E: list, L_vec: np.ndarray) -> np.ndarray:
    """Compute lattice shift vectors for each edge (for Bloch phases)."""
    n_E = len(E)
    shifts = np.zeros((n_E, 3), dtype=int)
    for idx, (i, j) in enumerate(E):
        delta = V[j] - V[i]
        shifts[idx] = np.round(delta / L_vec).astype(int)
    return shifts


def build_d0_bloch(V: np.ndarray, E: list, k: np.ndarray, L_vec: np.ndarray,
                   shifts: np.ndarray) -> np.ndarray:
    """Build d₀(k): gradient with Bloch phases."""
    n_V = len(V)
    n_E = len(E)
    d0 = np.zeros((n_E, n_V), dtype=complex)

    for idx, (i, j) in enumerate(E):
        phase = np.exp(1j * np.dot(k, shifts[idx] * L_vec))
        d0[idx, i] = -1.0
        d0[idx, j] = phase

    return d0


def build_d1_bloch(V: np.ndarray, E: list, F: list, k: np.ndarray, L_vec: np.ndarray,
                   d0_k: np.ndarray) -> np.ndarray:
    """Build d₁(k): curl with Bloch phases (exactness-preserving).

    Uses the exactness recurrence to ensure d₁(k) d₀(k) = 0.
    """
    n_E = len(E)
    n_F = len(F)

    edge_map = {}
    for idx, (i, j) in enumerate(E):
        edge_map[(i, j)] = (idx, +1)
        edge_map[(j, i)] = (idx, -1)

    d1 = np.zeros((n_F, n_E), dtype=complex)

    for f_idx, face in enumerate(F):
        n_verts = len(face)

        edges_info = []
        for v_pos in range(n_verts):
            i = face[v_pos]
            j = face[(v_pos + 1) % n_verts]
            e_idx, orient = edge_map[(i, j)]
            edges_info.append((e_idx, orient, i, j))

        # Exactness recurrence: phases such that d₁ d₀ = 0
        phases = [1.0 + 0j]
        for i in range(1, n_verts):
            e_prev_idx, orient_prev, _, _ = edges_info[i-1]
            e_curr_idx, orient_curr, _, _ = edges_info[i]
            v = face[i]

            d0_prev_v = d0_k[e_prev_idx, v]
            d0_curr_v = d0_k[e_curr_idx, v]

            phase_prev = phases[i-1]
            phase_curr = -orient_prev * phase_prev * d0_prev_v / (orient_curr * d0_curr_v)
            phases.append(phase_curr)

        for i, (e_idx, orient, _, _) in enumerate(edges_info):
            d1[f_idx, e_idx] = orient * phases[i]

    return d1


def extract_gauge_speeds(V: np.ndarray, E: list, F: list, L_vec: np.ndarray,
                         star1: np.ndarray, star2: np.ndarray,
                         directions: np.ndarray, k_mags: np.ndarray,
                         threshold_rel: float = 1e-12) -> Tuple[np.ndarray, int]:
    """Extract gauge wave speeds for multiple directions.

    Args:
        V, E, F: mesh data
        L_vec: box size vector
        star1, star2: Hodge stars from build_hodge_stars_voronoi
        directions: (n_dirs, 3) unit vectors
        k_mags: (n_k,) wave vector magnitudes
        threshold_rel: relative threshold for zero mode detection

    Returns:
        c_values: (n_dirs,) wave speeds
        n_zero: typical number of zero modes (should equal n_V)
    """
    shifts = compute_edge_shifts(V, E, L_vec)
    c_values = []
    n_zero_typical = None

    for k_hat in directions:
        omega_sq_1 = []
        omega_sq_2 = []

        for k_mag in k_mags:
            k = k_mag * k_hat

            d0_k = build_d0_bloch(V, E, k, L_vec, shifts)
            d1_k = build_d1_bloch(V, E, F, k, L_vec, d0_k)

            # K = d₁† *₂ d₁
            star2_diag = np.diag(star2)
            K = d1_k.conj().T @ star2_diag @ d1_k
            K = 0.5 * (K + K.conj().T)  # ensure hermitian

            # M = *₁
            M = np.diag(star1)

            eigvals, _ = eigh(K, M)
            eigvals = np.sort(np.real(eigvals))

            max_eig = np.max(np.abs(eigvals)) if len(eigvals) > 0 else 1.0
            threshold = max(max_eig * threshold_rel, 1e-14)

            n_zero = np.sum(np.abs(eigvals) < threshold)
            if n_zero_typical is None:
                n_zero_typical = n_zero

            physical = eigvals[np.abs(eigvals) >= threshold]
            if len(physical) >= 2:
                omega_sq_1.append(physical[0])
                omega_sq_2.append(physical[1])
            elif len(physical) >= 1:
                omega_sq_1.append(physical[0])
                omega_sq_2.append(physical[0])

        if len(omega_sq_1) >= 2:
            # Mean of 2 polarizations (consistent with elastic mean of 2 T modes)
            omega_sq_mean = (np.array(omega_sq_1) + np.array(omega_sq_2)) / 2
            k_sq = k_mags[:len(omega_sq_mean)]**2
            c_sq = np.sum(omega_sq_mean * k_sq) / np.sum(k_sq**2)
            c_values.append(np.sqrt(c_sq) if c_sq > 0 else 0)
        else:
            c_values.append(0)

    return np.array(c_values), n_zero_typical


def generate_sphere_directions(n_dirs: int, seed: int = 42) -> np.ndarray:
    """Generate uniformly distributed directions on unit sphere (Fibonacci)."""
    np.random.seed(seed)
    dirs = []
    phi = np.pi * (3 - np.sqrt(5))

    for i in range(n_dirs):
        y = 1 - (i / (n_dirs - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        dirs.append([x, y, z])

    return np.array(dirs)


def compute_anisotropy(values: np.ndarray) -> float:
    """Compute δ/mean anisotropy."""
    mean = np.mean(values)
    return (np.max(values) - np.min(values)) / mean if mean > 0 else 0


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    num = np.sum(x_c * y_c)
    denom = np.sqrt(np.sum(x_c**2) * np.sum(y_c**2))
    return num / denom if denom > 0 else 0


def compare_gauge_elastic(data: Dict, n_dirs: int = 50, k_L: float = 3.0, k_T: float = 1.0) -> Dict:
    """Compare gauge and elastic anisotropy patterns.

    Main function for gauge-elastic bridge comparison.

    Args:
        data: Dict from build_c15_with_dual_info or similar
        n_dirs: number of directions to sample
        k_L, k_T: elastic spring constants

    Returns:
        dict with:
            'r': Pearson correlation (should be ≈ 1.0)
            'alpha_aniso': ratio of anisotropies (should be ≈ 0.016)
            'aniso_gauge': gauge δc/c
            'aniso_elastic': elastic δv/v
            'v_T': elastic speeds by direction
            'c_gauge': gauge speeds by direction
    """
    from .bloch import DisplacementBloch
    from .hodge import build_hodge_stars_voronoi

    V = data['V']
    E = data['E']
    F = data['F']
    L = data['L']
    L_vec = data['L_vec']

    # Hodge stars
    star1, star2 = build_hodge_stars_voronoi(data)

    # Directions and k magnitudes
    directions = generate_sphere_directions(n_dirs)
    L_min = np.min(L_vec)
    k_mags = np.linspace(0.05, 0.2, 4) * (2 * np.pi / L_min)

    # Gauge speeds
    c_gauge, n_zero = extract_gauge_speeds(V, E, F, L_vec, star1, star2, directions, k_mags)

    # Elastic speeds
    bloch = DisplacementBloch(V, E, L, k_L=k_L, k_T=k_T)
    v_T = []
    for k_hat in directions:
        omega_T_list = []
        for k_mag in k_mags:
            k = k_mag * k_hat
            omega_T, _, _ = bloch.classify_modes(k)
            omega_T_list.append(np.mean(omega_T))
        omega_T_arr = np.array(omega_T_list)
        v_fit = np.sum(omega_T_arr * k_mags) / np.sum(k_mags**2)
        v_T.append(v_fit)
    v_T = np.array(v_T)

    # Compute metrics
    aniso_gauge = compute_anisotropy(c_gauge)
    aniso_elastic = compute_anisotropy(v_T)
    r = pearson_correlation(v_T, c_gauge)
    alpha = aniso_gauge / aniso_elastic if aniso_elastic > 0 else 0

    return {
        'r': r,
        'alpha_aniso': alpha,
        'aniso_gauge': aniso_gauge,
        'aniso_elastic': aniso_elastic,
        'v_T': v_T,
        'c_gauge': c_gauge,
        'n_zero': n_zero,
    }
