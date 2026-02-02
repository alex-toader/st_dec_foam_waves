"""
Bloch Boundary Conditions for Periodic DEC
==========================================

Implements twisted differential operators d₀(k), d₁(k) and the dynamical matrix
for phonon band structure calculations on periodic structures.

THEORY:
    For Bloch waves ψ(x+R) = e^{ik·R} ψ(x), the DEC operators get phase factors
    on edges/faces that cross periodic boundaries.

    d₀(k)[e, j] = +exp(i k·n_e·L)  for target vertex (if edge crosses boundary)
    d₀(k)[e, i] = -1               for source vertex

    where n_e is the crossing vector (+1 if crosses +boundary, -1 if crosses -, 0 if no crossing).

STRUCTURAL LIMITATION (d₀/d₁ operators):
    The twisted DEC operators d₀(k), d₁(k) break exactness at k≠0:
        d₁(k) @ d₀(k) ≠ 0  for k ≠ 0

    This is a STRUCTURAL property, not an implementation bug. The issue:
    - d₀(k) puts phase exp(ik·n·L) on TARGET vertex of boundary-crossing edges
    - For vertex v in face f, incoming edge has phase, outgoing edge has none
    - Phases don't cancel: exp(ik·n·L) - 1 ≠ 0

    Trade-off (cannot have both):
    - Current formulation: correct k-dependent spectrum, broken exactness
    - Gauge-covariant formulation: exactness preserved, k-independent spectrum

    This affects only BlochComplex (deprecated) and standalone d₀/d₁ operators.

RECOMMENDED CLASSES:
    DisplacementBloch - builds dynamical matrix DIRECTLY via spring network,
                        bypasses d₀/d₁ entirely, works correctly for all k.

    Bath operators (A, B in bath.py) - also built directly, not via d₀/d₁.

Jan 2026
"""

import warnings
import numpy as np
from typing import Tuple, Dict, List

from core_math_v2.operators.incidence import build_d0, build_d1

from .constants import (
    ZERO_K_THRESHOLD,
    ZERO_EIGENVALUE_THRESHOLD,
    COEFFICIENT_THRESHOLD,
    DISPERSION_K_MIN,
)


def compute_edge_crossings(vertices: np.ndarray,
                           edges: List[Tuple[int, int]],
                           L: float) -> np.ndarray:
    """
    Compute boundary crossing vectors for all edges.

    For edge (i, j) with i < j (canonical direction):
        n[axis] = +1 if edge crosses + boundary (physical path adds +L)
        n[axis] = -1 if edge crosses - boundary (physical path adds -L)
        n[axis] = 0  if no crossing

    Args:
        vertices: (V, 3) vertex positions
        edges: list of (i, j) tuples with i < j
        L: period of the cubic cell

    Returns:
        crossings: (E, 3) array of crossing vectors
    """
    crossings = []
    for (i, j) in edges:
        delta = vertices[j] - vertices[i]
        n = np.zeros(3, dtype=int)
        for axis in range(3):
            if delta[axis] < -L/2:
                n[axis] = +1  # wrapped from high to low
            elif delta[axis] > L/2:
                n[axis] = -1  # wrapped from low to high
        crossings.append(n)
    return np.array(crossings)


def compute_edge_geometry(vertices: np.ndarray,
                          edges: List[Tuple[int, int]],
                          L: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute edge unit vectors and boundary crossings for all edges.

    This is the unified function for computing edge geometry with periodic
    boundary conditions. Use this instead of duplicating the logic.

    Args:
        vertices: (V, 3) vertex positions
        edges: list of (i, j) tuples
        L: period of the cubic cell

    Returns:
        edge_vectors: (E, 3) unit vectors along edges (unwrapped)
        crossings: (E, 3) boundary crossing vectors
    """
    edge_vectors = []
    crossings = []

    for (i, j) in edges:
        delta = vertices[j] - vertices[i]
        n = np.zeros(3, dtype=int)

        # Unwrap periodic boundary
        for axis in range(3):
            if delta[axis] < -L/2:
                delta[axis] += L
                n[axis] = +1
            elif delta[axis] > L/2:
                delta[axis] -= L
                n[axis] = -1

        length = np.linalg.norm(delta)
        if length > ZERO_EIGENVALUE_THRESHOLD:
            edge_vectors.append(delta / length)
        else:
            edge_vectors.append(np.zeros(3))
        crossings.append(n)

    return np.array(edge_vectors), np.array(crossings)


def build_edge_lookup(edges: List[Tuple[int, int]],
                      crossings: np.ndarray) -> Dict:
    """
    Build lookup table for edges in both directions.

    Returns:
        edge_lookup[(v1, v2)] = (edge_index, sign, crossing_in_this_direction)
    """
    edge_lookup = {}
    for e_idx, (i, j) in enumerate(edges):
        edge_lookup[(i, j)] = (e_idx, +1, crossings[e_idx])
        edge_lookup[(j, i)] = (e_idx, -1, -crossings[e_idx])
    return edge_lookup


def build_d0_bloch(vertices: np.ndarray,
                   edges: List[Tuple[int, int]],
                   L: float,
                   k: np.ndarray,
                   crossings: np.ndarray = None) -> np.ndarray:
    """
    Build Bloch-twisted gradient operator d₀(k).

    d₀(k)[e, i] = -1            (source vertex)
    d₀(k)[e, j] = +exp(i k·n·L) (target vertex, with phase if crossing boundary)

    Args:
        vertices: (V, 3) vertex positions
        edges: list of (i, j) tuples
        L: period
        k: (3,) wave vector
        crossings: precomputed crossing vectors (optional)

    Returns:
        d0k: (E, V) complex matrix
    """
    V = len(vertices)
    E = len(edges)

    if crossings is None:
        crossings = compute_edge_crossings(vertices, edges, L)

    d0k = np.zeros((E, V), dtype=complex)

    for e_idx, (i, j) in enumerate(edges):
        n = crossings[e_idx]
        phase = np.exp(1j * np.dot(k, n * L))
        d0k[e_idx, i] = -1
        d0k[e_idx, j] = phase

    return d0k


def build_d1_bloch(vertices: np.ndarray,
                   edges: List[Tuple[int, int]],
                   faces: List[List[int]],
                   L: float,
                   k: np.ndarray,
                   edge_lookup: Dict = None,
                   crossings: np.ndarray = None) -> np.ndarray:
    """
    Build Bloch-twisted curl operator d₁(k).

    For face f using edge e in direction (v1 -> v2):
        d₁(k)[f, e] = sign × exp(i k·n_face·L)

    where n_face is the crossing vector in the face's edge direction.

    Args:
        vertices, edges, faces: mesh data
        L: period
        k: wave vector
        edge_lookup: precomputed lookup (optional)
        crossings: precomputed crossing vectors (optional)

    Returns:
        d1k: (F, E) complex matrix
    """
    E = len(edges)
    F = len(faces)

    if crossings is None:
        crossings = compute_edge_crossings(vertices, edges, L)
    if edge_lookup is None:
        edge_lookup = build_edge_lookup(edges, crossings)

    d1k = np.zeros((F, E), dtype=complex)

    for f_idx, face in enumerate(faces):
        n_verts = len(face)
        for idx in range(n_verts):
            v1 = face[idx]
            v2 = face[(idx + 1) % n_verts]

            e_idx, sign, n_face = edge_lookup[(v1, v2)]
            phase = np.exp(1j * np.dot(k, n_face * L))
            d1k[f_idx, e_idx] = sign * phase

    return d1k


def build_hodge_stars_uniform(V: int, E: int, F: int,
                              a: float = 1.0) -> Tuple[np.ndarray, ...]:
    """
    Build uniform Hodge star operators for cubic lattice.

    Simplified: ⋆₀ = a³·I, ⋆₁ = a²·I, ⋆₂ = a²·I

    Args:
        V, E, F: counts
        a: lattice constant

    Returns:
        star0, star1, star2, star0_inv, star1_inv
    """
    star0 = np.eye(V) * (a**3)
    star1 = np.eye(E) * (a**2)
    star2 = np.eye(F) * (a**2)

    star0_inv = np.eye(V) / (a**3)
    star1_inv = np.eye(E) / (a**2)

    return star0, star1, star2, star0_inv, star1_inv


def build_L_elastic(d0k: np.ndarray, d1k: np.ndarray,
                    star0_inv: np.ndarray, star1: np.ndarray,
                    star1_inv: np.ndarray, star2: np.ndarray,
                    K: float = 1.0, G: float = 1.0) -> Tuple[np.ndarray, ...]:
    """
    Build elastic Laplacian on 1-forms.

    L = G·L_shear + (K + 4G/3)·L_long

    where:
        L_long = d₀ ⋆₀⁻¹ d₀† ⋆₁
        L_shear = ⋆₁⁻¹ d₁† ⋆₂ d₁

    Args:
        d0k, d1k: Bloch operators
        star*: Hodge stars
        K, G: bulk and shear moduli

    Returns:
        L, L_long, L_shear
    """
    # L_long = d₀(k) ⋆₀⁻¹ d₀(k)† ⋆₁
    L_long = d0k @ star0_inv @ d0k.conj().T @ star1

    # L_shear = ⋆₁⁻¹ d₁(k)† ⋆₂ d₁(k)
    L_shear = star1_inv @ d1k.conj().T @ star2 @ d1k

    # Combined
    L = G * L_shear + (K + 4*G/3) * L_long

    return L, L_long, L_shear


class BlochComplex:
    """
    Precomputed Bloch DEC complex for a periodic structure.

    .. deprecated::
        BlochComplex has a known bug in crossing assignment - exactness
        d₁(k)d₀(k) = 0 fails for k ≠ 0. Use DisplacementBloch instead.
    """

    def __init__(self, vertices: np.ndarray,
                 edges: List[Tuple[int, int]],
                 faces: List[List[int]],
                 L: float,
                 a: float = None):
        """
        Initialize Bloch complex.

        Args:
            vertices, edges, faces: mesh data
            L: period
            a: lattice constant (default: estimated from edge lengths)
        """
        warnings.warn(
            "BlochComplex has a known exactness bug for k≠0. "
            "Use DisplacementBloch instead.",
            DeprecationWarning,
            stacklevel=2
        )

        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.L = L

        self.V = len(vertices)
        self.E = len(edges)
        self.F = len(faces)

        # Lattice constant - estimate from ALL edges, not just first 10
        if a is None:
            lengths = []
            crossings = compute_edge_crossings(vertices, edges, L)
            for e_idx, (i, j) in enumerate(edges):
                delta = vertices[j] - vertices[i]
                # Unwrap using precomputed crossings
                # Convention: delta_unwrapped = delta_raw + n*L
                delta = delta + crossings[e_idx] * L
                lengths.append(np.linalg.norm(delta))
            a = np.median(lengths)
        self.a = a

        # Precompute
        self.crossings = compute_edge_crossings(vertices, edges, L)
        self.edge_lookup = build_edge_lookup(edges, self.crossings)

        # Hodge stars
        self.star0, self.star1, self.star2, self.star0_inv, self.star1_inv = \
            build_hodge_stars_uniform(self.V, self.E, self.F, a)

        # Standard (k=0) operators for reference
        self.d0 = build_d0(vertices, edges)
        self.d1 = build_d1(vertices, edges, faces)

    def d0k(self, k: np.ndarray) -> np.ndarray:
        """Bloch gradient at wave vector k."""
        return build_d0_bloch(self.vertices, self.edges, self.L, k, self.crossings)

    def d1k(self, k: np.ndarray) -> np.ndarray:
        """Bloch curl at wave vector k."""
        return build_d1_bloch(self.vertices, self.edges, self.faces, self.L, k,
                              self.edge_lookup, self.crossings)

    def L_elastic(self, k: np.ndarray, K: float = 1.0, G: float = 1.0) -> np.ndarray:
        """Elastic Laplacian at wave vector k."""
        d0k = self.d0k(k)
        d1k = self.d1k(k)
        L, _, _ = build_L_elastic(d0k, d1k,
                                   self.star0_inv, self.star1,
                                   self.star1_inv, self.star2,
                                   K, G)
        return L

    def eigenvalues(self, k: np.ndarray, K: float = 1.0, G: float = 1.0) -> np.ndarray:
        """Sorted eigenvalues of L_elastic(k)."""
        L = self.L_elastic(k, K, G)
        eigs = np.linalg.eigvalsh(L)
        return np.sort(np.real(eigs))

    def check_exactness(self, k: np.ndarray) -> float:
        """Check ||d₁(k)d₀(k)|| (should be 0 for all k; nonzero indicates bug)."""
        d0k = self.d0k(k)
        d1k = self.d1k(k)
        return np.linalg.norm(d1k @ d0k)

    def check_hermitian(self, k: np.ndarray, K: float = 1.0, G: float = 1.0) -> float:
        """Check ||L - L†||."""
        L = self.L_elastic(k, K, G)
        return np.linalg.norm(L - L.conj().T)


# =============================================================================
# DISPLACEMENT-BASED FORMULATION (for proper acoustic phonons)
# =============================================================================

class DisplacementBloch:
    """
    Displacement-based Bloch formulation for elastic phonon bands.

    Unlike DEC 1-forms (1 DOF per edge), this uses displacement vectors (3 DOF per vertex).
    This is the standard solid-state physics formulation for phonon band structure.

    MODEL: Tensorial spring network
        Energy = (1/2) Σ_e [ k_L [(u_j - u_i)·ê]² + k_T |u_j - u_i - [(u_j-u_i)·ê]ê|² ]

    where:
        ê = unit vector along edge
        k_L = longitudinal stiffness (compression along edge)
        k_T = transverse stiffness (shear perpendicular to edge)

    The coupling matrix for edge e is: K_e = k_L (ê⊗ê) + k_T (I - ê⊗ê)

    When k_L = k_T: isotropic springs (degenerate T/L branches)
    When k_L ≠ k_T: proper T/L separation with v_L ≠ v_T
    """

    def __init__(self, vertices: np.ndarray,
                 edges: List[Tuple[int, int]],
                 L: float,
                 spring_k: float = 1.0,
                 mass: float = 1.0,
                 k_L: float = None,
                 k_T: float = None):
        """
        Initialize displacement-based Bloch system.

        Args:
            vertices: (V, 3) positions
            edges: list of (i, j) tuples
            L: period
            spring_k: spring constant (uniform, used if k_L/k_T not specified)
            mass: vertex mass (uniform)
            k_L: longitudinal spring constant (default: spring_k)
            k_T: transverse spring constant (default: spring_k)

        For proper T/L separation, set k_L ≠ k_T.
        Typical values for elastic medium with bulk modulus K and shear modulus G:
            k_L ∝ K + 4G/3 (longitudinal modulus)
            k_T ∝ G (shear modulus)
        """
        self.vertices = vertices
        self.edges = edges
        self.L = L
        self.spring_k = spring_k
        self.mass = mass

        # Tensorial spring constants
        self.k_L = k_L if k_L is not None else spring_k
        self.k_T = k_T if k_T is not None else spring_k

        self.V = len(vertices)
        self.E = len(edges)

        # Precompute edge vectors and crossings using unified function
        self.edge_vectors, self.crossings = compute_edge_geometry(vertices, edges, L)

    def build_dynamical_matrix(self, k: np.ndarray) -> np.ndarray:
        """
        Build dynamical matrix D(k) for wave vector k using tensorial springs.

        D(k) = (1/m) K(k)

        where K(k) is the Bloch-twisted stiffness matrix.

        The eigenvalue problem is: ω² u = D(k) u

        For each edge with direction ê, the coupling matrix is:
            K_e = k_L (ê⊗ê) + k_T (I - ê⊗ê)

        This gives:
            K_ab = k_L * ê_a * ê_b + k_T * (δ_ab - ê_a * ê_b)
                 = k_T * δ_ab + (k_L - k_T) * ê_a * ê_b

        Args:
            k: (3,) wave vector

        Returns:
            D: (3V, 3V) complex Hermitian matrix

        Physics:
            - When k_L = k_T: isotropic, all branches degenerate
            - When k_L > k_T: longitudinal branch faster (v_L > v_T)
            - When k_L < k_T: transverse branch faster (v_T > v_L)
        """
        D = np.zeros((3*self.V, 3*self.V), dtype=complex)

        for e_idx, (i, j) in enumerate(self.edges):
            e_hat = self.edge_vectors[e_idx]
            n = self.crossings[e_idx]

            # Phase factor for edge crossing boundary
            phase = np.exp(1j * np.dot(k, n * self.L))

            # Tensorial coupling: K_ab = k_T δ_ab + (k_L - k_T) ê_a ê_b
            for a in range(3):
                for b in range(3):
                    # K_ab = k_T * δ_ab + (k_L - k_T) * ê_a * ê_b
                    coeff = (self.k_L - self.k_T) * e_hat[a] * e_hat[b]
                    if a == b:
                        coeff += self.k_T

                    if abs(coeff) < COEFFICIENT_THRESHOLD:
                        continue

                    # Diagonal blocks (self-interaction)
                    D[3*i + a, 3*i + b] += coeff
                    D[3*j + a, 3*j + b] += coeff

                    # Off-diagonal blocks (interaction i-j)
                    D[3*i + a, 3*j + b] -= coeff * phase
                    D[3*j + a, 3*i + b] -= coeff * np.conj(phase)

        # Include mass (D = K/m)
        D /= self.mass

        return D

    def eigenvalues(self, k: np.ndarray) -> np.ndarray:
        """Get sorted eigenvalues ω² of D(k)."""
        D = self.build_dynamical_matrix(k)
        eigs = np.linalg.eigvalsh(D)
        return np.sort(np.real(eigs))

    def frequencies(self, k: np.ndarray) -> np.ndarray:
        """Get sorted frequencies ω = sqrt(eigenvalue)."""
        eigs = self.eigenvalues(k)
        return np.sqrt(np.maximum(eigs, 0))

    def eigenpairs(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get eigenvalues and eigenvectors of D(k).

        Returns:
            eigs: (3V,) eigenvalues ω², sorted
            vecs: (3V, 3V) eigenvectors as columns, same order as eigs
        """
        D = self.build_dynamical_matrix(k)
        eigs, vecs = np.linalg.eigh(D)
        # Sort by eigenvalue
        idx = np.argsort(np.real(eigs))
        return np.real(eigs[idx]), vecs[:, idx]

    def longitudinal_fraction(self, vec: np.ndarray, k: np.ndarray) -> float:
        """
        Compute longitudinal fraction f_L = u†P_L(k)u / u†u for a mode.

        Args:
            vec: (3V,) eigenvector
            k: (3,) wave vector

        Returns:
            f_L in [0, 1]: 0 = purely transverse, 1 = purely longitudinal
        """
        k_mag = np.linalg.norm(k)
        if k_mag < ZERO_K_THRESHOLD:
            return 0.0  # No preferred direction at Γ

        k_hat = k / k_mag

        # P_L projects each vertex's displacement onto k̂
        # f_L = Σ_i |u_i · k̂|² / Σ_i |u_i|²
        norm_sq = 0.0
        proj_sq = 0.0

        for i in range(self.V):
            u_i = vec[3*i:3*i+3]
            norm_sq += np.abs(np.vdot(u_i, u_i))
            proj = np.vdot(k_hat, u_i)
            proj_sq += np.abs(proj)**2

        if norm_sq < COEFFICIENT_THRESHOLD:
            return 0.0

        return float(np.real(proj_sq / norm_sq))

    def classify_modes(self, k: np.ndarray, n_modes: int = 6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Classify lowest modes as T1, T2, L based on longitudinal fraction.

        Args:
            k: (3,) wave vector
            n_modes: number of lowest modes to consider

        Returns:
            omega_T: (2,) transverse frequencies (lowest f_L)
            omega_L: (1,) longitudinal frequency (highest f_L among first 3)
            f_L_values: (3,) longitudinal fractions for the 3 acoustic modes
        """
        eigs, vecs = self.eigenpairs(k)
        omega = np.sqrt(np.maximum(eigs[:n_modes], 0))

        # Compute f_L for each mode
        f_L = np.array([self.longitudinal_fraction(vecs[:, i], k) for i in range(n_modes)])

        # Among first 3 modes (acoustic), find the one with highest f_L = longitudinal
        acoustic_f_L = f_L[:3]
        idx_L = np.argmax(acoustic_f_L)

        # The other two are transverse
        idx_T = [i for i in range(3) if i != idx_L]

        omega_T = omega[idx_T]
        omega_L = omega[idx_L:idx_L+1]

        return omega_T, omega_L, acoustic_f_L

    def frequencies_classified(self, k: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        """
        Get classified frequencies (T1, T2, L) with their f_L values.

        Returns:
            (omega_T1, omega_T2, omega_L, f_L_T1, f_L_T2, f_L_L)
        """
        omega_T, omega_L, f_L_all = self.classify_modes(k)

        # Sort transverse by frequency
        if omega_T[0] > omega_T[1]:
            omega_T = omega_T[::-1]

        idx_L = np.argmax(f_L_all)
        idx_T = [i for i in range(3) if i != idx_L]

        return (omega_T[0], omega_T[1], omega_L[0],
                f_L_all[idx_T[0]], f_L_all[idx_T[1]], f_L_all[idx_L])

    def check_hermitian(self, k: np.ndarray) -> float:
        """Check ||D - D†||."""
        D = self.build_dynamical_matrix(k)
        return np.linalg.norm(D - D.conj().T)

    def compute_band_structure(self, path: List[Tuple[str, np.ndarray]],
                               n_points: int = 30) -> Tuple[np.ndarray, ...]:
        """
        Compute band structure along a k-space path.

        Args:
            path: list of (label, k_point) tuples defining high-symmetry points
            n_points: points per segment

        Returns:
            k_dist: (N,) cumulative distance along path
            omega: (N, 3V) frequencies at each k
            tick_pos: positions for high-symmetry labels
            tick_labels: labels for ticks
        """
        all_k_dist = []
        all_omega = []
        tick_pos = [0]
        tick_labels = [path[0][0]]

        cumulative = 0.0

        for i in range(len(path) - 1):
            label1, k1 = path[i]
            label2, k2 = path[i + 1]

            segment_length = np.linalg.norm(k2 - k1)

            for j in range(n_points):
                t = j / n_points
                k = k1 + t * (k2 - k1)

                all_k_dist.append(cumulative + t * segment_length)
                all_omega.append(self.frequencies(k))

            cumulative += segment_length
            tick_pos.append(len(all_k_dist))
            tick_labels.append(label2)

        # Add final point
        all_k_dist.append(cumulative)
        all_omega.append(self.frequencies(path[-1][1]))

        return (np.array(all_k_dist), np.array(all_omega),
                tick_pos, tick_labels)

    def build_longitudinal_projector(self, k: np.ndarray) -> np.ndarray:
        """
        Build longitudinal projector P_L(k) for wave vector k.

        P_L projects displacement onto longitudinal direction (parallel to k).
        For each vertex i: (P_L u)_i = (u_i · k̂) k̂

        In matrix form: (P_L)_{3i+a, 3j+b} = δ_ij k̂_a k̂_b

        Args:
            k: (3,) wave vector (direction matters, magnitude doesn't)

        Returns:
            P_L: (3V, 3V) real symmetric matrix
        """
        k_mag = np.linalg.norm(k)
        if k_mag < ZERO_K_THRESHOLD:
            # At Γ, no preferred direction - return zero
            return np.zeros((3*self.V, 3*self.V))

        k_hat = k / k_mag

        # Build block-diagonal projector
        # Each 3×3 block is k̂ ⊗ k̂
        P_L = np.zeros((3*self.V, 3*self.V))

        block = np.outer(k_hat, k_hat)  # 3×3 outer product

        for i in range(self.V):
            P_L[3*i:3*i+3, 3*i:3*i+3] = block

        return P_L

    def build_dynamical_matrix_with_mass(self, k: np.ndarray,
                                         m_L: float = 0.0) -> np.ndarray:
        """
        Build dynamical matrix with longitudinal mass term [K mechanism].

        D'(k) = D(k) + m_L² P_L(k)

        This gaps the longitudinal mode: ω_L² = D eigenvalue + m_L²
        Transverse modes remain acoustic: ω_T² = D eigenvalue (P_T·P_L = 0)

        NOTE: With tensorial springs (k_L ≠ k_T), the [K] mechanism P_L(k)
        is ADDITIONAL to the natural T/L separation from the springs.

        NOTE: P_L(k=0) = 0 (no preferred direction at Γ), so at Γ exact
        there are still 3 zero-modes (translations). The gap appears as k→0⁺.

        Args:
            k: (3,) wave vector
            m_L: longitudinal mass (ω_L → m_L as k→0⁺, not at k=0 exact)

        Returns:
            D: (3V, 3V) complex Hermitian matrix
        """
        D = self.build_dynamical_matrix(k)

        if m_L > 0:
            P_L = self.build_longitudinal_projector(k)
            D = D + m_L**2 * P_L

        return D

    def frequencies_with_mass(self, k: np.ndarray, m_L: float = 0.0) -> np.ndarray:
        """Get sorted frequencies ω with longitudinal mass."""
        D = self.build_dynamical_matrix_with_mass(k, m_L=m_L)
        eigs = np.linalg.eigvalsh(D)
        return np.sqrt(np.maximum(eigs, 0))

    def analyze_dispersion(self, direction: np.ndarray,
                           k_max: float = 0.2,
                           n_points: int = 20) -> Dict:
        """
        Analyze dispersion along a specific direction.

        Returns:
            dict with: k_values, omega_1, omega_2, omega_3, velocities, linearity_error
        """
        direction = direction / np.linalg.norm(direction)
        k_values = np.linspace(DISPERSION_K_MIN, k_max, n_points)

        omega_all = []
        for kappa in k_values:
            k = kappa * direction
            omega = self.frequencies(k)[:3]
            omega_all.append(omega)

        omega_all = np.array(omega_all)

        # Compute velocities (ω/k)
        velocities = omega_all / k_values[:, None]

        # Linearity error: std(velocity) / mean(velocity)
        linearity = np.std(velocities, axis=0) / np.mean(velocities, axis=0)

        return {
            'k_values': k_values,
            'omega_1': omega_all[:, 0],
            'omega_2': omega_all[:, 1],
            'omega_3': omega_all[:, 2],
            'v_1': velocities[:, 0],
            'v_2': velocities[:, 1],
            'v_3': velocities[:, 2],
            'linearity_error': linearity,
            'birefringence': np.abs(omega_all[:, 0] - omega_all[:, 1]) / omega_all[:, 0].clip(1e-10)
        }
