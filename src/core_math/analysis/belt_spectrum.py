"""
Belt Spectrum — Discrete Laplacian on Equatorial Circuits
==========================================================

Standard discrete ring spectrum (Brillouin 1946, Nyquist sampling)
applied to foam cell equatorial belt geometry.

Utility module: computes eigenvalues and mode structure of the 1D
discrete Laplacian on periodic rings extracted from foam cell circuits.
No new physics — all spectral results are standard textbook properties
of tridiagonal circulant matrices.

Functions:
  1. Circuit geometry — get_circuit_segments
  2. Operators — build_laplacian_1d, build_mass_matrix_1d
  3. Spectrum — compute_mode_spectrum, has_m2, max_supported_mode
  4. Full belt — compute_belt_spectrum (face areas + edge couplings)

Two spectrum models:
  - compute_mode_spectrum: 1D FEM along belt path (segments = mid→center→mid)
  - compute_belt_spectrum: coupled oscillators (mass = face area, κ = L_edge/d_cc)
  These give different spectra; only frequency ratios are meaningful
  (absolute scale depends on undefined elastic constants).

Inputs:
  - circuit: tuple of local face indices (from cell_topology.find_simple_cycles)
  - face_data, adj: from cell_topology.get_cell_geometry

Outputs:
  - Eigenvalues ω² (relative units; absolute scale undefined)
  - Eigenvectors with zero-crossing counts → mode identification
  - Boolean m=2 existence check

Standard properties used:
  - Nyquist: mode m requires N >= 2m+1 (clean) or N >= 2m (degenerate)
  - Uniform ring dispersion: ω²_m = 4 sin²(mπ/N) / h² (FEM + lumped mass)
  - Mode gap: Δω/ω₁ = 2cos(π/N) - 1 (trigonometric identity)

Feb 2026
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════
# 1. CIRCUIT GEOMETRY
# ═══════════════════════════════════════════════════════════════

def get_circuit_segments(circuit, face_data, adj):
    """
    Extract physical segment lengths for a circuit.

    Each segment k passes through face circuit[k]:
      edge_midpoint_in → face_center → edge_midpoint_out

    Segment length h_k = |mid_in - center| + |center - mid_out|.

    Args:
        circuit: tuple/list of local face indices forming a cycle
        face_data: from get_cell_geometry
        adj: face adjacency dict

    Returns:
        segments: np.array of length N, segment lengths h_k
    """
    n = len(circuit)
    # Edge midpoints between consecutive circuit faces
    nodes = []
    for k in range(n):
        fi = circuit[k]
        fj = circuit[(k + 1) % n]
        e_key = adj[fi][fj]
        nodes.append(face_data[fi]['edges'][e_key])

    segments = np.zeros(n)
    for k in range(n):
        em_in = nodes[(k - 1) % n]
        fc = face_data[circuit[k]]['center']
        em_out = nodes[k]
        segments[k] = np.linalg.norm(em_in - fc) + np.linalg.norm(fc - em_out)

    return segments


# ═══════════════════════════════════════════════════════════════
# 2. OPERATORS
# ═══════════════════════════════════════════════════════════════

def build_laplacian_1d(segments):
    """
    Build FEM stiffness matrix K for -d²u/dx² on a non-uniform periodic ring.

    Standard 1D finite element assembly: each element k (length h_k = segments[k])
    connects node (k-1) to node k, contributing (1/h_k) * [[1,-1],[-1,1]].

    After assembly over all N elements (periodic):
      K[k,k]   = 1/h_k + 1/h_{k+1}
      K[k,k-1] = -1/h_k
      K[k,k+1] = -1/h_{k+1}

    where h_k = segments[k], h_{k+1} = segments[(k+1)%N].

    Symmetric and positive semi-definite by construction.
    Null space: constant vector (rigid translation).

    On uniform ring (h_k = h): eigenvalues = 4 sin²(mπ/N) / h.
    With mass matrix M = diag((h_{k-1}+h_k)/2), generalized eigenvalues
    of H = M^{-1/2} K M^{-1/2} are 4 sin²(mπ/N) / h² (uniform).

    Args:
        segments: array of N segment lengths

    Returns:
        K: NxN stiffness matrix (symmetric, positive semi-definite)
    """
    N = len(segments)
    K = np.zeros((N, N))
    # FEM assembly: element k has length h_k, connects nodes (k-1) and k
    for k in range(N):
        h_k = segments[k]
        n_left = (k - 1) % N
        n_right = k
        K[n_left, n_left] += 1.0 / h_k
        K[n_left, n_right] += -1.0 / h_k
        K[n_right, n_left] += -1.0 / h_k
        K[n_right, n_right] += 1.0 / h_k
    return K


def build_mass_matrix_1d(segments):
    """
    Build lumped mass matrix M for the 1D ring.

    Each node k gets mass = average of neighboring segments:
      m_k = (h_{k-1} + h_k) / 2

    Args:
        segments: array of N segment lengths

    Returns:
        M: NxN diagonal mass matrix
    """
    N = len(segments)
    m = np.zeros(N)
    for k in range(N):
        m[k] = (segments[(k - 1) % N] + segments[k]) / 2.0
    return np.diag(m)


# ═══════════════════════════════════════════════════════════════
# 3. SPECTRUM
# ═══════════════════════════════════════════════════════════════

def _count_zero_crossings(v):
    """Count sign changes in a cyclic vector."""
    N = len(v)
    return sum(1 for k in range(N) if v[k] * v[(k + 1) % N] < 0)


def compute_mode_spectrum(circuit, face_data, adj):
    """
    Compute full mode spectrum of the 1D Laplacian on a circuit.

    Solves generalized eigenvalue problem K φ = ω² M φ via symmetric
    transformation H = M^{-1/2} K M^{-1/2}, then φ = M^{-1/2} ψ.

    Returns dict with:
        N: number of faces in circuit
        segments: segment lengths
        L_total: total circuit length
        R_eff: effective radius L/(2π)
        eigenvalues: ω² array (sorted ascending)
        eigenvectors: NxN matrix (columns = modes, M-orthonormal)
        zero_crossings: list of zero-crossing counts per mode
        frequencies: ω = sqrt(ω²) array

    Args:
        circuit: tuple of local face indices
        face_data: from get_cell_geometry
        adj: face adjacency dict

    Returns:
        dict with spectrum data, or None if circuit invalid
    """
    N = len(circuit)
    if N < 3:
        return None

    segments = get_circuit_segments(circuit, face_data, adj)
    L_total = segments.sum()
    R_eff = L_total / (2 * np.pi)

    K = build_laplacian_1d(segments)
    M = build_mass_matrix_1d(segments)

    # Symmetric transformation: H = M^{-1/2} K M^{-1/2}
    m_diag = np.diag(M)
    M_sqrt_inv = np.diag(1.0 / np.sqrt(m_diag))

    H = M_sqrt_inv @ K @ M_sqrt_inv
    evals, evecs_psi = np.linalg.eigh(H)

    # Sort by eigenvalue
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs_psi = evecs_psi[:, idx]

    # Physical eigenvectors: φ = M^{-1/2} ψ (M-orthonormal)
    evecs_phi = M_sqrt_inv @ evecs_psi

    # Zero-crossing counts
    zc = [_count_zero_crossings(evecs_phi[:, col]) for col in range(N)]

    # Frequencies
    freqs = np.sqrt(np.maximum(evals, 0))

    return {
        'N': N,
        'segments': segments,
        'L_total': L_total,
        'R_eff': R_eff,
        'eigenvalues': evals,
        'eigenvectors': evecs_phi,
        'zero_crossings': zc,
        'frequencies': freqs,
    }


def has_m2(circuit, face_data, adj):
    """
    Check whether a circuit supports an m=2 mode.

    m=2 is identified by exactly 4 zero-crossings in the eigenvector.

    Args:
        circuit: tuple of local face indices
        face_data: from get_cell_geometry
        adj: face adjacency dict

    Returns:
        bool: True if m=2 mode exists
    """
    spec = compute_mode_spectrum(circuit, face_data, adj)
    if spec is None:
        return False
    return 4 in spec['zero_crossings']


def max_supported_mode(N):
    """
    Maximum angular mode number supported on an N-face ring.

    Nyquist limit: m_max = floor(N/2).
    Mode m requires 2m sample points → N >= 2m → m <= N/2.

    Key values:
      N=3: m_max=1 (no m=2)
      N=4: m_max=2 (Nyquist degenerate)
      N=5: m_max=2 (first clean m=2)
      N=8: m_max=4 (first m=4)
      N=9: m_max=4 (first clean m=4)

    Args:
        N: number of faces in circuit

    Returns:
        int: maximum supported mode number
    """
    return N // 2


# ═══════════════════════════════════════════════════════════════
# 4. FULL BELT SPECTRUM (face areas + edge couplings)
# ═══════════════════════════════════════════════════════════════

def compute_belt_spectrum(circuit, face_data, adj):
    """
    Compute spectrum using physical face areas and edge coupling constants.

    This is the "full" Lagrangian:
      L = Σ_k [½ μ A_k (du_k/dt)² - ½ κ_k (u_{k+1} - u_k)²]

    where:
      A_k = area of face k
      κ_k = L_edge / d_cc (shared edge length / center-center distance)

    Eigenvalues are ω² in units of c_s² (shear wave speed squared).

    Args:
        circuit: tuple of local face indices (must be ordered ring)
        face_data: from get_cell_geometry (needs 'vertices', 'center')
        adj: face adjacency dict

    Returns:
        dict with:
            N, areas, kappas, shared_edges, center_dists,
            L_circ, eigenvalues, eigenvectors, zero_crossings,
            frequencies, face_types
        or None if invalid
    """
    N = len(circuit)
    if N < 3:
        return None

    # Face areas (cross-product triangulation)
    areas = np.zeros(N)
    for i in range(N):
        fi = circuit[i]
        verts = face_data[fi]['vertices']
        area = 0.0
        for j in range(1, len(verts) - 1):
            area += 0.5 * np.linalg.norm(
                np.cross(verts[j] - verts[0], verts[j + 1] - verts[0]))
        areas[i] = area

    if np.any(areas <= 0):
        return None

    # Centers
    centers = np.array([face_data[circuit[i]]['center'] for i in range(N)])

    # Coupling constants: κ_k = shared_edge_length / center_distance
    kappas = np.zeros(N)
    shared_edges = np.zeros(N)
    center_dists = np.zeros(N)

    for i in range(N):
        fi = circuit[i]
        fj = circuit[(i + 1) % N]
        e_key = adj[fi][fj]

        # Shared edge length — look up vertex positions in fi, fallback to fj
        v1_idx, v2_idx = e_key
        p1 = p2 = None
        for face_idx in (fi, fj):
            fd = face_data[face_idx]
            vids = fd['vertex_ids']
            verts = fd['vertices']
            for k, vid in enumerate(vids):
                if vid == v1_idx and p1 is None:
                    p1 = verts[k]
                if vid == v2_idx and p2 is None:
                    p2 = verts[k]
            if p1 is not None and p2 is not None:
                break
        if p1 is None or p2 is None:
            return None
        shared_edges[i] = np.linalg.norm(p2 - p1)

        center_dists[i] = np.linalg.norm(centers[(i + 1) % N] - centers[i])
        if center_dists[i] <= 1e-10:
            return None
        kappas[i] = shared_edges[i] / center_dists[i]

    # Stiffness matrix K
    K = np.zeros((N, N))
    for i in range(N):
        ip = (i + 1) % N
        im = (i - 1) % N
        K[i, i] = kappas[im] + kappas[i]
        K[i, ip] = -kappas[i]
        K[i, im] = -kappas[im]

    # Mass matrix M = diag(areas)
    M = np.diag(areas)

    # Generalized eigenvalue K v = ω² M v
    # Symmetric form: H = M^{-1/2} K M^{-1/2}
    m_sqrt_inv = np.diag(1.0 / np.sqrt(areas))
    H = m_sqrt_inv @ K @ m_sqrt_inv
    evals, evecs_psi = np.linalg.eigh(H)

    idx = np.argsort(evals)
    evals = evals[idx]
    evecs_psi = evecs_psi[:, idx]

    evecs_phi = m_sqrt_inv @ evecs_psi

    zc = [_count_zero_crossings(evecs_phi[:, col]) for col in range(N)]
    freqs = np.sqrt(np.maximum(evals, 0))

    L_circ = np.sum(center_dists)
    face_types = tuple(face_data[circuit[i]]['n_sides'] for i in range(N))

    return {
        'N': N,
        'areas': areas,
        'kappas': kappas,
        'shared_edges': shared_edges,
        'center_dists': center_dists,
        'L_circ': L_circ,
        'eigenvalues': evals,
        'eigenvectors': evecs_phi,
        'zero_crossings': zc,
        'frequencies': freqs,
        'face_types': face_types,
    }
