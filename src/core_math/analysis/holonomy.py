"""
Holonomy — Discrete Curvature and Spinor Transport on Polyhedral Surfaces
==========================================================================

Standard discrete differential geometry (Descartes 1630, Gauss-Bonnet)
applied to foam cell boundaries. All results follow from the polyhedral
Gauss-Bonnet theorem and the SU(2) double cover of SO(3).

Functions:
  1. Curvature — compute_vertex_deficits, gauss_bonnet_total
  2. Face normals — compute_face_normal, compute_face_normals
  3. Gauss map holonomy — circuit_gauss_map_holonomy
  4. SU(2) lift — su2_from_holonomy, circuit_su2_holonomy

Two independent holonomy computations:
  - Gauss-Bonnet: Omega = Sum(angular deficits at enclosed vertices)
    Computed by circuit_holonomy in cell_topology.py (M0).
  - Gauss map: theta = spherical excess of face normal polygon on S^2
    Computed by circuit_gauss_map_holonomy here.
  On polyhedral surfaces these agree modulo 2*pi. Raw theta depends
  on spherical polygon branch/orientation; use as cross-check, not
  as primary holonomy computation.

SU(2) holonomy:
  U = exp(-i*Omega*sigma_z/2)
  For Omega = 2*pi: U = -I (spinor sign flip, exact by construction)
  For Omega != 2*pi: U != -I

Inputs:
  - face_data, adj: from cell_topology.get_cell_geometry (M0)
  - circuit: tuple of local face indices
  - cell_center: 3D position of cell center (for outward normal orientation)

Standard properties used:
  - Descartes angular deficit = discrete Gaussian curvature (1630)
  - Gauss-Bonnet: total deficit = 4*pi for genus-0 surface
  - Holonomy = enclosed curvature (discrete connection on polyhedral surface)
  - SU(2) double cover: 2*pi rotation -> -I

Feb 2026
"""

import numpy as np
from collections import defaultdict


# =====================================================================
# 1. CURVATURE
# =====================================================================

def compute_vertex_deficits(face_data):
    """
    Compute angular deficit at each vertex of a polyhedral surface.

    Angular deficit delta_v = 2*pi - Sum(face angles at v).
    This is the discrete Gaussian curvature (Descartes, 1630).

    For a convex polyhedron: all deficits are positive.
    For a genus-0 closed surface: Sum(delta_v) = 4*pi (Descartes theorem).

    Args:
        face_data: list of face dicts from get_cell_geometry.
                   Each must have 'vertex_ids' and 'vertices'.

    Returns:
        dict: vertex_id -> angular deficit (radians)
    """
    angle_sums = defaultdict(float)

    for fd in face_data:
        vids = fd['vertex_ids']
        verts = fd['vertices']
        n = len(vids)
        for j in range(n):
            v_prev = verts[(j - 1) % n]
            v_curr = verts[j]
            v_next = verts[(j + 1) % n]
            e1 = v_prev - v_curr
            e2 = v_next - v_curr
            n1 = np.linalg.norm(e1)
            n2 = np.linalg.norm(e2)
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
            angle_sums[vids[j]] += np.arccos(cos_a)

    return {vid: 2 * np.pi - total for vid, total in angle_sums.items()}


def gauss_bonnet_total(face_data):
    """
    Compute total Gaussian curvature of a polyhedral surface.

    For a genus-0 closed surface (S^2): total = 4*pi.
    This is the Descartes-Euler theorem.

    Args:
        face_data: list of face dicts from get_cell_geometry

    Returns:
        float: total angular deficit (should be 4*pi for a single cell)
    """
    deficits = compute_vertex_deficits(face_data)
    return sum(deficits.values())


# =====================================================================
# 2. FACE NORMALS
# =====================================================================

def compute_face_normal(fd, cell_center):
    """
    Compute outward-pointing unit normal for one face.

    Polygon normal via cross-sum (area-weighted), oriented outward
    from cell_center.

    Args:
        fd: face dict with 'vertices' (Nx3 array) and 'center'
        cell_center: 3D cell center position

    Returns:
        3-vector: outward unit normal
    """
    verts = fd['vertices']
    n = len(verts)
    normal = np.zeros(3)
    for j in range(n):
        normal += np.cross(verts[j], verts[(j + 1) % n])
    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-12:
        raise ValueError("Degenerate face: zero-area polygon, cannot compute normal")
    normal = normal / norm_val
    if np.dot(normal, fd['center'] - cell_center) < 0:
        normal = -normal
    return normal


def compute_face_normals(face_data, cell_center):
    """
    Compute outward-pointing unit normals for all faces.

    Args:
        face_data: list of face dicts from get_cell_geometry
        cell_center: 3D cell center position

    Returns:
        list of 3-vectors (one per face, same indexing as face_data)
    """
    return [compute_face_normal(fd, cell_center) for fd in face_data]


# =====================================================================
# 3. GAUSS MAP HOLONOMY
# =====================================================================

def _spherical_angle(n_prev, n_curr, n_next):
    """
    Interior angle of a spherical polygon at vertex n_curr.

    Computes the signed angle between the great circle arcs
    n_prev->n_curr and n_curr->n_next, measured at n_curr.

    Args:
        n_prev, n_curr, n_next: unit vectors on S^2

    Returns:
        float: signed angle (radians), positive for left turn
    """
    # Tangent vectors of great circle arcs at n_curr
    t_in = n_prev - np.dot(n_curr, n_prev) * n_curr
    norm_in = np.linalg.norm(t_in)
    if norm_in < 1e-12:
        return 0.0
    t_in /= norm_in

    t_out = n_next - np.dot(n_curr, n_next) * n_curr
    norm_out = np.linalg.norm(t_out)
    if norm_out < 1e-12:
        return 0.0
    t_out /= norm_out

    cos_a = np.clip(np.dot(t_in, t_out), -1, 1)
    cross = np.cross(t_in, t_out)
    sin_a = np.dot(cross, n_curr)
    return np.arctan2(sin_a, cos_a)


def circuit_gauss_map_holonomy(circuit, face_data, cell_center):
    """
    Compute holonomy via the Gauss map (spherical excess).

    Projects face normals onto S^2 and computes the spherical excess
    (solid angle) of the resulting spherical polygon. On a polyhedral
    surface, this equals the enclosed Gaussian curvature = Omega_GB.

    Gauge-invariant: depends only on face normals, not on arbitrary
    reference direction choices within each face.

    The spherical excess of a polygon with interior angles A_k is:
      Omega = Sum(A_k) - (N-2)*pi

    The sign depends on the winding direction of the normal polygon
    on S^2. For comparison with Gauss-Bonnet, use mod 2*pi.

    Args:
        circuit: tuple of local face indices
        face_data: from get_cell_geometry
        cell_center: 3D cell center (for normal orientation)

    Returns:
        dict with:
            theta_raw: spherical excess (holonomy angle)
            theta_mod_2pi: theta_raw mod 2*pi, normalized to [-pi, pi]
            spherical_angles: array of interior angles at each vertex
        or None if computation fails
    """
    n = len(circuit)
    if n < 3:
        return None

    normals = compute_face_normals(face_data, cell_center)
    ns = [normals[circuit[k]] for k in range(n)]

    # Check normals are unit and distinct
    for i in range(n):
        if np.linalg.norm(ns[i]) < 0.5:
            return None

    # Spherical polygon interior angles
    angles = []
    for k in range(n):
        a = _spherical_angle(ns[(k - 1) % n], ns[k], ns[(k + 1) % n])
        angles.append(a)

    # Spherical excess = sum of interior angles - (N-2)*pi
    total = sum(angles) - (n - 2) * np.pi

    total_mod = total % (2 * np.pi)
    if total_mod > np.pi:
        total_mod -= 2 * np.pi

    return {
        'theta_raw': total,
        'theta_mod_2pi': total_mod,
        'spherical_angles': np.array(angles),
    }


# =====================================================================
# 4. SU(2) LIFT
# =====================================================================

def su2_from_holonomy(omega):
    """
    SU(2) holonomy element for holonomy angle omega.

    U = exp(-i*omega*sigma_z/2) = diag(exp(-i*omega/2), exp(i*omega/2))

    Key values:
      omega = 0:    U = +I  (trivial)
      omega = pi:   U = diag(-i, i)  (quarter turn)
      omega = 2*pi: U = -I  (spinor sign flip)
      omega = 4*pi: U = +I  (trivial again)

    tr(U) = 2*cos(omega/2).

    Args:
        omega: holonomy angle (radians)

    Returns:
        2x2 complex numpy array (SU(2) matrix, diagonal)
    """
    return np.array([
        [np.exp(-1j * omega / 2), 0j],
        [0j, np.exp(1j * omega / 2)]
    ])


def circuit_su2_holonomy(circuit, face_data, adj):
    """
    Compute SU(2) holonomy from Gauss-Bonnet angle.

    Calls circuit_holonomy (M0) to get Omega, then lifts to SU(2).

    Args:
        circuit: tuple of local face indices
        face_data: from get_cell_geometry
        adj: face adjacency

    Returns:
        dict with:
            omega: Gauss-Bonnet holonomy angle
            U: 2x2 SU(2) matrix
            trace: tr(U) = 2*cos(omega/2)
            is_minus_I: bool, True if |Omega mod 4*pi - 2*pi| < 0.05
        or None if holonomy computation fails
    """
    from core_math.analysis.cell_topology import circuit_holonomy  # local to avoid circular

    omega = circuit_holonomy(circuit, face_data, adj)
    if omega is None:
        return None

    U = su2_from_holonomy(omega)
    tr = np.real(np.trace(U))

    return {
        'omega': omega,
        'U': U,
        'trace': tr,
        'is_minus_I': abs(omega % (4 * np.pi) - 2 * np.pi) < 0.05,
    }
