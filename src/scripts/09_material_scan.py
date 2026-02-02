#!/usr/bin/env python3
"""
SYSTEMATIC MATERIAL SCAN - Foam Isotropy Landscape
====================================================

Goal: Systematic scan of a catalog of cubic prototypes for elastic
      isotropy (margin >= 1x). Not exhaustive - covers 15 representative
      structures from common space groups.

Expected benchmarks: Kelvin ~4x, FCC ~1-2x, C15 ~27x.
We scan Voronoi complexes induced by crystal site sets;
Plateau-ness is checked separately as topology diagnostic.

CRITERION (elastic-only, no bridge)
------------------------------------
    delta_v/v = elastic anisotropy from Bloch analysis
    margin = 10^-18 * sqrt(M) / delta_v_v
    M ~ 6.2e34 (1m cavity, Planck grain)
    sqrt(M) ~ 2.5e17

    Passes if: delta_v_v < 10^-18 * sqrt(M) = 2.5e-1 = 25%
    i.e. delta_v_v < 25% passes (even FCC at 16.5% passes)

    For margin >= 4x (comfortable): delta_v_v < 6.3%

    NOTE: This is the elastic-only bound (delta_v/v). The full
    Lorentz bound uses delta_c/c = alpha * delta_v/v where alpha
    is the gauge-elastic bridge coefficient (alpha ~ 0.01-0.06).
    With bridge, margins improve by 15-100x. This scan uses the
    conservative elastic-only criterion.

SCREENING
---------
    Phase 1: I4 = <n_x^4 + n_y^4 + n_z^4>, isotropic = 0.600
             Cheap filter: |I4 - 0.6| correlates with delta_v_v
             I4 computed as unweighted mean over edge directions.
    Phase 2: Full Bloch delta_v_v for ALL structures (not just top)

ELASTIC PARAMETERS
------------------
    k_L = 3.0  (longitudinal spring constant)
    k_T = 1.0  (transverse spring constant)
    These are model parameters for the Bloch analysis; k_L/k_T ratio
    controls L-mode stiffening. Test T6 in 05_dispersion_grb.py shows
    varying these changes delta_v/v by < 5x, so ranking is robust.

STRUCTURES
----------
    7 existing + 13 new = 20 cubic crystal structures

Jan 2026
"""

import numpy as np
from scipy.spatial import Voronoi
from itertools import product
from typing import Tuple, List
from pathlib import Path
import sys
import time

# Path setup
_src_dir = Path(__file__).resolve().parents[2] / 'src'
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Topology thresholds
TOPO_RETRY_THRESH = 0.9   # retry with repeat=2 if frac < this
TOPO_OK_THRESH = 0.8      # flag TOPO? if frac < this


# =============================================================================
# VORONOI BUILDER (from 08_foam_isotropy_search.py)
# =============================================================================

def build_voronoi(points, L, repeat=1):
    """Build periodic Voronoi from points in [0,L)^3. Returns V, E, F.

    Args:
        repeat: number of shells around central cell.
                1 -> 3x3x3 (27 images), 2 -> 5x5x5 (125 images).
                Use repeat=2 for exotic structures if topology looks wrong.
    """
    n_pts = len(points)

    # Build periodic images
    offsets = list(product(range(-repeat, repeat+1), repeat=3))
    central_idx = offsets.index((0, 0, 0))

    images = []
    for di, dj, dk in offsets:
        offset = np.array([di, dj, dk]) * L
        images.append(points + offset)

    all_points = np.vstack(images)
    central_start = central_idx * n_pts
    central_end = central_start + n_pts

    vor = Voronoi(all_points)

    vertex_dict = {}
    vertices = []
    face_set = {}

    def wrap(pos):
        return tuple(round(x % L, 8) for x in pos)

    def get_idx(pos):
        w = wrap(pos)
        if w not in vertex_dict:
            vertex_dict[w] = len(vertices)
            vertices.append(np.array(w))
        return vertex_dict[w]

    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        rverts = vor.ridge_vertices[ridge_idx]
        if -1 in rverts:
            continue

        in_c1 = central_start <= p1 < central_end
        in_c2 = central_start <= p2 < central_end
        if not (in_c1 or in_c2):
            continue

        coords = np.array([vor.vertices[v] for v in rverts])
        center = coords.mean(axis=0)

        # Order vertices cyclically
        ref = coords[0] - center
        normal = np.cross(coords[1] - center, coords[2] - center)
        nl = np.linalg.norm(normal)
        if nl < 1e-14:
            continue
        normal /= nl

        vecs = coords - center
        angles = np.arctan2(
            np.einsum('ij,j->i', np.cross(vecs, ref[np.newaxis, :].repeat(len(vecs), 0)), normal),
            vecs @ ref
        )
        order = np.argsort(angles)

        face_indices = tuple(get_idx(coords[i]) for i in order)

        # Canonical form (start from min index, try both orientations)
        def canonical(f):
            n = len(f)
            best = f
            for start in range(n):
                rot = f[start:] + f[:start]
                if rot < best:
                    best = rot
            rev = f[::-1]
            for start in range(n):
                rot = rev[start:] + rev[:start]
                if rot < best:
                    best = rot
            return best

        cf = canonical(face_indices)
        if cf not in face_set:
            face_set[cf] = list(face_indices)

    V = np.array(vertices)
    faces = list(face_set.values())

    # Extract edges from faces
    edge_set = set()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            edge_set.add((min(a, b), max(a, b)))
    E = sorted(edge_set)

    return V, E, faces


# =============================================================================
# I4 COMPUTATION
# =============================================================================

def compute_I4(V, E, L):
    """Compute I4 from edge directions. Isotropic = 0.600.

    Unweighted mean over edges (each edge counts equally regardless
    of length). This matches the convention used in ST_8 analyses.
    """
    E_arr = np.asarray(E, dtype=int)
    diff = V[E_arr[:, 1]] - V[E_arr[:, 0]]
    diff = diff - L * np.round(diff / L)
    lengths = np.linalg.norm(diff, axis=1)
    mask = lengths > 1e-12
    dirs = diff[mask] / lengths[mask, np.newaxis]
    return np.mean(np.sum(dirs**4, axis=1))


def check_topology(V, E, faces):
    """Check vertex degree and faces per edge.

    Returns avg_deg, avg_fpe, is_plateau, topo_info.
    Plateau = all vertices degree 4, all edges shared by 3 faces.
    Uses fraction-based criterion (>= 95% of vertices/edges satisfy)
    rather than strict mean to handle boundary/degeneracy artifacts.
    """
    from collections import Counter

    # Vertex degree
    deg = Counter()
    for a, b in E:
        deg[a] += 1
        deg[b] += 1
    degrees = list(deg.values())
    avg_deg = np.mean(degrees) if degrees else 0
    frac_deg4 = sum(1 for d in degrees if d == 4) / len(degrees) if degrees else 0

    # Faces per edge
    edge_face_count = Counter()
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            edge_face_count[(min(a, b), max(a, b))] += 1
    fpe = list(edge_face_count.values())
    avg_fpe = np.mean(fpe) if fpe else 0
    frac_fpe3 = sum(1 for f in fpe if f == 3) / len(fpe) if fpe else 0

    is_plateau = (frac_deg4 >= 0.95 and frac_fpe3 >= 0.95)
    topo_info = {
        'frac_deg4': frac_deg4,
        'frac_fpe3': frac_fpe3,
        'deg_min': min(degrees) if degrees else 0,
        'deg_max': max(degrees) if degrees else 0,
    }
    return avg_deg, avg_fpe, is_plateau, topo_info


# =============================================================================
# SITE GENERATORS - EXISTING
# =============================================================================

def gen_sites(frac_positions, N=1, L_cell=4.0):
    """Generic: fractional positions -> physical coordinates."""
    L = N * L_cell
    points = []
    seen = set()
    for i, j, k in product(range(N), repeat=3):
        for f in frac_positions:
            p = tuple(round(((i + f[d]) * L_cell) % L, 8) for d in range(3))
            if p not in seen:
                seen.add(p)
                points.append(list(p))
    return np.array(points), L


def sites_bcc(N=1, L_cell=4.0):
    return gen_sites([[0,0,0], [0.5,0.5,0.5]], N, L_cell)

def sites_fcc(N=1, L_cell=4.0):
    return gen_sites([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]], N, L_cell)

def sites_sc(N=1, L_cell=4.0):
    return gen_sites([[0,0,0]], N, L_cell)

def sites_diamond(N=1, L_cell=4.0):
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    base = [[0,0,0], [0.25,0.25,0.25]]
    fracs = [[(b[d]+t[d])%1 for d in range(3)] for b in base for t in fcc]
    return gen_sites(fracs, N, L_cell)

def sites_a15(N=1, L_cell=4.0):
    """A15 / Weaire-Phelan. Pm-3n (223), 8 sites."""
    fracs = [
        [0,0,0], [0.5,0.5,0.5],  # 2a
        [0.25,0,0.5], [0.75,0,0.5],  # 6d
        [0.5,0.25,0], [0.5,0.75,0],
        [0,0.5,0.25], [0,0.5,0.75],
    ]
    return gen_sites(fracs, N, L_cell)

def sites_c15(N=1, L_cell=4.0):
    """C15 Laves. Fd-3m (227), 24 sites."""
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    base_8a = [[0,0,0], [0.25,0.25,0.25]]
    base_16d = [
        [5/8, 5/8, 5/8], [5/8, 3/8, 3/8],
        [3/8, 5/8, 3/8], [3/8, 3/8, 5/8],
    ]
    fracs = []
    for b in base_8a:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    for b in base_16d:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_clathrate_I(N=1, L_cell=4.0):
    """Type I clathrate. Pm-3n (223), 24 sites (2a+6c+16i)."""
    x = 0.184
    fracs = [
        [0,0,0], [0.5,0.5,0.5],  # 2a
        [0.25,0,0.5], [0.75,0,0.5],  # 6c
        [0.5,0.25,0], [0.5,0.75,0],
        [0,0.5,0.25], [0,0.5,0.75],
    ]
    # 16i: (x,x,x) + sign combos + BCC shift
    for sx, sy, sz in product([1,-1], repeat=3):
        fracs.append([sx*x % 1, sy*x % 1, sz*x % 1])
        fracs.append([(0.5+sx*x) % 1, (0.5+sy*x) % 1, (0.5+sz*x) % 1])
    # Deduplicate
    seen = set()
    unique = []
    for f in fracs:
        key = tuple(round(x, 6) for x in f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return gen_sites(unique, N, L_cell)


# =============================================================================
# SITE GENERATORS - NEW STRUCTURES
# =============================================================================

def sites_fluorite(N=1, L_cell=4.0):
    """Fluorite (CaF2). Fm-3m (225), 12 sites.
    4a: (0,0,0) + FCC
    8c: (1/4,1/4,1/4) + FCC
    """
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    base_4a = [[0,0,0]]
    base_8c = [[0.25,0.25,0.25], [0.75,0.75,0.75]]
    fracs = []
    for b in base_4a + base_8c:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_nacl(N=1, L_cell=4.0):
    """NaCl (rocksalt). Fm-3m (225), 8 sites.
    4a: (0,0,0) + FCC
    4b: (1/2,1/2,1/2) + FCC
    """
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    fracs = []
    for b in [[0,0,0], [0.5,0.5,0.5]]:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_perovskite(N=1, L_cell=4.0):
    """Perovskite (SrTiO3). Pm-3m (221), 5 sites.
    1a: (0,0,0)
    1b: (1/2,1/2,1/2)
    3d: (1/2,0,0), (0,1/2,0), (0,0,1/2)
    """
    fracs = [
        [0,0,0], [0.5,0.5,0.5],
        [0.5,0,0], [0,0.5,0], [0,0,0.5],
    ]
    return gen_sites(fracs, N, L_cell)

def sites_pyrochlore(N=1, L_cell=4.0):
    """Pyrochlore (16d sublattice of C15). Fd-3m (227), 16 sites.
    Tests: does removing 8a sites from C15 help or hurt?
    """
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    base_16d = [
        [5/8, 5/8, 5/8], [5/8, 3/8, 3/8],
        [3/8, 5/8, 3/8], [3/8, 3/8, 5/8],
    ]
    fracs = []
    for b in base_16d:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_sodalite(N=1, L_cell=4.0):
    """Sodalite framework. Im-3m (229), 12 sites.
    12d: (1/4,0,1/2) + permutations
    With BCC translations: (0,0,0) and (1/2,1/2,1/2)
    """
    base_12d = [
        [0.25, 0, 0.5], [0.75, 0, 0.5],
        [0.5, 0.25, 0], [0.5, 0.75, 0],
        [0, 0.5, 0.25], [0, 0.5, 0.75],
    ]
    bcc = [[0,0,0], [0.5,0.5,0.5]]
    fracs = []
    for b in base_12d:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)

def sites_gamma_brass(N=1, L_cell=4.0):
    """gamma-brass (Cu5Zn8). I-43m (217), 52 sites.
    IT: 8c(x,x,x) x=0.828
    OT: 8c(x,x,x) x=0.110
    OH: 12e(x,0,0) x=0.355
    CO: 24g(x,x,z) x=0.313, z=0.037
    All with BCC translations.
    """
    bcc = [[0,0,0], [0.5,0.5,0.5]]
    fracs = []

    # 8c: (x,x,x), (-x,-x,x), (-x,x,-x), (x,-x,-x) + BCC
    for x_val in [0.828, 0.110]:  # IT and OT
        base = [
            [x_val, x_val, x_val], [-x_val, -x_val, x_val],
            [-x_val, x_val, -x_val], [x_val, -x_val, -x_val],
        ]
        for b in base:
            for t in bcc:
                fracs.append([(b[d]+t[d])%1 for d in range(3)])

    # 12e: (x,0,0) + permutations + signs + BCC
    x_oh = 0.355
    base_12e = [
        [x_oh, 0, 0], [-x_oh, 0, 0],
        [0, x_oh, 0], [0, -x_oh, 0],
        [0, 0, x_oh], [0, 0, -x_oh],
    ]
    for b in base_12e:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])

    # 24g: ITC Wyckoff positions for I-43m (217)
    # Sign pattern: even number of negatives per position (not all combos)
    # 12 base positions + BCC centering = 24
    x_co, z_co = 0.313, 0.037
    base_24g = [
        [ x_co,  x_co,  z_co], [-x_co, -x_co,  z_co],
        [-x_co,  x_co, -z_co], [ x_co, -x_co, -z_co],
        [ z_co,  x_co,  x_co], [ z_co, -x_co, -x_co],
        [-z_co, -x_co,  x_co], [-z_co,  x_co, -x_co],
        [ x_co,  z_co,  x_co], [-x_co,  z_co, -x_co],
        [ x_co, -z_co, -x_co], [-x_co, -z_co,  x_co],
    ]
    for b in base_24g:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])

    # Deduplicate all
    seen = set()
    unique = []
    for f in fracs:
        key = tuple(round(x % 1, 6) for x in f)
        if key not in seen:
            seen.add(key)
            unique.append(f)

    if len(unique) != 52:
        print(f"  WARN gamma-brass: expected 52 unique positions, got {len(unique)}")

    return gen_sites(unique, N, L_cell)

def sites_beta_mn(N=1, L_cell=4.0):
    """beta-Mn. P4132 (213), 20 sites. Chiral cubic.
    8c: (x,x,x) x=0.064, with 4-fold screw axis
    12d: (1/8,y,1/4+y) y=0.203, with 2-fold axis
    """
    x = 0.064
    y = 0.203

    # 8c: generated by P4132 symmetry from (x,x,x)
    # The 4 positions in one asymmetric unit, then +1/2 shifts
    base_8c = [
        [x, x, x],
        [0.5-x, -x, 0.5+x],
        [-x, 0.5+x, 0.5-x],
        [0.5+x, 0.5-x, -x],
        [0.5+x, 0.5+x, 0.5+x],
        [1-x, 0.5-x, x],
        [0.5-x, x, 1-x],
        [x, 1-x, 0.5-x],
    ]

    # 12d: generated from (1/8, y, 1/4+y)
    base_12d = [
        [1/8, y, 0.25+y],
        [3/8, -y, 0.75+y],
        [7/8, -y, 0.75-y],
        [5/8, y, 0.25-y],
        [0.25+y, 1/8, y],
        [0.75+y, 3/8, -y],
        [0.75-y, 7/8, -y],
        [0.25-y, 5/8, y],
        [y, 0.25+y, 1/8],
        [-y, 0.75+y, 3/8],
        [-y, 0.75-y, 7/8],
        [y, 0.25-y, 5/8],
    ]

    fracs = [[c % 1 for c in pos] for pos in base_8c + base_12d]
    return gen_sites(fracs, N, L_cell)


def sites_spinel(N=1, L_cell=4.0):
    """Spinel (MgAl2O4). Fd-3m (227), 56 sites.
    8a:  (0,0,0) + FCC, basis (0,0,0),(1/4,1/4,1/4)
    16d: (5/8,5/8,5/8) + FCC, basis (5/8,5/8,5/8),(5/8,3/8,3/8),(3/8,5/8,3/8),(3/8,3/8,5/8)
    32e: (x,x,x) x=0.3875 + FCC, all sign combos (x,x,x),(-x,-x,x),(-x,x,-x),(x,-x,-x),
         then +(1/4,1/4,1/4) for each
    """
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    # 8a
    base_8a = [[0,0,0], [0.25,0.25,0.25]]
    # 16d
    base_16d = [[5/8,5/8,5/8], [5/8,3/8,3/8], [3/8,5/8,3/8], [3/8,3/8,5/8]]
    # 32e: x=0.3875 (oxide parameter for ideal spinel)
    x = 0.3875
    base_32e_half = [[x,x,x], [-x,-x,x], [-x,x,-x], [x,-x,-x]]
    base_32e = base_32e_half + [[(b[d]+0.25)%1 for d in range(3)] for b in base_32e_half]
    fracs = []
    for b in base_8a + base_16d + base_32e:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)


def sites_alpha_mn(N=1, L_cell=4.0):
    """alpha-Mn. I-43m (217), 58 sites.
    2a:  (0,0,0) + BCC
    8c:  (x,x,x) x=0.317 + BCC (same sign pattern as gamma-brass 8c)
    24g: (x,x,z) x=0.357, z=0.034 + BCC (ITC sign pattern)
    24g: (x,x,z) x=0.089, z=0.278 + BCC (ITC sign pattern)
    """
    bcc = [[0,0,0], [0.5,0.5,0.5]]
    fracs = []

    # 2a
    for t in bcc:
        fracs.append(t[:])

    # 8c: x=0.317
    x8 = 0.317
    base_8c = [[x8,x8,x8], [-x8,-x8,x8], [-x8,x8,-x8], [x8,-x8,-x8]]
    for b in base_8c:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])

    # Two 24g orbits (ITC sign pattern for I-43m)
    for xg, zg in [(0.357, 0.034), (0.089, 0.278)]:
        base_24g = [
            [ xg,  xg,  zg], [-xg, -xg,  zg],
            [-xg,  xg, -zg], [ xg, -xg, -zg],
            [ zg,  xg,  xg], [ zg, -xg, -xg],
            [-zg, -xg,  xg], [-zg,  xg, -xg],
            [ xg,  zg,  xg], [-xg,  zg, -xg],
            [ xg, -zg, -xg], [-xg, -zg,  xg],
        ]
        for b in base_24g:
            for t in bcc:
                fracs.append([(b[d]+t[d])%1 for d in range(3)])

    # Dedup
    seen = set()
    unique = []
    for f in fracs:
        key = tuple(round(v % 1, 6) for v in f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return gen_sites(unique, N, L_cell)


def sites_th3p4(N=1, L_cell=4.0):
    """Th3P4-type. I-43d (220), 28 sites.
    12a: (3/8,0,1/4) + symmetry ops + BCC
    16c: (x,x,x) x=0.083 + sign pattern + BCC
    """
    bcc = [[0,0,0], [0.5,0.5,0.5]]
    fracs = []

    # 12a: I-43d Wyckoff 12a
    # (3/8,0,1/4), (1/8,0,3/4), (1/4,3/8,0), (3/4,1/8,0), (0,1/4,3/8), (0,3/4,1/8)
    base_12a = [
        [3/8, 0, 1/4], [1/8, 0, 3/4],
        [1/4, 3/8, 0], [3/4, 1/8, 0],
        [0, 1/4, 3/8], [0, 3/4, 1/8],
    ]
    for b in base_12a:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])

    # 16c: (x,x,x) with I-43d symmetry, x=0.083
    x = 0.083
    base_16c = [
        [x, x, x], [-x, 0.5-x, 0.5+x],
        [0.5-x, 0.5+x, -x], [0.5+x, -x, 0.5-x],
        [0.25+x, 0.25+x, 0.25+x], [0.25-x, 0.75-x, 0.75+x],
        [0.75-x, 0.75+x, 0.25-x], [0.75+x, 0.25-x, 0.75-x],
    ]
    for b in base_16c:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])

    seen = set()
    unique = []
    for f in fracs:
        key = tuple(round(v % 1, 6) for v in f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return gen_sites(unique, N, L_cell)


def sites_pyrite(N=1, L_cell=4.0):
    """Pyrite (FeS2). Pa-3 (205), 12 sites.
    4a: (0,0,0) + FCC translations
    8c: (x,x,x) x=0.386, Pa-3 sign pattern + FCC
    """
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    # 4a
    base_4a = [[0, 0, 0]]
    # 8c: Pa-3 has (x,x,x), (-x+1/2,-x,x+1/2), (x+1/2,-x+1/2,-x), (-x,x+1/2,-x+1/2)
    # These are the 2 positions in primitive cell
    x = 0.386
    base_8c = [
        [x, x, x],
        [0.5-x, -x, 0.5+x],
    ]
    fracs = []
    for b in base_4a + base_8c:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)


def sites_skutterudite(N=1, L_cell=4.0):
    """Skutterudite (CoAs3). Im-3 (204), 32 sites.
    8c: (x,x,x) x=0.0 (origin) + BCC  [Co at corners]
    24g: (0,y,z) y=0.150, z=0.350 + Im-3 symmetry + BCC
    """
    bcc = [[0,0,0], [0.5,0.5,0.5]]
    fracs = []

    # 8c: (0,0,0) site + 3-fold axis
    # In Im-3, 8c: (x,x,x),(-x,-x,x),(-x,x,-x),(x,-x,-x) + BCC
    # For x=0, this gives 2 sites (0,0,0)+(1/2,1/2,1/2) but conventionally
    # skutterudite has metal at 8c with x≈0, so effectively BCC sublattice
    # Use small x to avoid degeneracy
    for t in bcc:
        fracs.append(t[:])
    # Add the 6 equivalent positions from Im-3 8c with x=0 but via FCC-like
    # Actually in Im-3: 8c is (1/4,1/4,1/4) + sign combos + BCC
    base_8c = [[1/4,1/4,1/4], [-1/4,-1/4,1/4], [-1/4,1/4,-1/4], [1/4,-1/4,-1/4]]
    for b in base_8c:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])

    # 24g: (0,y,z) + permutations + signs + BCC
    y, z = 0.150, 0.350
    base_24g = [
        [0, y, z], [0, -y, -z], [0, -y, z], [0, y, -z],
        [z, 0, y], [-z, 0, -y], [z, 0, -y], [-z, 0, y],
        [y, z, 0], [-y, -z, 0], [-y, z, 0], [y, -z, 0],
    ]
    for b in base_24g:
        for t in bcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])

    seen = set()
    unique = []
    for f in fracs:
        key = tuple(round(v % 1, 6) for v in f)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return gen_sites(unique, N, L_cell)


def sites_half_heusler(N=1, L_cell=4.0):
    """Half-Heusler. F-43m (216), 12 sites.
    4a: (0,0,0) + FCC
    4b: (1/2,1/2,1/2) + FCC  [NOT occupied in half-Heusler, skip]
    4c: (1/4,1/4,1/4) + FCC
    4d: (3/4,3/4,3/4) + FCC
    """
    fcc = [[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]
    base = [[0,0,0], [0.25,0.25,0.25], [0.75,0.75,0.75]]
    fracs = []
    for b in base:
        for t in fcc:
            fracs.append([(b[d]+t[d])%1 for d in range(3)])
    return gen_sites(fracs, N, L_cell)


# =============================================================================
# STRUCTURE CATALOG
# =============================================================================

STRUCTURES = [
    # Existing (baseline)
    ('SC',          sites_sc,          'Pm-3m (221)',  1),
    ('BCC (Kelvin)',sites_bcc,         'Im-3m (229)',  2),
    ('FCC',         sites_fcc,         'Fm-3m (225)',  4),
    ('Diamond',     sites_diamond,     'Fd-3m (227)',  8),
    ('A15 (WP)',    sites_a15,         'Pm-3n (223)',  8),
    ('Clathrate-I', sites_clathrate_I, 'Pm-3n (223)', 24),
    ('C15 (Laves)', sites_c15,         'Fd-3m (227)', 24),
    # New
    ('NaCl',        sites_nacl,        'Fm-3m (225)',  8),
    ('Perovskite',  sites_perovskite,  'Pm-3m (221)',  5),
    ('Fluorite',    sites_fluorite,    'Fm-3m (225)', 12),
    ('Pyrochlore',  sites_pyrochlore,  'Fd-3m (227)', 16),
    ('Sodalite',    sites_sodalite,    'Im-3m (229)', 12),
    ('beta-Mn',     sites_beta_mn,     'P4132 (213)', 20),
    ('gamma-brass', sites_gamma_brass, 'I-43m (217)', 52),
    # New batch 2
    ('Spinel',      sites_spinel,      'Fd-3m (227)', 56),
    ('alpha-Mn',    sites_alpha_mn,    'I-43m (217)', 58),
    ('Th3P4',       sites_th3p4,       'I-43d (220)', 28),
    ('Pyrite',      sites_pyrite,      'Pa-3 (205)',  12),
    ('Skutterudite',sites_skutterudite,'Im-3 (204)',  32),
    ('Half-Heusler',sites_half_heusler,'F-43m (216)', 12),
]


# =============================================================================
# MAIN SCAN
# =============================================================================

def run_phase1():
    """Phase 1: I4 screening for all structures."""
    print("=" * 80)
    print("PHASE 1: I4 SCREENING")
    print("=" * 80)
    print()

    results = []

    for name, site_fn, sg, n_sites_expected in STRUCTURES:
        try:
            pts, L = site_fn(N=1, L_cell=4.0)
            n_sites = len(pts)

            # Verify site count
            if n_sites != n_sites_expected:
                print(f"  {name:20s}  WARN: expected {n_sites_expected} sites, got {n_sites}")

            # Build Voronoi; auto-retry with repeat=2 if topology is bad
            t0 = time.time()
            V, E, F = build_voronoi(pts, L, repeat=1)
            avg_deg, avg_fpe, is_plateau, topo = check_topology(V, E, F)

            used_repeat = 1
            if topo['frac_deg4'] < TOPO_RETRY_THRESH or topo['frac_fpe3'] < TOPO_RETRY_THRESH:
                V2, E2, F2 = build_voronoi(pts, L, repeat=2)
                avg_deg2, avg_fpe2, is_plateau2, topo2 = check_topology(V2, E2, F2)
                score1 = topo['frac_deg4'] + topo['frac_fpe3']
                score2 = topo2['frac_deg4'] + topo2['frac_fpe3']
                if score2 >= score1:
                    V, E, F = V2, E2, F2
                    avg_deg, avg_fpe, is_plateau, topo = avg_deg2, avg_fpe2, is_plateau2, topo2
                    used_repeat = 2

            dt = time.time() - t0

            I4 = compute_I4(V, E, L)
            dI4 = abs(I4 - 0.600)

            results.append({
                'name': name,
                'sg': sg,
                'n_sites': n_sites,
                'n_sites_expected': n_sites_expected,
                'V': len(V), 'E': len(E), 'F': len(F),
                'I4': I4,
                'dI4': dI4,
                'avg_deg': avg_deg,
                'avg_fpe': avg_fpe,
                'plateau': is_plateau,
                'topo': topo,
                'time': dt,
                'site_fn': site_fn,
                'repeat': used_repeat,
            })
            plat = 'Y' if is_plateau else 'N'
            rep = f' R={used_repeat}' if used_repeat > 1 else ''
            print(f"  {name:20s}  sites={n_sites:3d}  V={len(V):4d} E={len(E):4d}  "
                  f"I4={I4:.4f}  |dI4|={dI4:.4f}  deg={avg_deg:.1f} fpe={avg_fpe:.1f}  "
                  f"Plateau={plat}{rep}  ({dt:.2f}s)")

        except Exception as e:
            print(f"  {name:20s}  ERROR: {e}")
            results.append({'name': name, 'error': str(e)})

    print()

    # Sort by |I4 - 0.6|
    valid = [r for r in results if 'I4' in r and not np.isnan(r['I4'])]
    valid.sort(key=lambda r: r['dI4'])

    # Dynamic threshold: use C15's dI4 from this run
    c15_dI4 = next((r['dI4'] for r in valid if 'C15' in r['name']), 0.015)

    print("-" * 80)
    print("RANKING BY |I4 - 0.600| (smaller = more isotropic)")
    print("-" * 80)
    print(f"{'#':>3s}  {'Structure':20s}  {'SG':15s}  {'Sites':>5s}  {'V':>5s}  {'E':>5s}  "
          f"{'I4':>7s}  {'|dI4|':>7s}  {'Plateau':>7s}")
    for i, r in enumerate(valid):
        plat = 'Y' if r['plateau'] else 'N'
        marker = ' **' if r['dI4'] < c15_dI4 else ('  *' if r['dI4'] < 0.060 else '   ')
        print(f"{i+1:3d}  {r['name']:20s}  {r['sg']:15s}  {r['n_sites']:5d}  "
              f"{r['V']:5d}  {r['E']:5d}  {r['I4']:7.4f}  {r['dI4']:7.4f}  "
              f"{plat:>7s}{marker}")

    print()
    print(f"  ** = beats C15 (|dI4| < {c15_dI4:.4f})")
    print("   * = promising (|dI4| < 0.060)")
    print()

    return valid


def run_phase2(phase1_results):
    """Phase 2: Full Bloch delta_v/v for all valid structures."""
    from physics.christoffel import compute_delta_v_direct
    from physics.bloch import DisplacementBloch

    print("=" * 80)
    print("PHASE 2: BLOCH delta_v/v (ALL STRUCTURES, elastic-only, k_L=3.0 k_T=1.0)")
    print("=" * 80)
    print()

    SQRT_M = 2.5e17  # sqrt(6.2e34)

    results_full = []

    for r in phase1_results:
        if 'error' in r:
            continue

        name = r['name']
        try:
            pts, L = r['site_fn'](N=1, L_cell=4.0)
            repeat = r.get('repeat', 1)
            V, E, F = build_voronoi(pts, L, repeat=repeat)

            db = DisplacementBloch(V, E, L, k_L=3.0, k_T=1.0)
            result = compute_delta_v_direct(db, L, n_directions=50)
            dv = result['delta_v_over_v']
            margin = 1e-18 * SQRT_M / dv if dv > 0 else float('inf')

            r['delta_v_v'] = dv
            r['margin'] = margin
            # Recalculate topology from Phase 2 Voronoi (may differ from Phase 1)
            _, _, _, topo_p2 = check_topology(V, E, F)
            r['topo_ok'] = (topo_p2.get('frac_deg4', 0) >= TOPO_OK_THRESH and
                            topo_p2.get('frac_fpe3', 0) >= TOPO_OK_THRESH)
            results_full.append(r)

            status = 'PASS' if margin >= 1 else 'FAIL'
            topo_flag = '' if r['topo_ok'] else '  [TOPO?]'
            print(f"  {name:20s}  delta_v/v={dv*100:6.2f}%  margin={margin:6.1f}x  {status}{topo_flag}")

        except Exception as e:
            print(f"  {name:20s}  ERROR: {e}")

    print()

    # Sort by margin (descending)
    results_full.sort(key=lambda r: -r.get('margin', 0))

    print("-" * 80)
    print("FULL RANKING BY MARGIN (descending)")
    print("-" * 80)
    print(f"{'#':>3s}  {'Structure':20s}  {'SG':15s}  {'Sites':>5s}  {'I4':>7s}  "
          f"{'delta_v/v':>10s}  {'Margin':>8s}  {'Status':>8s}")
    for i, r in enumerate(results_full):
        dv = r['delta_v_v']
        m = r['margin']
        status = 'PASS' if m >= 1 else 'FAIL'
        if not r.get('topo_ok', True):
            status += ' T?'
        print(f"{i+1:3d}  {r['name']:20s}  {r['sg']:15s}  {r['n_sites']:5d}  "
              f"{r['I4']:7.4f}  {dv*100:9.2f}%  {m:7.1f}x  {status:>8s}")

    # Summary
    n_pass = sum(1 for r in results_full if r['margin'] >= 1)
    n_comfortable = sum(1 for r in results_full if r['margin'] >= 4)
    best_all = results_full[0]
    best_clean = next((r for r in results_full if r.get('topo_ok', True) and r['margin'] >= 1), best_all)
    print()
    print(f"  PASS (margin >= 1x):   {n_pass}/{len(results_full)}")
    print(f"  Comfortable (>= 4x):   {n_comfortable}/{len(results_full)}")
    print(f"  Best overall: {best_all['name']} ({best_all['margin']:.1f}x)")
    if best_clean != best_all:
        print(f"  Best (clean topo): {best_clean['name']} ({best_clean['margin']:.1f}x)")
    print()

    return results_full


if __name__ == '__main__':
    print()
    print("SYSTEMATIC MATERIAL SCAN FOR FOAM ISOTROPY")
    print("=" * 80)
    print()

    phase1 = run_phase1()
    phase2 = run_phase2(phase1)
