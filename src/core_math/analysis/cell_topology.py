"""
Cell Topology Utilities
========================

Shared infrastructure for topological analysis of foam cells.
Consolidates functions duplicated across 30+ investigation scripts.

Groups:
  1. Periodic geometry — periodic_delta
  2. Cell extraction — get_cell_geometry, build_cell_adjacency
  3. Circuit finding — find_simple_cycles, equatorial_test,
                       circuit_holonomy, find_best_belt
  4. Cap assignment — assign_caps_bfs, classify_faces_signed
  5. Belt geometry — get_belt_polygon, get_belt_normal
  6. Intersection — point_in_polygon_2d, count_segment_belt_crossings

Reference implementations from script 45 (A5, tested on 156k+ paths)
and script 59 v3 (signed-distance classification, 100% validated).

Feb 2026
"""

import numpy as np
from collections import defaultdict, deque


# ═══════════════════════════════════════════════════════════════
# 1. PERIODIC GEOMETRY
# ═══════════════════════════════════════════════════════════════

def periodic_delta(a, b, L):
    """
    Minimum-image displacement b - a under periodic boundary conditions.

    Args:
        a, b: points (arrays or scalars)
        L: box size (scalar for cubic box)

    Returns:
        displacement vector (b - a) wrapped to [-L/2, L/2]
    """
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return d - L * np.round(d / L)


# ═══════════════════════════════════════════════════════════════
# 2. CELL EXTRACTION
# ═══════════════════════════════════════════════════════════════

def get_cell_geometry(cell_idx, cell_center, vertices, faces, cfi, L):
    """
    Extract geometry for one cell: unwrap vertices, compute face centers,
    build face adjacency graph.

    Args:
        cell_idx: index into cfi
        cell_center: 3D center of cell
        vertices: global vertex array
        faces: global face list (each face = list of vertex indices)
        cfi: cell-face-incidence (list of lists of (face_idx, orient))
        L: box size

    Returns:
        face_data: list of dicts per face:
            'f_idx': global face index
            'center': 3D face center (unwrapped)
            'n_sides': number of vertices
            'edges': dict of edge_key → edge midpoint
            'vertices': Nx3 unwrapped vertex positions
            'vertex_ids': list of global vertex indices
        adj: dict of dict, face adjacency via shared edges
            adj[i][j] = edge_key for adjacent faces i, j
    """
    face_list = cfi[cell_idx]
    face_data = []
    edge_to_faces = defaultdict(list)

    for local_i, (f_idx, orient) in enumerate(face_list):
        fv = faces[f_idx]
        unwrapped = np.array([
            cell_center + periodic_delta(cell_center, vertices[v], L)
            for v in fv
        ])
        fc = unwrapped.mean(axis=0)

        edges_of_face = {}
        for j in range(len(fv)):
            v1_idx, v2_idx = fv[j], fv[(j + 1) % len(fv)]
            e_key = (min(v1_idx, v2_idx), max(v1_idx, v2_idx))
            mid = (unwrapped[j] + unwrapped[(j + 1) % len(unwrapped)]) / 2
            edges_of_face[e_key] = mid
            edge_to_faces[e_key].append(local_i)

        face_data.append({
            'f_idx': f_idx,
            'center': fc,
            'n_sides': len(fv),
            'edges': edges_of_face,
            'vertices': unwrapped,
            'vertex_ids': list(fv),
        })

    adj = defaultdict(dict)
    for e_key, fl in edge_to_faces.items():
        if len(fl) == 2:
            i, j = fl
            adj[i][j] = e_key
            adj[j][i] = e_key

    return face_data, dict(adj)


def build_cell_adjacency(cfi, n_cells):
    """
    Build cell-cell adjacency from shared faces.

    Args:
        cfi: cell-face-incidence
        n_cells: number of cells

    Returns:
        adj: dict mapping cell_idx → list of (neighbor_cell, shared_face_idx)
        face_to_cells: dict mapping face_idx → list of cell indices
    """
    face_to_cells = defaultdict(list)
    for ci in range(n_cells):
        for f_idx, orient in cfi[ci]:
            face_to_cells[f_idx].append(ci)

    adj = defaultdict(list)
    for f_idx, cells in face_to_cells.items():
        if len(cells) == 2:
            ci, cj = cells
            adj[ci].append((cj, f_idx))
            adj[cj].append((ci, f_idx))

    return dict(adj), dict(face_to_cells)


# ═══════════════════════════════════════════════════════════════
# 3. CIRCUIT FINDING
# ═══════════════════════════════════════════════════════════════

def find_simple_cycles(adj, n_faces, max_length=8):
    """
    Enumerate all simple cycles on face adjacency graph up to max_length.

    Uses DFS with canonical rotation to avoid duplicates.

    Args:
        adj: face adjacency dict (from get_cell_geometry)
        n_faces: total number of faces
        max_length: maximum cycle length

    Returns:
        list of tuples, each a cycle of face indices (canonically ordered)
    """
    cycles_found = set()

    def dfs(path, start, current, visited, max_len):
        if len(path) > max_len:
            return
        for neighbor in adj.get(current, {}):
            if neighbor == start and len(path) >= 3:
                cycle = tuple(path)
                min_idx = cycle.index(min(cycle))
                rotated = cycle[min_idx:] + cycle[:min_idx]
                if len(rotated) > 1 and rotated[-1] < rotated[1]:
                    rotated = (rotated[0],) + tuple(reversed(rotated[1:]))
                cycles_found.add(rotated)
            elif neighbor not in visited and neighbor > start:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(path, start, neighbor, visited, max_len)
                path.pop()
                visited.remove(neighbor)

    for start in range(n_faces):
        visited = {start}
        dfs([start], start, start, visited, max_length)

    return sorted(cycles_found, key=lambda c: len(c))


def equatorial_test(circuit, face_data, adj, cell_center, n_total_faces):
    """
    Heuristic filter: test whether a circuit is equatorial (passes near
    cell center and splits remaining faces into balanced caps). Used for
    candidate selection in find_best_belt; holonomy is computed separately
    by circuit_holonomy.

    Args:
        circuit: tuple of local face indices forming a cycle
        face_data: from get_cell_geometry
        adj: face adjacency
        cell_center: 3D cell center
        n_total_faces: total faces on cell

    Returns:
        dict with:
            'is_equatorial': bool (offset_ratio < 0.3 and balance > 0.3)
            'offset_ratio': float
            'balance': float
        or None if circuit edges not connected
    """
    n = len(circuit)
    points = []
    for k in range(n):
        fi = circuit[k]
        fj = circuit[(k + 1) % n]
        if fj not in adj.get(fi, {}):
            return None
        e_key = adj[fi][fj]
        points.append(face_data[fi]['edges'][e_key])

    points = np.array(points)
    centroid = points.mean(axis=0)
    centered = points - centroid
    U, S, Vt = np.linalg.svd(centered)
    plane_normal = Vt[-1]

    offset = abs(np.dot(cell_center - centroid, plane_normal))
    R_cell = np.mean([
        np.linalg.norm(fd['center'] - cell_center) for fd in face_data
    ])
    offset_ratio = offset / R_cell if R_cell > 0 else 999

    circuit_set = set(circuit)
    remaining = [i for i in range(n_total_faces) if i not in circuit_set]
    cap1, cap2 = [], []
    for fi in remaining:
        proj = np.dot(face_data[fi]['center'] - centroid, plane_normal)
        if proj >= 0:
            cap1.append(fi)
        else:
            cap2.append(fi)

    n1, n2 = len(cap1), len(cap2)
    balance = min(n1, n2) / max(n1, n2) if max(n1, n2) > 0 else 0

    return {
        'is_equatorial': (offset_ratio < 0.3) and (balance > 0.3),
        'offset_ratio': offset_ratio,
        'balance': balance,
    }


def circuit_holonomy(circuit, face_data, adj):
    """
    Compute Gauss-Bonnet holonomy Ω for one cap of a circuit.

    Uses T8b method: BFS cap assignment, vertex deficit angles,
    boundary vertex assignment via Jacobi neighbor voting.

    Note on convention dependence: for Ω≈2π circuits, the boundary
    vertex assignment is unambiguous (Jacobi voting and face-incidence
    counting produce identical results on all tested cell types).
    For non-2π circuits, the numeric Ω value depends on the assignment
    convention. Only Ω≈2π membership is used for physics claims.
    See ST_9/wip/w_6_release_infra/03_all_circuits_A_vs_B.py.

    Args:
        circuit: tuple of local face indices
        face_data: from get_cell_geometry
        adj: face adjacency

    Returns:
        float Ω (holonomy angle) or None if circuit doesn't split cell
    """
    cap1_faces, cap2_faces, belt_faces = assign_caps_bfs(
        face_data, circuit, adj)

    if not cap1_faces or not cap2_faces:
        return None

    # Vertex angle sums — keyed by global vertex ID
    all_verts = {}
    for fi, fd in enumerate(face_data):
        verts = fd['vertices']
        vids = fd['vertex_ids']
        for j, vid in enumerate(vids):
            v = verts[j]
            if vid not in all_verts:
                all_verts[vid] = []
            prev_v = verts[(j - 1) % len(verts)]
            next_v = verts[(j + 1) % len(verts)]
            e1 = prev_v - v
            e2 = next_v - v
            n1 = np.linalg.norm(e1)
            n2 = np.linalg.norm(e2)
            if n1 < 1e-12 or n2 < 1e-12:
                continue
            cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
            all_verts[vid].append(np.arccos(cos_a))

    # Assign vertices to caps — keyed by vertex ID
    vert_to_cap = {}
    belt_vids = set()
    cap1_vids = set()
    cap2_vids = set()

    for fi in cap1_faces:
        for vid in face_data[fi]['vertex_ids']:
            cap1_vids.add(vid)
    for fi in cap2_faces:
        for vid in face_data[fi]['vertex_ids']:
            cap2_vids.add(vid)
    for fi in belt_faces:
        for vid in face_data[fi]['vertex_ids']:
            belt_vids.add(vid)

    for vid in cap1_vids - belt_vids:
        vert_to_cap[vid] = 1
    for vid in cap2_vids - belt_vids:
        vert_to_cap[vid] = 2

    # Jacobi voting for boundary vertices (shared with belt).
    # Collect all new assignments per round, apply at end of round.
    # This makes the result independent of iteration order.
    boundary = belt_vids & (cap1_vids | cap2_vids)
    all_edges_adj = defaultdict(set)
    for fi, fd_item in enumerate(face_data):
        vids = fd_item['vertex_ids']
        for j in range(len(vids)):
            v1 = vids[j]
            v2 = vids[(j + 1) % len(vids)]
            all_edges_adj[v1].add(v2)
            all_edges_adj[v2].add(v1)

    changed = True
    while changed:
        new_assignments = {}
        for vid in boundary:
            if vid in vert_to_cap:
                continue
            neighbors = all_edges_adj.get(vid, set())
            c1 = sum(1 for nb in neighbors if vert_to_cap.get(nb) == 1)
            c2 = sum(1 for nb in neighbors if vert_to_cap.get(nb) == 2)
            if c1 > c2:
                new_assignments[vid] = 1
            elif c2 > c1:
                new_assignments[vid] = 2
        vert_to_cap.update(new_assignments)
        changed = len(new_assignments) > 0

    for vid in boundary:
        if vid not in vert_to_cap:
            vert_to_cap[vid] = 1

    # Vertex deficits
    deficits = {}
    for vid, angles in all_verts.items():
        deficits[vid] = 2 * np.pi - sum(angles)

    cap1_all = {vid for vid, c in vert_to_cap.items() if c == 1}
    omega = sum(deficits.get(vid, 0) for vid in cap1_all)
    return omega


def find_best_belt(face_data, adj, cell_center, hol_threshold=0.3):
    """
    Find the best Ω≈2π equatorial circuit for one cell.

    Searches all simple cycles up to length 8, filters for equatorial
    geometry, selects the one closest to Ω = 2π.

    Args:
        face_data: from get_cell_geometry
        adj: face adjacency
        cell_center: 3D cell center
        hol_threshold: max |Ω| - 2π error (default 0.3)

    Returns:
        tuple (circuit, holonomy) or None if no Ω≈2π belt found
    """
    n_f = len(face_data)
    cycles = find_simple_cycles(adj, n_f, max_length=8)

    best = None
    best_err = 999

    for cyc in cycles:
        eq = equatorial_test(cyc, face_data, adj, cell_center, n_f)
        if not (eq and eq['is_equatorial']):
            continue
        hol = circuit_holonomy(cyc, face_data, adj)
        if hol is None:
            continue
        err = abs(abs(hol) - 2 * np.pi)
        if err < hol_threshold and err < best_err:
            best_err = err
            best = (cyc, hol)

    return best


# ═══════════════════════════════════════════════════════════════
# 4. CAP ASSIGNMENT
# ═══════════════════════════════════════════════════════════════

def assign_caps_bfs(face_data, circuit, adj):
    """
    Assign non-belt faces to cap1 or cap2 using BFS.

    Starting from the first non-belt face, BFS finds all reachable
    faces without crossing the belt → cap1. Remaining non-belt faces
    → cap2. The belt separates the cell into two caps.

    Args:
        face_data: from get_cell_geometry
        circuit: tuple of local face indices forming the belt
        adj: face adjacency

    Returns:
        cap1: set of local face indices
        cap2: set of local face indices
        belt: set of local face indices (= set(circuit))
    """
    circuit_set = set(circuit)
    n_faces = len(face_data)
    cap_faces = [i for i in range(n_faces) if i not in circuit_set]

    if not cap_faces:
        return set(), set(), circuit_set

    visited = set()
    cap1_faces = set()
    queue = deque([cap_faces[0]])
    visited.add(cap_faces[0])
    cap1_faces.add(cap_faces[0])

    while queue:
        f = queue.popleft()
        for nb in adj.get(f, {}):
            if nb not in visited and nb not in circuit_set:
                visited.add(nb)
                cap1_faces.add(nb)
                queue.append(nb)

    cap2_faces = set(cap_faces) - cap1_faces
    return cap1_faces, cap2_faces, circuit_set


def classify_faces_signed(face_data, circuit, adj):
    """
    Classify ALL faces (including belt) into cap1(+1) or cap2(-1)
    using signed distance to the belt plane.

    Belt faces sit approximately on the belt plane; their signed
    distance determines which cap they belong to. Non-belt faces
    are guaranteed to match the BFS assignment (validated in script
    59 v3: 100% agreement on 210 non-belt faces, 0 disagreements).

    Args:
        face_data: from get_cell_geometry
        circuit: tuple of local face indices forming the belt
        adj: face adjacency

    Returns:
        dict mapping local_face_idx → +1 (cap1 side) or -1 (cap2 side)
        or None if belt doesn't split cell
    """
    belt_polygon = get_belt_polygon(circuit, face_data, adj)
    centroid = belt_polygon.mean(axis=0)
    centered = belt_polygon - centroid
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]

    cap1, cap2, belt_set = assign_caps_bfs(face_data, circuit, adj)
    if not cap1 or not cap2:
        return None

    # Signed distances for all faces
    face_side = {}
    for fi in range(len(face_data)):
        face_side[fi] = np.dot(face_data[fi]['center'] - centroid, normal)

    # Ensure cap1 is on positive side
    avg_cap1 = np.mean([face_side[fi] for fi in cap1])
    avg_cap2 = np.mean([face_side[fi] for fi in cap2])

    if avg_cap1 < avg_cap2:
        for fi in face_side:
            face_side[fi] = -face_side[fi]

    # Assign each face to +1 (cap1 side) or -1 (cap2 side)
    face_cap = {}
    for fi in range(len(face_data)):
        face_cap[fi] = +1 if face_side[fi] >= 0 else -1

    return face_cap


# ═══════════════════════════════════════════════════════════════
# 5. BELT GEOMETRY
# ═══════════════════════════════════════════════════════════════

def get_belt_polygon(circuit, face_data, adj):
    """
    Extract the 3D polygon formed by belt edge midpoints.

    Each consecutive pair of belt faces shares an edge; the midpoint
    of that edge is a vertex of the belt polygon.

    Args:
        circuit: tuple of local face indices
        face_data: from get_cell_geometry
        adj: face adjacency

    Returns:
        Nx3 numpy array of edge midpoints
    """
    n = len(circuit)
    points = []
    for k in range(n):
        fi = circuit[k]
        fj = circuit[(k + 1) % n]
        e_key = adj[fi][fj]
        points.append(face_data[fi]['edges'][e_key])
    return np.array(points)


def get_belt_normal(belt_polygon):
    """
    Compute belt plane normal via SVD.

    Args:
        belt_polygon: Nx3 array from get_belt_polygon

    Returns:
        3-vector, unit normal to best-fit plane
    """
    centered = belt_polygon - belt_polygon.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]
    return normal / np.linalg.norm(normal)


# ═══════════════════════════════════════════════════════════════
# 6. INTERSECTION / CROSSING
# ═══════════════════════════════════════════════════════════════

def point_in_polygon_2d(point_2d, polygon_2d):
    """
    Winding number test for point-in-polygon (2D).

    Args:
        point_2d: (x, y) tuple
        polygon_2d: list of (x, y) tuples, vertices in order

    Returns:
        True if point is inside polygon
    """
    n = len(polygon_2d)
    winding = 0
    px, py = point_2d

    for i in range(n):
        x1, y1 = polygon_2d[i]
        x2, y2 = polygon_2d[(i + 1) % n]
        if y1 <= py:
            if y2 > py:
                cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
                if cross > 0:
                    winding += 1
        else:
            if y2 <= py:
                cross = (x2 - x1) * (py - y1) - (px - x1) * (y2 - y1)
                if cross < 0:
                    winding -= 1

    return winding != 0


def count_segment_belt_crossings(p_start, p_end, belt_polygon, belt_normal):
    """
    Count how many times segment [p_start, p_end] crosses belt polygon.

    Uses ray-plane intersection + polygon interior test.

    Args:
        p_start, p_end: 3D points (numpy arrays)
        belt_polygon: Nx3 array from get_belt_polygon
        belt_normal: 3-vector from get_belt_normal

    Returns:
        0 or 1
    """
    assert len(belt_polygon) >= 3, (
        f"belt_polygon must have >= 3 points, got {len(belt_polygon)}"
    )

    normal = belt_normal / np.linalg.norm(belt_normal)
    center = belt_polygon.mean(axis=0)

    ray_dir = p_end - p_start
    length = np.linalg.norm(ray_dir)
    if length < 1e-12:
        return 0

    denom = np.dot(normal, ray_dir)
    if abs(denom) < 1e-12:
        return 0  # parallel to belt plane

    eps = 1e-8
    s = np.dot(normal, center - p_start) / denom
    if s < eps or s > 1.0 - eps:
        return 0  # intersection outside segment

    hit = p_start + s * ray_dir

    # Project to 2D for polygon interior test
    e1 = belt_polygon[1] - belt_polygon[0]
    n1 = np.linalg.norm(e1)
    assert n1 > 1e-12, "degenerate belt polygon: first two points coincide"
    e1 = e1 / n1
    e2 = np.cross(normal, e1)
    e2 = e2 / np.linalg.norm(e2)

    poly_2d = [
        (np.dot(v - center, e1), np.dot(v - center, e2))
        for v in belt_polygon
    ]
    dh = hit - center
    hit_2d = (np.dot(dh, e1), np.dot(dh, e2))

    return 1 if point_in_polygon_2d(hit_2d, poly_2d) else 0


def count_path_belt_crossings(path, face_data, belt_polygon, belt_normal,
                               belt_set, belt_plane_eps=0.1):
    """
    Count belt-plane crossings for a face-hopping path within one cell.

    Uses signed-distance approach with belt-face perturbation.
    For each consecutive pair of faces, checks if their face centers
    are on opposite sides of the belt plane (sign change → crossing),
    then verifies the crossing point is inside the belt polygon.

    Belt faces sit near the belt plane (|d| < 0.1) and are perturbed
    to d = -1e-6 (cap2 side). This is a convention (A5-style) to break
    belt-face degeneracy: without it, a path visiting belt faces would
    have d ≈ 0 and sign detection would be unreliable. The perturbation
    direction is arbitrary but must be consistent across all belt faces.
    Validated on 156k+ paths with 0 parity failures (script 45).

    Args:
        path: list of local face indices
        face_data: from get_cell_geometry
        belt_polygon: Nx3 array from get_belt_polygon
        belt_normal: 3-vector from get_belt_normal
        belt_set: set of local face indices forming the belt
        belt_plane_eps: distance threshold for belt-face perturbation
            (default 0.1, in lattice units)

    Returns:
        int, number of crossings
    """
    normal = belt_normal / np.linalg.norm(belt_normal)
    centroid = belt_polygon.mean(axis=0)

    # Signed distances with belt-face perturbation
    distances = []
    for fi in path:
        fc = face_data[fi]['center']
        d = np.dot(fc - centroid, normal)
        if fi in belt_set and abs(d) < belt_plane_eps:
            d = -1e-6  # perturb belt faces consistently to one side
        distances.append(d)

    # Build 2D projection basis for polygon-interior test
    e1 = belt_polygon[1] - belt_polygon[0]
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    e2 = e2 / np.linalg.norm(e2)
    poly_2d = [
        (np.dot(v - centroid, e1), np.dot(v - centroid, e2))
        for v in belt_polygon
    ]

    crossings = 0
    for i in range(len(path) - 1):
        d_i = distances[i]
        d_j = distances[i + 1]

        if d_i * d_j >= 0:
            continue  # same side, no crossing

        # Sign change → compute intersection point
        s = d_i / (d_i - d_j)
        p_i = face_data[path[i]]['center']
        p_j = face_data[path[i + 1]]['center']
        hit = p_i + s * (p_j - p_i)

        dh = hit - centroid
        hit_2d = (np.dot(dh, e1), np.dot(dh, e2))

        if point_in_polygon_2d(hit_2d, poly_2d):
            crossings += 1

    return crossings


def enumerate_paths(adj, start_set, target_set, max_length=14):
    """
    Enumerate all simple paths from start_set faces to target_set faces.

    DFS on the face adjacency graph within one cell. Each path stops
    at the first target face reached (no backtracking past target).

    Args:
        adj: face adjacency dict {face_idx: {neighbor_idx: edge_key, ...}}
        start_set: set/list of starting face indices
        target_set: set/list of target face indices
        max_length: maximum path length (number of faces visited)

    Returns:
        list of paths, each path = list of face indices [start, ..., target]
    """
    all_paths = []
    target_s = set(target_set)

    def dfs(current, path, visited):
        if len(path) > max_length:
            return
        for nb in sorted(adj.get(current, {}).keys()):
            if nb in visited:
                continue
            new_path = path + [nb]
            if nb in target_s:
                all_paths.append(new_path)
            else:
                visited.add(nb)
                dfs(nb, new_path, visited)
                visited.remove(nb)

    for src in sorted(start_set):
        visited = {src}
        dfs(src, [src], visited)

    return all_paths
