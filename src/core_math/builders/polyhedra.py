"""
Generic Polyhedra Construction
==============================

Build various polyhedra for κ comparison tests.

POLYHEDRA INCLUDED:
    - Cube (V=8, E=12, F=6)
    - Octahedron (V=6, E=12, F=8)
    - Tetrahedron (V=4, E=6, F=4)

These are used to verify the surgery formula: Δκ = 4ΔE - Δdim_bridge

κ = 137 NOTE:
    κ = 137 comes from topology (V,E,F) = (24,36,14), not specific shape.
    Both Kelvin AND Truncated Cube give κ = 137 (same V,E,F, same dim_bridge=7).
    Kelvin is unique among SPACE-FILLING foam cells (tiles R³).
    Truncated Cube does NOT tile space.
"""

import numpy as np
from typing import Tuple, List, Dict, Any

from ..spec.constants import EPS_CLOSE, EDGE_TOL_POLY, FACE_TOL_POLY


def build_cube() -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], Dict[tuple, int]]:
    """
    Build a unit cube centered at origin.

    TOPOLOGY:
        V = 8 vertices (corners of cube)
        E = 12 edges
        F = 6 faces (squares)
        χ = V - E + F = 8 - 12 + 6 = 2

    Returns:
        vertices: (8, 3) array
        edges: list of 12 edge tuples
        faces: list of 6 faces
        v_to_idx: vertex lookup dict
    """
    # 8 corners of unit cube at (±1, ±1, ±1)
    vertices = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                vertices.append((x, y, z))

    vertices = sorted(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    vertices_arr = np.array(vertices, dtype=float)

    # Edges: distance = 2 (between adjacent corners)
    edges = []
    for i in range(8):
        for j in range(i+1, 8):
            d2 = np.sum((vertices_arr[i] - vertices_arr[j])**2)
            if abs(d2 - 4.0) < EPS_CLOSE:  # distance = 2
                edges.append((i, j))

    # 6 faces (squares at ±1 on each axis)
    faces = []
    for axis in range(3):
        for sign in [-1, 1]:
            face_idx = [i for i, v in enumerate(vertices) if v[axis] == sign]
            # Order vertices CCW
            coords = vertices_arr[face_idx]
            centroid = coords.mean(axis=0)
            normal = np.zeros(3)
            normal[axis] = sign

            # Build local frame
            if abs(normal[0]) < 0.9:
                u = np.cross(normal, [1, 0, 0])
            else:
                u = np.cross(normal, [0, 1, 0])
            u = u / np.linalg.norm(u)
            v = np.cross(normal, u)

            angles = [np.arctan2(np.dot(coords[k] - centroid, v),
                                 np.dot(coords[k] - centroid, u))
                      for k in range(len(face_idx))]
            order = np.argsort(angles)
            faces.append([face_idx[o] for o in order])

    if len(vertices) != 8:
        raise ValueError(f"Expected 8 vertices, got {len(vertices)}")
    if len(edges) != 12:
        raise ValueError(f"Expected 12 edges, got {len(edges)}")
    if len(faces) != 6:
        raise ValueError(f"Expected 6 faces, got {len(faces)}")

    return vertices_arr, edges, faces, v_to_idx


def build_octahedron() -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], Dict[tuple, int]]:
    """
    Build a regular octahedron centered at origin.

    TOPOLOGY:
        V = 6 vertices (on axes at ±1)
        E = 12 edges
        F = 8 faces (triangles)
        χ = V - E + F = 6 - 12 + 8 = 2

    Returns:
        vertices: (6, 3) array
        edges: list of 12 edge tuples
        faces: list of 8 triangular faces
        v_to_idx: vertex lookup dict
    """
    # 6 vertices on coordinate axes
    vertices = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    vertices = sorted(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    vertices_arr = np.array(vertices, dtype=float)

    # Edges: distance = √2 (between adjacent vertices)
    edges = []
    for i in range(6):
        for j in range(i+1, 6):
            d2 = np.sum((vertices_arr[i] - vertices_arr[j])**2)
            if abs(d2 - 2.0) < EPS_CLOSE:  # distance = √2
                edges.append((i, j))

    # 8 triangular faces (one in each octant)
    faces = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                # Find vertices in this octant
                face_verts = []
                for i, v in enumerate(vertices):
                    # Vertex is on face if it has the right signs
                    if (v[0] == sx or v[0] == 0) and \
                       (v[1] == sy or v[1] == 0) and \
                       (v[2] == sz or v[2] == 0) and \
                       sum(abs(c) for c in v) == 1:
                        face_verts.append(i)

                if len(face_verts) == 3:
                    # Order CCW from octant direction
                    normal = np.array([sx, sy, sz], dtype=float)
                    normal = normal / np.linalg.norm(normal)
                    coords = vertices_arr[face_verts]
                    centroid = coords.mean(axis=0)

                    if abs(normal[0]) < 0.9:
                        u = np.cross(normal, [1, 0, 0])
                    else:
                        u = np.cross(normal, [0, 1, 0])
                    u = u / np.linalg.norm(u)
                    v = np.cross(normal, u)

                    angles = [np.arctan2(np.dot(coords[k] - centroid, v),
                                         np.dot(coords[k] - centroid, u))
                              for k in range(3)]
                    order = np.argsort(angles)
                    faces.append([face_verts[o] for o in order])

    if len(vertices) != 6:
        raise ValueError(f"Expected 6 vertices, got {len(vertices)}")
    if len(edges) != 12:
        raise ValueError(f"Expected 12 edges, got {len(edges)}")
    if len(faces) != 8:
        raise ValueError(f"Expected 8 faces, got {len(faces)}")

    return vertices_arr, edges, faces, v_to_idx


def build_truncated_cube() -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], Dict[tuple, int]]:
    """
    Build the Archimedean truncated cube.

    CONSTRUCTION (verified heuristic, not pure derivation like Kelvin):
        - Triangles: "3 closest to corner" heuristic (finds cube corners)
        - Octagons: vertices with coord ≈ ±1 on one axis (finds cube faces)

    VERIFIED BY TEST SUITE (tests_math/test_all.py):
        - T0: All face segments are valid edges ✓
        - T9: All faces planar (max deviation 1.2e-16) ✓
        - Vertex degree: all 3 (Archimedean) ✓
        - Edge uniformity: all equal length ✓
        - κ = 137 ✓

    TOPOLOGY:
        V = 24 vertices
        E = 36 edges
        F = 14 faces (8 triangles + 6 octagons)
        χ = V - E + F = 24 - 36 + 14 = 2

    NOTE: This polyhedron also gives κ = 137 (same topology as Kelvin).
    κ = 137 comes from (V,E,F) = (24,36,14), not the specific shape.
    Kelvin is selected by OTHER criteria (space-filling, BCC optimal).
    Truncated Cube does NOT tile R³.

    Vertices are at (±ξ, ±1, ±1), (±1, ±ξ, ±1), (±1, ±1, ±ξ)
    where ξ = √2 - 1 ≈ 0.414 for the Archimedean solid.

    Returns:
        vertices: (24, 3) array
        edges: list of 36 edge tuples
        faces: list of 14 faces
        v_to_idx: vertex lookup dict
    """
    xi = np.sqrt(2) - 1  # ≈ 0.414

    vertices = set()
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                # (±ξ, ±1, ±1)
                vertices.add((round(sx * xi, 10), round(sy * 1.0, 10), round(sz * 1.0, 10)))
                # (±1, ±ξ, ±1)
                vertices.add((round(sx * 1.0, 10), round(sy * xi, 10), round(sz * 1.0, 10)))
                # (±1, ±1, ±ξ)
                vertices.add((round(sx * 1.0, 10), round(sy * 1.0, 10), round(sz * xi, 10)))

    vertices = sorted(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    vertices_arr = np.array(vertices)

    # Find edge length (minimum distance)
    min_dist = float('inf')
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            d = np.linalg.norm(vertices_arr[i] - vertices_arr[j])
            if d > 0.1 and d < min_dist:
                min_dist = d

    # Build edges
    edges = []
    for i in range(len(vertices)):
        for j in range(i+1, len(vertices)):
            d = np.linalg.norm(vertices_arr[i] - vertices_arr[j])
            if abs(d - min_dist) < EDGE_TOL_POLY:
                edges.append((i, j))

    # Build faces (8 triangles at cube corners + 6 octagons on cube faces)
    # For now, use simplified face detection
    faces = []

    # 8 triangular faces (at corners of original cube)
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                corner = np.array([sx, sy, sz])
                # Find 3 vertices closest to this corner
                dists = [np.linalg.norm(vertices_arr[i] - corner) for i in range(24)]
                closest = np.argsort(dists)[:3]
                faces.append(list(closest))

    # 6 octagonal faces (one per cube face)
    for axis in range(3):
        for sign in [-1, 1]:
            # Vertices on this face have coordinate[axis] = ±1
            face_verts = [i for i, v in enumerate(vertices)
                         if abs(v[axis] - sign) < FACE_TOL_POLY]
            if len(face_verts) == 8:
                # Order by angle around face center
                coords = vertices_arr[face_verts]
                centroid = coords.mean(axis=0)
                normal = np.zeros(3)
                normal[axis] = sign

                if abs(normal[0]) < 0.9:
                    u = np.cross(normal, [1, 0, 0])
                else:
                    u = np.cross(normal, [0, 1, 0])
                u = u / np.linalg.norm(u)
                v = np.cross(normal, u)

                angles = [np.arctan2(np.dot(coords[k] - centroid, v),
                                     np.dot(coords[k] - centroid, u))
                          for k in range(len(face_verts))]
                order = np.argsort(angles)
                faces.append([face_verts[o] for o in order])

    if len(vertices) != 24:
        raise ValueError(f"Expected 24 vertices, got {len(vertices)}")
    if len(edges) != 36:
        raise ValueError(f"Expected 36 edges, got {len(edges)}")
    if len(faces) != 14:
        raise ValueError(f"Expected 14 faces, got {len(faces)}")

    # Verify no duplicate faces (important for heuristic construction)
    face_sets = [frozenset(f) for f in faces]
    if len(face_sets) != len(set(face_sets)):
        raise ValueError(f"Duplicate faces detected: {len(faces)} faces but {len(set(face_sets))} unique")

    # Verify face types: 8 triangles + 6 octagons
    triangles = [f for f in faces if len(f) == 3]
    octagons = [f for f in faces if len(f) == 8]
    if len(triangles) != 8:
        raise ValueError(f"Expected 8 triangles, got {len(triangles)}")
    if len(octagons) != 6:
        raise ValueError(f"Expected 6 octagons, got {len(octagons)}")

    return vertices_arr, edges, faces, v_to_idx


def build_tetrahedron() -> Tuple[np.ndarray, List[Tuple[int, int]], List[List[int]], Dict[tuple, int]]:
    """
    Build a regular tetrahedron centered at origin.

    TOPOLOGY:
        V = 4 vertices
        E = 6 edges
        F = 4 faces (triangles)
        χ = V - E + F = 4 - 6 + 4 = 2

    Returns:
        vertices, edges, faces, v_to_idx
    """
    # Regular tetrahedron vertices (at alternating corners of cube)
    vertices = [
        (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)
    ]
    vertices = sorted(vertices)
    v_to_idx = {v: i for i, v in enumerate(vertices)}
    vertices_arr = np.array(vertices, dtype=float)

    # All pairs are edges (complete graph K4)
    edges = [(i, j) for i in range(4) for j in range(i+1, 4)]

    # 4 triangular faces
    faces = [
        [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]
    ]

    # Order faces correctly
    ordered_faces = []
    for face in faces:
        coords = vertices_arr[face]
        centroid = coords.mean(axis=0)
        normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
        # Ensure outward normal
        if np.dot(normal, centroid) < 0:
            normal = -normal
        normal = normal / np.linalg.norm(normal)

        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1, 0, 0])
        else:
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)

        angles = [np.arctan2(np.dot(coords[k] - centroid, v),
                             np.dot(coords[k] - centroid, u))
                  for k in range(3)]
        order = np.argsort(angles)
        ordered_faces.append([face[o] for o in order])

    return vertices_arr, edges, ordered_faces, v_to_idx


# NOTE: compute_kappa_for_polyhedron moved to analysis/kappa.py (layer separation)


# Self-test
if __name__ == "__main__":
    from ..analysis.kappa import compute_kappa_for_polyhedron

    print("=" * 60)
    print("POLYHEDRA CONSTRUCTION")
    print("=" * 60)

    for name, builder in [("Cube", build_cube),
                          ("Octahedron", build_octahedron),
                          ("Tetrahedron", build_tetrahedron)]:
        v, e, f, idx = builder()
        result = compute_kappa_for_polyhedron(v, e, f, idx)
        print(f"\n{name}:")
        print(f"  V={result['V']}, E={result['E']}, F={result['F']}, χ={result['chi']}")
        print(f"  dim(H) = {result['dim_H']}")
        print(f"  dim(bridge) = {result['dim_bridge']}, dim(ring) = {result['dim_ring']}")
        print(f"  κ = 4×{result['E']} - {result['dim_bridge']} = {result['kappa']}")
        print(f"  Centrosymmetric: {result['is_centrosymmetric']}")
