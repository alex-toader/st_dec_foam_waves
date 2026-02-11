"""
Mesh Contract - THE critical piece
===================================

All builders MUST return a mesh dict conforming to this contract.
All tests MUST read faces_per_edge from mesh, NEVER hardcode.
"""

import numpy as np
from typing import List, Tuple
from .constants import COMPLEX_SURFACE, COMPLEX_FOAM, COMPLEX_TILING, COMPLEX_SOLID, FACES_PER_EDGE


def canonical_face(face: List[int]) -> Tuple[tuple, int]:
    """
    Return canonical representation of a face cycle and its relative orientation.

    The canonical form:
        1. Starts at the minimum vertex index
        2. Goes in the direction that makes the second element smaller

    Args:
        face: list of vertex indices forming a cycle (e.g., [3, 1, 4, 2])

    Returns:
        (canonical_tuple, orientation)
        - canonical_tuple: face vertices in canonical order
        - orientation: +1 if input matches canonical direction, -1 if reversed

    Example:
        canonical_face([3, 1, 4, 2]) → ((1, 4, 2, 3), +1) or ((1, 3, 2, 4), -1)
        depending on which direction is lexicographically smaller.

    Use case:
        For face deduplication while preserving orientation information.
        Two cells sharing a face will have opposite orientations relative to canonical.
    """
    if len(face) < 3:
        raise ValueError(f"Face must have at least 3 vertices, got {len(face)}")

    # Rotate to start at minimum vertex
    min_idx = face.index(min(face))
    rotated = face[min_idx:] + face[:min_idx]

    # Compare forward vs reverse direction (both start at min vertex)
    reversed_rot = [rotated[0]] + rotated[1:][::-1]

    if tuple(rotated) < tuple(reversed_rot):
        return tuple(rotated), +1
    else:
        return tuple(reversed_rot), -1


class MeshContract:
    """
    Documents the required fields for a mesh dict.

    Required fields:
        V : np.ndarray (N×3) or list
            Vertex coordinates
        E : list of tuples (i, j)
            Edges with i < j convention
        F : list of lists
            Faces as cycles of vertex indices

        complex_type : str
            "surface" (2-manifold) or "foam" (3D Plateau)
        faces_per_edge : int
            2 for surface, 3 for foam

    Optional metadata:
        name : str
            Human-readable name
        n_cells : int
            Number of cells (for supercell)
        periodic : bool
            Whether periodic boundary conditions
        seed : int
            Random seed if applicable

    Optional for d₂ support (foams/solids with 3-cells):
        cell_face_incidence : list of list of (face_idx, orientation)
            For each cell, list of (face_idx, ±1) pairs.
            Orientation: +1 if face normal points outward from cell, -1 if inward.
            Enables building d₂ (cell-face boundary operator).

    Face orientation convention:
        Faces stored in F use canonical ordering (min vertex first, lexicographic direction).
        The canonical_face() helper converts any face cycle to this form.
    """

    REQUIRED_FIELDS = ['V', 'E', 'F', 'complex_type', 'faces_per_edge']
    VALID_COMPLEX_TYPES = [COMPLEX_SURFACE, COMPLEX_FOAM, COMPLEX_TILING, COMPLEX_SOLID]


def validate_mesh(mesh: dict, strict: bool = True) -> tuple[bool, list[str]]:
    """
    Validate a mesh dict against the contract.

    Args:
        mesh: The mesh dict to validate
        strict: If True, raise on first error

    Returns:
        (is_valid, list of error messages)
    """
    errors = []

    # Check required fields
    for field in MeshContract.REQUIRED_FIELDS:
        if field not in mesh:
            errors.append(f"Missing required field: {field}")

    if errors and strict:
        raise ValueError(f"Mesh contract violation: {errors}")

    if 'complex_type' in mesh:
        ct = mesh['complex_type']
        if ct not in MeshContract.VALID_COMPLEX_TYPES:
            errors.append(f"Invalid complex_type: {ct}")

        # Check consistency
        if 'faces_per_edge' in mesh:
            expected = FACES_PER_EDGE.get(ct)
            actual = mesh['faces_per_edge']
            if expected and actual != expected:
                errors.append(
                    f"Inconsistent: complex_type={ct} expects "
                    f"faces_per_edge={expected}, got {actual}"
                )

    # Check vertex index bounds in edges
    if 'V' in mesh and 'E' in mesh:
        n_V = len(mesh['V'])
        for idx, (i, j) in enumerate(mesh['E']):
            if i < 0 or i >= n_V or j < 0 or j >= n_V:
                errors.append(f"Edge {idx}: ({i},{j}) has index out of bounds [0, {n_V-1}]")
                break  # Don't spam

    # Check edge convention (i < j)
    if 'E' in mesh:
        for idx, (i, j) in enumerate(mesh['E']):
            if i >= j:
                errors.append(f"Edge {idx}: ({i},{j}) violates i<j convention")
                break  # Don't spam

    # Check face-edge validity: every face segment must be in edge set
    if 'E' in mesh and 'F' in mesh:
        edge_set = set()
        for (i, j) in mesh['E']:
            edge_set.add((min(i, j), max(i, j)))

        n_V = len(mesh['V']) if 'V' in mesh else float('inf')
        for f_idx, face in enumerate(mesh['F']):
            if len(face) < 3:
                errors.append(f"Face {f_idx}: has < 3 vertices")
                continue
            # Check for repeated vertices in face
            if len(face) != len(set(face)):
                errors.append(f"Face {f_idx}: has repeated vertices")
                continue
            # Check vertex index bounds in face
            if any(v < 0 or v >= n_V for v in face):
                errors.append(f"Face {f_idx}: vertex index out of bounds [0, {n_V-1}]")
                continue
            # Check each segment
            n = len(face)
            for k in range(n):
                v1, v2 = face[k], face[(k + 1) % n]
                edge = (min(v1, v2), max(v1, v2))
                if edge not in edge_set:
                    errors.append(f"Face {f_idx}: segment ({v1},{v2}) not in edge list")
                    break  # Don't spam per face

    if errors and strict:
        raise ValueError(f"Mesh contract violation: {errors}")

    return (len(errors) == 0, errors)


def create_mesh(
    V, E, F,
    complex_type: str,
    name: str = "unnamed",
    n_cells: int = 1,
    periodic: bool = False,
    seed: int = None,
    cell_face_incidence: list = None,
    period_L: float = None
) -> dict:
    """
    Helper to create a contract-compliant mesh dict.

    Args:
        V: Vertex coordinates
        E: Edge list [(i,j), ...]
        F: Face list [cycle, ...]
        complex_type: "surface", "foam", or "solid"
        name: Human-readable name
        n_cells: Number of cells
        periodic: Periodic boundaries?
        seed: Random seed if applicable
        cell_face_incidence: For d₂ support. List of lists of (face_idx, ±1).
            For each cell, the faces it contains and their orientations
            (+1 = outward, -1 = inward).
        period_L: For periodic structures, the period length (box size).
            Formula depends on builder:
            - BCC foam: L = 4N
            - SC solid: L = 2N
            - FCC solid: L = 4N

    Returns:
        Contract-compliant mesh dict
    """
    if complex_type not in MeshContract.VALID_COMPLEX_TYPES:
        raise ValueError(f"Invalid complex_type: {complex_type}")

    mesh = {
        'V': np.array(V) if not isinstance(V, np.ndarray) else V,
        'E': list(E),
        'F': list(F),
        'complex_type': complex_type,
        'faces_per_edge': FACES_PER_EDGE[complex_type],
        'name': name,
        'n_cells': n_cells,
        'periodic': periodic,
        'seed': seed,
    }

    # Add cell-face incidence if provided (enables d₂)
    if cell_face_incidence is not None:
        mesh['cell_face_incidence'] = cell_face_incidence

    # Add period length if provided (for periodic structures)
    if period_L is not None:
        mesh['period_L'] = period_L

    # Add derived counts
    mesh['n_V'] = len(mesh['V'])
    mesh['n_E'] = len(mesh['E'])
    mesh['n_F'] = len(mesh['F'])

    # Validate
    validate_mesh(mesh, strict=True)

    return mesh
