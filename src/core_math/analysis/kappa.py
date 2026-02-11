"""
κ (Kappa) Computation
=====================

Compute κ = 4E - dim(bridge) for polyhedra.

This is in analysis/ layer because it depends on operators (incidence, parity).

Date: Jan 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Any


def compute_kappa_for_polyhedron(vertices: np.ndarray,
                                  edges: List[Tuple[int, int]],
                                  faces: List[List[int]],
                                  v_to_idx: Dict[tuple, int]) -> Dict[str, Any]:
    """
    Compute κ = 4E - dim(bridge) for any centrosymmetric polyhedron.

    Args:
        vertices, edges, faces, v_to_idx from build_* functions

    Returns:
        dict with V, E, F, dim_H, dim_bridge, dim_ring, kappa

    NOTE: If polyhedron is NOT centrosymmetric, returns dim_bridge = 0.
    """
    from ..operators.incidence import build_incidence_matrices, get_cycle_space
    from ..operators.parity import build_parity_operator, parity_decomposition

    V = len(vertices)
    E = len(edges)
    F = len(faces)
    chi = V - E + F

    d0, d1 = build_incidence_matrices(vertices, edges, faces)
    H = get_cycle_space(d0)
    dim_H = H.shape[1]

    # Check if centrosymmetric
    # Handle both integer and floating point coordinates
    is_centrosymmetric = True
    for v in vertices:
        neg_v = -np.array(v)
        # Try exact match first
        neg_v_tuple = tuple(neg_v)
        if neg_v_tuple in v_to_idx:
            continue
        # Try rounded (for floating point)
        neg_v_rounded = tuple(round(x, 10) for x in neg_v)
        if neg_v_rounded in v_to_idx:
            continue
        # Try integer (for Kelvin)
        neg_v_int = tuple(neg_v.astype(int))
        if neg_v_int in v_to_idx:
            continue
        # Not found
        is_centrosymmetric = False
        break

    if is_centrosymmetric:
        try:
            P = build_parity_operator(vertices, edges, v_to_idx)
            H_bridge, H_ring, dim_bridge, dim_ring = parity_decomposition(H, P)
        except (ValueError, AssertionError) as e:
            # Parity decomposition failed - don't invent dimensions, fail explicitly
            raise ValueError(f"Parity decomposition failed for centrosymmetric polyhedron: {e}")
    else:
        # No parity symmetry
        dim_bridge = 0
        dim_ring = dim_H

    kappa = 4 * E - dim_bridge

    return {
        'V': V, 'E': E, 'F': F, 'chi': chi,
        'dim_H': dim_H,
        'dim_bridge': dim_bridge,
        'dim_ring': dim_ring,
        'kappa': kappa,
        'is_centrosymmetric': is_centrosymmetric
    }
