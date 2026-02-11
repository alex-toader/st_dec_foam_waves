"""
Weaire-Phelan κ Computation
===========================

Compute κ = 4E - dim(bridge) for Weaire-Phelan cells.

This is in analysis/ layer because it depends on operators (incidence, parity).

WP geometry verified by test suite (tests_math/test_all.py):
    - test_T0_wp_type_a, test_T0_wp_type_b (face-edge validity)
    - test_T9_wp_type_a_planarity, test_T9_wp_type_b_planarity

Date: Jan 2026
"""

import numpy as np
from typing import Dict, List, Tuple, Any

from ..operators.incidence import build_incidence_matrices, get_cycle_space
from ..operators.parity import parity_decomposition
from ..spec.constants import EPS_CLOSE

# Precision for vertex coordinate rounding (matches parity.py pattern)
COORD_DECIMALS = 10


def compute_wp_kappa(cell_type: str = 'A') -> Dict[str, Any]:
    """
    Compute κ = 4E - bridge for Weaire-Phelan cell.

    Args:
        cell_type: 'A' for dodecahedron, 'B' for tetrakaidecahedron

    Returns:
        dict with V, E, F, dim_H, dim_bridge, kappa, plus detailed parity info:
            - is_centrosymmetric: whether all vertices have reflected partners
            - parity_failed: whether parity decomposition failed
            - parity_error: error message if failed
            - n_fixed_vertices: vertices mapped to themselves
            - n_fixed_edges: edges mapped to themselves
            - free_involution: True if no fixed points
            - trace_P: Tr(P|_H) for Lefschetz check

    NOTE: WP geometry verified by test suite (see tests_math/test_all.py).
    """
    from ..builders.weaire_phelan import build_wp_type_a, build_wp_type_b

    if cell_type.upper() == 'A':
        vertices, edges, faces, v_to_idx = build_wp_type_a()
    else:
        vertices, edges, faces, v_to_idx = build_wp_type_b()

    V = len(vertices)
    E = len(edges)
    F = len(faces)
    chi = V - E + F

    d0, d1 = build_incidence_matrices(vertices, edges, faces)
    L1 = d0 @ d0.T + d1.T @ d1
    Tr_L1 = np.trace(L1)

    H = get_cycle_space(d0)
    dim_H = H.shape[1]

    # Build rounded vertex lookup (matches parity.py pattern)
    center = np.mean(vertices, axis=0)
    v_to_idx_rounded = {}
    for i, v in enumerate(vertices):
        key = tuple(round(float(x), COORD_DECIMALS) for x in v)
        v_to_idx_rounded[key] = i

    # Check centrosymmetry via dict lookup (not distance search)
    is_centrosymmetric = True
    v_map = {}  # vertex index -> reflected vertex index
    for i, v in enumerate(vertices):
        neg_v = 2 * center - v
        neg_key = tuple(round(float(x), COORD_DECIMALS) for x in neg_v)
        if neg_key in v_to_idx_rounded:
            v_map[i] = v_to_idx_rounded[neg_key]
        else:
            is_centrosymmetric = False
            break

    # Initialize parity tracking
    parity_failed = False
    parity_error = None
    n_fixed_vertices = 0
    n_fixed_edges = 0
    trace_P = None
    dim_bridge = 0
    dim_ring = dim_H

    if is_centrosymmetric:
        try:
            # Count fixed points
            n_fixed_vertices = sum(1 for i in range(V) if v_map[i] == i)

            edge_to_idx = {tuple(sorted(e)): k for k, e in enumerate(edges)}
            for i, j in edges:
                mi, mj = v_map[i], v_map[j]
                if tuple(sorted((mi, mj))) == tuple(sorted((i, j))):
                    n_fixed_edges += 1

            # Build parity operator
            P = _build_centered_parity_from_map(vertices, edges, v_map)

            # Parity decomposition
            H_bridge, H_ring, dim_bridge, dim_ring = parity_decomposition(H, P, strict=False)
            trace_P = dim_bridge - dim_ring

        except Exception as e:
            parity_failed = True
            parity_error = str(e)
            dim_bridge = 0
            dim_ring = dim_H

    kappa = 4 * E - dim_bridge
    free_involution = (n_fixed_vertices == 0 and n_fixed_edges == 0) if is_centrosymmetric else False

    return {
        'cell_type': cell_type,
        'V': V,
        'E': E,
        'F': F,
        'chi': chi,
        'Tr_L1': Tr_L1,
        'dim_H': dim_H,
        'dim_bridge': dim_bridge,
        'dim_ring': dim_ring,
        'kappa': kappa,
        'is_centrosymmetric': is_centrosymmetric,
        'parity_failed': parity_failed,
        'parity_error': parity_error,
        'n_fixed_vertices': n_fixed_vertices,
        'n_fixed_edges': n_fixed_edges,
        'free_involution': free_involution,
        'trace_P': trace_P,
    }


def _build_centered_parity_from_map(vertices: np.ndarray, edges: List[Tuple[int, int]],
                                    v_map: Dict[int, int]) -> np.ndarray:
    """
    Build parity operator for reflection using pre-computed vertex map.

    Args:
        vertices: (V, 3) array
        edges: list of (i, j) tuples
        v_map: dict mapping vertex index to its reflected partner index

    Returns:
        P: (E, E) parity operator matrix

    PROPERTIES:
        P² = I (involution)
        P is orthogonal
    """
    n_E = len(edges)
    edge_to_idx = {tuple(sorted(e)): k for k, e in enumerate(edges)}

    P = np.zeros((n_E, n_E))
    for k, (i, j) in enumerate(edges):
        mi, mj = v_map[i], v_map[j]
        mapped_edge = tuple(sorted((mi, mj)))
        if mapped_edge not in edge_to_idx:
            raise ValueError(
                f"Edge ({i},{j}) maps to ({mi},{mj}) which is not an edge. "
                f"Parity is NOT a graph automorphism for this polyhedron."
            )
        k2 = edge_to_idx[mapped_edge]
        # Sign: +1 if orientation preserved (mi < mj), -1 if reversed
        sign = 1 if mi < mj else -1
        P[k2, k] = sign

    # Verify P² = I
    P2 = P @ P
    if not np.allclose(P2, np.eye(n_E), atol=EPS_CLOSE):
        raise ValueError(f"P² ≠ I, ||P² - I|| = {np.linalg.norm(P2 - np.eye(n_E))}")

    return P


def compare_wp_kelvin() -> None:
    """Compare WP cells with Kelvin cell."""
    from ..builders.kelvin import build_kelvin_cell
    from .kappa import compute_kappa_for_polyhedron

    print("=" * 70)
    print("WEAIRE-PHELAN vs KELVIN COMPARISON")
    print("=" * 70)

    # Kelvin
    kelvin = compute_kappa_for_polyhedron(*build_kelvin_cell())
    print(f"\nKelvin cell:")
    print(f"  V={kelvin['V']}, E={kelvin['E']}, F={kelvin['F']}")
    print(f"  bridge={kelvin['dim_bridge']}, ring={kelvin['dim_ring']}")
    print(f"  κ = 4×{kelvin['E']} - {kelvin['dim_bridge']} = {kelvin['kappa']}")

    def _print_wp_result(wp: Dict[str, Any], name: str) -> None:
        """Print WP result with full parity diagnostics."""
        print(f"\n{name}:")
        print(f"  V={wp['V']}, E={wp['E']}, F={wp['F']}")
        print(f"  Centrosymmetric: {wp['is_centrosymmetric']}")
        if wp['is_centrosymmetric']:
            print(f"  Fixed vertices: {wp['n_fixed_vertices']}, Fixed edges: {wp['n_fixed_edges']}")
            print(f"  Free involution: {wp['free_involution']}")
            if wp['parity_failed']:
                print(f"  Parity FAILED: {wp['parity_error']}")
            else:
                print(f"  Tr(P|_H) = {wp['trace_P']}")
        print(f"  bridge={wp['dim_bridge']}, ring={wp['dim_ring']}")
        print(f"  κ = 4×{wp['E']} - {wp['dim_bridge']} = {wp['kappa']}")

    # WP Type A
    try:
        wp_a = compute_wp_kappa('A')
        _print_wp_result(wp_a, "WP Type A (dodecahedron)")
    except Exception as e:
        print(f"\nWP Type A: Error - {e}")

    # WP Type B
    try:
        wp_b = compute_wp_kappa('B')
        _print_wp_result(wp_b, "WP Type B (tetrakaidecahedron)")
    except Exception as e:
        print(f"\nWP Type B: Error - {e}")

    # Weighted average for WP foam
    print("\n" + "-" * 70)
    print("WP FOAM AVERAGE (2 Type A + 6 Type B per unit cell):")
    print("-" * 70)


if __name__ == "__main__":
    compare_wp_kelvin()
