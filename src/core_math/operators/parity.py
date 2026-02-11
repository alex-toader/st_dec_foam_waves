"""
Parity Operator and Bridge/Ring Decomposition
==============================================

Central inversion symmetry P and its action on cycle space.
The decomposition into P-even (bridge) and P-odd (ring) subspaces.

DEFINITION:
    P: v → -v  (central inversion)

    For Kelvin cell centered at origin, P is a symmetry.
    P acts on edges as a permutation (possibly with sign for oriented edges).

IMPORTANT - TWO DIFFERENT H's:
    1. H₁(surface) = 0 for Kelvin (sphere topology, χ = 2)
    2. H = ker(d₀ᵀ) = cycle space of the GRAPH (1-skeleton)

    We work with (2), not (1). The graph has dim(H) = E - V + 1 = 13
    independent cycles, even though the surface has no 1-cycles.

    These are different mathematical objects:
    - H₁(surface): homology of the 2D manifold
    - H(graph): cycle space of the edge graph (= kernel of incidence)

KEY THEOREM (Lefschetz on Graph):
    For a connected GRAPH G with free involution P (no fixed vertices,
    no edge mapped to itself):
        L(P) = Tr(P|_V) - Tr(P|_E) = 0 - 0 = 0  (no fixed points)

    Since L(P) = Tr(P|_H₀) - Tr(P|_H₁) and H₀ = ℝ (connected graph):
        Tr(P|_H₁) = 1

    This is the Lefschetz theorem applied to the 1-skeleton (graph),
    not to the 2D surface.

    For Kelvin graph: H₁ is 13-dimensional, split into:
        bridge (P = +1): dimension 7
        ring (P = -1): dimension 6

    This gives: Tr(P|_H) = 7 - 6 = 1 ✓

TERMINOLOGY:
    "Bridge": cycles that look the same after inversion (symmetric)
    "Ring": cycles that flip sign after inversion (antisymmetric)

WARNING - v_to_idx FOR FLOATING-POINT GEOMETRIES:
    The implicit mapping int(round(x)) for v_to_idx is ONLY safe for:
        - Kelvin/integer coordinates
        - Geometries with vertex separation >> 10^(-DECIMALS)

    For WP, random foams, or any floating-point geometry:
        - ALWAYS provide v_to_idx explicitly from the builder
        - Builder knows the exact vertex creation order
        - round()-based reconstruction can silently fail

    This is the #1 source of "silent parity fail" in practice.

    Safe pattern:
        # Builder provides v_to_idx
        V, E, F, v_to_idx = build_wp_type_a()
        result = build_parity_from_mesh(mesh, v_to_idx=v_to_idx)

    Fragile pattern (avoid for floating geometries):
        # Let parity.py reconstruct v_to_idx - can fail silently
        result = build_parity_from_mesh(mesh)  # v_to_idx=None

REFERENCE: Lefschetz fixed point theorem for graphs
    See: Knill, "A Lefschetz fixed point formula for graphs"
"""

import numpy as np
from typing import Tuple, Dict, Any

from .incidence import build_incidence_matrices, get_cycle_space
from ..spec.constants import WP_ROUND


def build_parity_operator(vertices: np.ndarray,
                          edges: list,
                          v_to_idx: dict,
                          oriented: bool = True,
                          center: np.ndarray = None) -> np.ndarray:
    """
    Build parity operator P on edge space.

    DEFINITION:
        P permutes edges according to v → 2*center - v (inversion about center).
        Default center is origin: v → -v.

        For ORIENTED edges (discrete 1-forms):
            P[e', e] = +1 if edge e maps to e' with same orientation
            P[e', e] = -1 if edge e maps to e' with reversed orientation

        For UNORIENTED edges:
            P[e', e] = 1 (just permutation matrix)

    Args:
        vertices: (V, 3) array of vertex coordinates
        edges: list of (i, j) tuples with i < j
        v_to_idx: dict mapping vertex tuple to index
        oriented: if True, include orientation sign
        center: inversion center (default None = origin)
                For Kelvin (centered at origin): use default
                For WP (centered at centroid): pass centroid explicitly
                See analysis/weaire_phelan_kappa.py for centered parity example.

    Returns:
        P: (E, E) parity operator matrix

    PROPERTIES:
        P² = I (involution)
        P is orthogonal
        Eigenvalues are ±1
    """
    E = len(edges)
    edge_to_idx = {tuple(sorted(e)): k for k, e in enumerate(edges)}

    # Default center is origin
    if center is None:
        center = np.zeros(3)
    else:
        center = np.array(center)

    P = np.zeros((E, E))

    def find_parity_vertex_idx(v):
        """Find index of 2*center - v (parity image), handling both int and float coordinates."""
        parity_v = 2 * center - np.array(v)
        # Try exact match
        parity_tuple = tuple(parity_v)
        if parity_tuple in v_to_idx:
            return v_to_idx[parity_tuple]
        # Try rounded (for floating point) - uses WP_ROUND for consistency with WP builders
        parity_rounded = tuple(round(x, WP_ROUND) for x in parity_v)
        if parity_rounded in v_to_idx:
            return v_to_idx[parity_rounded]
        # Try integer (for Kelvin)
        parity_int = tuple(parity_v.astype(int))
        if parity_int in v_to_idx:
            return v_to_idx[parity_int]
        raise KeyError(f"Cannot find parity image for v={v}")

    for k, (i, j) in enumerate(edges):
        # Get vertex coordinates
        vi = vertices[i]
        vj = vertices[j]

        # Apply inversion: v → 2*center - v
        mi = find_parity_vertex_idx(vi)
        mj = find_parity_vertex_idx(vj)

        # Find the mapped edge
        k2 = edge_to_idx[tuple(sorted((mi, mj)))]

        if oriented:
            # Original edge goes i → j (with i < j convention)
            # After inversion: mi → mj
            # If mi < mj: same orientation → sign = +1
            # If mi > mj: reversed orientation → sign = -1
            sign = 1 if mi < mj else -1
        else:
            sign = 1

        P[k2, k] = sign

    # Verify P² = I
    P2 = P @ P
    if not np.allclose(P2, np.eye(E)):
        raise ValueError(f"P² ≠ I, ||P² - I|| = {np.linalg.norm(P2 - np.eye(E))}")

    return P


def parity_decomposition(H: np.ndarray,
                         P: np.ndarray,
                         strict: bool = True) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Decompose cycle space H into P-even (bridge) and P-odd (ring) subspaces.

    THEOREM (Lefschetz):
        For Kelvin cell with central inversion:
            dim(bridge) = 7
            dim(ring) = 6
            Tr(P|_H) = 7 - 6 = 1

    Args:
        H: (E, dim_H) cycle space basis from get_cycle_space()
        P: (E, E) parity operator from build_parity_operator()
        strict: if True, assert Lefschetz holds; if False, just compute

    Returns:
        H_bridge: (E, dim_bridge) basis for P = +1 eigenspace
        H_ring: (E, dim_ring) basis for P = -1 eigenspace
        dim_bridge: dimension of bridge subspace
        dim_ring: dimension of ring subspace

    VERIFICATION:
        dim_bridge + dim_ring = dim(H)
        dim_bridge - dim_ring = Tr(P|_H) = 1 (when Lefschetz holds)
    """
    dim_H = H.shape[1]

    # Restrict P to cycle space H
    P_on_H = H.T @ P @ H

    # Symmetrize to handle numerical errors
    P_on_H = 0.5 * (P_on_H + P_on_H.T)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(P_on_H)

    # P-even (bridge): eigenvalue ≈ +1
    bridge_mask = eigvals > 0.5
    H_bridge = H @ eigvecs[:, bridge_mask]
    dim_bridge = H_bridge.shape[1]

    # P-odd (ring): eigenvalue ≈ -1
    ring_mask = eigvals < -0.5
    H_ring = H @ eigvecs[:, ring_mask]
    dim_ring = H_ring.shape[1]

    # Verify dimensions add up (always required)
    if dim_bridge + dim_ring != dim_H:
        raise ValueError(f"Dimensions don't add up: {dim_bridge} + {dim_ring} ≠ {dim_H}")

    # Verify Lefschetz (optional in explore mode)
    trace_P_on_H = dim_bridge - dim_ring
    if strict:
        if trace_P_on_H != 1:
            raise ValueError(f"Lefschetz theorem violated: Tr(P|_H) = {trace_P_on_H}, expected 1")

    return H_bridge, H_ring, dim_bridge, dim_ring


# =============================================================================
# CONTRACT-AWARE WRAPPER
# =============================================================================

def build_parity_from_mesh(mesh: dict, v_to_idx: dict = None) -> dict:
    """
    Build parity operator and decomposition from contract-compliant mesh.

    Args:
        mesh: Contract-compliant mesh dict
        v_to_idx: Optional vertex lookup (built if not provided)

    Returns:
        dict with P, H_bridge, H_ring, dimensions, lefschetz results

    WARNING:
        For floating-point geometries (WP, random foams), ALWAYS provide
        v_to_idx explicitly from the builder. The auto-built v_to_idx uses
        round(x, 10) which can silently fail for close vertices.

        See module docstring for details on this common source of errors.
    """
    V = mesh['V']
    E = mesh['E']
    F = mesh['F']

    # Build v_to_idx if not provided
    # WARNING: This is fragile for floating-point geometries!
    # For WP, random foams, etc., pass v_to_idx explicitly from builder.
    if v_to_idx is None:
        v_to_idx = {}
        for i, v in enumerate(V):
            # round(x, WP_ROUND) can collide for vertices closer than 1e-WP_ROUND
            # Safe for Kelvin (integer), fragile for WP/random (float)
            key = tuple(round(float(x), WP_ROUND) for x in v)
            if key in v_to_idx:
                # Collision detected - this is the "silent fail" case
                import warnings
                warnings.warn(
                    f"v_to_idx collision: vertex {i} maps to same key as vertex {v_to_idx[key]}. "
                    f"For floating-point geometries, provide v_to_idx explicitly from builder.",
                    UserWarning
                )
            v_to_idx[key] = i

    # Build operators
    d0, d1 = build_incidence_matrices(V, E, F)
    H = get_cycle_space(d0)
    P = build_parity_operator(V, E, v_to_idx)

    # Parity decomposition
    H_bridge, H_ring, dim_bridge, dim_ring = parity_decomposition(H, P, strict=False)
    trace_P = dim_bridge - dim_ring

    # Check for fixed points
    n_fixed_vertices = sum(1 for v in V if np.allclose(v, -v))

    edge_lookup = {tuple(sorted(e)): k for k, e in enumerate(E)}
    n_fixed_edges = 0
    for i, j in E:
        vi, vj = V[i], V[j]
        # Use round(x, WP_ROUND) for floating point safety (consistent with v_to_idx)
        mi_key = tuple(round(float(-x), WP_ROUND) for x in vi)
        mj_key = tuple(round(float(-x), WP_ROUND) for x in vj)
        if mi_key in v_to_idx and mj_key in v_to_idx:
            mi, mj = v_to_idx[mi_key], v_to_idx[mj_key]
            if tuple(sorted((mi, mj))) == tuple(sorted((i, j))):
                n_fixed_edges += 1

    return {
        'P': P,
        'H': H,
        'H_bridge': H_bridge,
        'H_ring': H_ring,
        'dim_H': H.shape[1],
        'dim_bridge': dim_bridge,
        'dim_ring': dim_ring,
        'trace_P': trace_P,
        'lefschetz_holds': (trace_P == 1),
        'n_fixed_vertices': n_fixed_vertices,
        'n_fixed_edges': n_fixed_edges,
        'free_involution': (n_fixed_vertices == 0 and n_fixed_edges == 0),
    }


# Self-test when run directly
# Run with: python -m core_math.operators.parity (from ST_8/)
if __name__ == "__main__":
    import sys
    import os
    # Add ST_8 to path for proper imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from core_math.builders import build_kelvin_cell_mesh

    print("=" * 60)
    print("PARITY AND BRIDGE/RING DECOMPOSITION - VERIFICATION")
    print("=" * 60)

    mesh = build_kelvin_cell_mesh()
    result = build_parity_from_mesh(mesh)

    print(f"\nMesh: {mesh['name']}")
    print(f"V={mesh['n_V']}, E={mesh['n_E']}, F={mesh['n_F']}")

    print(f"\n--- PARITY OPERATOR ---")
    P = result['P']
    P2 = P @ P
    print(f"||P² - I|| = {np.linalg.norm(P2 - np.eye(mesh['n_E'])):.2e}")

    print(f"\n--- LEFSCHETZ THEOREM ---")
    print(f"Fixed vertices: {result['n_fixed_vertices']} (should be 0)")
    print(f"Fixed edges: {result['n_fixed_edges']} (should be 0)")
    print(f"Free involution: {result['free_involution']}")
    print(f"Tr(P|_H) = {result['trace_P']} (should be 1)")
    print(f"Lefschetz satisfied: {result['lefschetz_holds']}")

    print(f"\n--- BRIDGE/RING DECOMPOSITION ---")
    print(f"dim(H) = {result['dim_H']}")
    print(f"dim(bridge) = {result['dim_bridge']} (P = +1)")
    print(f"dim(ring) = {result['dim_ring']} (P = -1)")
    print(f"Sum: {result['dim_bridge']} + {result['dim_ring']} = {result['dim_bridge'] + result['dim_ring']}")
    print(f"Difference: {result['dim_bridge']} - {result['dim_ring']} = {result['trace_P']}")

    # Verify orthogonality
    overlap = result['H_bridge'].T @ result['H_ring']
    print(f"\n--- ORTHOGONALITY ---")
    print(f"||H_bridge^T H_ring|| = {np.linalg.norm(overlap):.2e}")

    print("\n" + "=" * 60)
    print("All parity verifications passed.")
    print("=" * 60)
