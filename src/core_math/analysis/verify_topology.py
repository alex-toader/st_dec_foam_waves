"""
Topology Verification Functions
===============================

Verify structural properties using DEC operators (d1 matrix).

These functions are in analysis/ layer because they depend on operators.

Date: Jan 2026
"""

import numpy as np
from typing import Dict

from ..spec.constants import EPS_CLOSE


def verify_sc_solid_structure(d1: np.ndarray, E: int) -> Dict:
    """
    Verify SC solid structure (4 faces per edge, NOT foam).

    Args:
        d1: (F, E) incidence matrix
        E: number of edges

    Returns:
        dict with verification results
    """
    # Count how many faces each edge bounds
    faces_per_edge = np.sum(np.abs(d1), axis=0)

    all_bound_4 = np.all(faces_per_edge == 4)
    min_bound = np.min(faces_per_edge)
    max_bound = np.max(faces_per_edge)

    # Trace theorem for SC solid: Tr(d₁ᵀd₁) = 4E
    d1td1 = d1.T @ d1
    trace = np.trace(d1td1)
    expected_trace = 4 * E
    trace_ok = abs(trace - expected_trace) < EPS_CLOSE

    return {
        'is_sc_solid': all_bound_4,
        'min_faces_per_edge': int(min_bound),
        'max_faces_per_edge': int(max_bound),
        'trace_d1td1': int(trace),
        'expected_trace': expected_trace,
        'sc_trace_theorem_holds': trace_ok
    }


def verify_fcc_structure(d1: np.ndarray, E: int) -> Dict:
    """
    Verify FCC structure (3 faces per edge, same k as foam).

    Args:
        d1: (F, E) incidence matrix
        E: number of edges

    Returns:
        dict with verification results
    """
    faces_per_edge = np.sum(np.abs(d1), axis=0)

    all_bound_3 = np.all(faces_per_edge == 3)
    min_bound = np.min(faces_per_edge)
    max_bound = np.max(faces_per_edge)

    # Trace theorem for FCC: Tr(d₁ᵀd₁) = 3E (same k as foam)
    d1td1 = d1.T @ d1
    trace = np.trace(d1td1)
    expected_trace = 3 * E
    trace_ok = abs(trace - expected_trace) < EPS_CLOSE

    return {
        'is_fcc_k3': all_bound_3,  # k=3 tiling (same trace as foam, different physics)
        'min_faces_per_edge': int(min_bound),
        'max_faces_per_edge': int(max_bound),
        'trace_d1td1': int(trace),
        'expected_trace': expected_trace,
        'fcc_trace_theorem_holds': trace_ok
    }
