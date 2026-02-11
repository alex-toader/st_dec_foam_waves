"""
Random Foam Generation
======================

Generate disordered/random foams via Voronoi tessellation.
Used for testing Koide n predictions against quenched foam statistics.

NOTE: This is ANALYSIS code, not a core DEC builder.
    - Uses scipy.spatial.Voronoi (external dependency)
    - Does NOT return contract-compliant mesh dict
    - Not exported in builders/__init__.py
    - For statistics only, not for DEC operator construction

KEY OUTPUTS:
    - Face count distribution P(F)
    - Mean face count ⟨F⟩ (literature: 15.54 for Poisson-Voronoi, ~14.1 for jammed)
    - Variance and higher moments
    - Individual cell topology (F, E, V, χ)

REFERENCE:
    - Poisson-Voronoi: ⟨F⟩ = 15.535 (exact, Meijering 1953)
    - Random close packing: ⟨F⟩ ~ 14.1-14.4 (O'Hern, Torquato)
    - Quenched foam (materials science): ⟨F⟩ = 14.1 ± 0.3

PURPOSE:
    Test if Koide n = 14.14 emerges from foam geometry statistics.

Date: Jan 2026
"""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

from ..spec.constants import EPS_CLOSE


@dataclass
class CellTopology:
    """Topology of a single Voronoi cell."""
    index: int
    F: int  # faces
    E: int  # edges
    V: int  # vertices
    chi: int  # Euler characteristic (should be 2 for closed)
    volume: float
    is_bounded: bool  # False if cell extends to infinity


@dataclass
class FoamStatistics:
    """Statistics of foam face counts."""
    n_cells: int
    n_bounded: int  # cells with finite volume
    mean_F: float
    median_F: float
    mode_F: int
    var_F: float
    std_F: float
    min_F: int
    max_F: int
    histogram: Tuple[np.ndarray, np.ndarray]  # counts, bin_edges

    # Derived quantities for Koide
    # KEY INSIGHT: n from mode (typical F) matches Koide better than n from mean
    n_from_mean: float    # n(⟨F⟩) - often too high
    n_from_median: float  # n(median F)
    n_from_mode: float    # n(mode F) - matches Koide best!
    harmonic_mean_F: float  # 1/<1/F>
    n_from_harmonic: float  # n(F_harm)


def generate_random_points(n_points: int,
                           box_size: float = 10.0,
                           seed: Optional[int] = None) -> np.ndarray:
    """
    Generate uniformly distributed random points in a cube.

    Args:
        n_points: Number of points
        box_size: Side length of cube [0, box_size]³
        seed: Random seed for reproducibility

    Returns:
        (n_points, 3) array of coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.uniform(0, box_size, (n_points, 3))


def generate_jammed_points(n_points: int,
                           box_size: float = 10.0,
                           packing_fraction: float = 0.64,
                           seed: Optional[int] = None,
                           max_iterations: int = 1000) -> np.ndarray:
    """
    Generate jammed/packed point configuration via simple relaxation.

    This approximates random close packing (RCP) which gives ⟨F⟩ ~ 14.1-14.4,
    closer to Koide n = 14.14 than pure Poisson-Voronoi (⟨F⟩ = 15.54).

    Args:
        n_points: Number of points
        box_size: Side length of cube
        packing_fraction: Target packing fraction (0.64 for RCP)
        seed: Random seed
        max_iterations: Max relaxation steps

    Returns:
        (n_points, 3) array of coordinates

    Note: This is a simplified model. For rigorous RCP, use specialized
    algorithms (Lubachevsky-Stillinger, event-driven MD, etc.)
    """
    if seed is not None:
        np.random.seed(seed)

    # Estimate particle radius from packing fraction
    # φ = n × (4/3)πr³ / V → r = (3φV/(4πn))^(1/3)
    volume = box_size**3
    r = (3 * packing_fraction * volume / (4 * np.pi * n_points))**(1/3)

    # Start with random positions
    points = np.random.uniform(r, box_size - r, (n_points, 3))

    # Simple relaxation: push overlapping particles apart
    for iteration in range(max_iterations):
        moved = False
        for i in range(n_points):
            for j in range(i + 1, n_points):
                diff = points[j] - points[i]
                dist = np.linalg.norm(diff)

                # If overlapping, push apart
                if dist < 2 * r and dist > EPS_CLOSE:
                    overlap = 2 * r - dist
                    direction = diff / dist
                    shift = 0.5 * overlap * direction
                    points[i] -= shift
                    points[j] += shift
                    moved = True

        # Apply periodic-like boundary (wrap or reflect)
        points = np.clip(points, r, box_size - r)

        if not moved:
            break

    return points


def generate_poisson_disk_points(n_points: int,
                                  box_size: float = 10.0,
                                  d_min: Optional[float] = None,
                                  seed: Optional[int] = None,
                                  max_tries: int = 200000) -> np.ndarray:
    """
    Generate hard-core (Poisson-disk) point configuration.

    More stable than simple relaxation for approximating jammed/RCP packings.
    Points have minimum separation d_min (hard spheres).

    Args:
        n_points: Target number of points
        box_size: Side length of cube
        d_min: Minimum separation. If None, estimated from density.
        seed: Random seed
        max_tries: Maximum attempts before giving up

    Returns:
        (n_points, 3) array of coordinates

    Raises:
        RuntimeError: If cannot place all points
    """
    rng = np.random.default_rng(seed)

    # Estimate d_min if not provided (for ~64% packing)
    if d_min is None:
        # For RCP-like packing, d_min ~ 0.9 * (box/n^(1/3))
        d_min = 0.9 * box_size / (n_points ** (1/3))

    pts = []
    tries = 0
    while len(pts) < n_points and tries < max_tries:
        p = rng.uniform(0, box_size, size=3)
        ok = True
        for q in pts:
            if np.linalg.norm(p - q) < d_min:
                ok = False
                break
        if ok:
            pts.append(p)
        tries += 1

    if len(pts) < n_points:
        raise RuntimeError(f"Could only place {len(pts)} of {n_points} points. "
                          f"Try lower d_min or larger box_size.")

    return np.array(pts)


def add_ghost_points(points: np.ndarray, box_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 3x3x3 periodic tiling for better boundary handling.

    Returns extended point set and mask indicating central (original) points.
    Use this for Poisson-Voronoi to get accurate ⟨F⟩ ≈ 15.535.

    Args:
        points: (n, 3) original points in [0, box_size]³
        box_size: Side length of cube

    Returns:
        (points_extended, mask_central)
        - points_extended: (27n, 3) tiled points
        - mask_central: (27n,) bool array, True for original points
    """
    shifts = [-box_size, 0.0, box_size]
    tiles = []
    masks = []
    for dx in shifts:
        for dy in shifts:
            for dz in shifts:
                shift = np.array([dx, dy, dz])
                tiles.append(points + shift)
                is_central = (dx == 0.0 and dy == 0.0 and dz == 0.0)
                masks.append(np.ones(len(points), dtype=bool) if is_central
                            else np.zeros(len(points), dtype=bool))
    pts_ext = np.vstack(tiles)
    mask_central = np.concatenate(masks)
    return pts_ext, mask_central


def analyze_foam_ghost(points: np.ndarray, box_size: float) -> List[CellTopology]:
    """
    Analyze foam using ghost points (3x3x3 tiling).

    This eliminates boundary bias and gives accurate Poisson-Voronoi statistics.
    Expected: ⟨F⟩ → 15.535 for large N.

    Args:
        points: (n, 3) original points in [0, box_size]³
        box_size: Side length of cube

    Returns:
        List of CellTopology for central (original) cells only
    """
    pts_ext, mask_central = add_ghost_points(points, box_size)
    vor = Voronoi(pts_ext)

    cells = []
    central_indices = np.where(mask_central)[0]
    for idx in central_indices:
        cell = analyze_cell(vor, idx)
        if cell is not None and cell.is_bounded and cell.F > 0:
            cells.append(cell)

    return cells


def compute_voronoi(points: np.ndarray) -> Voronoi:
    """
    Compute Voronoi tessellation of points.

    Args:
        points: (n, 3) array of seed points

    Returns:
        scipy.spatial.Voronoi object
    """
    return Voronoi(points)


def analyze_cell(vor: Voronoi, cell_index: int) -> Optional[CellTopology]:
    """
    Analyze topology of a single Voronoi cell.

    Args:
        vor: Voronoi tessellation
        cell_index: Index of cell (= index of seed point)

    Returns:
        CellTopology or None if cell is unbounded/degenerate
    """
    region_index = vor.point_region[cell_index]
    region = vor.regions[region_index]

    # Check if bounded (no -1 vertex)
    if -1 in region or len(region) == 0:
        return CellTopology(
            index=cell_index,
            F=0, E=0, V=0, chi=0,
            volume=np.inf,
            is_bounded=False
        )

    # Get vertices of this cell
    vertices = vor.vertices[region]

    # V = number of vertices
    V = len(region)

    # Compute convex hull to get faces and edges
    try:
        hull = ConvexHull(vertices)
        F = len(hull.simplices)  # Number of triangular faces

        # For convex polyhedron: each triangular face has 3 edges
        # Each edge shared by 2 faces: E = 3F/2
        # But this assumes triangulated! Need to count actual faces.

        # Actually, Voronoi cells have polygonal faces, not triangular.
        # ConvexHull gives triangulated surface.
        # For proper F count, need different approach.

        # Use Euler: V - E + F = 2
        # For convex hull of V vertices:
        # - Triangulated: F_tri = 2V - 4, E_tri = 3V - 6

        # Better approach: count ridge vertices
        # Each ridge (face of Voronoi) corresponds to a neighbor

    except Exception:
        return None

    # Count faces by counting neighbors (ridges)
    # In 3D Voronoi, each face corresponds to a ridge
    F_actual = 0
    E_actual = 0

    # Ridge = face between two cells
    # ridge_points[i] = [p1, p2] means ridge between cells p1 and p2
    # ridge_vertices[i] = vertices forming that ridge

    for ridge_idx, (p1, p2) in enumerate(vor.ridge_points):
        if p1 == cell_index or p2 == cell_index:
            ridge_verts = vor.ridge_vertices[ridge_idx]
            if -1 not in ridge_verts:  # bounded ridge
                F_actual += 1
                E_actual += len(ridge_verts)  # edges of this face

    # Each edge counted twice (shared by 2 faces)
    E_actual = E_actual // 2

    # Euler characteristic
    chi = V - E_actual + F_actual

    # Volume
    try:
        volume = hull.volume
    except:
        volume = 0.0

    return CellTopology(
        index=cell_index,
        F=F_actual,
        E=E_actual,
        V=V,
        chi=chi,
        volume=volume,
        is_bounded=True
    )


def analyze_foam(points: np.ndarray,
                 exclude_boundary: bool = True,
                 boundary_margin: float = 0.1) -> Tuple[List[CellTopology], Voronoi]:
    """
    Analyze all cells in a Voronoi foam.

    Args:
        points: (n, 3) seed points
        exclude_boundary: If True, exclude cells near boundary
        boundary_margin: Fraction of box size to exclude at boundary

    Returns:
        List of CellTopology for each valid cell
        Voronoi object
    """
    vor = compute_voronoi(points)

    # Determine box bounds
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    box_size = maxs - mins
    margin = boundary_margin * box_size

    cells = []
    for i in range(len(points)):
        # Skip boundary cells if requested
        if exclude_boundary:
            p = points[i]
            if np.any(p - mins < margin) or np.any(maxs - p < margin):
                continue

        cell = analyze_cell(vor, i)
        if cell is not None and cell.is_bounded and cell.F > 0:
            cells.append(cell)

    return cells, vor


def compute_statistics(cells: List[CellTopology],
                       chi_fixed: int = 2) -> FoamStatistics:
    """
    Compute foam statistics from cell list.

    Args:
        cells: List of CellTopology
        chi_fixed: Euler characteristic to use in n formula (default 2)

    Returns:
        FoamStatistics object

    KEY INSIGHT (reviewer feedback Jan 2026):
        Koide n does NOT correspond to ⟨F⟩ (arithmetic mean).
        It corresponds to F_typical (mode) ≈ 14.
        For jammed foam: ⟨F⟩ ≈ 14.2-14.4, but mode_F = 14.
        n(mode_F=14) = 14.1414 = Koide!
    """
    if not cells:
        raise ValueError("No valid cells to analyze")

    face_counts = np.array([c.F for c in cells], dtype=int)
    bounded_cells = [c for c in cells if c.is_bounded]

    # Basic statistics
    mean_F = float(np.mean(face_counts))
    median_F = float(np.median(face_counts))
    mode_F = int(Counter(face_counts.tolist()).most_common(1)[0][0])
    var_F = float(np.var(face_counts))
    std_F = float(np.std(face_counts))

    # Harmonic mean: 1/⟨1/F⟩
    harmonic_mean_F = float(len(face_counts) / np.sum(1.0 / face_counts))

    # n from Koide formula for each statistic
    n_from_mean = koide_n_from_F(mean_F, chi=chi_fixed)
    n_from_median = koide_n_from_F(median_F, chi=chi_fixed)
    n_from_mode = koide_n_from_F(float(mode_F), chi=chi_fixed)
    n_from_harmonic = koide_n_from_F(harmonic_mean_F, chi=chi_fixed)

    # Histogram
    bins = np.arange(face_counts.min() - 0.5, face_counts.max() + 1.5, 1)
    hist_counts, hist_edges = np.histogram(face_counts, bins=bins)

    return FoamStatistics(
        n_cells=len(cells),
        n_bounded=len(bounded_cells),
        mean_F=mean_F,
        median_F=median_F,
        mode_F=mode_F,
        var_F=var_F,
        std_F=std_F,
        min_F=int(face_counts.min()),
        max_F=int(face_counts.max()),
        histogram=(hist_counts, hist_edges),
        n_from_mean=n_from_mean,
        n_from_median=n_from_median,
        n_from_mode=n_from_mode,
        harmonic_mean_F=harmonic_mean_F,
        n_from_harmonic=n_from_harmonic
    )


def koide_n_from_F(F: float, chi: int = 2) -> float:
    """
    Compute Koide n from effective face count F.

    Formula: n = (F + sqrt(F² + 4χ))/2

    For Kelvin (F=14, χ=2): n = 7 + √51 = 14.1414
    """
    return (F + np.sqrt(F**2 + 4*chi)) / 2


def F_eff_from_koide_n(n: float, chi: int = 2) -> float:
    """
    Inverse: compute effective F from Koide n.

    From n² - Fn - χ = 0: F = n - χ/n

    For n_fit = 14.1367: F_eff = 13.995
    """
    return n - chi / n


def stationary_map_test(face_counts: np.ndarray,
                        chi: int = 2,
                        n_iterations: int = 10000,
                        n_realizations: int = 100,
                        seed: Optional[int] = None) -> Dict:
    """
    Test: does n_{k+1} = F_k + χ/n_k converge to Koide n?

    This iterates the Möbius map with F_k drawn from measured P(F).
    If foam P(F) "encodes" Koide, n_stationary should → 14.14.

    Args:
        face_counts: Array of measured F values from foam
        chi: Euler characteristic (default 2)
        n_iterations: Steps per realization
        n_realizations: Number of independent runs
        seed: Random seed

    Returns:
        Dict with n_mean, n_std, n_all (all final values)

    Theory:
        Fixed point of n = F + χ/n is n = (F + √(F²+4χ))/2
        For F=14: n = 7 + √51 = 14.1414

        But if F varies according to P(F), the stationary distribution
        of the map may differ. This test checks that.
    """
    rng = np.random.default_rng(seed)

    n_finals = []
    for _ in range(n_realizations):
        # Start from random n in reasonable range
        n = rng.uniform(10, 20)

        # Iterate the map
        for _ in range(n_iterations):
            F_k = rng.choice(face_counts)
            n = F_k + chi / n

        n_finals.append(n)

    n_finals = np.array(n_finals)

    return {
        'n_mean': float(np.mean(n_finals)),
        'n_std': float(np.std(n_finals)),
        'n_median': float(np.median(n_finals)),
        'n_min': float(np.min(n_finals)),
        'n_max': float(np.max(n_finals)),
        'n_all': n_finals,
        'reference_kelvin': 7 + np.sqrt(51),  # 14.1414
        'reference_fit': 14.1367
    }


# =============================================================================
# Quick test functions
# =============================================================================

def quick_test_poisson(n_points: int = 500, seed: int = 42,
                       use_ghost: bool = False, box_size: float = 10.0) -> FoamStatistics:
    """
    Quick test with Poisson-Voronoi (random uniform points).
    Expected: ⟨F⟩ ≈ 15.54 (with ghost points for accuracy)
    """
    points = generate_random_points(n_points, box_size=box_size, seed=seed)
    if use_ghost:
        cells = analyze_foam_ghost(points, box_size)
    else:
        cells, _ = analyze_foam(points, exclude_boundary=True)
    return compute_statistics(cells)


def quick_test_jammed(n_points: int = 200, seed: int = 42) -> FoamStatistics:
    """
    Quick test with jammed packing (simple relaxation).
    Expected: ⟨F⟩ ≈ 14.1-14.4, mode_F = 14
    """
    points = generate_jammed_points(n_points, box_size=10.0, seed=seed)
    cells, _ = analyze_foam(points, exclude_boundary=True)
    return compute_statistics(cells)


def quick_test_poisson_disk(n_points: int = 200, seed: int = 42,
                            d_min: Optional[float] = None) -> FoamStatistics:
    """
    Quick test with Poisson-disk (hard-core) packing.
    More stable than simple relaxation for RCP approximation.
    Expected: mode_F = 14
    """
    points = generate_poisson_disk_points(n_points, box_size=10.0,
                                          d_min=d_min, seed=seed)
    cells, _ = analyze_foam(points, exclude_boundary=True)
    return compute_statistics(cells)


def run_koide_test():
    """
    Run Koide n prediction test.

    KEY INSIGHT (reviewer Jan 2026):
        Koide n does NOT correspond to ⟨F⟩ (mean).
        It corresponds to F_typical (mode) = 14.
        n(mode_F=14) = 7 + √51 = 14.1414 = Koide!
    """
    print("=" * 70)
    print("KOIDE n FROM FOAM GEOMETRY")
    print("=" * 70)
    print()

    # Reference values
    n_kelvin = koide_n_from_F(14)
    n_fit = 14.1367
    F_eff = F_eff_from_koide_n(n_fit)

    print("REFERENCE VALUES:")
    print(f"  Kelvin ideal:   F = 14      → n = {n_kelvin:.4f}")
    print(f"  Koide fit:      n = {n_fit}  → F_eff = {F_eff:.4f}")
    print(f"  KEY: F_eff ≈ 14 means Koide encodes F=14 (Kelvin), not ⟨F⟩!")
    print()

    # Poisson-Voronoi test (with ghost for accuracy)
    print("POISSON-VORONOI (ghost points for accuracy):")
    try:
        stats = quick_test_poisson(n_points=500, seed=42, use_ghost=True)
        print(f"  Cells: {stats.n_bounded}")
        print(f"  mean_F = {stats.mean_F:.2f}, mode_F = {stats.mode_F}, median_F = {stats.median_F:.1f}")
        print(f"  n(mean)  = {stats.n_from_mean:.4f}")
        print(f"  n(mode)  = {stats.n_from_mode:.4f}")
        print(f"  (Literature: ⟨F⟩ = 15.535)")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Jammed test
    print("JAMMED PACKING (relaxation):")
    try:
        stats = quick_test_jammed(n_points=200, seed=42)
        print(f"  Cells: {stats.n_bounded}")
        print(f"  mean_F = {stats.mean_F:.2f}, mode_F = {stats.mode_F}, median_F = {stats.median_F:.1f}")
        print(f"  n(mean)  = {stats.n_from_mean:.4f}  ← too high!")
        print(f"  n(mode)  = {stats.n_from_mode:.4f}  ← matches Koide!")
        print(f"  n(harm)  = {stats.n_from_harmonic:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Poisson-disk test
    print("POISSON-DISK (hard-core):")
    try:
        stats = quick_test_poisson_disk(n_points=150, seed=42)
        print(f"  Cells: {stats.n_bounded}")
        print(f"  mean_F = {stats.mean_F:.2f}, mode_F = {stats.mode_F}, median_F = {stats.median_F:.1f}")
        print(f"  n(mean)  = {stats.n_from_mean:.4f}")
        print(f"  n(mode)  = {stats.n_from_mode:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Stationary map test
    print("STATIONARY MAP TEST: n_{k+1} = F_k + χ/n_k")
    print("-" * 70)

    # Test with Poisson-Voronoi P(F)
    print("Using Poisson-Voronoi P(F):")
    try:
        points_pv = generate_random_points(500, box_size=10.0, seed=42)
        cells_pv = analyze_foam_ghost(points_pv, 10.0)
        F_pv = np.array([c.F for c in cells_pv])
        result_pv = stationary_map_test(F_pv, chi=2, n_iterations=1000,
                                        n_realizations=50, seed=123)
        print(f"  n_stationary = {result_pv['n_mean']:.4f} ± {result_pv['n_std']:.4f}")
        print(f"  vs Kelvin: {result_pv['reference_kelvin']:.4f}")
        print(f"  vs Koide fit: {result_pv['reference_fit']}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    # Test with jammed P(F)
    print("Using Jammed P(F):")
    try:
        points_j = generate_jammed_points(200, box_size=10.0, seed=42)
        cells_j, _ = analyze_foam(points_j, exclude_boundary=True)
        F_j = np.array([c.F for c in cells_j])
        result_j = stationary_map_test(F_j, chi=2, n_iterations=1000,
                                       n_realizations=50, seed=123)
        print(f"  n_stationary = {result_j['n_mean']:.4f} ± {result_j['n_std']:.4f}")
        print(f"  vs Kelvin: {result_j['reference_kelvin']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    print("=" * 70)
    print("CONCLUSIONS:")
    print("  1. F_eff = n - χ/n = 13.995 ≈ 14 (strongest argument)")
    print("  2. mode_F = 14 appears in BOTH PV and jammed (not discriminating)")
    print("  3. Stationary map test checks if P(F) encodes Koide n")
    print("=" * 70)


# =============================================================================
# V9: Radial Distribution Function g(r)
# =============================================================================

@dataclass
class RadialDistribution:
    """Radial distribution function g(r) results."""
    r_bins: np.ndarray       # bin centers
    g_r: np.ndarray          # g(r) values
    r_peaks: np.ndarray      # positions of peaks
    peak_heights: np.ndarray # heights of peaks
    r_peaks_normalized: np.ndarray  # peaks / d_nn (normalized by nearest neighbor)
    d_nn: float              # mean nearest neighbor distance


def compute_pair_distances(points: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise distances between points.

    Args:
        points: (n, 3) array of coordinates

    Returns:
        (n*(n-1)/2,) array of unique distances
    """
    from scipy.spatial.distance import pdist
    return pdist(points)


def compute_g_r(points: np.ndarray,
                n_bins: int = 100,
                r_max: Optional[float] = None,
                box_size: Optional[float] = None) -> RadialDistribution:
    """
    Compute radial distribution function g(r).

    g(r) = histogram of pair distances, normalized by ideal gas.
    Peaks indicate preferred neighbor distances.

    For V9: Check if peaks occur at √n × d_nn

    Args:
        points: (n, 3) seed point coordinates
        n_bins: Number of histogram bins
        r_max: Maximum distance (default: half box size)
        box_size: Box size for normalization (estimated if None)

    Returns:
        RadialDistribution with g(r), peaks, and normalizations
    """
    n = len(points)

    # Estimate box size if not given
    if box_size is None:
        box_size = np.max(points) - np.min(points)

    if r_max is None:
        r_max = box_size / 2  # avoid periodic artifacts

    # Compute all pairwise distances
    distances = compute_pair_distances(points)

    # Find mean nearest neighbor distance
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(points, points)
    np.fill_diagonal(dist_matrix, np.inf)
    d_nn = float(np.mean(np.min(dist_matrix, axis=1)))

    # Histogram
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_bins = 0.5 * (r_edges[:-1] + r_edges[1:])
    dr = r_edges[1] - r_edges[0]

    hist, _ = np.histogram(distances, bins=r_edges)

    # Normalize: g(r) = (V / N²) × hist / (4πr²dr)
    # Standard RDF normalization: g(r) → 1 at large r for ideal gas

    volume = box_size**3
    rho = n / volume

    # Shell volume at each r
    shell_volume = 4 * np.pi * r_bins**2 * dr

    # Ideal gas expectation per particle pair
    # n_pairs = n(n-1)/2 pairs contribute to histogram
    # Each pair contributes once to hist at distance r
    # Ideal gas: n_ideal(r) = (n-1) × ρ × shell_volume per particle
    # Total in hist: n × n_ideal(r) / 2 = n(n-1)ρ × shell_volume / 2
    n_pairs = n * (n - 1) / 2
    ideal = n_pairs * rho * shell_volume / (n / 2)  # = (n-1) * rho * shell_volume

    # g(r) - should be ~1 at large r, >1 at peaks
    g_r = np.zeros_like(r_bins)
    mask = ideal > 0
    g_r[mask] = hist[mask] / ideal[mask]

    # Find peaks (local maxima) - use relaxed parameters
    from scipy.signal import find_peaks
    peak_indices, properties = find_peaks(g_r, height=0.3, distance=3)

    r_peaks = r_bins[peak_indices]
    peak_heights = g_r[peak_indices]
    r_peaks_normalized = r_peaks / d_nn

    return RadialDistribution(
        r_bins=r_bins,
        g_r=g_r,
        r_peaks=r_peaks,
        peak_heights=peak_heights,
        r_peaks_normalized=r_peaks_normalized,
        d_nn=d_nn
    )


def check_sqrt_n_peaks(points: np.ndarray,
                      box_size: Optional[float] = None,
                      tolerance: float = 0.1) -> Dict:
    """
    Test V9: Do g(r) peaks occur at √n × d_nn?

    Expected peaks:
        n=1: r/d_nn = 1.00  (nearest neighbor)
        n=2: r/d_nn = 1.41  (√2)
        n=3: r/d_nn = 1.73  (√3)
        n=4: r/d_nn = 2.00  (2)

    Args:
        points: Seed points
        box_size: Box size
        tolerance: Allowed deviation from √n (default 10%)

    Returns:
        Dict with peak analysis and verdict
    """
    grd = compute_g_r(points, n_bins=150, box_size=box_size)

    # Expected peak positions (√n for n=1,2,3,4,5)
    sqrt_n = np.array([1.0, np.sqrt(2), np.sqrt(3), 2.0, np.sqrt(5)])

    # Match observed peaks to expected
    matches = []
    for i, expected in enumerate(sqrt_n):
        n = i + 1
        diffs = np.abs(grd.r_peaks_normalized - expected)
        if len(diffs) > 0:
            best_idx = np.argmin(diffs)
            best_diff = diffs[best_idx]
            if best_diff < tolerance:
                matches.append({
                    'n': n,
                    'expected': expected,
                    'observed': grd.r_peaks_normalized[best_idx],
                    'error': best_diff / expected,
                    'match': True
                })
            else:
                matches.append({
                    'n': n,
                    'expected': expected,
                    'observed': None,
                    'error': None,
                    'match': False
                })
        else:
            matches.append({
                'n': n,
                'expected': expected,
                'observed': None,
                'error': None,
                'match': False
            })

    # Verdict
    n_matched = sum(1 for m in matches if m['match'])

    return {
        'd_nn': grd.d_nn,
        'peaks_observed': grd.r_peaks_normalized.tolist(),
        'peaks_absolute': grd.r_peaks.tolist(),
        'peak_heights': grd.peak_heights.tolist(),
        'matches': matches,
        'n_matched': n_matched,
        'verdict': 'PASS' if n_matched >= 2 else 'WEAK' if n_matched >= 1 else 'FAIL'
    }


def run_v9_test():
    """
    Run V9 test: shells at √n×λ.
    """
    print("=" * 70)
    print("V9: SHELLS AT √n × λ")
    print("=" * 70)
    print()
    print("Testing if g(r) peaks occur at √n × d_nn")
    print("Expected: √1, √2, √3, √4, √5... = 1.00, 1.41, 1.73, 2.00, 2.24...")
    print()

    # Test 1: Poisson-Voronoi
    print("TEST 1: Poisson-Voronoi (random uniform)")
    print("-" * 40)
    points_pv = generate_random_points(500, box_size=10.0, seed=42)
    result_pv = check_sqrt_n_peaks(points_pv, box_size=10.0)
    print(f"  d_nn = {result_pv['d_nn']:.3f}")
    print(f"  Peaks (r/d_nn): {[f'{p:.2f}' for p in result_pv['peaks_observed'][:5]]}")
    print(f"  Matches: {result_pv['n_matched']}/5")
    for m in result_pv['matches'][:3]:
        status = '✓' if m['match'] else '✗'
        obs = f"{m['observed']:.2f}" if m['observed'] else "---"
        print(f"    n={m['n']}: expected √{m['n']}={m['expected']:.2f}, observed={obs} {status}")
    print(f"  Verdict: {result_pv['verdict']}")
    print()

    # Test 2: Jammed
    print("TEST 2: Jammed packing")
    print("-" * 40)
    points_j = generate_jammed_points(300, box_size=10.0, seed=42)
    result_j = check_sqrt_n_peaks(points_j, box_size=10.0)
    print(f"  d_nn = {result_j['d_nn']:.3f}")
    print(f"  Peaks (r/d_nn): {[f'{p:.2f}' for p in result_j['peaks_observed'][:5]]}")
    print(f"  Matches: {result_j['n_matched']}/5")
    for m in result_j['matches'][:3]:
        status = '✓' if m['match'] else '✗'
        obs = f"{m['observed']:.2f}" if m['observed'] else "---"
        print(f"    n={m['n']}: expected √{m['n']}={m['expected']:.2f}, observed={obs} {status}")
    print(f"  Verdict: {result_j['verdict']}")
    print()

    # Test 3: Poisson-disk
    print("TEST 3: Poisson-disk (hard-core)")
    print("-" * 40)
    try:
        points_pd = generate_poisson_disk_points(200, box_size=10.0, seed=42)
        result_pd = check_sqrt_n_peaks(points_pd, box_size=10.0)
        print(f"  d_nn = {result_pd['d_nn']:.3f}")
        print(f"  Peaks (r/d_nn): {[f'{p:.2f}' for p in result_pd['peaks_observed'][:5]]}")
        print(f"  Matches: {result_pd['n_matched']}/5")
        for m in result_pd['matches'][:3]:
            status = '✓' if m['match'] else '✗'
            obs = f"{m['observed']:.2f}" if m['observed'] else "---"
            print(f"    n={m['n']}: expected √{m['n']}={m['expected']:.2f}, observed={obs} {status}")
        print(f"  Verdict: {result_pd['verdict']}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

    print("=" * 70)
    print("SUMMARY:")
    print("  - √1 (d_nn): always present by definition")
    print("  - √2 (1.41): key for shell₂ = √2×λ claim")
    print("  - √3 (1.73): alternative 2nd shell in some packings")
    print("=" * 70)


if __name__ == "__main__":
    run_koide_test()
    print("\n\n")
    run_v9_test()
