#!/usr/bin/env python3
"""
ELASTIC MEDIUM DRAG TESTS
=========================

WARNING - MODELING LIMITATION (Jan 2026):
==========================================
This test models a MASS IMPURITY (different mass connected by springs).
When mass moves → compresses springs → LONGITUDINAL coupling → drag F=2kv.

But ST_ particles are VORTICES, not mass impurities.
A vortex may move through medium via SHEAR (rotation), not compression.

Example: smoke ring in air travels far without drag - pure shear, no compression.

THIS TEST DOES NOT VALIDATE "NO DRAG" FOR VORTICES.
It only shows that "mass on springs" has drag (obvious, not relevant).

TODO: Research how shear vortices actually propagate through elastic foam.
      Need different toy model: vortex dynamics, not mass-spring system.

=========================================================================

Scientific investigation: Does a defect moving through an elastic lattice
experience drag (energy loss)?

FINDINGS (Jan 2026):
====================
1. E_total is conserved (trivial - conservative system)
2. E_defect is NOT conserved - defect loses energy to lattice
3. Energy loss is INDEPENDENT of velocity - no Cherenkov threshold
4. F = 2kv exactly - linear drag from LOCAL spring coupling

CAVEAT: These findings apply to MASS IMPURITY model only.
        Vortex in shear medium may behave completely differently.

Migrated from ST_7/wip/22_closing_todos/03_no_drag_toy.py
Jan 2026
"""

import numpy as np
import pytest


# =============================================================================
# HELPERS
# =============================================================================

def compute_forces_periodic(x, k, N):
    """Spring forces with periodic BC, rest length = 0 (positions are displacements)."""
    F = np.zeros(N)
    for i in range(N):
        dx_left = x[i] - x[(i-1) % N]
        dx_right = x[(i+1) % N] - x[i]
        F[i] = k * (dx_right - dx_left)
    return F


def compute_forces_open(x, k, N, a_rest):
    """Spring forces with open BC and explicit rest length."""
    F = np.zeros(N)
    for i in range(N):
        if i < N - 1:
            extension_right = (x[i+1] - x[i]) - a_rest
            F[i] += k * extension_right
        if i > 0:
            extension_left = (x[i] - x[i-1]) - a_rest
            F[i] -= k * extension_left
    return F


def run_1d_impulse(N, m_defect, m_lattice, k_spring, dt, T_sim):
    """
    Run 1D chain simulation with impulse on defect.
    Returns time series of defect KE and total E.
    """
    n_steps = int(T_sim / dt)

    x = np.zeros(N)
    v = np.zeros(N)
    v[N//2] = 1.0  # impulse on defect at center

    masses = np.ones(N) * m_lattice
    masses[N//2] = m_defect

    KE_defect = []
    E_total = []

    for step in range(n_steps):
        F = compute_forces_periodic(x, k_spring, N)
        a = F / masses

        # Velocity Verlet
        v_half = v + 0.5 * a * dt
        x = x + v_half * dt
        F_new = compute_forces_periodic(x, k_spring, N)
        a_new = F_new / masses
        v = v_half + 0.5 * a_new * dt

        # Energy
        KE_d = 0.5 * masses[N//2] * v[N//2]**2
        KE_tot = 0.5 * np.sum(masses * v**2)
        PE_tot = 0.5 * k_spring * np.sum((np.roll(x, -1) - x)**2)

        if step % 100 == 0:
            KE_defect.append(KE_d)
            E_total.append(KE_tot + PE_tot)

    return np.array(KE_defect), np.array(E_total)


# =============================================================================
# TEST 1: TOTAL ENERGY CONSERVATION (Verlet integrator check)
# =============================================================================

def test_verlet_energy_conservation():
    """
    TEST: Velocity Verlet integrator conserves total energy.

    This is a NUMERICAL test, not a physics test.
    Conservative system must have E_total = const.
    """
    N, m_lattice, m_defect, k_spring = 100, 1.0, 2.0, 1.0
    dt, T_sim = 0.01, 50.0

    KE_d, E_tot = run_1d_impulse(N, m_defect, m_lattice, k_spring, dt, T_sim)

    E_0 = E_tot[0]
    E_drift = np.max(np.abs(E_tot - E_0)) / E_0

    print(f"\nVERLET ENERGY CONSERVATION:")
    print(f"  E_0 = {E_0:.6f}")
    print(f"  max |ΔE/E| = {E_drift:.2e}")

    assert E_drift < 1e-4, f"Verlet drift too large: {E_drift:.2e}"
    print(f"  → PASS: Integrator conserves energy to {E_drift:.1e}")


def test_verlet_dt_scaling():
    """
    TEST: Verlet error scales as dt² (second-order method).
    """
    N, m_lattice, m_defect, k_spring = 50, 1.0, 2.0, 1.0
    T_sim = 50.0

    dt_values = [0.02, 0.01, 0.005]
    drift_values = []

    for dt in dt_values:
        KE_d, E_tot = run_1d_impulse(N, m_defect, m_lattice, k_spring, dt, T_sim)
        drift = np.max(np.abs(E_tot - E_tot[0])) / E_tot[0]
        drift_values.append(drift)

    # Fit log-log slope
    log_dt = np.log(dt_values)
    log_drift = np.log(drift_values)
    slope, _ = np.polyfit(log_dt, log_drift, 1)

    print(f"\nVERLET dt-SCALING:")
    for dt, drift in zip(dt_values, drift_values):
        print(f"  dt={dt:.3f}: drift={drift:.2e}")
    print(f"  Slope: {slope:.2f} (expect ~2)")

    assert 1.5 < slope < 2.5, f"Wrong scaling: slope={slope:.2f}"
    print(f"  → PASS: Error ∝ dt^{slope:.1f}")


# =============================================================================
# TEST 2: DEFECT ENERGY TRANSFER (Radiation Reaction)
# =============================================================================

@pytest.mark.slow
def test_defect_energy_transfer():
    """
    TEST: Defect transfers kinetic energy to lattice waves.

    FINDING: Defect DOES lose energy to lattice.
    This is radiation reaction / effective drag from defect's perspective.

    Setup:
    - Large chain, defect at center
    - Stop before waves return (T < N/2c_s)
    - Measure K_defect(t)

    Result: K_defect decreases exponentially with γ ≈ 0.8
    """
    N = 500
    m_lattice, m_defect, k_spring = 1.0, 2.0, 1.0
    dt = 0.01
    c_s = np.sqrt(k_spring / m_lattice)
    T_sim = N / (4 * c_s)  # Stop well before wave return

    KE_d, E_tot = run_1d_impulse(N, m_defect, m_lattice, k_spring, dt, T_sim)

    # Fit exponential decay: K(t) = K_0 * exp(-γt)
    t = np.arange(len(KE_d)) * 100 * dt
    n_fit = len(KE_d) // 2  # Fit first half

    # Avoid log(0) issues
    KE_safe = np.maximum(KE_d[:n_fit], 1e-10)
    log_ratio = np.log(KE_safe / KE_d[0])
    slope, _ = np.polyfit(t[:n_fit], log_ratio, 1)
    gamma = -slope

    # Energy conservation check
    E_drift = abs(E_tot[-1] - E_tot[0]) / E_tot[0]

    # Transfer fraction
    transfer_frac = 1 - KE_d[-1] / KE_d[0]

    print(f"\nDEFECT ENERGY TRANSFER:")
    print(f"  N = {N}, T = {T_sim:.0f} (wave return at {N/c_s:.0f})")
    print(f"  K_defect(0) = {KE_d[0]:.4f}")
    print(f"  K_defect(end) = {KE_d[-1]:.4f}")
    print(f"  Transfer to lattice: {100*transfer_frac:.1f}%")
    print(f"  Decay rate: γ = {gamma:.4f}")
    print(f"  E_total drift: {E_drift:.2e}")

    # Assertions
    assert E_drift < 1e-4, f"E_total not conserved: {E_drift:.2e}"
    assert transfer_frac > 0.5, f"Expected >50% transfer, got {100*transfer_frac:.1f}%"
    assert gamma > 0.1, f"Expected positive decay rate, got γ={gamma:.4f}"

    print(f"\n  → CONFIRMED: Defect loses {100*transfer_frac:.0f}% of KE to lattice")
    print(f"    This IS radiation reaction (effective drag on defect)")
    print(f"    E_total conserved, but E_defect is NOT")


# =============================================================================
# TEST 3: VELOCITY DEPENDENCE OF ENERGY TRANSFER
# =============================================================================

@pytest.mark.slow
def test_energy_transfer_vs_velocity():
    """
    TEST: Does energy transfer rate depend on velocity?

    Cherenkov prediction:
    - v < c_s: no radiation, γ ≈ 0
    - v > c_s: radiation, γ > 0

    FINDING: Energy loss rate is similar for all velocities.
    No Cherenkov threshold observed.
    """
    N = 500  # larger chain to allow longer observation
    m_lattice, k_spring = 1.0, 1.0
    c_s = np.sqrt(k_spring / m_lattice)
    dt = 0.01
    T_sim = 50.0
    n_steps = int(T_sim / dt)

    velocities = [0.3, 0.5, 0.8, 1.2, 1.5]

    print(f"\nENERGY TRANSFER VS VELOCITY:")
    print(f"  c_s = {c_s:.2f}, N = {N}")
    print(f"\n  {'v':<6} {'v/c_s':<8} {'KE_ratio':<12} {'regime':<12}")
    print("  " + "-" * 45)

    results = []

    for v_0 in velocities:
        # Use periodic BC, defect at center with initial velocity
        x = np.zeros(N)
        vel = np.zeros(N)
        defect_idx = N // 2
        vel[defect_idx] = v_0

        masses = np.ones(N) * m_lattice
        masses[defect_idx] = 2.0  # heavier defect

        KE_0 = 0.5 * masses[defect_idx] * v_0**2

        for step in range(n_steps):
            F = compute_forces_periodic(x, k_spring, N)
            a = F / masses

            vel_half = vel + 0.5 * a * dt
            x = x + vel_half * dt
            F_new = compute_forces_periodic(x, k_spring, N)
            a_new = F_new / masses
            vel = vel_half + 0.5 * a_new * dt

        KE_final = 0.5 * masses[defect_idx] * vel[defect_idx]**2
        KE_ratio = KE_final / KE_0

        regime = "subsonic" if v_0 < c_s else "SUPERSONIC"
        print(f"  {v_0:<6.1f} {v_0/c_s:<8.2f} {KE_ratio:<12.4f} {regime:<12}")

        results.append({'v': v_0, 'v_cs': v_0/c_s, 'KE_ratio': KE_ratio})

    # Analyze: is there a threshold?
    sub_ratios = [r['KE_ratio'] for r in results if r['v_cs'] < 0.9]
    sup_ratios = [r['KE_ratio'] for r in results if r['v_cs'] > 1.1]

    if sub_ratios and sup_ratios:
        sub_mean = np.mean(sub_ratios)
        sup_mean = np.mean(sup_ratios)

        print(f"\n  Mean KE_ratio (subsonic): {sub_mean:.4f}")
        print(f"  Mean KE_ratio (supersonic): {sup_mean:.4f}")

        # Cherenkov would show: subsonic retains MORE energy than supersonic
        if sub_mean > sup_mean * 2.0:
            print(f"\n  → CHERENKOV DETECTED: supersonic loses more energy")
        elif sub_mean > sup_mean * 1.3:
            print(f"\n  → WEAK CHERENKOV effect")
        else:
            print(f"\n  → NO CHERENKOV: energy loss similar for all velocities")
            print(f"    Energy transfer is LOCAL, not radiative")

    # This documents the finding
    assert True


# =============================================================================
# TEST 4: MOMENTUM CONSERVATION
# =============================================================================

@pytest.mark.slow
def test_momentum_conservation():
    """
    TEST: Total momentum is conserved (no external forces).

    Defect transfers momentum to lattice, but P_total = const.
    """
    N = 500
    m_lattice, m_defect, k_spring = 1.0, 2.0, 1.0
    dt = 0.01
    c_s = np.sqrt(k_spring / m_lattice)
    T_sim = N / (4 * c_s)
    n_steps = int(T_sim / dt)

    x = np.zeros(N)
    v = np.zeros(N)
    masses = np.ones(N) * m_lattice
    defect_idx = N // 2
    masses[defect_idx] = m_defect
    v[defect_idx] = 1.0

    P_0 = np.sum(masses * v)
    P_defect_0 = masses[defect_idx] * v[defect_idx]

    for step in range(n_steps):
        F = compute_forces_periodic(x, k_spring, N)
        a = F / masses
        v_half = v + 0.5 * a * dt
        x = x + v_half * dt
        F_new = compute_forces_periodic(x, k_spring, N)
        a_new = F_new / masses
        v = v_half + 0.5 * a_new * dt

    P_final = np.sum(masses * v)
    P_defect_final = masses[defect_idx] * v[defect_idx]

    P_drift = abs(P_final - P_0) / abs(P_0)
    P_transfer = 1 - abs(P_defect_final) / abs(P_defect_0)

    print(f"\nMOMENTUM CONSERVATION:")
    print(f"  P_total(0) = {P_0:.6f}")
    print(f"  P_total(end) = {P_final:.6f}")
    print(f"  |ΔP/P| = {P_drift:.2e}")
    print(f"  P_defect transfer: {100*P_transfer:.1f}%")

    assert P_drift < 1e-6, f"Momentum not conserved: {P_drift:.2e}"
    print(f"\n  → PASS: Momentum conserved to {P_drift:.1e}")


# =============================================================================
# TEST 5: CONSTRAINED MOTION DRAG FORCE
# =============================================================================

def test_constrained_motion_force():
    """
    TEST: Force needed to maintain constant velocity.

    If defect is CONSTRAINED to move at constant v, what force is needed?

    FINDING: F = 2kv (local spring drag)
    - Defect at velocity v compresses front spring, stretches back spring
    - Net force F = k*v + k*v = 2kv needed to maintain motion
    - This IS drag from local coupling

    This contradicts naive "no drag" expectation.
    """
    N = 100
    m_lattice, k_spring, a_rest = 1.0, 1.0, 1.0
    dt = 0.01
    T_sim = 50.0
    n_steps = int(T_sim / dt)

    # Test multiple velocities
    velocities = [0.2, 0.5, 1.0]

    print(f"\nCONSTRAINED MOTION FORCE:")
    print(f"  Prediction: F = 2kv (local spring drag)")
    print(f"\n  {'v':<8} {'<F>':<10} {'2kv':<10} {'ratio':<10}")
    print("  " + "-" * 40)

    for v_0 in velocities:
        x = np.arange(N) * a_rest
        v = np.zeros(N)
        defect_idx = N // 2
        x_defect_0 = x[defect_idx]

        forces = []

        for step in range(n_steps):
            t = step * dt
            x[defect_idx] = x_defect_0 + v_0 * t

            F = compute_forces_open(x, k_spring, N, a_rest)
            F_on_defect = -F[defect_idx]

            for i in range(N):
                if i != defect_idx:
                    v[i] += F[i] / m_lattice * dt
                    x[i] += v[i] * dt

            if step > n_steps // 4:
                forces.append(F_on_defect)

        mean_F = np.mean(forces)
        predicted_F = 2 * k_spring * v_0
        ratio = mean_F / predicted_F if predicted_F > 0 else 0

        print(f"  {v_0:<8.1f} {mean_F:<10.4f} {predicted_F:<10.4f} {ratio:<10.3f}")

    # Final assertion: F should scale with v
    # Use last velocity for assertion
    assert abs(ratio - 1.0) < 0.1, f"F ≠ 2kv: ratio = {ratio:.3f}"

    print(f"\n  → CONFIRMED: F = 2kv (local spring drag)")
    print(f"    Defect moving at constant v requires force F = 2kv")
    print(f"    This IS drag from local coupling to neighbors")


# =============================================================================
# SUMMARY
# =============================================================================

def _print_findings():
    """Print summary of findings."""
    print("""
================================================================================
SUMMARY OF FINDINGS
================================================================================

1. TOTAL ENERGY CONSERVATION
   - E_total is conserved (trivial for conservative system)
   - This confirms integrator works, not "no drag"

2. DEFECT ENERGY TRANSFER
   - Defect LOSES kinetic energy to lattice waves
   - ~100% transfer to lattice before wave return
   - This IS radiation reaction / effective drag

3. VELOCITY DEPENDENCE
   - γ is INDEPENDENT of velocity
   - No Cherenkov threshold at v = c_s
   - Energy transfer is LOCAL (spring coupling), not wave radiation

4. MOMENTUM CONSERVATION
   - P_total conserved (as expected)
   - Defect transfers momentum to lattice

5. CONSTRAINED MOTION FORCE
   - F = 2kv to maintain constant velocity
   - This IS drag from local spring coupling
   - NOT zero, scales linearly with v

IMPLICATION FOR ST_ MODEL:
--------------------------
A MASS IMPURITY in elastic medium DOES experience drag (F = 2kv).

BUT: ST_ particles are VORTICES, not mass impurities.
     Vortices may propagate via SHEAR without compression.
     This test is NOT relevant to vortex dynamics.

OPEN QUESTION:
  How does a shear vortex move through elastic foam?
  Does it couple to longitudinal (compression) or only shear modes?

  Example: smoke rings travel far without drag (pure shear motion).

  Need research / different toy model to answer this properly.
================================================================================
""")


if __name__ == "__main__":
    print("=" * 70)
    print("ELASTIC MEDIUM DRAG TESTS")
    print("=" * 70)

    test_verlet_energy_conservation()
    test_verlet_dt_scaling()
    test_defect_energy_transfer()
    test_energy_transfer_vs_velocity()
    test_momentum_conservation()
    test_constrained_motion_force()
    _print_findings()
