#!/usr/bin/env python3
"""
Thermal Crosstalk Mitigation Analysis for Photonic Attention MZI Array.

Evaluates 4 mitigation strategies against thermal crosstalk:
  1. Guard trenches — deep-etched thermal isolation between heaters
  2. Substrate undercut — remove Si handle wafer under heater region
  3. Pitch increase — larger MZI spacing
  4. Active compensation — feedback control to null crosstalk

Each method is modeled with physics-based coupling reduction factors,
then verified via Monte Carlo against attention ranking fidelity (Spearman ρ).

Physical basis:
  - Guard trench: creates a thermal barrier by removing SiO2 between heaters,
    forcing heat to travel through a narrow bridge (BOX layer remnant).
  - Substrate undercut: removes the Si handle wafer (κ_Si ≈ 148 W/m·K),
    leaving only SiO2 (κ ≈ 1.38 W/m·K) as the lateral conduction medium.
  - Pitch increase: straightforward geometric separation.
  - Active compensation: feedback loop measures ΔT at each heater and
    applies compensating phase offset with finite gain and noise.
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
import os
import sys
import time

# ============================================================
# Reuse core attention + MZI models from thermal_crosstalk.py
# ============================================================
def mzi_T_ideal(delta_phi):
    return np.sin(delta_phi / 2.0) ** 2


class RealMZI:
    def __init__(self, csv_path="~/mzi_transmission.csv"):
        p = os.path.expanduser(csv_path)
        if os.path.exists(p):
            data = np.loadtxt(p, delimiter=",", skiprows=1)
            self._interp = interp1d(data[:, 0], data[:, 2], kind="cubic",
                                     bounds_error=False, fill_value="extrapolate")
        else:
            self._interp = None
    def __call__(self, delta_phi):
        if self._interp is None:
            return mzi_T_ideal(delta_phi)
        phi_w = np.mod(delta_phi, 2 * np.pi)
        return np.clip(self._interp(phi_w), 0.0, 1.0)


def photonic_attention(Q, K, V, mzi_fn, beta=1.0):
    D = Q.shape[0]
    S = Q @ K.T / np.sqrt(D)
    phi_bias = np.pi / 2
    delta_phi = beta * S + phi_bias
    delta_phi = np.clip(delta_phi, 0, np.pi)
    raw_T = mzi_fn(delta_phi)
    scores = np.exp(raw_T - raw_T.max(axis=1, keepdims=True))
    scores = scores / scores.sum(axis=1, keepdims=True)
    return scores, raw_T, scores @ V


# ============================================================
# Base thermal model (from thermal_crosstalk.py, refactored)
# ============================================================
def build_base_coupling_matrix(shape, pitch_um, kappa_SiO2=1.38,
                                dn_dT=1.86e-4, wavelength_um=1.55,
                                L_phase_um=3.0, r_heater_um=1.5, r_thermal_um=30.0):
    """
    Build thermal CROSS-coupling matrix C where:
      δφ_i = Σ_{j≠i} C_{ij} · P_j

    Self-heating (C_ii = 0) is NOT included — the phase encoding
    φ = β·S + π/2 already represents the desired phase shift.
    The coupling matrix captures only UNWANTED perturbation from neighbors.
    """
    N = shape[0] * shape[1]
    rows, cols = shape
    C = np.zeros((N, N), dtype=np.float64)

    xs = np.arange(cols) * pitch_um
    ys = np.arange(rows) * pitch_um
    X, Y = np.meshgrid(xs, ys)
    positions = np.column_stack([X.ravel(), Y.ravel()])

    P_mW = 1.0
    P_W = P_mW * 1e-3
    L_m = L_phase_um * 1e-6
    prefactor = (2 * np.pi / wavelength_um) * dn_dT * L_phase_um * P_W / (2 * np.pi * kappa_SiO2 * L_m)

    # Self-heating reference value (for stats, not for C_ii)
    self_ref = prefactor * np.log(r_thermal_um / r_heater_um)

    for i in range(N):
        xi, yi = positions[i]
        C[i, i] = 0.0  # Self-heating already in phase encoding
        for j in range(N):
            if i == j:
                continue
            xj, yj = positions[j]
            r = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if r < r_thermal_um:
                C[i, j] = prefactor * np.log(r_thermal_um / max(r, r_heater_um))

    return C, positions, self_ref


def apply_crosstalk_to_phases(phi_nominal, coupling_matrix, D=None):
    """Compute thermally-perturbed phases given coupling matrix."""
    if D is None:
        N = coupling_matrix.shape[0]
        D = int(np.sqrt(N))
    base_power = 5.0   # mW at quadrature bias
    power_swing = 5.0   # mW for π shift
    phi_rel = phi_nominal - np.pi / 2
    powers = (base_power + phi_rel / np.pi * power_swing).flatten()
    delta_phi_flat = coupling_matrix @ powers
    delta_phi = delta_phi_flat.reshape(D, D)
    perturbed = phi_nominal + delta_phi
    return np.clip(perturbed, 0, np.pi)


# ============================================================
# Mitigation 1: Guard Trenches
# ============================================================
class GuardTrenchMitigation:
    """
    Guard trenches: deep-etched gaps between adjacent phase shifters.

    Physical model:
      - Trench depth D_trench into SiO2 BOX layer
      - Total SiO2 thickness T_SiO2 ≈ 2-3 μm (BOX)
      - Remaining thermal bridge thickness = T_SiO2 - D_trench
      - Heat flux through remaining bridge is proportional to cross-section
      - Coupling attenuation: α = (T_SiO2 - D_trench) / T_SiO2 · f_geometry

    Additionally, the trench creates a longer heat path (around the trench
    bottom corner), modeled as an effective thermal path length increase.

    Typical values:
      - Trench width: 2-5 μm
      - Trench depth: 2-3 μm (into BOX, not through buried oxide)
      - SiO2 BOX thickness: 2 μm (SOI platform)
      - Si handle wafer: 725 μm below

    The trench primarily affects nearest-neighbor coupling. For N>1 neighbors,
    heat can go around the trench laterally (2D effect).
    """

    def __init__(self, trench_depth_um=2.0, trench_width_um=3.0,
                 sio2_thickness_um=3.0):
        self.trench_depth = trench_depth_um
        self.trench_width = trench_width_um
        self.sio2_thickness = sio2_thickness_um

        # Bridge factor: remaining SiO2 cross-section
        bridge_ratio = max(0.01, (sio2_thickness_um - trench_depth_um) / sio2_thickness_um)
        # Geometric factor: longer path around trench corner
        # ΔL_effective ≈ 2 * trench_width / (2π) added to path
        path_extension = trench_width_um / np.pi
        # Combined attenuation
        self.nn_attenuation = bridge_ratio * np.exp(-path_extension / sio2_thickness_um)

    def apply(self, C, positions):
        """Apply guard trench attenuation to coupling matrix."""
        C_mitigated = C.copy()
        N = C.shape[0]
        pitch = np.sqrt((positions[1, 0] - positions[0, 0])**2 +
                        (positions[1, 1] - positions[0, 1])**2)

        for i in range(N):
            xi, yi = positions[i]
            for j in range(N):
                if i == j:
                    continue
                xj, yj = positions[j]
                r = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                # Nearest-neighbor coupling (Manhattan distance = 1 pitch unit)
                dist_units = (abs(xi - xj) + abs(yi - yj)) / pitch
                if dist_units <= 1.1:  # Nearest neighbors (including diagonal)
                    C_mitigated[i, j] *= self.nn_attenuation
                elif dist_units <= 2.1:  # Next-nearest
                    # Heat goes around trenches but still partially blocked
                    C_mitigated[i, j] *= (self.nn_attenuation ** 0.5)

        return C_mitigated


# ============================================================
# Mitigation 2: Substrate Undercut
# ============================================================
class SubstrateUndercutMitigation:
    """
    Substrate undercut: remove Si handle wafer from under the heater region.

    Physical model:
      - Before undercut: heat spreads laterally through both SiO2 cladding
        AND Si substrate. Si (κ ≈ 148 W/m·K) dominates.
      - Effective lateral κ ≈ (κ_SiO2·t_SiO2 + κ_Si·t_si_eff) / t_total
        where t_si_eff accounts for the Si substrate thermal spreading depth.
      - After undercut: only SiO2 remains for lateral conduction.
        κ_eff' ≈ κ_SiO2 ≈ 1.38 W/m·K.

    The attenuation factor:
      α_undercut = κ_eff_after / κ_eff_before
      ≈ κ_SiO2 / (κ_SiO2 + κ_Si · t_si_eff / t_SiO2)

    For SOI with 2 μm BOX + 725 μm Si substrate:
      κ_eff_before ≈ (1.38·2 + 148·r_thermal) / (2 + r_thermal)
      with r_thermal ≈ 30 μm: κ_eff_before ≈ (2.76 + 4440) / 32 ≈ 139 W/m·K
      κ_eff_after ≈ 1.38 W/m·K
      α_undercut ≈ 1.38 / 139 ≈ 0.01

    This is extremely effective because Si dominates heat spreading.
    """

    def __init__(self, undercut_extent_um=40.0, si_kappa=148.0, sio2_kappa=1.38,
                 box_thickness_um=2.0, si_device_thickness_um=0.22,
                 r_thermal_before=30.0, r_thermal_after=5.0):
        self.undercut_extent = undercut_extent_um
        # Effective κ before undercut
        self.kappa_eff_before = ((sio2_kappa * box_thickness_um + si_kappa * r_thermal_before) /
                                  (box_thickness_um + r_thermal_before))
        # Effective κ after undercut (only SiO2 in lateral path)
        self.kappa_eff_after = sio2_kappa
        self.attenuation = self.kappa_eff_after / self.kappa_eff_before
        self.r_thermal_after = r_thermal_after
        self.si_kappa = si_kappa
        self.sio2_kappa = sio2_kappa

    def apply(self, C, positions):
        """Apply undercut attenuation by reducing lateral coupling."""
        C_mitigated = C.copy()
        N = C.shape[0]
        for i in range(N):
            for j in range(N):
                if i != j:
                    C_mitigated[i, j] *= self.attenuation
        # Also reduce self-heating slightly (heat is more confined)
        diag_reduction = self.kappa_eff_before / (self.kappa_eff_before + self.sio2_kappa)
        for i in range(N):
            C_mitigated[i, i] *= diag_reduction  # ~5% increase in self-heating sensitivity
        return C_mitigated


# ============================================================
# Mitigation 3: Pitch Increase
# ============================================================
class PitchIncreaseMitigation:
    """
    Increase MZI pitch to reduce thermal coupling.

    From scaling study results:
      - 20 μm: NN coupling = 7.8% of self-heating → catastrophic
      - 40 μm: NN coupling < 0.01% → negligible
      - 60 μm: NN coupling < 0.001% → non-existent

    This is the simplest and most reliable mitigation.
    """

    def __init__(self, target_pitch_um=40.0):
        self.target_pitch = target_pitch_um

    # Pitch increase is applied by rebuilding the coupling matrix
    # at the new pitch — no modification to existing C needed.


# ============================================================
# Mitigation 4: Active Thermal Compensation
# ============================================================
class ActiveCompensationMitigation:
    """
    Active thermal compensation using feedback control.

    Physical model:
      - Temperature sensors (integrated diodes or ring resonators)
        at each MZI measure the local temperature
      - A feedback controller applies compensating phase offsets
      - Residual error from: sensor noise, finite loop gain, bandwidth limits

    The compensated coupling matrix:
      C_comp = C · (1 - G) where G is the feedback gain matrix

    With perfect knowledge and infinite gain:
      C_comp → 0 (complete cancellation)

    Realistically:
      - Gain error: σ_G ≈ 2-5%
      - Sensor noise: σ_T ≈ 0.01 K RMS
      - Bandwidth limit: τ_control ≈ 1-10 μs
      - Residual NN coupling: ~2-5% of uncompensated

    We model residual as C * (1 - effective_gain) + measurement_noise.
    """

    def __init__(self, loop_gain=0.95, gain_error_pct=3.0,
                 sensor_noise_K=0.01, bandwidth_kHz=100.0, thermal_tau_us=10.0):
        """
        Args:
            loop_gain: feedback loop gain (0-1, 1=perfect cancellation)
            gain_error_pct: RMS gain variation (%)
            sensor_noise_K: temperature sensor RMS noise (K)
            bandwidth_kHz: control loop bandwidth (kHz)
            thermal_tau_us: thermal time constant (μs)
        """
        self.loop_gain = loop_gain
        self.gain_error = gain_error_pct / 100.0
        self.sensor_noise = sensor_noise_K
        self.bandwidth = bandwidth_kHz * 1e3
        self.thermal_tau = thermal_tau_us * 1e-6
        # Bandwidth-limited compensation effectiveness
        self.bw_factor = 1.0 / (1.0 + 1.0 / (2 * np.pi * self.bandwidth * self.thermal_tau))

    def apply(self, C, rng=None):
        """
        Apply active compensation to coupling matrix.
        Returns compensated C and residual noise std.
        """
        if rng is None:
            rng = np.random.default_rng(42)

        N = C.shape[0]
        # Effective residual coupling after compensation
        residual_factor = (1 - self.loop_gain * self.bw_factor)
        # Per-element gain variation
        gain_variation = rng.normal(0, self.gain_error, (N, N))
        C_comp = C * (residual_factor + gain_variation)

        # Add sensor measurement noise (converted to phase noise)
        # δφ_noise = (2π/λ) · dn/dT · L_phase · σ_T_sensor
        dphi_sensor = (2 * np.pi / 1.55) * 1.86e-4 * 3.0 * self.sensor_noise
        # Noise appears as random phase perturbation per MZI
        self.residual_noise_std = dphi_sensor * 0.1  # partially averaged by loop filter

        return C_comp


# ============================================================
# Comprehensive mitigation evaluator
# ============================================================
def evaluate_mitigation(mitigation_name, C_mitigated, D=64, N_trials=50, seed_base=42):
    """
    Run Monte Carlo evaluation for a specific mitigated coupling matrix.
    Returns Spearman ρ statistics.
    """
    base_rng = np.random.default_rng(seed_base)
    ideal_fn = mzi_T_ideal
    real_mzi = RealMZI("~/mzi_transmission.csv")

    sigma_dphi = 0.05
    sigma_coupling = 0.02
    sigma_resp = 0.03
    thermal_range = 0.01

    rhos = np.zeros(N_trials)

    for i in range(N_trials):
        rng_qkv = np.random.default_rng(seed_base + i)
        rng_proc = np.random.default_rng(seed_base + 1000 + i)

        Q = rng_qkv.normal(0, 1, (D, D)).astype(np.float64)
        K = rng_qkv.normal(0, 1, (D, D)).astype(np.float64)
        V = rng_qkv.normal(0, 1, (D, D)).astype(np.float64)

        # Ideal reference
        scores_ideal, _, _ = photonic_attention(Q, K, V, ideal_fn)

        # Process variations
        kappa1 = np.clip(0.5 + rng_proc.normal(0, sigma_coupling, (D, D)), 0.01, 0.99)
        kappa2 = np.clip(0.5 + rng_proc.normal(0, sigma_coupling, (D, D)), 0.01, 0.99)
        dphi_err = rng_proc.normal(0, sigma_dphi, (D, D))
        eta_det = rng_proc.normal(0, sigma_resp, (D, D))
        dphi_thermal_residual = rng_proc.uniform(-thermal_range, thermal_range, (D, D))

        S = Q @ K.T / np.sqrt(D)
        phi_nominal = np.clip(S + np.pi / 2, 0, np.pi)
        phi_proc = phi_nominal + dphi_err + dphi_thermal_residual
        phi_proc = np.clip(phi_proc, 0, np.pi)

        # Apply mitigated thermal crosstalk
        phi_therm = apply_crosstalk_to_phases(phi_proc, C_mitigated, D)

        t1 = np.sqrt(1 - kappa1)
        c1 = np.sqrt(kappa1)
        t2 = np.sqrt(1 - kappa2)
        c2 = np.sqrt(kappa2)
        E_bar = t1 * t2 * np.exp(1j * phi_therm) - c1 * c2
        T_out = np.abs(E_bar) ** 2 * (1 + eta_det)
        T_out = np.clip(T_out, 0, None)

        scores_out = np.exp(T_out - T_out.max(axis=1, keepdims=True))
        scores_out = scores_out / scores_out.sum(axis=1, keepdims=True)

        rho, _ = spearmanr(scores_ideal.flatten(), scores_out.flatten())
        rhos[i] = rho

    return {
        "name": mitigation_name,
        "rho_mean": rhos.mean(),
        "rho_std": rhos.std(),
        "rho_min": rhos.min(),
        "rho_max": rhos.max(),
        "yield_099": (rhos >= 0.99).mean(),
        "yield_095": (rhos >= 0.95).mean(),
    }


def run_mitigation_study():
    """Evaluate all mitigation strategies and combinations."""
    print("=" * 70)
    print("THERMAL CROSSTALK MITIGATION ANALYSIS")
    print("=" * 70)
    D = 64
    N_trials = 50

    results = []

    # ---- Baseline: no mitigation, 20 μm pitch ----
    print("\n--- Baseline (20 μm pitch, no mitigation) ---")
    C_base, pos_base, _ = build_base_coupling_matrix((D, D), pitch_um=20.0)
    r_base = evaluate_mitigation("1. Baseline (20μm, no mitigation)", C_base, D, N_trials)
    results.append(r_base)
    print(f"  ρ = {r_base['rho_mean']:.4f} ± {r_base['rho_std']:.4f}  "
          f"min={r_base['rho_min']:.4f}  yield_099={r_base['yield_099']:.0%}")

    # ---- Baseline: no crosstalk (ideal reference) ----
    C_zero = np.zeros_like(C_base)  # all zeros = no thermal crosstalk
    r_zero = evaluate_mitigation("0. Process only (no crosstalk)", C_zero, D, N_trials)
    results.append(r_zero)
    print(f"  ρ = {r_zero['rho_mean']:.4f} ± {r_zero['rho_std']:.4f}  "
          f"(ideal — no crosstalk)")

    # ---- Mitigation 1: Guard Trenches ----
    print("\n--- Mitigation 1: Guard Trenches ---")
    for depth in [1.0, 2.0, 2.5]:
        trench = GuardTrenchMitigation(trench_depth_um=depth, trench_width_um=3.0, sio2_thickness_um=3.0)
        C_trench = trench.apply(C_base, pos_base)
        r_trench = evaluate_mitigation(f"Guard trench {depth}μm deep", C_trench, D, N_trials)
        results.append(r_trench)
        print(f"  depth={depth}μm, attenuation={trench.nn_attenuation:.4f} → "
              f"ρ = {r_trench['rho_mean']:.4f} ± {r_trench['rho_std']:.4f}  "
              f"yield_099={r_trench['yield_099']:.0%}")

    # ---- Mitigation 2: Substrate Undercut ----
    print("\n--- Mitigation 2: Substrate Undercut ---")
    for extent in [20.0, 40.0]:
        undercut = SubstrateUndercutMitigation(undercut_extent_um=extent)
        C_undercut = undercut.apply(C_base, pos_base)
        r_undercut = evaluate_mitigation(f"Substrate undercut {extent}μm", C_undercut, D, N_trials)
        results.append(r_undercut)
        print(f"  extent={extent}μm, attenuation={undercut.attenuation:.4f} → "
              f"ρ = {r_undercut['rho_mean']:.4f} ± {r_undercut['rho_std']:.4f}  "
              f"yield_099={r_undercut['yield_099']:.0%}")

    # ---- Mitigation 3: Pitch Increase ----
    print("\n--- Mitigation 3: Pitch Increase ---")
    for pitch in [30.0, 40.0, 60.0]:
        C_pitch, pos_pitch, _ = build_base_coupling_matrix((D, D), pitch_um=pitch)
        r_pitch = evaluate_mitigation(f"Pitch {pitch:.0f}μm", C_pitch, D, N_trials)
        results.append(r_pitch)
        # Area estimate
        area_per_cell = (pitch * D) ** 2 * 1e-6  # mm²
        area_40pct = area_per_cell * 1.4
        print(f"  pitch={pitch:.0f}μm → ρ = {r_pitch['rho_mean']:.4f} ± {r_pitch['rho_std']:.4f}  "
              f"yield_099={r_pitch['yield_099']:.0%}  "
              f"area≈{area_40pct:.1f}mm² (w/ routing)")

    # ---- Mitigation 4: Active Compensation ----
    print("\n--- Mitigation 4: Active Compensation ---")
    for gain in [0.80, 0.90, 0.95, 0.98]:
        active = ActiveCompensationMitigation(loop_gain=gain)
        C_active = active.apply(C_base)
        r_active = evaluate_mitigation(f"Active comp gain={gain}", C_active, D, N_trials)
        results.append(r_active)
        print(f"  gain={gain:.2f} → ρ = {r_active['rho_mean']:.4f} ± {r_active['rho_std']:.4f}  "
              f"yield_099={r_active['yield_099']:.0%}")

    # ---- Combinations ----
    print("\n--- Combined Mitigations ---")

    # Guard trench + pitch increase (practical combo)
    for pitch in [30.0, 35.0]:
        for depth in [2.0, 2.5]:
            C_combo, pos_combo, _ = build_base_coupling_matrix((D, D), pitch_um=pitch)
            trench_combo = GuardTrenchMitigation(trench_depth_um=depth)
            C_combo_mit = trench_combo.apply(C_combo, pos_combo)
            r_combo = evaluate_mitigation(f"Pitch {pitch:.0f}μm + Trench {depth}μm",
                                          C_combo_mit, D, N_trials)
            results.append(r_combo)
            area_cell = (pitch * D)**2 * 1e-6 * 1.4
            print(f"  pitch={pitch:.0f}μm + trench={depth}μm → "
                  f"ρ = {r_combo['rho_mean']:.4f} ± {r_combo['rho_std']:.4f}  "
                  f"yield_099={r_combo['yield_099']:.0%}  area≈{area_cell:.1f}mm²")

    # Substrate undercut + pitch (ultimate solution)
    pitch_ult = 25.0
    C_ult, pos_ult, _ = build_base_coupling_matrix((D, D), pitch_um=pitch_ult)
    undercut_ult = SubstrateUndercutMitigation(undercut_extent_um=40.0)
    C_ult_mit = undercut_ult.apply(C_ult, pos_ult)
    r_ult = evaluate_mitigation(f"Pitch 25μm + Undercut", C_ult_mit, D, N_trials)
    results.append(r_ult)
    area_ult = (pitch_ult * D)**2 * 1e-6 * 1.4
    print(f"  pitch=25μm + undercut → ρ = {r_ult['rho_mean']:.4f}  "
          f"yield_099={r_ult['yield_099']:.0%}  area≈{area_ult:.1f}mm²")

    # ============================================================
    # Summary Table
    # ============================================================
    print(f"\n{'='*70}")
    print("MITIGATION SUMMARY — ranked by Spearman ρ")
    print(f"{'='*70}")
    print(f"  {'Mitigation':<45s} {'ρ_mean':>7s} {'yield_99':>8s} {'Area':>7s}")
    print(f"  {'─'*45} {'─'*7} {'─'*8} {'─'*7}")

    # Sort by ρ_mean
    results_sorted = sorted(results, key=lambda r: r['rho_mean'], reverse=True)
    for r in results_sorted:
        name = r['name'][:44]
        area_str = ""
        if "Pitch" in name or "pitch" in name:
            # Extract pitch value
            import re
            m = re.search(r'([\d.]+)μm', name)
            if m:
                p = float(m.group(1)) if 'Pitch' in name else 20.0
                area_str = f"{(p*64)**2*1e-6*1.4:.0f}mm²"
        status = "✅" if r['yield_099'] >= 0.99 else ("⚠️" if r['yield_099'] >= 0.90 else "❌")
        print(f"  {name:<45s} {r['rho_mean']:7.4f} {r['yield_099']:7.0%} {area_str:>7s} {status}")

    # ---- Feasibility Recommendations ----
    print(f"\n{'='*70}")
    print("FEASIBILITY RECOMMENDATIONS")
    print(f"{'='*70}")

    # Find viable options
    viable = [r for r in results_sorted if r['yield_099'] >= 0.99]
    print(f"\n  Viable mitigations (yield ρ≥0.99 ≥ 99%): {len(viable)} options")
    for r in viable:
        print(f"    ✅ {r['name']}: ρ={r['rho_mean']:.4f}, yield={r['yield_099']:.0%}")

    if not viable:
        print(f"  ⚠️  No single mitigation achieves yield ≥ 99%")
        marginal = [r for r in results_sorted if r['yield_099'] >= 0.90]
        if marginal:
            print(f"  Marginal mitigations (yield ≥ 90%): {len(marginal)} options")
            for r in marginal[:3]:
                print(f"    ⚠️  {r['name']}: ρ={r['rho_mean']:.4f}, yield={r['yield_099']:.0%}")

    print(f"\n  Recommended approach (ranked by practicality):")
    print(f"    1. Shallow guard trenches (2μm) at 30μm pitch")
    print(f"       — simplest fabrication, moderate area cost")
    print(f"    2. 40μm pitch (no other changes)")
    print(f"       — zero process changes, area cost 4×")
    print(f"    3. Substrate undercut at 25μm pitch")
    print(f"       — highest performance, requires MEMS release step")
    print(f"    4. Active compensation as supplementary")
    print(f"       — adds control complexity, best combined with physical isolation")

    return results


if __name__ == "__main__":
    results = run_mitigation_study()
    print("\nDone.")
