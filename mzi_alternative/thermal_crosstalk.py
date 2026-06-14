#!/usr/bin/env python3
"""
Thermal Crosstalk Simulation for Photonic Attention MZI Array.

Models heat diffusion from thermo-optic phase shifters to adjacent MZIs
in a dense array, computes the thermal coupling matrix, and evaluates
the impact on attention ranking fidelity.

Physical model:
  - 2D steady-state heat equation in SiO₂ cladding
  - Point heat sources at each MZI phase shifter
  - Temperature rise: ΔT(r) = P_heater / (2π · κ_SiO2) · ln(r_thermal / r)
  - Thermo-optic phase shift: δφ = (2π/λ) · (dn/dT)_Si · ΔT · L_phase
  - Thermal coupling matrix C: δφ_i = Σ_j C_{ij} · P_j

Key parameters:
  - κ_SiO2 = 1.38 W/(m·K)  (thermal conductivity)
  - (dn/dT)_Si = 1.86×10⁻⁴ K⁻¹ (thermo-optic coefficient)
  - Heater power: 5-20 mW per MZI
  - MZI pitch: 20 μm (from layout)
"""

import numpy as np
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
import os
import sys
import time

# ============================================================
# 0. Ideal MZI and attention core (imported from photonic_attention_sim)
# ============================================================
def mzi_T_ideal(delta_phi):
    """Ideal MZI bar-port transmission: T_bar = sin²(Δφ/2)."""
    return np.sin(delta_phi / 2.0) ** 2


class RealMZI:
    """Realistic MZI transfer from simulation data."""
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
    """Standard photonic attention (from photonic_attention_sim.py)."""
    D = Q.shape[0]
    S = Q @ K.T / np.sqrt(D)
    phi_bias = np.pi / 2
    delta_phi = beta * S + phi_bias
    delta_phi = np.clip(delta_phi, 0, np.pi)
    raw_T = mzi_fn(delta_phi)
    scores = np.exp(raw_T - raw_T.max(axis=1, keepdims=True))
    scores = scores / scores.sum(axis=1, keepdims=True)
    output = scores @ V
    return scores, raw_T, output


# ============================================================
# 1. Thermal physics model
# ============================================================
class ThermalCrosstalkModel:
    """
    Models thermal crosstalk between MZI phase shifters in a 2D array.

    Physical parameters (SiO₂ cladding, Si waveguide):
      - κ_SiO2: thermal conductivity of SiO₂ [W/(m·K)]
      - dn_dT:  thermo-optic coefficient of Si [K⁻¹]
      - r_heater: effective heater radius [μm]
      - r_thermal: thermal decay length [μm] (distance at which ΔT → 0)
    """

    def __init__(self, array_shape=(64, 64), mzi_pitch_um=20.0,
                 kappa_SiO2=1.38, dn_dT=1.86e-4, wavelength_um=1.55,
                 L_phase_um=3.0, r_heater_um=1.5, r_thermal_um=30.0):
        """
        Args:
            array_shape: (rows, cols) of the MZI attention array
            mzi_pitch_um: center-to-center pitch between MZIs [μm]
            kappa_SiO2: SiO₂ thermal conductivity [W/(m·K)]
            dn_dT: Si thermo-optic coefficient [K⁻¹]
            wavelength_um: operating wavelength [μm]
            L_phase_um: phase shifter length [μm]
            r_heater_um: effective heater radius [μm]
            r_thermal_um: thermal decay length [μm]
        """
        self.shape = array_shape
        self.pitch = mzi_pitch_um
        self.kappa = kappa_SiO2
        self.dn_dT = dn_dT
        self.wavelength = wavelength_um
        self.L_phase = L_phase_um
        self.r_heater = r_heater_um
        self.r_thermal = r_thermal_um

        # Pre-compute thermal coupling matrix
        self._build_coupling_matrix()

    def _temperature_rise(self, r_um, P_mW):
        """
        Steady-state 2D temperature rise from a point heat source.

        ΔT(r) = P / (2π · κ · L_phase) · ln(r_thermal / r)   for r_heater < r < r_thermal

        Args:
            r_um: distance from heater center [μm]
            P_mW: heater power [mW]

        Returns:
            ΔT in Kelvin
        """
        if isinstance(r_um, np.ndarray):
            r_eff = np.maximum(r_um, self.r_heater)  # avoid singularity at r=0
            inside_mask = r_um < self.r_thermal
            dT = np.zeros_like(r_um, dtype=np.float64)
            # P in W, L_phase in m
            P_W = P_mW * 1e-3
            L_m = self.L_phase * 1e-6
            prefactor = P_W / (2 * np.pi * self.kappa * L_m)
            dT[inside_mask] = prefactor * np.log(self.r_thermal / r_eff[inside_mask])
            return dT
        else:
            if r_um < self.r_heater:
                r_um = self.r_heater
            if r_um >= self.r_thermal:
                return 0.0
            P_W = P_mW * 1e-3
            L_m = self.L_phase * 1e-6
            return P_W / (2 * np.pi * self.kappa * L_m) * np.log(self.r_thermal / r_um)

    def _build_coupling_matrix(self):
        """
        Build the thermal CROSS-coupling matrix C where:
          δφ_i = Σ_{j≠i} C_{ij} · P_j

        C_{ij} = (2π/λ) · dn/dT · L_phase · ΔT_{ij}(1 mW)  for i≠j
        C_{ii} = 0  (self-heating is already accounted for in phase encoding)

        NOTE: Self-heating (C_ii) is NOT included because the phase encoding
        φ = β·S + π/2 already represents the desired thermo-optic phase shift.
        The coupling matrix only captures the UNWANTED perturbation from
        neighboring heaters.
        """
        N = self.shape[0] * self.shape[1]
        rows, cols = self.shape
        self.C = np.zeros((N, N), dtype=np.float64)

        # MZI positions (grid)
        xs = np.arange(cols) * self.pitch
        ys = np.arange(rows) * self.pitch
        X, Y = np.meshgrid(xs, ys)
        positions = np.column_stack([X.ravel(), Y.ravel()])

        # Compute self-heating for reference (reported in stats, not used in C)
        self_heating_dphi = (2 * np.pi / self.wavelength) * self.dn_dT * \
                            self._temperature_rise(self.r_heater, 1.0) * self.L_phase

        for i in range(N):
            xi, yi = positions[i]
            for j in range(N):
                if i == j:
                    self.C[i, j] = 0.0  # No self-coupling (already in phase encoding)
                else:
                    xj, yj = positions[j]
                    r = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    if r < self.r_thermal:
                        dT = self._temperature_rise(r, 1.0)
                        dphi_per_mW = (2 * np.pi / self.wavelength) * self.dn_dT * dT * self.L_phase
                        self.C[i, j] = dphi_per_mW

        self.self_heating_dphi_per_mw = self_heating_dphi
        self.coupling_matrix = self.C

    def compute_crosstalk_phase(self, heater_powers, base_phases):
        """
        Compute the thermally-perturbed phase for each MZI.

        Args:
            heater_powers: [N] array of heater powers (mW) — from attention computation
            base_phases:   [D, D] array of nominal MZI phases

        Returns:
            perturbed_phases: [D, D] phases with thermal crosstalk
        """
        D = base_phases.shape[0]
        # Map each attention element to a heater power
        # Heater power proportional to phase shift (P ∝ Δφ for thermo-optic)
        # Baseline: π/2 bias requires ~5 mW, full swing 0→π requires ~0→10 mW
        base_power = 5.0  # mW at quadrature
        power_swing = 5.0  # mW for π phase shift

        # Phase relative to quadrature
        phi_rel = base_phases - np.pi / 2
        powers = base_power + phi_rel / np.pi * power_swing
        powers_flat = powers.flatten()

        # Apply thermal coupling
        delta_phi_flat = self.coupling_matrix @ powers_flat

        # Reshape and add to base phases
        delta_phi = delta_phi_flat.reshape(D, D)
        perturbed = base_phases + delta_phi

        return np.clip(perturbed, 0, np.pi)

    def get_coupling_stats(self):
        """Report thermal coupling statistics."""
        # Self-coupling is stored separately (not in C matrix)
        self_val = self.self_heating_dphi_per_mw

        # Nearest-neighbor coupling (pitch distance)
        nn_mask = np.zeros_like(self.C, dtype=bool)
        N = self.shape[0] * self.shape[1]
        cols = self.shape[1]
        for i in range(N):
            row_i, col_i = divmod(i, cols)
            for j in range(N):
                if i != j:
                    row_j, col_j = divmod(j, cols)
                    dist = max(abs(row_i - row_j), abs(col_i - col_j))
                    if dist == 1:
                        nn_mask[i, j] = True

        nn_vals = self.C[nn_mask]

        stats = {
            "self_heating_dphi_per_mW": self_val,
            "nn_coupling_dphi_per_mW": np.mean(np.abs(nn_vals)) if len(nn_vals) > 0 else 0,
            "nn_to_self_ratio": np.mean(np.abs(nn_vals)) / self_val if self_val > 0 and len(nn_vals) > 0 else 0,
            "max_coupling": np.max(np.abs(self.C)),
            "mean_coupling": np.mean(np.abs(self.C)),
        }
        return stats


# ============================================================
# 2. Monte Carlo with thermal crosstalk
# ============================================================
def run_thermal_monte_carlo(N=200, D=64, seed_base=42):
    """
    Monte Carlo analysis comparing attention fidelity:
      (a) Process variations only (baseline, from monte_carlo_process.py)
      (b) Process variations + thermal crosstalk
    """
    print(f"\n{'='*60}")
    print(f"THERMAL CROSSTALK MONTE CARLO")
    print(f"{'='*60}")
    print(f"  Trials: {N}, D: {D}")
    print()

    # Initialize thermal model
    thermal = ThermalCrosstalkModel(
        array_shape=(D, D),
        mzi_pitch_um=20.0,
        kappa_SiO2=1.38,
        dn_dT=1.86e-4,
        wavelength_um=1.55,
        L_phase_um=3.0,
        r_heater_um=1.5,
        r_thermal_um=30.0,
    )

    # Report coupling stats
    stats = thermal.get_coupling_stats()
    print("Thermal Coupling Matrix Statistics:")
    print(f"  Self-heating:        {stats['self_heating_dphi_per_mW']:.4f} rad/mW")
    print(f"  NN coupling:         {stats['nn_coupling_dphi_per_mW']:.4f} rad/mW")
    print(f"  NN/self ratio:       {stats['nn_to_self_ratio']:.4f} ({stats['nn_to_self_ratio']*100:.2f}%)")
    print(f"  Max inter-MZI coupling: {stats['max_coupling']:.4f} rad/mW")
    print()

    # Process variation parameters (same as monte_carlo_process.py)
    sigma_dphi = 0.05
    sigma_coupling = 0.02
    sigma_resp = 0.03
    thermal_range = 0.01

    real_mzi = RealMZI("~/mzi_transmission.csv")
    ideal_fn = mzi_T_ideal
    base_rng = np.random.default_rng(seed_base)

    rhos_baseline = np.zeros(N)
    rhos_thermal = np.zeros(N)

    t_start = time.time()

    for i in range(N):
        trial_seed_qkv = seed_base + i
        trial_seed_proc = seed_base + 1000 + i

        rng_qkv = np.random.default_rng(trial_seed_qkv)
        Q = rng_qkv.normal(0, 1, (D, D)).astype(np.float64)
        K = rng_qkv.normal(0, 1, (D, D)).astype(np.float64)
        V = rng_qkv.normal(0, 1, (D, D)).astype(np.float64)

        # Ideal reference
        scores_ideal, _, _ = photonic_attention(Q, K, V, ideal_fn)

        # ---- (a) Process variations only ----
        rng_proc = np.random.default_rng(trial_seed_proc)

        # Per-MZI process variations
        kappa1 = np.clip(0.5 + rng_proc.normal(0, sigma_coupling, (D, D)), 0.01, 0.99)
        kappa2 = np.clip(0.5 + rng_proc.normal(0, sigma_coupling, (D, D)), 0.01, 0.99)
        dphi_err = rng_proc.normal(0, sigma_dphi, (D, D))
        eta_det = rng_proc.normal(0, sigma_resp, (D, D))
        dphi_thermal_residual = rng_proc.uniform(-thermal_range, thermal_range, (D, D))

        S = Q @ K.T / np.sqrt(D)
        phi_nominal = np.clip(S + np.pi / 2, 0, np.pi)
        phi_proc = phi_nominal + dphi_err + dphi_thermal_residual
        phi_proc = np.clip(phi_proc, 0, np.pi)

        t1 = np.sqrt(1 - kappa1)
        c1 = np.sqrt(kappa1)
        t2 = np.sqrt(1 - kappa2)
        c2 = np.sqrt(kappa2)
        E_bar = t1 * t2 * np.exp(1j * phi_proc) - c1 * c2
        T_proc = np.abs(E_bar) ** 2 * (1 + eta_det)
        T_proc = np.clip(T_proc, 0, None)

        scores_proc = np.exp(T_proc - T_proc.max(axis=1, keepdims=True))
        scores_proc = scores_proc / scores_proc.sum(axis=1, keepdims=True)

        rho_base, _ = spearmanr(scores_ideal.flatten(), scores_proc.flatten())
        rhos_baseline[i] = rho_base

        # ---- (b) Process variations + thermal crosstalk ----
        # Apply thermal crosstalk on top of process-varied phases
        phi_thermal = thermal.compute_crosstalk_phase(None, phi_proc)

        E_bar_thermal = t1 * t2 * np.exp(1j * phi_thermal) - c1 * c2
        T_thermal = np.abs(E_bar_thermal) ** 2 * (1 + eta_det)
        T_thermal = np.clip(T_thermal, 0, None)

        scores_thermal = np.exp(T_thermal - T_thermal.max(axis=1, keepdims=True))
        scores_thermal = scores_thermal / scores_thermal.sum(axis=1, keepdims=True)

        rho_therm, _ = spearmanr(scores_ideal.flatten(), scores_thermal.flatten())
        rhos_thermal[i] = rho_therm

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            eta = (N - i - 1) * elapsed / (i + 1)
            print(f"  [{i+1:4d}/{N}]  ρ_baseline={rhos_baseline[:i+1].mean():.6f}  "
                  f"ρ_thermal={rhos_thermal[:i+1].mean():.6f}  "
                  f"Δρ={rhos_baseline[:i+1].mean() - rhos_thermal[:i+1].mean():.6f}  "
                  f"ETA={eta:.0f}s")

    elapsed = time.time() - t_start

    # ---- Results ----
    delta_rho = rhos_baseline - rhos_thermal  # positive = degradation from crosstalk

    print(f"\n{'='*60}")
    print("THERMAL CROSSTALK RESULTS")
    print(f"{'='*60}")
    print(f"  Baseline (proc only):")
    print(f"    ρ mean: {rhos_baseline.mean():.6f} ± {rhos_baseline.std():.6f}")
    print(f"    ρ min:  {rhos_baseline.min():.6f}")
    print()
    print(f"  With thermal crosstalk:")
    print(f"    ρ mean: {rhos_thermal.mean():.6f} ± {rhos_thermal.std():.6f}")
    print(f"    ρ min:  {rhos_thermal.min():.6f}")
    print()
    print(f"  Crosstalk-induced ρ degradation:")
    print(f"    Mean Δρ: {delta_rho.mean():.6f}")
    print(f"    Max Δρ:  {delta_rho.max():.6f}")
    print(f"    Std Δρ:  {delta_rho.std():.6f}")
    print()
    print(f"  Yield (ρ ≥ 0.99):")
    print(f"    Baseline: {(rhos_baseline >= 0.99).mean()*100:.1f}%")
    print(f"    Thermal:  {(rhos_thermal >= 0.99).mean()*100:.1f}%")

    # ---- Interpretation ----
    degradation = delta_rho.mean()
    if degradation < 0.001:
        print(f"\n  ✅ Thermal crosstalk impact is NEGLIGIBLE (Δρ < 0.001)")
    elif degradation < 0.005:
        print(f"\n  ✅ Thermal crosstalk impact is SMALL (Δρ < 0.005)")
    elif degradation < 0.01:
        print(f"\n  ⚠️  Thermal crosstalk impact is MODERATE (Δρ < 0.01)")
        print(f"     Consider guard trenches or thermal isolation.")
    else:
        print(f"\n  ⚠️  Thermal crosstalk impact is SIGNIFICANT (Δρ ≥ 0.01)")
        print(f"     Thermal isolation required for deployment.")

    print(f"  Elapsed: {elapsed:.1f}s")

    return {
        "N": N, "D": D,
        "rhos_baseline": rhos_baseline,
        "rhos_thermal": rhos_thermal,
        "delta_rho_mean": delta_rho.mean(),
        "delta_rho_max": delta_rho.max(),
        "coupling_stats": stats,
        "yield_baseline": (rhos_baseline >= 0.99).mean(),
        "yield_thermal": (rhos_thermal >= 0.99).mean(),
    }


# ============================================================
# 3. Thermal crosstalk scaling study
# ============================================================
def run_scaling_study():
    """Study how thermal crosstalk scales with array size and pitch."""
    print(f"\n{'='*60}")
    print("THERMAL CROSSTALK SCALING STUDY")
    print(f"{'='*60}")

    D_vals = [16, 32, 64, 128]
    pitch_vals = [10.0, 20.0, 40.0, 60.0]

    print(f"\n  {'D':>5s}  {'Pitch':>8s}  {'NN/Self':>10s}  {'ρ_degradation':>15s}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*10}  {'─'*15}")

    results = []
    for D in D_vals:
        for pitch in pitch_vals:
            if D * pitch > 2000:  # skip unrealistic sizes
                continue
            thermal = ThermalCrosstalkModel(
                array_shape=(D, D),
                mzi_pitch_um=pitch,
            )
            stats = thermal.get_coupling_stats()
            results.append({
                "D": D, "pitch": pitch,
                "nn_self_ratio": stats["nn_to_self_ratio"],
                "max_coupling": stats["max_coupling"],
            })
            print(f"  {D:5d}  {pitch:6.0f} μm  {stats['nn_to_self_ratio']:10.4f}  {'—':>15s}")

    print(f"\n  → Larger pitch and smaller arrays reduce thermal crosstalk.")
    print(f"  → At 60 μm pitch, nearest-neighbor coupling is < 1% of self-heating.")
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Full Monte Carlo analysis
    results = run_thermal_monte_carlo(N=200, D=64)

    # Scaling study (quick)
    scaling = run_scaling_study()

    # Save data
    np.savez(
        os.path.expanduser("~/thermal_crosstalk_data.npz"),
        rhos_baseline=results["rhos_baseline"],
        rhos_thermal=results["rhos_thermal"],
        delta_rho_mean=results["delta_rho_mean"],
        delta_rho_max=results["delta_rho_max"],
        yield_baseline=results["yield_baseline"],
        yield_thermal=results["yield_thermal"],
    )
    print("\nData saved to ~/thermal_crosstalk_data.npz")
    print("Done.")
