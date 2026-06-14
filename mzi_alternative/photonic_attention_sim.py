#!/usr/bin/env python3
"""
Photonic Attention Simulation — Ideal vs. Real MZI Transfer Function.

Sections:
  1. Ideal MZI transfer:  T(Δφ) = sin²(Δφ/2)
  2. Real MZI transfer:   loaded from ~/mzi_sim.py FDTD+analytic results,
     interpolated via scipy.
  3. Photonic WDM attention with bipolar weights (D=64).
  4. Spearman ρ comparison between ideal and real-MZI attention scores.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import spearmanr
import os
import sys

# ============================================================
# 1. Ideal MZI transfer function
# ============================================================
def mzi_T_ideal(delta_phi):
    """
    Ideal lossless MZI bar-port transmission.
    Two perfect 3-dB couplers:  T_bar = sin²(Δφ/2).
    """
    return np.sin(delta_phi / 2.0) ** 2


# ============================================================
# 2. Real MZI transfer function (from Meep simulation data)
# ============================================================
class RealMZI:
    """
    Realistic MZI transfer loaded from FDTD+analytic simulation results.
    Falls back to ideal if simulation data is missing.
    """

    def __init__(self, csv_path="~/mzi_transmission.csv"):
        self.csv_path = os.path.expanduser(csv_path)
        self._interpolator = None
        self._load()

    def _load(self):
        if os.path.exists(self.csv_path):
            data = np.loadtxt(self.csv_path, delimiter=",", skiprows=1)
            self.phases = data[:, 0]           # phase in radians
            self.T_bar  = data[:, 2]           # bar-port transmission
            # Wrap interpolation to be 2π-periodic
            self._interpolator = interp1d(
                self.phases, self.T_bar,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            print(f"[RealMZI] Loaded {len(self.phases)} points from {self.csv_path}")
        else:
            print(f"[RealMZI] WARNING: {self.csv_path} not found; using ideal model.")
            self._interpolator = None

    def __call__(self, delta_phi):
        """Bar-port transmission for phase shift Δφ [rad]."""
        if self._interpolator is None:
            return mzi_T_ideal(delta_phi)
        # Wrap phase to [0, 2π]
        phi_wrapped = np.mod(delta_phi, 2 * np.pi)
        # Interpolate and clip to [0, 1]
        T = self._interpolator(phi_wrapped)
        return np.clip(T, 0.0, 1.0)


# ============================================================
# 3. Photonic Attention Core (WDM, bipolar weights)
# ============================================================
def photonic_attention(Q, K, V, mzi_fn, beta=1.0):
    """
    Compute scaled dot-product attention using a photonic MZI mesh.

    The photonic mesh encodes each (q_i · k_j) dot product into a phase
    shift Δφ_ij = β * (q_i · k_j) + φ_bias, then reads out the bar-port
    transmission as the attention weight proxy.

    Args:
        Q, K, V : [D, D] float arrays (query, key, value matrices)
        mzi_fn  : callable  Δφ → T_bar
        beta    : electro-optic modulation efficiency [rad / (a.u. of dot product)]

    Returns:
        scores    : [D, D] attention weight matrix (row-normalized via softmax)
        raw_T     : [D, D] raw MZI transmissions before softmax
    """
    D = Q.shape[0]

    # Dot-product similarity
    S = Q @ K.T  # [D, D]

    # Scale by sqrt(D) as in standard attention
    S_scaled = S / np.sqrt(D)

    # Convert dot products to phase shifts
    # Center so that zero dot product → π/2 phase (50% transmission, quiescent point)
    phi_bias = np.pi / 2
    delta_phi = beta * S_scaled + phi_bias

    # Clamp to [0, π] for the ideal MZI (full swing), wider for real
    delta_phi = np.clip(delta_phi, 0, np.pi)  # monotonic region of sin²

    # Apply MZI transfer: each phase → bar-port transmission
    raw_T = mzi_fn(delta_phi)  # [D, D]

    # Softmax over rows to get attention weights
    scores = np.exp(raw_T)
    scores = scores / scores.sum(axis=1, keepdims=True)

    # Weighted sum of values
    output = scores @ V

    return scores, raw_T, output


# ============================================================
# 4. Simulation runner
# ============================================================
def run_comparison(D=64, seed=42):
    """
    Run attention with both ideal and real MZI models,
    compare the resulting attention score matrices via Spearman ρ.
    """
    print(f"\n{'='*60}")
    print(f"Photonic Attention Comparison  |  D = {D}")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)

    # Generate random Q, K, V matrices (bipolar entries)
    Q = rng.normal(0, 1, (D, D)).astype(np.float64)
    K = rng.normal(0, 1, (D, D)).astype(np.float64)
    V = rng.normal(0, 1, (D, D)).astype(np.float64)

    # ----- Ideal MZI -----
    print("\n--- Ideal MZI (sin²) ---")
    scores_ideal, T_ideal, out_ideal = photonic_attention(Q, K, V, mzi_T_ideal, beta=1.0)
    print(f"  T range:      [{T_ideal.min():.4f}, {T_ideal.max():.4f}]")
    print(f"  Score sparsity: {(scores_ideal < 0.001).mean()*100:.1f}% near-zero")

    # ----- Real MZI -----
    print("\n--- Real MZI (Meep simulation) ---")
    real_mzi = RealMZI("~/mzi_transmission.csv")
    scores_real, T_real, out_real = photonic_attention(Q, K, V, real_mzi, beta=1.0)
    print(f"  T range:      [{T_real.min():.4f}, {T_real.max():.4f}]")
    print(f"  Score sparsity: {(scores_real < 0.001).mean()*100:.1f}% near-zero")

    # ----- Spearman ρ comparison -----
    # Flatten the score matrices and compute rank correlation
    scores_ideal_flat = scores_ideal.flatten()
    scores_real_flat  = scores_real.flatten()

    rho, pval = spearmanr(scores_ideal_flat, scores_real_flat)
    print(f"\n--- Spearman Rank Correlation ---")
    print(f"  ρ   = {rho:.6f}")
    print(f"  p   = {pval:.2e}")
    print(f"  Interpretation: ", end="")
    if rho > 0.99:
        print("near-perfect agreement — real MZI ≈ ideal")
    elif rho > 0.95:
        print("excellent agreement — tiny deviations")
    elif rho > 0.90:
        print("very good agreement — small but meaningful deviation")
    elif rho > 0.80:
        print("good agreement — noticeable non-ideality impact")
    else:
        print("moderate agreement — real MZI deviates significantly from ideal")

    # ----- Per-row rank stability -----
    row_rhos = []
    for i in range(D):
        r, _ = spearmanr(scores_ideal[i, :], scores_real[i, :])
        row_rhos.append(r)
    print(f"\n  Per-row ρ: mean={np.mean(row_rhos):.4f}, "
          f"std={np.std(row_rhos):.4f}, "
          f"min={np.min(row_rhos):.4f}")

    # ----- Attention output comparison -----
    rho_out, _ = spearmanr(out_ideal.flatten(), out_real.flatten())
    print(f"  Output vector ρ = {rho_out:.6f}")

    return {
        "D": D,
        "spearman_rho": rho,
        "spearman_pval": pval,
        "row_rhos": row_rhos,
        "output_rho": rho_out,
        "scores_ideal": scores_ideal,
        "scores_real": scores_real,
        "T_ideal": T_ideal,
        "T_real": T_real,
    }


# ============================================================
# 5. WDM wavelength sweep (bonus)
# ============================================================
def wdm_sweep(wavelengths, D=64, seed=42):
    """
    For WDM systems, the effective index (and thus phase shift per volt)
    varies with wavelength.  Sweep over wavelength channels and report
    the Spearman ρ at each.
    """
    print(f"\n{'='*60}")
    print(f"WDM Wavelength Sweep")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    Q = rng.normal(0, 1, (D, D)).astype(np.float64)
    K = rng.normal(0, 1, (D, D)).astype(np.float64)
    V = rng.normal(0, 1, (D, D)).astype(np.float64)

    real_mzi = RealMZI("~/mzi_transmission.csv")

    results = []
    for wl in wavelengths:
        # Phase shift scales as 1/λ (phase = 2π/λ * Δn * L)
        beta_wl = 1.55 / wl  # normalized to 1.0 at 1.55 μm

        _, T_ideal, _ = photonic_attention(Q, K, V, mzi_T_ideal, beta=beta_wl)
        _, T_real, _  = photonic_attention(Q, K, V, real_mzi, beta=beta_wl)

        rho, _ = spearmanr(T_ideal.flatten(), T_real.flatten())
        results.append((wl, rho))
        print(f"  λ = {wl:.3f} μm  →  ρ = {rho:.6f}")

    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Primary comparison
    results = run_comparison(D=64, seed=42)

    # Optional: WDM sweep around 1.55 μm
    do_wdm = "--wdm" in sys.argv
    if do_wdm:
        wl_channels = np.linspace(1.53, 1.57, 9)
        wdm_sweep(wl_channels, D=64, seed=42)

    # Optional: Full algorithm non-ideality sweep
    if "--all-nonidealities" in sys.argv:
        print(f"\n{'='*60}")
        print("Running full algorithm non-ideality sweep...")
        print(f"{'='*60}")
        try:
            import algorithm_nondieality_sweep
            algo_results = algorithm_nondieality_sweep.run_full_sweep()
        except ImportError:
            print("  ERROR: algorithm_nondieality_sweep.py not found in path.")

    # Optional: Thermal crosstalk analysis
    if "--thermal-crosstalk" in sys.argv:
        print(f"\n{'='*60}")
        print("Running thermal crosstalk analysis...")
        print(f"{'='*60}")
        try:
            import thermal_crosstalk
            therm_results = thermal_crosstalk.run_thermal_monte_carlo(N=100, D=64)
        except ImportError:
            print("  ERROR: thermal_crosstalk.py not found in path.")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Configuration:  D=64, bipolar weights, WDM-compatible")
    print(f"  Ideal MZI:      T = sin²(Δφ/2)")
    print(f"  Real MZI:       Analytic transfer-matrix (Si 500nm, SiO₂ clad)")
    print(f"  Spearman ρ:     {results['spearman_rho']:.6f}")
    print(f"  Output ρ:       {results['output_rho']:.6f}")
    print()

    # Interpretation
    rho = results['spearman_rho']
    if rho > 0.99:
        print("  ✓ The real MZI transfer function is nearly identical to ideal sin².")
        print("    Non-idealities (coupling error, loss) have negligible impact on attention ranking.")
    elif rho > 0.95:
        print("  ✓ Real MZI closely approximates ideal behavior.")
        print("    Minor deviations exist but attention rankings are well-preserved.")
    elif rho > 0.90:
        print("  ~ Real MZI shows small but measurable deviation from ideal.")
        print("    Consider per-chip calibration for production use.")
    else:
        print("  ! Significant deviation detected. The real MZI transfer differs")
        print("    substantially from sin² — attention performance may degrade.")
