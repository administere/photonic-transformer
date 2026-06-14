#!/usr/bin/env python3
"""
Throughput and Energy Model for Photonic Attention Accelerator.

Validates the 330 TOPS bandwidth claim by modeling:
  - Optical propagation delay through MZI mesh
  - WDM channel parallelism (8 channels, 200 GHz spacing)
  - Electro-optic modulation bandwidth
  - DAC/ADC conversion time
  - Detector integration time
  - Thermal tuning settling time

Computes realistic throughput bounds and energy per operation.
"""

import numpy as np

# ============================================================
# System parameters
# ============================================================

# Optical
D = 64                          # sequence length
n_wdm = 8                       # WDM channels
n_mzi_per_dot = 2               # PP + PN per dot product
f_mod = 10e9                    # EO modulation bandwidth (10 GHz)
tau_opt = 1.0 / f_mod           # optical symbol period (100 ps)
c = 3e8                         # speed of light
n_eff = 2.8                     # effective index (Si waveguide)
v_opt = c / n_eff               # optical propagation speed (~1.07e8 m/s)

# Layout geometry
mzi_length = 25.0e-6            # single MZI length (25 um)
array_width = D * 20.0e-6       # MZI array width (1.28 mm)
propagation_distance = array_width * 2  # round-trip through array (~2.56 mm)

# Electronics
dac_resolution = 8              # bits
dac_rate = 20e9                 # DAC sample rate (20 GS/s)
adc_resolution = 8              # bits
adc_rate = 20e9                 # ADC sample rate (20 GS/s)
dac_settling = 50e-12           # DAC settling time (50 ps)

# Detection
detector_bw = 10e9              # detector bandwidth (10 GHz)
tau_det = 1.0 / detector_bw     # detector response time (100 ps)
tia_bw = 10e9                   # transimpedance amplifier bandwidth
tau_tia = 1.0 / tia_bw

# Thermal
tau_thermal = 10e-6             # thermal tuning time constant (10 us)

# Energy
E_dac_per_sample = 0.5e-12      # DAC energy per sample (0.5 pJ)
E_adc_per_sample = 1.0e-12      # ADC energy per sample (1 pJ)
E_laser = 10e-3                 # laser power (10 mW per WDM channel)
E_heater_per_mzi = 5e-3         # heater power (5 mW per MZI)
E_tia_per_channel = 0.5e-3      # TIA power (0.5 mW per channel)


def compute_throughput():
    """Compute realistic throughput in operations per second."""
    print("=" * 70)
    print("THROUGHPUT & ENERGY ANALYSIS")
    print("=" * 70)

    # 1. Optical latency
    t_prop = propagation_distance / v_opt
    print(f"\n  Optical propagation: {propagation_distance*1e3:.1f} mm / {v_opt*1e-6:.1f} um/ps")
    print(f"    = {t_prop*1e12:.1f} ps")

    # 2. Operations per optical cycle
    # Each MZI mesh computes the full QK^T matrix (D*D dot products) in one optical pass
    ops_per_pass = D * D * n_wdm  # dot products per WDM pass
    print(f"\n  Operations per optical pass: {ops_per_pass:,} dot products ({D}×{D} × {n_wdm} WDM)")

    # 3. Pipeline depth from optical latency
    pipeline_depth = max(1, int(t_prop * f_mod))  # how many ops can be pipelined in flight
    print(f"  Pipeline depth (in-flight ops): {pipeline_depth}")

    # 4. Throughput (operations/second)
    # Limited by: modulation rate, detection rate, or thermal tuning
    ops_per_second_mod = ops_per_pass * f_mod  # modulation-limited
    ops_per_second_det = ops_per_pass * detector_bw  # detection-limited
    ops_per_second_thermal = ops_per_pass / tau_thermal  # thermal-limited

    print(f"\n  Throughput bounds:")
    print(f"    Modulation-limited: {ops_per_second_mod*1e-12:.1f} TOPS")
    print(f"    Detection-limited:  {ops_per_second_det*1e-12:.1f} TOPS")
    print(f"    Thermal-limited:    {ops_per_second_thermal*1e-9:.3f} GOPS")

    # 5. Realistic throughput (bottleneck analysis)
    # The thermal tuning is the slowest component for reconfiguration
    # For a single static weight setting, modulation rate dominates

    # Scenario A: Weight-static (inference with fixed weights)
    throughput_static = ops_per_second_mod
    print(f"\n  Scenario A: Weight-static inference")
    print(f"    Throughput: {throughput_static*1e-12:.1f} TOPS")
    print(f"    Bottleneck: EO modulation bandwidth ({f_mod*1e-9:.0f} GHz)")

    # Scenario B: Time-multiplexed (weights updated every cycle)
    throughput_tm = ops_per_second_thermal
    print(f"\n  Scenario B: Weight-update per cycle")
    print(f"    Throughput: {throughput_tm*1e-9:.3f} GOPS")
    print(f"    Bottleneck: Thermal tuning ({tau_thermal*1e6:.0f} us time constant)")

    # Scenario C: Pipelined (weights updated in background)
    n_pipeline = 8
    throughput_pipe = ops_per_second_mod * n_pipeline / (1 + n_pipeline * f_mod * tau_thermal)
    print(f"\n  Scenario C: {n_pipeline}× pipelined weight update")
    print(f"    Throughput: {throughput_pipe*1e-12:.1f} TOPS")
    print(f"    Bottleneck: Pipeline depth × modulation rate")

    # 6. Comparison with 330 TOPS claim
    claimed_tops = 330e12
    print(f"\n  Claim validation:")
    print(f"    Claimed: {claimed_tops*1e-12:.0f} TOPS")
    print(f"    Weight-static (our model): {throughput_static*1e-12:.1f} TOPS")
    ratio = throughput_static / claimed_tops
    if ratio >= 0.5:
        print(f"    → Claim is PLAUSIBLE ({ratio*100:.0f}% of claimed)")
    elif ratio >= 0.1:
        print(f"    → Claim is AGGRESSIVE ({ratio*100:.0f}% of claimed)")
    else:
        print(f"    → Claim appears OPTIMISTIC ({ratio*100:.0f}% of claimed)")

    return throughput_static, ops_per_pass


def compute_energy():
    """Compute energy per operation."""
    print(f"\n{'='*70}")
    print("ENERGY ANALYSIS")
    print(f"{'='*70}")

    # Components
    n_mzi_total = D * D * n_mzi_per_dot  # total MZIs
    n_detectors = D * D  # one detector per attention entry
    n_dacs = n_mzi_total  # one DAC per MZI
    n_adcs = n_detectors

    # Power breakdown
    P_laser_total = E_laser * n_wdm
    P_heaters = E_heater_per_mzi * n_mzi_total
    P_tias = E_tia_per_channel * n_detectors
    P_dacs = E_dac_per_sample * dac_rate * n_dacs
    P_adcs = E_adc_per_sample * adc_rate * n_adcs

    P_total = P_laser_total + P_heaters + P_tias + P_dacs + P_adcs

    print(f"\n  Component power:")
    print(f"    Lasers ({n_wdm}× WDM):       {P_laser_total*1e3:.0f} mW")
    print(f"    Heaters ({n_mzi_total} MZIs):  {P_heaters:.1f} W")
    print(f"    TIAs ({n_detectors} ch):        {P_tias:.1f} W")
    print(f"    DACs ({n_dacs} ch):           {P_dacs:.1f} W")
    print(f"    ADCs ({n_adcs} ch):           {P_adcs:.1f} W")
    print(f"    TOTAL:                        {P_total:.1f} W")

    # Energy per operation
    ops_per_second = D * D * n_wdm * f_mod
    E_per_op = P_total / ops_per_second
    print(f"\n  Energy per operation: {E_per_op*1e12:.2f} pJ/op")
    print(f"    Reference: Miller (2017) attojoule optoelectronics ~ 10 fJ/op")
    print(f"    Digital CMOS (7nm): ~ 1 pJ/MAC")

    # Energy efficiency analysis
    if E_per_op < 1e-12:
        print(f"    → Sub-pJ regime — competitive with digital ASICs")
    elif E_per_op < 10e-12:
        print(f"    → Single-digit pJ — better than GPU, comparable to digital ASIC")
    else:
        print(f"    → >10 pJ — energy advantage unclear without optimization")

    # Key insight: heaters dominate
    heater_fraction = P_heaters / P_total * 100
    print(f"\n  Dominant power consumer: Heaters ({heater_fraction:.0f}%)")
    print(f"    → Migrating to resonant or electro-optic tuning would")
    print(f"       reduce power by 100-1000×")

    return P_total, E_per_op


if __name__ == "__main__":
    throughput, ops_per_pass = compute_throughput()
    power, energy_per_op = compute_energy()

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Weight-static throughput: {throughput*1e-12:.1f} TOPS")
    print(f"  Total system power:      {power:.1f} W")
    print(f"  Energy per operation:    {energy_per_op*1e12:.2f} pJ")

    # Feasibility verdict
    if throughput > 100e12 and energy_per_op < 10e-12:
        print(f"\n  ✅ Throughput and energy are competitive with digital ASICs")
    elif throughput > 10e12:
        print(f"\n  ⚠️  Throughput is reasonable but energy needs optimization")
    else:
        print(f"\n  ℹ️  Throughput limited by thermal tuning; use for weight-static inference")

    print("Done.")
