#!/usr/bin/env python3
"""
Complete Photonic-Electronic Transformer Processor Architecture.

A Transformer processor requires 6 subsystems per layer:
  1. Multi-Head Attention (MHA) — QK^T on photonic MZI, rest electronic
  2. Feed-Forward Network (FFN) — 2-layer MLP, fully electronic
  3. Layer Normalization — electronic, O(N) complexity
  4. Residual Connections — electronic, O(N) addition
  5. Activation Storage — SRAM register file between layers
  6. Pipeline Controller — orchestrates photonic + electronic timing

Photonic-Electronic Partition:
  ┌─────────────────────────────────────────────────┐
  │  Transformer Layer (Pre-LN architecture)         │
  │                                                  │
  │  x ──→ [LayerNorm] ──→ [MHA] ──(+)──→ [LayerNorm] ──→ [FFN] ──(+)──→  │
  │        (electronic)    ↑↓        ↑                (electronic)   ↑       │
  │                    ┌───┴──┐  ┌──┴──┐                        ┌──┴──┐    │
  │                    │Photonic│  │Elec │                        │Elec │    │
  │                    │  MZI   │  │QKV, │                        │MLP  │    │
  │                    │ QK^T   │  │softm│                        │2-lay│    │
  │                    │  mesh  │  │ax,V │                        │er   │    │
  │                    └───────┘  └─────┘                        └─────┘    │
  └──────────────────────────────────────────────────────────────────────┘

This hybrid architecture is standard in photonic AI accelerator literature
(Shen et al. 2017, Shastri et al. 2021): photonics handles the O(N^2)
matrix multiply, electronics handles everything else.

This module models the COMPLETE processor, not just the attention engine.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import time

# ============================================================
# 1. System Parameters
# ============================================================
@dataclass
class TransformerProcessorConfig:
    """Complete Transformer processor configuration."""

    # Model architecture
    d_model: int = 64            # hidden dimension
    n_heads: int = 4             # attention heads
    d_ff: int = 256              # FFN hidden dimension (4× expansion)
    n_layers: int = 2            # number of Transformer layers
    vocab_size: int = 500        # vocabulary size
    max_seq_len: int = 64        # maximum sequence length
    d_k: int = 16                # per-head key dimension (= d_model // n_heads)

    # Photonic subsystem
    n_wdm: int = 8              # WDM channels
    mzi_pitch_um: float = 30.0   # MZI pitch (thermally mitigated)
    f_mod_GHz: float = 10.0      # EO modulation bandwidth
    tau_thermal_us: float = 10.0 # thermal tuning time constant

    # Electronic subsystem
    f_clk_GHz: float = 2.0       # digital logic clock
    sram_bw_GBps: float = 100.0  # SRAM bandwidth
    dac_bits: int = 8            # DAC resolution
    adc_bits: int = 8            # ADC resolution

    # Technology node
    tech_node_nm: int = 28       # CMOS technology node

    def __post_init__(self):
        self.d_k = self.d_model // self.n_heads


# ============================================================
# 2. Subsystem Models
# ============================================================
class PhotonicAttentionSubsystem:
    """
    Photonic MZI mesh for QK^T computation.

    Computes: S = Q @ K^T / sqrt(d_k)  (the O(N²) bottleneck)
    Dimensions: [B, H, N, d_k] @ [B, H, d_k, N] → [B, H, N, N]

    Operations per pass: N² × d_k MACs per head, H heads
    Total MACs: B × H × N² × d_k
    """

    def __init__(self, config: TransformerProcessorConfig):
        self.cfg = config

    def compute_macs(self, batch_size=1, seq_len=None):
        """MAC operations performed optically."""
        N = seq_len or self.cfg.max_seq_len
        return batch_size * self.cfg.n_heads * N * N * self.cfg.d_k

    def compute_latency(self):
        """
        Optical latency for QK^T matrix multiply.

        Components:
        - EO modulation: encoding Q, K vectors onto optical carriers
        - Optical propagation: light travel through MZI mesh
        - OE detection: photodetector + TIA readout
        - Thermal settling: phase stabilization (dominant for reconfig)
        """
        # EO modulation: parallel encoding of all elements
        t_mod = 1.0 / (self.cfg.f_mod_GHz * 1e9)  # 100 ps per symbol

        # Optical propagation through MZI array
        # MZI path length ≈ 25 μm per MZI, array width ≈ N * pitch
        N = self.cfg.max_seq_len
        array_width = N * self.cfg.mzi_pitch_um * 1e-6  # meters
        c_opt = 3e8 / 2.8  # speed in Si waveguide
        t_prop = array_width / c_opt  # ~12 ps for N=64, 30μm pitch

        # OE detection
        t_det = 1.0 / (self.cfg.f_mod_GHz * 1e9)  # 100 ps (matched to modulation)

        # ADC conversion
        t_adc = 1.0 / (20e9)  # 50 ps at 20 GS/s

        # Total photonic latency (single pass, weight-static)
        t_photonic = t_mod + t_prop + t_det + t_adc
        # Dominant for reconfiguration: thermal settling
        t_thermal = self.cfg.tau_thermal_us * 1e-6

        return {
            't_modulation': t_mod,
            't_propagation': t_prop,
            't_detection': t_det,
            't_adc': t_adc,
            't_photonic_total': t_photonic,
            't_thermal_settling': t_thermal,
            't_weight_static_pass': t_photonic,
            't_weight_update_pass': t_thermal + t_photonic,
        }

    def compute_energy(self):
        """Energy per QK^T pass (optical only)."""
        N = self.cfg.max_seq_len
        n_mzi = N * N * 2  # PP + PN pairs
        # Laser power: 10 mW per WDM channel
        E_laser = 10e-3 * self.cfg.n_wdm * self.compute_latency()['t_photonic_total']
        # Heater power: 5 mW per MZI (only during phase update)
        E_heaters = 5e-3 * n_mzi * self.cfg.tau_thermal_us * 1e-6
        # Detector + TIA: ~0.5 mW per channel
        E_det = 0.5e-3 * N * N * self.compute_latency()['t_photonic_total']
        return {
            'E_laser_per_pass': E_laser,
            'E_heaters_per_update': E_heaters,
            'E_detectors_per_pass': E_det,
            'E_photonic_per_pass': E_laser + E_det,
            'E_reconfig': E_heaters,
        }


class ElectronicAttentionSubsystem:
    """
    Electronic portion of Multi-Head Attention.

    Operations:
    - Q = W_q @ x  (linear projection)
    - K = W_k @ x
    - V = W_v @ x
    - softmax(QK^T / sqrt(d_k))  (row-wise)
    - Attention @ V  (weighted sum)
    - W_o @ concat(heads)  (output projection)

    Note: QK^T is done optically; this module handles Q,K,V projections,
    softmax, V weighting, and output projection.
    """

    def __init__(self, config: TransformerProcessorConfig):
        self.cfg = config

    def compute_macs(self, batch_size=1, seq_len=None):
        """Electronic MACs for MHA (excluding QK^T)."""
        N = seq_len or self.cfg.max_seq_len
        d = self.cfg.d_model
        # Q,K,V projections: 3 × (N·d·d)
        macs_qkv = 3 * N * d * d
        # Output projection: N·d·d
        macs_out = N * d * d
        # V weighting: N·N·d
        macs_v = N * N * d
        return batch_size * (macs_qkv + macs_out + macs_v)

    def compute_latency(self):
        """Electronic latency for MHA."""
        t_clk = 1.0 / (self.cfg.f_clk_GHz * 1e9)
        N = self.cfg.max_seq_len
        d = self.cfg.d_model
        # QKV projections: can be parallelized across heads
        macs_qkv = 3 * N * d * d
        cycles_qkv = macs_qkv / (d * self.cfg.n_heads)  # parallel across heads
        t_qkv = cycles_qkv * t_clk * 1e-12  # scale to practical

        # Softmax: O(N²) exponentials, O(N) per row
        t_softmax = N * N * t_clk * 0.1  # ~10 cycles per softmax element

        # V weighting: N·N·d (done after photonic QK^T)
        t_v_weight = N * N * d * t_clk / 16  # 16 MAC/cycle (SIMD)

        return {
            't_qkv_projections': N * d * d * t_clk / 64,  # 64-way parallel
            't_softmax': t_softmax,
            't_v_weighting': t_v_weight,
            't_output_projection': N * d * d * t_clk / 64,
            't_electronic_mha_total': N * d * d * t_clk / 64 * 2 + t_softmax + t_v_weight,
        }


class FeedForwardSubsystem:
    """
    Feed-Forward Network (fully electronic).

    Architecture: Linear(d→d_ff) → GELU → Linear(d_ff→d)
    MACs: 2 × N × d × d_ff (two linear layers)
    """

    def __init__(self, config: TransformerProcessorConfig):
        self.cfg = config

    def compute_macs(self, batch_size=1, seq_len=None):
        N = seq_len or self.cfg.max_seq_len
        # Two linear layers: d→d_ff and d_ff→d
        return batch_size * 2 * N * self.cfg.d_model * self.cfg.d_ff

    def compute_latency(self):
        N = self.cfg.max_seq_len
        d = self.cfg.d_model
        d_ff = self.cfg.d_ff
        t_clk = 1.0 / (self.cfg.f_clk_GHz * 1e9)
        # MACs per layer: N · d · d_ff
        # With 64-way SIMD parallelism
        t_ffn = 2 * N * d * d_ff * t_clk / 64
        return {
            't_ffn_total': t_ffn,
            'macs': 2 * N * d * d_ff,
        }


class LayerNormSubsystem:
    """
    Layer Normalization (electronic).

    LN(x) = (x - μ) / σ · γ + β
    Complexity: O(N·d) per LN call
    Two LN calls per Transformer layer.
    """

    def __init__(self, config: TransformerProcessorConfig):
        self.cfg = config

    def compute_latency(self):
        N = self.cfg.max_seq_len
        d = self.cfg.d_model
        t_clk = 1.0 / (self.cfg.f_clk_GHz * 1e9)
        # Mean + variance + normalize + scale: ~20 ops per element
        t_ln = N * d * 20 * t_clk / 64
        return {'t_layernorm': t_ln}


class ResidualAndStorage:
    """
    Residual connections + activation storage.

    Residual: y = x + F(LN(x))
    Storage: SRAM register file holding activations between layers.
    """

    def __init__(self, config: TransformerProcessorConfig):
        self.cfg = config

    def compute_storage(self):
        """Activation storage requirements per layer."""
        N = self.cfg.max_seq_len
        d = self.cfg.d_model
        # Store: x (input), LN(x), MHA output, LN(MHA_out), FFN output
        bytes_per_element = 2  # FP16
        storage_per_layer = 5 * N * d * bytes_per_element
        return {
            'storage_per_layer_bytes': storage_per_layer,
            'storage_total_bytes': storage_per_layer * self.cfg.n_layers,
            'storage_per_layer_KB': storage_per_layer / 1024,
        }

    def compute_latency(self):
        N = self.cfg.max_seq_len
        d = self.cfg.d_model
        t_clk = 1.0 / (self.cfg.f_clk_GHz * 1e9)
        # Element-wise addition: N·d additions
        t_residual = N * d * t_clk / 64
        # SRAM read/write
        t_sram = N * d * 2 * 1e-9 / (self.cfg.sram_bw_GBps)  # 2 bytes per FP16
        return {
            't_residual_add': t_residual,
            't_sram_rw': t_sram,
        }


# ============================================================
# 3. Complete Transformer Layer Timing Model
# ============================================================
@dataclass
class LayerTiming:
    """Timing breakdown for one Transformer layer."""
    # Photonic
    t_qkt_photonic: float = 0.0        # QK^T optical compute
    t_eo_modulation: float = 0.0        # EO encoding
    t_oe_detection: float = 0.0         # OE detection
    t_thermal_settling: float = 0.0     # Phase reconfiguration (if needed)

    # Electronic
    t_qkv_projection: float = 0.0       # Q,K,V linear projections
    t_softmax: float = 0.0              # Softmax (electronic)
    t_v_weighting: float = 0.0          # V weighted sum
    t_output_projection: float = 0.0    # Output linear
    t_ffn: float = 0.0                  # Feed-forward network
    t_layernorm: float = 0.0            # 2× LayerNorm
    t_residual: float = 0.0             # 2× residual add

    # Totals
    t_total_weight_static: float = 0.0  # Fixed weights
    t_total_weight_update: float = 0.0  # Phase reconfig each pass

    # MACs breakdown
    macs_photonic: float = 0.0          # Optical MACs
    macs_electronic: float = 0.0         # Electronic MACs
    macs_total: float = 0.0              # Total MACs

    # Photonic fraction
    photonic_mac_pct: float = 0.0       # % of MACs done optically


class TransformerLayer:
    """
    Complete Transformer layer with photonic attention + electronic FFN.
    """

    def __init__(self, config: TransformerProcessorConfig):
        self.cfg = config
        self.photonic_attn = PhotonicAttentionSubsystem(config)
        self.electronic_attn = ElectronicAttentionSubsystem(config)
        self.ffn = FeedForwardSubsystem(config)
        self.ln = LayerNormSubsystem(config)
        self.residual = ResidualAndStorage(config)

    def model_timing(self, batch_size=1, seq_len=None, weight_update=False):
        """Complete timing model for one Transformer layer."""
        N = seq_len or self.cfg.max_seq_len

        # Photonic attention (QK^T)
        t_phot = self.photonic_attn.compute_latency()
        t_elec_attn = self.electronic_attn.compute_latency()
        t_ffn = self.ffn.compute_latency()
        t_ln = self.ln.compute_latency()
        t_res = self.residual.compute_latency()

        timing = LayerTiming()

        # Photonic portion
        timing.t_eo_modulation = t_phot['t_modulation']
        timing.t_oe_detection = t_phot['t_detection'] + t_phot['t_adc']
        timing.t_qkt_photonic = t_phot['t_weight_static_pass']
        timing.t_thermal_settling = t_phot['t_thermal_settling']

        # Electronic portion
        timing.t_qkv_projection = t_elec_attn['t_qkv_projections']
        timing.t_softmax = t_elec_attn['t_softmax']
        timing.t_v_weighting = t_elec_attn['t_v_weighting']
        timing.t_output_projection = t_elec_attn['t_output_projection']
        timing.t_ffn = t_ffn['t_ffn_total']
        timing.t_layernorm = 2 * t_ln['t_layernorm']  # pre-MHA + pre-FFN
        timing.t_residual = 2 * t_res['t_residual_add']

        # Total latency (approximate: critical path)
        # QKV projections can overlap with photonic QK^T
        t_mha = max(timing.t_qkt_photonic,
                     timing.t_qkv_projection) + timing.t_softmax + timing.t_v_weighting + timing.t_output_projection
        # LayerNorm + MHA + residual
        t_mha_block = timing.t_layernorm / 2 + t_mha + timing.t_residual / 2
        # LayerNorm + FFN + residual
        t_ffn_block = timing.t_layernorm / 2 + timing.t_ffn + timing.t_residual / 2
        # Total layer
        timing.t_total_weight_static = t_mha_block + t_ffn_block
        # With weight update (thermal settling dominates)
        timing.t_total_weight_update = timing.t_total_weight_static + timing.t_thermal_settling

        # MACs
        timing.macs_photonic = self.photonic_attn.compute_macs(batch_size, N)
        timing.macs_electronic = (self.electronic_attn.compute_macs(batch_size, N) +
                                   self.ffn.compute_macs(batch_size, N))
        timing.macs_total = timing.macs_photonic + timing.macs_electronic
        timing.photonic_mac_pct = timing.macs_photonic / timing.macs_total * 100

        return timing


# ============================================================
# 4. Multi-Layer Pipeline Model
# ============================================================
class TransformerPipeline:
    """
    Multi-layer Transformer processor pipeline.

    Layers can be pipelined: while layer i processes token t,
    layer i+1 processes token t-1.

    Throughput is limited by the slowest layer stage.
    """

    def __init__(self, config: TransformerProcessorConfig):
        self.cfg = config
        self.layer = TransformerLayer(config)

    def model_pipeline(self, batch_size=1, seq_len=None, n_tokens=1):
        """Model pipelined execution of multiple tokens through all layers."""
        N = seq_len or self.cfg.max_seq_len
        n_layers = self.cfg.n_layers

        # Single layer timing
        layer_timing = self.layer.model_timing(batch_size, N, weight_update=False)
        t_layer = layer_timing.t_total_weight_static

        # Pipeline: first token takes n_layers * t_layer
        # Subsequent tokens emerge every t_layer (pipeline bubble)
        t_first_token = n_layers * t_layer
        t_subsequent = (n_tokens - 1) * t_layer
        t_total_pipeline = t_first_token + t_subsequent

        # Throughput
        tokens_per_second = n_tokens / t_total_pipeline if t_total_pipeline > 0 else float('inf')

        return {
            't_layer': t_layer,
            't_first_token': t_first_token,
            't_total_pipeline': t_total_pipeline,
            'tokens_per_second': tokens_per_second,
            'n_layers': n_layers,
            'n_tokens': n_tokens,
            'layer_timing': layer_timing,
        }


# ============================================================
# 5. Full Chip Energy/Power Model
# ============================================================
class ChipPowerModel:
    """
    Complete chip power model including all subsystems.
    """

    def __init__(self, config: TransformerProcessorConfig):
        self.cfg = config
        self.photonic = PhotonicAttentionSubsystem(config)
        self.layer = TransformerLayer(config)

    def compute_total_power(self, batch_size=1, seq_len=None):
        """Total chip power for inference."""
        N = seq_len or self.cfg.max_seq_len

        # Photonic
        E_phot = self.photonic.compute_energy()
        t_layer = self.layer.model_timing(batch_size, N).t_total_weight_static
        P_photonic = E_phot['E_photonic_per_pass'] / max(t_layer, 1e-12)

        # Electronic (estimate based on CMOS technology)
        # 28nm CMOS: ~0.5 pJ per MAC
        macs_electronic = self.layer.model_timing(batch_size, N).macs_electronic
        P_electronic = macs_electronic * 0.5e-12 / max(t_layer, 1e-12)

        # Heater power (static, for maintaining phase)
        n_mzi = N * N * 2  # PP + PN
        P_heaters_static = 5e-3 * n_mzi  # 5 mW per heater

        # Lasers
        P_lasers = 10e-3 * self.cfg.n_wdm

        # SRAM leakage + active
        storage = ResidualAndStorage(self.cfg).compute_storage()
        P_sram = storage['storage_total_bytes'] * 1e-9  # ~1 nW per byte (28nm)

        total = P_photonic + P_electronic + P_heaters_static + P_lasers + P_sram

        return {
            'P_photonic_W': P_photonic,
            'P_electronic_W': P_electronic,
            'P_heaters_W': P_heaters_static,
            'P_lasers_W': P_lasers,
            'P_sram_W': P_sram,
            'P_total_W': total,
        }


# ============================================================
# 6. End-to-End System Model
# ============================================================
class TransformerProcessorSystem:
    """
    Complete photonic-electronic Transformer processor.

    Models the full system: embedding → N layers → classifier.
    """

    def __init__(self, config: TransformerProcessorConfig = None):
        self.cfg = config or TransformerProcessorConfig()
        self.pipeline = TransformerPipeline(self.cfg)
        self.power = ChipPowerModel(self.cfg)
        self.storage = ResidualAndStorage(self.cfg)

    def full_system_report(self, batch_size=1, seq_len=None, n_tokens=128):
        """Generate complete system report."""
        N = seq_len or self.cfg.max_seq_len
        cfg = self.cfg

        print("=" * 75)
        print("PHOTONIC-ELECTRONIC TRANSFORMER PROCESSOR — SYSTEM MODEL")
        print("=" * 75)

        # Architecture summary
        print(f"\n  Architecture:")
        print(f"    Model:       {cfg.n_layers}-layer Transformer")
        print(f"    d_model:     {cfg.d_model}")
        print(f"    d_ff:        {cfg.d_ff}")
        print(f"    n_heads:     {cfg.n_heads}")
        print(f"    d_k:         {cfg.d_k}")
        print(f"    seq_len:     {N}")
        print(f"    Vocab:       {cfg.vocab_size}")

        print(f"\n  Technology:")
        print(f"    Photonic:    Si/SiO₂, {cfg.n_wdm}× WDM, {cfg.mzi_pitch_um}μm pitch")
        print(f"    Electronic:  {cfg.tech_node_nm}nm CMOS, {cfg.f_clk_GHz} GHz")
        print(f"    DAC/ADC:     {cfg.dac_bits}-bit @ 20 GS/s")

        # Single layer timing
        layer_t = self.pipeline.layer.model_timing(batch_size, N)
        print(f"\n{'─'*75}")
        print(f"  SINGLE LAYER TIMING")
        print(f"{'─'*75}")
        print(f"    Photonic QK^T:     {layer_t.t_qkt_photonic*1e12:8.1f} ps")
        print(f"      EO modulation:   {layer_t.t_eo_modulation*1e12:8.1f} ps")
        print(f"      OE detection:    {layer_t.t_oe_detection*1e12:8.1f} ps")
        print(f"    Electronic MHA:    {layer_t.t_qkv_projection*1e9:8.1f} ns (QKV+softmax+V)")
        print(f"    FFN (electronic):  {layer_t.t_ffn*1e9:8.1f} ns")
        print(f"    2× LayerNorm:      {layer_t.t_layernorm*1e9:8.1f} ns")
        print(f"    2× Residual:       {layer_t.t_residual*1e9:8.1f} ns")
        print(f"    ─────────────────────────────────")
        print(f"    TOTAL (weight-static): {layer_t.t_total_weight_static*1e9:8.1f} ns")
        print(f"    TOTAL (weight-update): {layer_t.t_total_weight_update*1e6:8.1f} μs")

        # MACs partition
        print(f"\n  MACs Partition (per layer):")
        print(f"    Photonic (QK^T):  {layer_t.macs_photonic:,.0f} ({layer_t.photonic_mac_pct:.1f}%)")
        print(f"    Electronic:    {layer_t.macs_electronic:,.0f} ({100-layer_t.photonic_mac_pct:.1f}%)")
        print(f"    Total MACs:    {layer_t.macs_total:,.0f}")

        # Pipeline throughput
        pipe = self.pipeline.model_pipeline(batch_size, N, n_tokens)
        print(f"\n{'─'*75}")
        print(f"  MULTI-TOKEN PIPELINE ({n_tokens} tokens, {cfg.n_layers} layers)")
        print(f"{'─'*75}")
        print(f"    Latency (1st token): {pipe['t_first_token']*1e6:8.1f} μs")
        print(f"    Latency (steady):    {pipe['t_layer']*1e9:8.1f} ns/token")
        print(f"    Throughput:          {pipe['tokens_per_second']:,.0f} tokens/s")
        print(f"    Throughput:          {pipe['tokens_per_second']*N:,.0f} elements/s")

        # Power
        pwr = self.power.compute_total_power(batch_size, N)
        print(f"\n{'─'*75}")
        print(f"  CHIP POWER (weight-static inference)")
        print(f"{'─'*75}")
        for k, v in pwr.items():
            bar = '█' * int(v / max(pwr['P_total_W'], 1) * 30)
            print(f"    {k:<20s} {v:8.2f} W  {bar}")
        # Energy per token
        E_per_token = pwr['P_total_W'] / max(pipe['tokens_per_second'], 1)
        print(f"\n    Energy per token:  {E_per_token*1e6:8.1f} μJ")
        print(f"    Energy per MAC:    {E_per_token/layer_t.macs_total*1e12:8.2f} pJ")

        # Storage
        stor = self.storage.compute_storage()
        print(f"\n{'─'*75}")
        print(f"  ON-CHIP STORAGE")
        print(f"{'─'*75}")
        print(f"    Per layer:      {stor['storage_per_layer_KB']:8.1f} KB")
        print(f"    Total ({cfg.n_layers} layers): {stor['storage_total_bytes']/1024:8.1f} KB")

        # Comparison
        print(f"\n{'─'*75}")
        print(f"  COMPARISON: Digital vs Photonic-Electronic Hybrid")
        print(f"{'─'*75}")
        # Digital baseline: all-electronic Transformer
        macs_total_per_token = layer_t.macs_total * cfg.n_layers
        # Digital: 28nm, 2 GHz, 64 MAC/cycle → 128 GMAC/s
        digital_mac_rate = 128e9
        digital_latency = macs_total_per_token / digital_mac_rate
        digital_energy = macs_total_per_token * 0.5e-12  # 0.5 pJ/MAC at 28nm

        hybrid_latency = pipe['t_total_pipeline'] / n_tokens  # per-token latency
        hybrid_energy = pwr['P_total_W'] / pipe['tokens_per_second']

        print(f"    {'':20s} {'Digital':>15s} {'Hybrid':>15s} {'Ratio':>10s}")
        print(f"    {'─'*20} {'─'*15} {'─'*15} {'─'*10}")
        print(f"    {'Latency/token':20s} {digital_latency*1e6:14.1f} μs {hybrid_latency*1e6:14.1f} μs {digital_latency/hybrid_latency:9.1f}x")
        print(f"    {'Energy/token':20s} {digital_energy*1e6:14.1f} μJ {hybrid_energy*1e6:14.1f} μJ {digital_energy/max(hybrid_energy,1e-12):9.1f}x")
        print(f"    {'Throughput':20s} {1/digital_latency:14,.0f} tok/s {pipe['tokens_per_second']:14,.0f} tok/s {pipe['tokens_per_second']/(1/digital_latency):9.1f}x")

        # Photonic bottleneck analysis
        print(f"\n{'─'*75}")
        print(f"  PHOTONIC BOTTLENECK")
        print(f"{'─'*75}")
        t_phot = layer_t.t_qkt_photonic
        t_elec = layer_t.t_total_weight_static - t_phot
        if t_phot > t_elec:
            print(f"    ⚠️  Photonic is the bottleneck ({t_phot*1e12:.0f} ps vs {t_elec*1e12:.0f} ps electronic)")
        else:
            print(f"    ✅ Electronic is the bottleneck ({t_elec*1e9:.1f} ns vs {t_phot*1e12:.0f} ps photonic)")
            print(f"       → Photonic QK^T is NOT on the critical path for D={N}")

        # Energy efficiency: thermal vs resonant
        print(f"\n{'─'*75}")
        print(f"  ENERGY EFFICIENCY: Thermal vs Resonant Tuning")
        print(f"{'─'*75}")
        n_mzi = N * N * 2
        # Thermal: 5 mW per heater, continuous
        P_thermal = 5e-3 * n_mzi
        E_thermal_per_token = P_thermal / pipe['tokens_per_second']
        # Resonant: ~1 μW per ring (Q ~ 10,000, voltage-tuned)
        P_resonant = 1e-6 * n_mzi
        E_resonant_per_token = P_resonant / pipe['tokens_per_second']
        # Electro-optic: ~10 fJ per switch, only during reconfig
        E_eo_per_token = 10e-15 * n_mzi  # per weight update

        print(f"    Thermal tuning (current design):")
        print(f"      {P_thermal:.1f} W static, {E_thermal_per_token*1e6:.0f} μJ/token — IMPRACTICAL")
        print(f"    Resonant tuning (rings, Q~10k):")
        print(f"      {P_resonant*1e3:.1f} mW static, {E_resonant_per_token*1e6:.2f} μJ/token — VIABLE")
        print(f"    Electro-optic (plasmonic/MOS, reconfig only):")
        print(f"      ~0 W static, {E_eo_per_token*1e12:.2f} pJ/token — OPTIMAL")
        print(f"    → Thermal tuning is for PROOF-OF-CONCEPT only.")
        print(f"    → Production requires resonant or EO tuning.")

        # Scaling analysis
        print(f"\n{'─'*75}")
        print(f"  SCALING WITH SEQUENCE LENGTH")
        print(f"{'─'*75}")
        print(f"    {'N':>5s}  {'Photonic MACs':>15s}  {'Total MACs':>15s}  {'Photonic %':>12s}  {'Speedup':>10s}")
        for n_test in [16, 32, 64, 128, 256, 512]:
            t_n = self.pipeline.layer.model_timing(1, n_test)
            speedup = (t_n.macs_total * 0.5e-12) / (pwr['P_total_W'] / max(pipe['tokens_per_second'], 1e-12))
            speedup = t_n.macs_total / max(t_n.macs_electronic, 1)
            print(f"    {n_test:5d}  {t_n.macs_photonic:15,.0f}  {t_n.macs_total:15,.0f}  {t_n.photonic_mac_pct:11.1f}%  {speedup:9.1f}x")

        return {
            'layer_timing': layer_t,
            'pipeline': pipe,
            'power': pwr,
            'storage': stor,
            'hybrid_latency': hybrid_latency,
            'hybrid_energy': hybrid_energy,
        }


# ============================================================
# 7. Practical Scaling Analysis: D=64 vs GPT-scale
# ============================================================
def scaling_analysis():
    """Analyze how the photonic advantage scales from D=64 to GPT-scale."""
    print(f"\n{'='*75}")
    print("PHOTONIC ADVANTAGE SCALING (D=64 → GPT-scale)")
    print(f"{'='*75}")

    configs = {
        "Tiny (this work)":    TransformerProcessorConfig(d_model=64, d_ff=256, n_heads=4, max_seq_len=64),
        "BERT-base":           TransformerProcessorConfig(d_model=768, d_ff=3072, n_heads=12, max_seq_len=512),
        "GPT-2 small":         TransformerProcessorConfig(d_model=768, d_ff=3072, n_heads=12, max_seq_len=1024),
        "GPT-3 scale":         TransformerProcessorConfig(d_model=12288, d_ff=49152, n_heads=96, max_seq_len=2048),
    }

    print(f"\n  {'Model':<20s} {'N':>5s} {'d':>5s} {'Photonic MACs':>15s} {'Total MACs':>15s} {'Photonic %':>10s} {'O(N²) fraction':>12s}")
    print(f"  {'─'*20} {'─'*5} {'─'*5} {'─'*15} {'─'*15} {'─'*10} {'─'*12}")

    for name, cfg in configs.items():
        layer = TransformerLayer(cfg)
        t = layer.model_timing(1, cfg.max_seq_len)
        # O(N²) operations: QK^T (N²·d_k) + softmax (N²) + V weighting (N²·d)
        o_n2_macs = cfg.max_seq_len**2 * (cfg.d_k + 1 + cfg.d_model)
        o_n2_pct = o_n2_macs / t.macs_total * 100
        print(f"  {name:<20s} {cfg.max_seq_len:5d} {cfg.d_model:5d} {t.macs_photonic:15,.0f} {t.macs_total:15,.0f} {t.photonic_mac_pct:9.1f}% {o_n2_pct:11.1f}%")

    # Key insight (updated)
    print(f"\n  Key insight:")
    print(f"    D=64:  Photonic is only {configs['Tiny (this work)'].max_seq_len**2*configs['Tiny (this work)'].d_k / (configs['Tiny (this work)'].max_seq_len**2*configs['Tiny (this work)'].d_k + 2*configs['Tiny (this work)'].max_seq_len*configs['Tiny (this work)'].d_model*configs['Tiny (this work)'].d_ff)*100:.0f}% of total MACs → modest speedup")
    print(f"    N=2048: O(N²) attention dominates FFN at long sequence lengths")
    print(f"    Production requires resonant/EO tuning (not thermal) for energy efficiency")
    print(f"    Current design is a PROOF-OF-CONCEPT test chip, not a product")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    # Default: D=64 configuration from this work
    config = TransformerProcessorConfig(
        d_model=64, d_ff=256, n_heads=4,
        max_seq_len=64, n_layers=2,
        mzi_pitch_um=30.0,   # thermally mitigated
        f_mod_GHz=10.0,
        f_clk_GHz=2.0,
        tech_node_nm=28,
    )

    system = TransformerProcessorSystem(config)
    report = system.full_system_report(batch_size=1, seq_len=64, n_tokens=128)

    # Scaling analysis
    scaling_analysis()

    print("\nDone.")
