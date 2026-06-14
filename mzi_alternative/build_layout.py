#!/usr/bin/env python3
"""
Dot Product Cell Layout using gdsfactory 8.x.
- 2 MZIs (PP pair, exploiting symmetry for PN)
- 1 microring demux placeholder
- Exports GDS and reports area estimation.
"""

import gdsfactory as gf
import numpy as np
import os
import sys

print(f"gdsfactory version: {gf.__version__}")

# ============================================================
# 1. Component parameters (based on our device simulation)
# ============================================================
wg_width      = 0.500   # Silicon waveguide width [μm]
dc_gap        = 0.200   # Directional coupler gap [μm]
dc_length     = 10.0    # 3-dB coupling length from FDTD [μm]
arm_length    = 5.0     # Interferometer arm between DCs [μm]
phase_length  = 3.0     # Thermo-optic phase shifter length [μm]
bend_radius   = 5.0     # Waveguide bend radius [μm]
io_pitch      = 2.0     # I/O port pitch [μm]
demux_radius  = 6.0     # Microring demux radius [μm]
arm_sep       = 4.0     # MZI arm separation center-to-center [μm]
mzi_pitch     = 20.0    # pitch between two MZIs [μm]

# Cross-section for Si waveguide
wg_xs = gf.cross_section.strip(width=wg_width, layer=(1, 0))

# Cross-section for heater (phase shifter)
heater_layer = (2, 0)

# Cross-section for ring demux
ring_layer = (3, 0)
ring_xs = gf.cross_section.strip(width=wg_width, layer=ring_layer)


# ============================================================
# 2. Build a single MZI using gdsfactory built-in components
# ============================================================
def build_mzi(name="mzi", x0=0, y0=0, use_gf_mzi=True):
    """
    Create one MZI sub-component.

    Two approaches:
      (a) use_gf_mzi=True: use gdsfactory's built-in MZI component
          (requires gf.components.mzi to be available)
      (b) use_gf_mzi=False: manual assembly from straight/coupler primitives
          (fallback for older gdsfactory versions)
    """
    mzi = gf.Component(name)

    if use_gf_mzi:
        try:
            # Attempt to use gdsfactory's built-in MZI
            # gf.components.mzi2x2 or mzi_arm or mzi
            if hasattr(gf.components, 'mzi2x2_2x2'):
                gf_mzi = gf.components.mzi2x2_2x2(
                    delta_length=arm_length,
                    length_y=arm_sep,
                    bend=gf.components.bend_euler(radius=bend_radius),
                    coupler=gf.components.coupler(
                        gap=dc_gap, length=dc_length, cross_section=wg_xs,
                    ),
                )
            elif hasattr(gf.components, 'mzi'):
                gf_mzi = gf.components.mzi(
                    delta_length=0,
                    length_x=dc_length,
                    length_y=arm_sep,
                    bend=gf.components.bend_euler(radius=bend_radius),
                    splitter=gf.components.mmi1x2(),
                )
            else:
                raise AttributeError("No built-in MZI found")

            ref = mzi << gf_mzi
            ref.movex(x0)
            ref.movey(y0)
            print(f"    Using gdsfactory built-in MZI component: {type(gf_mzi).__name__}")
            return mzi

        except Exception as e:
            print(f"    Built-in MZI not available ({e}), using manual assembly.")
            # Fall through to manual assembly

    # ----- Manual assembly from primitives -----
    y_top = y0 + arm_sep / 2
    y_bot = y0 - arm_sep / 2

    # Use Euler bends for realistic curve geometry
    try:
        bend = gf.components.bend_euler(radius=bend_radius, cross_section=wg_xs)
        has_bends = True
    except Exception:
        has_bends = False

    if has_bends:
        # --- Realistic layout with bends ---
        # Input waveguides → bends → DC1 → arm → DC2 → bends → output
        # Build top arm
        input_top = gf.components.straight(length=dc_length / 2, cross_section=wg_xs)
        ref = mzi << input_top
        ref.movex(x0).movey(y_top)

        # DC1 top
        dc1_top = gf.components.straight(length=dc_length, cross_section=wg_xs)
        ref = mzi << dc1_top
        ref.movex(x0 + dc_length / 2).movey(y_top)

        # Arm top (between DCs)
        arm_top = gf.components.straight(length=arm_length, cross_section=wg_xs)
        ref = mzi << arm_top
        ref.movex(x0 + dc_length + dc_length / 2).movey(y_top)

        # DC2 top
        dc2_top = gf.components.straight(length=dc_length, cross_section=wg_xs)
        ref = mzi << dc2_top
        ref.movex(x0 + dc_length + arm_length + dc_length / 2).movey(y_top)

        # Bottom arm — same structure
        dc1_bot = gf.components.straight(length=dc_length, cross_section=wg_xs)
        ref = mzi << dc1_bot
        ref.movex(x0 + dc_length / 2).movey(y_bot)

        arm_bot = gf.components.straight(length=arm_length, cross_section=wg_xs)
        ref = mzi << arm_bot
        ref.movex(x0 + dc_length + dc_length / 2).movey(y_bot)

        dc2_bot = gf.components.straight(length=dc_length, cross_section=wg_xs)
        ref = mzi << dc2_bot
        ref.movex(x0 + dc_length + arm_length + dc_length / 2).movey(y_bot)

        # Add bend-based entry/exit at I/O ports
        # (Bend entries from the sides to avoid sharp corners)
        entry_top = gf.components.bend_euler(radius=bend_radius, cross_section=wg_xs)
        ref_entry = mzi << entry_top
        ref_entry.movex(x0 - dc_length / 2).movey(y_top + bend_radius)

        exit_top = gf.components.bend_euler(radius=bend_radius, cross_section=wg_xs)
        ref_exit = mzi << exit_top
        ref_exit.movex(x0 + 2 * dc_length + arm_length + dc_length / 2).movey(y_top + bend_radius)

        entry_bot = gf.components.bend_euler(radius=bend_radius, cross_section=wg_xs)
        ref_entry = mzi << entry_bot
        ref_entry.movex(x0 - dc_length / 2).movey(y_bot - bend_radius)

        exit_bot = gf.components.bend_euler(radius=bend_radius, cross_section=wg_xs)
        ref_exit = mzi << exit_bot
        ref_exit.movex(x0 + 2 * dc_length + arm_length + dc_length / 2).movey(y_bot - bend_radius)

        total_len = 2 * dc_length + arm_length + 2 * bend_radius
    else:
        # Fallback to simple straight blocks (original implementation)
        total_len = dc_length * 2 + arm_length
        # ... use original straight block approach but keep it simple
        for y in [y_top, y_bot]:
            for x_start in [x0, x0 + dc_length, x0 + dc_length + arm_length]:
                wg_len = dc_length if x_start < x0 + dc_length else (
                    arm_length if x_start < x0 + dc_length + arm_length else dc_length)
                wg = gf.components.straight(length=wg_len, cross_section=wg_xs)
                ref = mzi << wg
                ref.movex(x_start).movey(y)

    # Phase shifter (heater) on top arm — drawn as rectangle
    # Positioned over the arm section, offset slightly for visibility
    ps = gf.components.rectangle(
        size=(phase_length, wg_width + 2.0),
        layer=heater_layer,
    )
    ref = mzi << ps
    ref.movex(x0 + dc_length + (arm_length - phase_length) / 2)
    ref.movey(y_top - 1.0)

    mzi_total_len = has_bends and (2 * dc_length + arm_length + 2 * bend_radius) or (2 * dc_length + arm_length)

    # Add ports
    mzi.add_port(name="in_top",  center=(x0 - 2.0, y_top), width=wg_width, layer=(1, 0),
                 orientation=180, port_type="optical")
    mzi.add_port(name="in_bot",  center=(x0 - 2.0, y_bot), width=wg_width, layer=(1, 0),
                 orientation=180, port_type="optical")
    mzi.add_port(name="out_top", center=(x0 + mzi_total_len + 2.0, y_top),
                 width=wg_width, layer=(1, 0), orientation=0, port_type="optical")
    mzi.add_port(name="out_bot", center=(x0 + mzi_total_len + 2.0, y_bot),
                 width=wg_width, layer=(1, 0), orientation=0, port_type="optical")

    return mzi


# ============================================================
# 3. Build dot product cell
# ============================================================
def build_dot_product_cell():
    """Assemble two MZIs + microring demux into one cell."""

    cell = gf.Component("dot_product_cell")

    total_mzi_len = 2 * dc_length + arm_length  # 25 μm

    # MZI A (PP) at y = +mzi_pitch/2
    mzi_a = build_mzi("mzi_PP", x0=5.0, y0=mzi_pitch / 2)
    ref_a = cell << mzi_a

    # MZI B (PN) at y = -mzi_pitch/2
    mzi_b = build_mzi("mzi_PN", x0=5.0, y0=-mzi_pitch / 2)
    ref_b = cell << mzi_b

    # Microring demux placeholder (ring + bus waveguide)
    ring = gf.components.ring(radius=demux_radius, width=wg_width, layer=ring_layer)
    ref_ring = cell << ring
    # Position to the right of the MZIs
    ring_x = 5.0 + total_mzi_len + demux_radius + 3.0
    ref_ring.movex(ring_x)
    ref_ring.movey(0)

    # Bus waveguide for ring
    bus_len = demux_radius * 3
    bus_wg = gf.components.straight(length=bus_len, cross_section=ring_xs)
    ref_bus = cell << bus_wg
    ref_bus.movex(ring_x - bus_len / 2 + demux_radius)
    ref_bus.movey(demux_radius + 0.1)  # evanescently coupled

    # Add cell-level ports
    mzi_x0 = 5.0
    mzi_x1 = mzi_x0 + total_mzi_len
    y_a = mzi_pitch / 2
    y_b = -mzi_pitch / 2

    cell.add_port(name="mziA_in_top",  center=(mzi_x0 - 2, y_a + arm_sep/2),
                  width=wg_width, layer=(1,0), orientation=180, port_type="optical")
    cell.add_port(name="mziA_in_bot",  center=(mzi_x0 - 2, y_a - arm_sep/2),
                  width=wg_width, layer=(1,0), orientation=180, port_type="optical")
    cell.add_port(name="mziA_out_top", center=(mzi_x1 + 2, y_a + arm_sep/2),
                  width=wg_width, layer=(1,0), orientation=0, port_type="optical")
    cell.add_port(name="mziA_out_bot", center=(mzi_x1 + 2, y_a - arm_sep/2),
                  width=wg_width, layer=(1,0), orientation=0, port_type="optical")
    cell.add_port(name="mziB_in_top",  center=(mzi_x0 - 2, y_b + arm_sep/2),
                  width=wg_width, layer=(1,0), orientation=180, port_type="optical")
    cell.add_port(name="mziB_in_bot",  center=(mzi_x0 - 2, y_b - arm_sep/2),
                  width=wg_width, layer=(1,0), orientation=180, port_type="optical")
    cell.add_port(name="mziB_out_top", center=(mzi_x1 + 2, y_b + arm_sep/2),
                  width=wg_width, layer=(1,0), orientation=0, port_type="optical")
    cell.add_port(name="mziB_out_bot", center=(mzi_x1 + 2, y_b - arm_sep/2),
                  width=wg_width, layer=(1,0), orientation=0, port_type="optical")

    return cell


# ============================================================
# 4. Main: Build & export
# ============================================================
if __name__ == "__main__":
    print("Building dot product cell layout...")
    cell = build_dot_product_cell()

    # Compute bounding box
    bbox = cell.bbox()
    if bbox is not None:
        # KLayout Box in database units (nm), convert to μm
        DBU = 1000.0  # 1 μm = 1000 nm
        xmin = float(bbox.left) / DBU
        xmax = float(bbox.right) / DBU
        ymin = float(bbox.bottom) / DBU
        ymax = float(bbox.top) / DBU
    else:
        xmin, ymin, xmax, ymax = 0, -mzi_pitch, 50, mzi_pitch

    area = (xmax - xmin) * (ymax - ymin)

    print(f"\n  Cell: {cell.name}")
    print(f"  Bounding box: x=[{xmin:.1f}, {xmax:.1f}], y=[{ymin:.1f}, {ymax:.1f}] μm")
    print(f"  Dimensions:   {xmax-xmin:.1f} × {ymax-ymin:.1f} μm")
    print(f"  Die area:     {area:.1f} μm²")
    print(f"  Ports:        {len(cell.ports)}")

    # Save GDS
    out_path = os.path.expanduser("~/dot_product_cell.gds")
    cell.write_gds(out_path)
    print(f"  GDS saved to: {out_path}")

    # ---- Area scaling estimate ----
    D = 64
    n_dot_units = (D * D) // 2  # exploiting symmetry of QK^T
    total_area_mm2 = n_dot_units * area * 1e-6  # μm² → mm²

    print(f"\n  Area scaling (D={D}):")
    print(f"    Dot-product units needed:  {n_dot_units}")
    print(f"    Single cell area:          {area:.1f} μm² = {area*1e-6:.4f} mm²")
    print(f"    Core area (parallel):      {total_area_mm2:.2f} mm²")
    print(f"    + 40% routing overhead:    {total_area_mm2 * 1.4:.2f} mm²")

    # Also estimate with pipelining — not all units need to be on-die simultaneously
    # A typical photonic tensor core pipelines depth-wise
    n_pipeline_stages = 8  # parallelism in time domain
    pipelined_area = total_area_mm2 / n_pipeline_stages
    total_with_routing = pipelined_area * 1.4
    print(f"    With 8x pipelining:        {pipelined_area:.2f} mm² (core)")
    print(f"    Pipelined + routing:       {total_with_routing:.2f} mm²")
    print(f"    → Realistic target:        ≤ 25 mm² (5×5 mm die)")

    # Save area estimate
    with open(os.path.expanduser("~/layout_area.txt"), "w") as f:
        f.write(f"Dot Product Cell Layout — Area Estimate\n")
        f.write(f"========================================\n")
        f.write(f"Single cell: {area:.1f} μm² ({area*1e-6:.4f} mm²)\n")
        f.write(f"Cell dimensions: {xmax-xmin:.1f} × {ymax-ymin:.1f} μm\n")
        f.write(f"Dot-product units (D=64, sym): {n_dot_units}\n")
        f.write(f"Core area (full parallel): {total_area_mm2:.2f} mm²\n")
        f.write(f"Core area (8x pipelined): {pipelined_area:.2f} mm²\n")
        f.write(f"Pipelined + 40% routing: {total_with_routing:.2f} mm²\n")
        f.write(f"Realistic target: ≤ 25 mm² (5×5 mm die)\n")
    print("  Estimate saved to ~/layout_area.txt")
    print("\nDone.")
