# Zenodo Upload Instructions

## Files

| File | Size | Note |
|------|------|------|
| `main.pdf` | 275 KB | 已编译论文 (required) |
| `photonic-attention-zenodo.zip` | 673 KB | 代码+数据+报告 (recommended) |

---

## Basic Information

| Field | Value |
|-------|-------|
| DOI | **Leave blank** (Zenodo auto-assigns) |
| Resource type | `Preprint` |
| Title | `Full-Stack Pre-Tapeout Verification of a WDM Bipolar Photonic Attention Accelerator` |
| Publication date | `2026-06-14` |
| Authors | `Wayne` |
| Affiliation | `Independent Researcher` |
| ORCID | Optional |

## Description

```
We present a complete pre-tapeout verification of a photonic dot-product
engine for Transformer attention, built on a WDM bipolar Mach-Zehnder
interferometer (MZI) mesh in silicon photonics (500nm waveguides, SiO2
cladding, 1.55um). The verification stack spans five layers: algorithm-level
robustness (worst-case Spearman rho >= 0.977), device-physics FDTD (rho =
0.999833 vs ideal sin^2), process-corners Monte Carlo (N=500, 100% yield),
gdsfactory layout (3.62 mm^2), and downstream task impact (0.04% process
noise accuracy loss). All code and data at github.com/administere/photonic-attention.

全产业AI替代之后的光子注意力加速器流片前全栈验证。包含算法鲁棒性、
器件物理FDTD、工艺角蒙特卡洛、版图、下游任务五层验证。全部通过。
```

---

## License

**`Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International`** (CC BY-NC-ND 4.0)

Same as arXiv. If you prefer CC BY only, use `Creative Commons Attribution 4.0 International`.

---

## Keywords

```
photonic computing
attention accelerator
Mach-Zehnder interferometer
WDM
pre-tapeout verification
FDTD
silicon photonics
```

---

## Related Works

| Relation | Identifier | Scheme |
|----------|-----------|--------|
| `is supplemented by` | `https://github.com/administere/photonic-attention` | `URL` |

After arXiv goes live, add:

| Relation | Identifier | Scheme |
|----------|-----------|--------|
| `is identical to` | `arXiv:XXXX.XXXXX` | `arXiv` |

---

## Software

| Field | Value |
|-------|-------|
| Repository URL | `https://github.com/administere/photonic-attention` |
| Programming language | `Python` |
| Development Status | `Active` |

---

## Publishing Information

**Leave all blank** for preprint upload.

After arXiv is active, optionally add:

| Field | Value |
|-------|-------|
| Journal | `arXiv preprint` |
| Page range | `arXiv:XXXX.XXXXX` |

---

## Summary Checklist

- [ ] Upload `main.pdf` (required)
- [ ] Upload `photonic-attention-zenodo.zip` (recommended)
- [ ] Resource type: `Preprint`
- [ ] License: `CC BY-NC-ND 4.0`
- [ ] DOI: leave blank for auto-assignment
- [ ] Related: GitHub URL
- [ ] Click **Publish**
