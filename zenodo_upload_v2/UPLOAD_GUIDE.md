# Zenodo Upload — 热蒸汽三叠式光子 Transformer 处理器

## 编译论文

任选一种:
- **Overleaf**: 上传 main.tex + IEEEtran.cls + references.bib → 编译 → 下载 PDF
- **本地**: `pdflatex main.tex && pdflatex main.tex`

## 上传 Zenodo

1. 打开 https://zenodo.org/ → 登录 → "New Upload"
2. 上传 `main.pdf` (必传)
3. 上传 `source_code.zip` (推荐)

### 基本信息

| 字段 | 值 |
|------|-----|
| Resource type | `Preprint` |
| Title | `A Photonic Transformer Processor: Thermal-Optical Screening with Direct CMOS Detection` |
| Publication date | `2026-06-14` |
| Authors | `Wayne` |
| Affiliation | `Independent Researcher` |
| DOI | **留空** (Zenodo 自动分配) |

### 摘要

```
We present a photonic Transformer attention processor based on a
thermal-optical screening architecture with direct CMOS photodetection.
Unlike silicon-photonic MZI mesh approaches, our design uses free-space
optical paths where light passes through thermo-optic phase shifters
encoding query vectors, is attenuated by electro-optic modulators
encoding key vectors, and is directly detected by a CMOS Ge-on-Si
photodetector array. Bipolar differential detection provides intrinsic
common-mode noise rejection. We verify 25 validation items spanning
device physics, algorithm robustness, thermal-field simulation,
electro-optic bandwidth, temperature drift, detector physics, optical
link budget, and full-system Monte Carlo. At D=128, bipolar dot-product
Spearman rho=0.9971, and complete Transformer attention achieves
rho>=0.995 across all configurations.
```

### Keywords

```
photonic computing
Transformer attention
thermal-optical screening
CMOS photodetection
differential detection
```

### License

`Creative Commons Attribution 4.0 International` (CC BY 4.0)

### Related Works

| Relation | Identifier | Scheme |
|----------|-----------|--------|
| `is supplemented by` | `https://github.com/administere/photonic-attention` | `URL` |

## 与之前论文的区别

之前 Zenodo (MZI 波导网格):
- `Full-Stack Pre-Tapeout Verification of a WDM Bipolar Photonic Attention Accelerator`
- MZI 波导干涉仪路线 (已归档至 mzi_alternative/)

本篇 (热蒸汽三叠式):
- `A Photonic Transformer Processor: Thermal-Optical Screening with Direct CMOS Detection`
- 热蒸汽筛选 → CMOS 直接探测, 三叠式垂直堆叠
- **独立新 DOI, 不覆盖旧论文**
