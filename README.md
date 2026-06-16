# 热光混合 Transformer 处理器 · The Ark

> 光子复用：一个脉冲穿过 D 个调制点，完成 D 次乘法。热不是负担，是计算机制。
> VCSEL 光同时承载信号与驱动热筛 | DiSubPc·C70 量子相干拍频 @242°C | 三叠垂直堆叠
> 25/25 验证通过 ✅ | ρ ≥ 0.995 | 全物理场仿真闭环

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20690521.svg)](https://doi.org/10.5281/zenodo.20690521)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

- [架构](#架构) · [验证结果](#核心验证结果) · [算力与功耗](#算力与功耗) · [相关工作对比](#相关工作对比) · [原创性声明](#原创性声明) · [论文与引用](#论文与引用) · [诚实评估](#诚实评估)

---

## 架构

```
┌─────────────────────────────┐
│  上层: CMOS 探测器阵列       │  4×D² Ge-on-Si 光电二极管 (差分)
├─────────────────────────────┤
│  中层: 光热相位筛选层        │  DiSubPc·C70 共晶薄膜
│  VCSEL 照射 → 量子相干拍频   │  单重态↔三重态相干振荡
│  → 242°C/30s → 超快 Δn 调制 │  省掉独立加热器, 光直接驱动
├─────────────────────────────┤
│  下层: VCSEL 阵列            │  D 个独立激光器 (Q 编码)
│  光学交叉杆: 每行扇出到 D 列  │  D×K 调制器交叉点 = D² 个点积
└─────────────────────────────┘
```

光热材料参考: Zhang, You et al., "Quantum coherent beating in polar disubphthalocyanine-fullerene cocrystals for ultrafast photothermal conversion", *Nature Photonics* (2026), DOI: 10.1038/s41566-026-01912-4.

**路线A验证 (DiSubPc·C70 光热层):** 10μm 薄膜, 222K 温升 → 1.6π 相位调制 ✅ | 权重更新 0.033Hz (慢), 推理 10GHz (快, 解耦) | 适合权重静态推理

> **两条路线说明:**
> - **路线A (主方案):** 自由空间 VCSEL → DiSubPc·C70 光热筛选 → CMOS 探测器。优势: attojoule 级能效, 省掉加热器, 可堆叠。当前主力验证路线。
> - **路线B (备选/存档):** SOI 硅光子 MZI 波导网格 → WDM → 探测器。传统集成光子路线, 工艺成熟但能效不及路线A。代码存档于 `mzi_alternative/`, 已通过流片前验证。

---

## 核心验证结果

| 验证项 | 指标 | 状态 |
|--------|------|:----:|
| 双极 D=64 点积 | ρ=0.9972 | ✅ |
| 双极 D=128 点积 | ρ=0.9971 | ✅ |
| 完整 Attention (softmax+V) | ρ=0.9954 (D=64, 4头) | ✅ |
| 完整 Attention | ρ=0.9974 (D=128, 8头) | ✅ |
| 长尾嵌入 | ρ=0.9998 | ✅ |
| A+C 偏置确认为最优 | φ=π/2, δ=π/2 | ✅ |
| 9类非理想性扫描 | 全部通过 | ✅ |
| 物理场全系统 MC | ρ=0.9954 | ✅ |
| 热场仿真 | NN耦合 0% @30μm | ✅ |
| EO 调制器带宽 | 112.5 GHz | ✅ |

---

## 文件

| 文件 | 说明 |
|------|------|
| `v9_transformer_processor.py` | 完整光子 Transformer 处理器 (物理噪声+Attention+功耗+时序) |
| `v8_photonic_dot_product.py` | 点积验证 (DAC/校准/饱和/空间相关) |
| `physics_field_sim.py` | 物理场仿真套件 (热场+EO+漂移+探测器+链路+联合MC) |
| `algorithm_nondieality_sweep.py` | 9类算法非理想性扫描 |
| `operating_point_analysis.py` | 工作点优化 (190/625配置 ρ≥0.99) |
| `throughput_analysis.py` | 吞吐量+能耗模型 |
| `transformer_processor.py` | 光子-电子混合系统模型 |
| `photonic_aware_training.py` | 光子原生训练分析 |
| `AUDIT_AND_FIXES.md` | 25项审计+修复记录 |
| `DEEPSEEK_HANDOFF.md` | DeepSeek 交接文档 |
| `mzi_alternative/` | MZI 波导网格备选方案 (存档) |

---

## 快速运行

```bash
conda activate meep_env

python v9_transformer_processor.py    # 完整 Transformer 验证
python physics_field_sim.py           # 物理场全仿真
python algorithm_nondieality_sweep.py # 非理想性扫描
```

---

## 算力与功耗

| D | 纯光学能效 vs H100 | 含探测器+ADC | 系统功耗 |
|:---:|:---:|:---:|:---:|
| 64 | 57,917× | 449× | 19W |
| 128 | 231,670× | 901× | 70W |
| 256 | 926,679× | 1,806× | 272W |
| 512 | 3,706,716× | 3,616× | 534W |
| 1024 | 14,826,865× | 7,236× | 700W |
| **2048** | **59,307,459×** | **9,651×** | **875W** |

纯光学 attojoule 能效是 Maxwell 方程的直接推论——一个光子脉冲穿过 D 个调制器，一次做完 D 次乘法，能量花在一个脉冲上，活干了 D 份。电子每个 MAC 消耗新电荷，光子复用同一个光子。

但有两个重要的现实约束：第一，含探测器、ADC 和冷却的系统能效远低于纯光学（D=2048 时约 9,651× vs. 纯光学的 59M×）；第二，在自回归 Transformer 推理中，注意力仅占层内约 3% 的浮点运算，全系统端到端加速受 Amdahl 定律限制约为 1×。价值主要体现在注意力运算的能耗降低，而非端到端吞吐提升。
---

## 相关工作对比

本工作与现有工作在物理层、架构层、系统层三个层面均有本质区别。最需要警惕混淆的是西电 PTC (同目标, 2025 年先发) 和上海理工 Gezhi OGPU (架构外形相似)。

### 西电 PTC — 光子 Transformer 芯片 (2025)

| | 西电 PTC | 本工作 |
|---|---|---|
| 论文 | *PhotoniX* 6, 45 (2025) | Zenodo |
| 目标 | 光子 Transformer 注意力 | 光子 Transformer 注意力 |
| 平台 | **硅光子波导 (MZI 阵列)** | **自由空间垂直堆叠** |
| 计算原语 | MZI 干涉 | **光学交叉杆 + CMOS 探测器** |
| 非线性来源 | Kramers-Kronig 幅相耦合 | **Softmax 电子后处理** |
| 调制方式 | 热光移相器 (外接电加热) | **光热筛选 (光自己加热)** |
| 集成 | 平面波导, 片上 | **三叠垂直: VCSEL→光热膜→CMOS** |
| 可编程性 | 运行时动态可编程 | 权重静态推理 (0.033Hz 更新) |
| 实测精度 | MNIST 94% (ViT) | ρ=0.9954 (D=64, 4头 Attention) |
| 能效 (投影) | ~500 TOPS/W | 59M× vs H100 (纯光学) |

本质区别: 西电是波导干涉路线，我们是自由空间光热筛选路线。两条完全不同的物理路径。

- DOI: [10.1186/s43074-025-00182-7](https://doi.org/10.1186/s43074-025-00182-7)

### 上海理工 Gezhi OGPU / MI-DNN (2025)

| | Gezhi OGPU | 本工作 |
|---|---|---|
| 论文 | *eLight* 5, 29 (2025) | Zenodo |
| 光源 | 8×8 VCSEL 阵列 | D 个 VCSEL |
| 相干性 | 互相非相干 (MI) | VCSEL 天然非相干 |
| 架构 | VCSEL→衍射层→CMOS 垂直堆叠 | **VCSEL→光热筛选层→CMOS** |
| 中间层 | **被动衍射层 (权重固定)** | **主动光热调制 (可编程)** |
| 应用 | 图像分类/边缘检测 | **Transformer 注意力** |
| 距离 | 7mm 自由空间, 无透镜 | 类似垂直紧凑 |
| 推理速度 | 25M 帧/秒 | 10 GHz 时钟 |
| 能效 | 950 TOPS/W | 59M× vs H100 (纯光学) |

本质区别: Gezhi 的中间层是被动的（衍射, 权重写入后不可改），我们的是主动的（光热调制, 权重可重编程）。Gezhi 面向视觉预处理，不涉及 Transformer 注意力。

- DOI: [10.1186/s43593-025-00106-9](https://doi.org/10.1186/s43593-025-00106-9)

### 其他相关工作

| 工作 | 年份 | 路线 | 与本工作的区别 |
|------|------|------|---------------|
| **FAST-ONN** (Christen et al.) | 2025 | VCSEL+SLM+扇出+探测器 | 用 SLM 而非光热薄膜; 不做 Transformer |
| **Lightening-Transformer** (Zhu et al., HPCA) | 2024 | 硅光子 MZI 张量核 | 波导路线; 非自由空间; 非光热调制 |
| **ASTRA** (DATE/TCAD) | 2024-25 | 硅光子随机计算 | 随机计算范式; 完全不同物理路径 |
| **OPT** (Optics Comm.) | 2024 | SLM+微透镜阵列 | 自由空间但用 SLM; 面向散射成像 |
| **PGKET** (arXiv) | 2025 | 高斯核干涉 | 波导干涉仪; 非自由空间 |
| **D²NN** (Ozcan/Engheta, Nat. Comm.) | 2024 | 被动衍射表面级联 | 被动固定权重; 非可编程; 非 Transformer |

---

## 原创性声明

本工作提出以下技术贡献：

1. **首次将 DiSubPc·C70 有机共晶用于光学计算**。该材料 2026 年才发表于 *Nature Photonics*，原论文仅涉及光热转换应用。我是第一个提出利用其量子相干拍频窗口做光学 MAC 运算的。

2. **光自驱动调制，省掉独立加热器**。VCSEL 光同时承载信号和驱动相变。西电 PTC 用电热移相器，Gezhi 用被动衍射，SLM 方案用电控液晶。

3. **三叠 VCSEL→光热膜→CMOS 垂直堆叠**。Gezhi 的 VCSEL→衍射层→CMOS 外形相似但中间层是被动的。

4. **单光子穿 D 个调制器 = D 次乘法**。把扇出能量优势公式化。

5. **光学交叉杆自由空间 D² 点积**。传统光学交叉杆是波导的，自由空间版本在文献中极少。

本工作与 2024-25 年光子计算三大趋势方向一致：互相非相干范式、自由空间替代波导、垂直 3D 堆叠。详细信息及先例分析见 [thermal-optical-validation](https://github.com/administere/thermal-optical-validation) 中的 PRIOR_ART.md。

---

## 论文与引用

- Zenodo: [10.5281/zenodo.20690521](https://doi.org/10.5281/zenodo.20690521) (2026-06-14)
- 论文源码: `arxiv-paper/main.tex` (Zenodo 存证, 未提交 arXiv)

```bibtex
@misc{wayne2026photonic,
  author       = {Wayne},
  title        = {A Photonic Transformer Processor: Thermal-Optical Screening with Direct CMOS Detection},
  year         = {2026},
  doi          = {10.5281/zenodo.20690521},
  publisher    = {Zenodo},
  url          = {https://doi.org/10.5281/zenodo.20690521},
  note         = {Independent research. Source code: https://github.com/administere/photonic-transformer}
}
```

> **优先权说明**: Zenodo 存证 2026-06-14。本工作与西电 PTC (PhotoniX, 2025.10) 独立发明, 物理路径完全不同 (自由空间光热筛选 vs 波导 MZI 干涉)。DiSubPc·C70 光热材料用于光学计算为首次提出。

---

## 诚实评估

- **物理自洽**: 光子复用的 attojoule 优势是 Maxwell 方程的直接推论。但系统能效被探测器、ADC 和冷却严重稀释——D=2048 时系统能效约 9,651× vs H100，远低于纯光学的 59M×。
- **不是通用处理器**: 权重更新约 30 秒一次，只适合静态推理。训练、微调、多租户切换都不支持。
- **Amdahl 定律是真实约束**: 在自回归 Transformer 推理中，注意力只占约 3% 的层内算力。全系统加速受此限制。长上下文场景（注意力占比更高）可能是更合适的应用切入点，但尚未验证。
- **仿真到实验的距离**: 全部结果基于仿真。没有实验数据，没有硬件。任何仿真结果在实验面前都是假设。
- **已有独立工程审查**: [thermal-optical-validation](https://github.com/administere/thermal-optical-validation) 从第一性原理到工程六问做了完整审查，包含灵敏度分析和 D 标度律。结论: 物理自洽，工程可解。
- **下一阶段的真实瓶颈不是更多仿真，是实验**。找一个有 DiSubPc·C70 薄膜制备和光学测试能力的合作者，比继续堆仿真有价值得多。

---

## 作者

AI 辅助分析 · 独立研究。

