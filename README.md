# CNN Hardware Accelerator for Solar Panel Fault Classification
### B.Tech Capstone Project (FYP) — Machine Learning Pipeline & Zero-Aware FPGA Accelerator Datapath

[![Project: B.Tech Capstone](https://img.shields.io/badge/Project-B.Tech%20Capstone%20FYP-blue.svg)](docs/Next-Gen(20261013-193018).pptx)
[![Dataset: 21660 Images](https://img.shields.io/badge/Dataset-21%2C660%20Multispectral%20Images-green.svg)](docs/Review_II_Presentation_Summary.md)
[![Model: EfficientNetV2-B0](https://img.shields.io/badge/Model-EfficientNetV2--B0%20(92.36%25%20Val)-orange.svg)](ml_pipeline/train_solar_efficientnet.py)
[![Hardware: Zero-Aware MAC](https://img.shields.io/badge/Hardware-Zero--Aware%20MAC%20RTL-purple.svg)](rtl/zero_aware_mac.v)
[![License: MIT](https://img.shields.io/badge/License-MIT-darkgreen.svg)](LICENSE)

An end-to-end edge AI hardware acceleration and computer vision project for autonomous photovoltaic (PV) defect classification. Guided at the School of Electronics Engineering (SENSE), Vellore Institute of Technology (VIT), Vellore.

The project addresses large-scale solar farm maintenance by combining a **92.36% validation accuracy EfficientNetV2-B0 convolutional neural network** trained on **21,660 multispectral aerial images** with a custom **Zero-Aware Multiply-Accumulate (MAC) FPGA datapath** designed to exploit activation and weight sparsity for high-efficiency edge inference.

---

## 1. System Overview & Multi-Stage Architecture

```
                                  Stage 1: Deep Learning Training
                        ┌─────────────────────────────────────────────────┐
                        │ 21,660 Photovoltaic Images (4 Defect Classes)   │
                        │                       │                         │
                        │                       ▼                         │
                        │ EfficientNetV2-B0 Transfer Learning (Kaggle 2xT4│
                        │ Val Acc: 92.36% | Macro F1: 92.21% | Test: 84.9%│
                        └───────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                                  Stage 2: Compression & Quantization
                        ┌─────────────────────────────────────────────────┐
                        │ Structured Channel Pruning & Activation Analysis│
                        │ Post-Training Quantization: FP32 -> INT8 (Q7)   │
                        │ Activation Sparsity Characterization (40%-60% 0)│
                        └───────────────────────┬─────────────────────────┘
                                                │
                                                ▼
                                  Stage 3: Hardware Acceleration (FPGA)
                        ┌─────────────────────────────────────────────────┐
                        │ Zero-Aware MAC Datapath (rtl/zero_aware_mac.v)  │
                        │ Combinational NOR Zero-Detection Logic          │
                        │ Clock-Gated Multipliers (Zero Dynamic Switching)│
                        │ Line-Buffer Convolutional Memory Architecture   │
                        └─────────────────────────────────────────────────┘
```

---

## 2. Dataset Distribution & Defect Classification

The model processes **21,660 high-resolution aerial electroluminescence and infrared images** partitioned across 4 defect categories:

| Class ID | Defect Category | Image Count | Physical Impact on PV Module |
| :---: | :--- | :---: | :--- |
| `0` | **Clean / Normal** | 5,415 | Optimal photovoltaic conversion efficiency ($> 20\%$). |
| `1` | **Dust / Soiling Accumulation** | 5,415 | Uniform surface obscuration causing $10\%\text{--}25\%$ energy yield reduction. |
| `2` | **Electrical / Hotspot Faults** | 5,415 | Shunted cells, bypass diode failures, severe localized thermal risk. |
| `3` | **Physical Cracking & Micro-Cracks** | 5,415 | Silicon wafer fractures from thermal cycling or hail impact. |
| **Total** | **Benchmark Corpus** | **21,660** | **Balanced 4-Class Partition: 70% Train, 15% Val, 15% Test** |

---

## 3. Machine Learning Architecture & Performance

### Transfer Learning Formulation
The feature extractor utilizes inverted residual bottlenecks with progressive learning regularization. Loss is optimized via categorical cross-entropy with label smoothing ($\epsilon = 0.1$):

$$L_{\text{CE}}(y, \hat{y}) = -\sum_{c=1}^4 \left[ (1 - \epsilon) y_c + \frac{\epsilon}{4} \right] \log(\hat{y}_c)$$

### Empirical Results across Architectures

| Model Architecture | Precision | Parameters | Validation Accuracy | Macro F1-Score | Held-Out Test Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50 Baseline** | FP32 | 25.6M | 86.40% | 85.80% | 79.20% |
| **MobileNetV3-Large** | FP32 | 5.4M | 89.15% | 88.70% | 81.60% |
| **EfficientNetV2-B0 (Proposed)** | **FP32** | **7.1M** | **92.36%** | **92.21%** | **84.91%** |
| **Quantized EfficientNetV2-B0** | **INT8** | **7.1M** | **91.80%** | **91.65%** | **84.30%** |

- **Validation Accuracy**: **$92.36\%$**
- **Macro F1-Score**: **$92.21\%$**
- **Held-Out Test Accuracy**: **$84.91\%$**
- Experimental confusion matrices, ROC curves, and training loss plots are documented in [`docs/Review_II_Presentation_Summary.md`](docs/Review_II_Presentation_Summary.md).

---

## 4. Hardware Accelerator: Zero-Aware MAC Architecture

In quantized convolutional neural networks, activation sparsity (zeros produced by non-linear activations like ReLU/GELU) combined with weight sparsity commonly exceeds $40\%\text{--}60\%$. Standard systolic multipliers dissipate power executing redundant multiplications ($0 \times W = 0$ and $A \times 0 = 0$).

```
                         Activation [7:0]   Weight [7:0]
                                │                 │
                                ├─────────┐       ├─────────┐
                                ▼         ▼       ▼         ▼
                           Zero Detector Zero Detector
                                │         │       │         │
                                └────► OR ◄───────┘         │
                                       │                    │
                                       ▼                    ▼
                                 Clock Gate / ────► 8x8 Signed Multiplier
                                 Bypass Mux                 │
                                                            ▼
                                                   32-Bit Accumulator
```

### Dynamic Power Savings Model
CMOS dynamic switching power dissipation is governed by:

$$P_{\text{dynamic}} = \alpha \cdot C_L \cdot V_{DD}^2 \cdot f$$

By gating the operand registers when $\text{zero\_detected} = (\text{act} == 0) \lor (\text{wt} == 0)$, the effective switching activity factor $\alpha_{\text{eff}}$ scales down proportionally with sparsity ($S$):

$$\alpha_{\text{eff}} = \alpha_0 \cdot (1 - S)$$

For a measured $45\%$ activation sparsity, dynamic power in the multiplier array is reduced by up to **$45\%$**.

---

## 5. Repository Directory Layout

```
Solar-Panel-Fault-Hardware-Accelerator/
├── README.md                           # Master project documentation
├── .gitignore                          # Python, cache, and simulation exclusions
├── rtl/
│   └── zero_aware_mac.v                # Synthesizable zero-aware MAC Verilog tile
├── tb/
│   └── tb_zero_aware_mac.v             # Zero-aware MAC self-checking testbench
├── hardware_roadmap/
│   └── zero_aware_mac_architecture.md  # Detailed hardware microarchitecture spec
├── ml_pipeline/
│   ├── train_solar_efficientnet.py     # EfficientNetV2-B0 training script
│   └── macro_sparsity_sweep.py         # Activation sparsity profiler
└── docs/
    ├── Next-Gen(20261013-193018).pptx  # 17-Slide Review-II defense presentation
    ├── Review_II_Presentation_Summary.md # Complete technical review transcript
    └── figures/                        # 11 Empirical training & evaluation charts
```

---

## 6. Execution Quick Start

### 1. Run Machine Learning Training & Evaluation
```bash
python ml_pipeline/train_solar_efficientnet.py
```

### 2. Profile Activation & Weight Sparsity
```bash
python ml_pipeline/macro_sparsity_sweep.py
```

### 3. Simulate Zero-Aware Hardware Accelerator
```bash
iverilog -o sim_mac.out rtl/zero_aware_mac.v tb/tb_zero_aware_mac.v
vvp sim_mac.out
```

---

## 7. Capstone Authors & Citation

- **Lead Engineer**: Hari Kruthik SK (Reg. 23BEC0358)  
- **Institution**: School of Electronics Engineering (SENSE), Vellore Institute of Technology (VIT), Vellore  
- **Project Milestone**: B.Tech Capstone Project (FYP) — Review II Completed & Verified  
- **License**: Released under the [MIT License](LICENSE).
