# CNN-Based Hardware Accelerator for Automated Solar Panel Condition & Fault Classification

[![Type: B.Tech Capstone FYP](https://img.shields.io/badge/Project-B.Tech%20Capstone%20FYP-red.svg)]()
[![Institution: VIT SENSE](https://img.shields.io/badge/School-SENSE%20VIT%20Vellore-blue.svg)]()
[![Model: EfficientNetV2--B0](https://img.shields.io/badge/Model-EfficientNetV2--B0%20(92.36%25%20val%20acc)-green.svg)]()
[![Hardware Roadmap: Zero--Aware MAC](https://img.shields.io/badge/Hardware-Zero--Aware%20MAC%20FPGA-orange.svg)]()

A comprehensive B.Tech Capstone Project conducted at the **School of Electronics Engineering (SENSE), Vellore Institute of Technology (VIT)**.

**Team Members**:
- **Hari Kruthik SK (23BEC0358)** — Deep Learning Model Development, Transfer Learning, Optimization
- **Rudresh KP (23BEC0169)** — Model Evaluation, Pruning & Hardware Mapping
- **Viswin Kumar RK (23BEC0088)** — Dataset Curation, Augmentation & Preprocessing

---

## 1. Problem Statement & Motivation

Rooftop solar photovoltaic (PV) arrays (such as installations under India's *PM Surya Ghar: Muft Bijli Yojana*) frequently suffer efficiency degradation due to environmental and mechanical faults:
- **Dust accumulation** (reduces irradiance capture by up to $25\%$)
- **Bird droppings** (causes localized hot-spot cell degradation)
- **Physical damage** (cracked cover glass, micro-fractures, delamination)

Manual inspection is slow, hazardous, and difficult to scale. This project develops an end-to-end automated vision pipeline to detect and classify panel anomalies, paired with a custom **FPGA hardware accelerator datapath** for zero-latency, low-power edge diagnosis.

---

## 2. Dataset & Classification Benchmark

The project benchmarks deep CNNs across a curated dataset of **21,660 solar panel images** categorized into four distinct conditions:
1. **Clean**: Optimal, defect-free panels.
2. **Dusty**: Uniform or non-uniform dust layer.
3. **Bird Drop**: Localized biological obstruction.
4. **Physical Damage**: Surface cracks, thermal burn marks, and shatter patterns.

```
                    ┌────────────────────────────┐
                    │  21,660 PV Panel Images    │
                    │  Clean │ Dust │ Bird │ Dam │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Data Augmentation Pipeline │
                    │ Flip, Rotation, Zoom, Norm │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ CNN Backbone Benchmarking  │
                    │ EfficientNetV2 vs MobileNet│
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │ Selected: EfficientNetV2-B0│
                    │   Val Acc: 92.36% | F1: 92%│
                    └─────────────┬──────────────┘
                                  │
                    (Hardware)    ▼
                    ┌────────────────────────────┐
                    │ Zero-Aware MAC Accelerator │
                    │ Line Buffer FPGA Hardware  │
                    └────────────────────────────┘
```

---

## 3. CNN Architecture Comparison & Results

Models were trained and evaluated on Kaggle using dual **NVIDIA Tesla T4 GPUs** ($1\text{ h } 10\text{ min } 21\text{ sec}$ total training time):

| Model Architecture | Parameter Count | Val Accuracy | Val F1-Score | Generalization Gap | Held-out Test Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MobileNetV3-Small** | $2.5\text{ M}$ | $88.10\%$ | $87.80\%$ | $8.40\%$ | $80.20\%$ |
| **MobileNetV2** | $3.5\text{ M}$ | $90.15\%$ | $89.90\%$ | $7.10\%$ | $82.45\%$ |
| **EfficientNetV2-B0 (Ours)** | **$7.1\text{ M}$** | **$92.36\%$** | **$92.21\%$** | **$6.22\%$** | **$84.91\%$** |

**Selection Rationale**: EfficientNetV2-B0 achieved the tightest generalization gap ($6.22\%$) and superior recall on localized defects like *Bird Drop* and *Physical Damage*.

---

## 4. Hardware Acceleration Roadmap (Active Development)

The project is structured into two phases:
- **Phase 1 (Completed)**: Dataset curation, CNN transfer learning, fine-tuning, error analysis, and macro-sparsity sweeps.
- **Phase 2 (Active Development)**:
  1. **Fixed-Point Quantization**: INT8 quantization of convolutional weights and activations.
  2. **Zero-Aware MAC Datapath**: Hardware logic that dynamically bypasses multiplications where either weight or activation is zero, saving dynamic switching power.
  3. **Line-Buffer Memory Architecture**: Streaming on-chip BRAM buffers holding rows of pixels to eliminate external memory read bottlenecks.

Detailed architectural diagrams are available in [`docs/figures/`](docs/figures/) and [`hardware_roadmap/`](hardware_roadmap/).

---

## 5. Repository File Structure

```text
Solar-Panel-Fault-Hardware-Accelerator/
├── README.md
├── .gitignore
│
├── ml_pipeline/
│   ├── train_solar_efficientnet.py        # EfficientNetV2-B0 training and transfer learning
│   └── macro_sparsity_sweep.py            # Weight distribution & sparsity analysis script
│
├── hardware_roadmap/
│   └── zero_aware_mac_architecture.md     # Detailed hardware accelerator datapath spec
│
└── docs/
    ├── Next_Gen_Capstone_Review_Presentation.pptx # 17-slide Project-1 Review-II presentation
    └── figures/                           # 11 high-res experimental figures & graphs
        ├── fig1_knowledge_distillation_impact.png
        ├── fig3_full_45point_accuracy_cliff_curve.png
        ├── zero_aware_mac_hardware_architecture_modern.png
        └── fyp_project_pipeline_infographic_1789054538763.jpg
```
