# Hardware Accelerator Roadmap: Zero-Aware Multiply-Accumulate (MAC) Architecture

## Overview
This architectural roadmap details the planned hardware acceleration phase for edge FPGA deployment of the Solar Panel Fault Classification model.

---

## 1. Zero-Aware MAC Datapath Concept

In quantized convolutional neural networks, activation sparsity (zeros produced by ReLU/GELU activations) and weight sparsity (near-zero pruned weights) often exceed $40\%\text{--}60\%$. Standard systolic multipliers waste energy performing multiplications where one or both operands are zero:

$$0 \times W = 0 \quad \text{and} \quad A \times 0 = 0$$

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

### Key Engineering Features
1. **Zero-Detection Logic**: An 8-input NOR tree monitors incoming 8-bit operands. If either operand is zero, the clock enable to the multiplier registers is gated, reducing dynamic switching power ($P = \alpha C V^2 f$).
2. **Line-Buffer Memory Architecture**: Uses on-chip BRAM line buffers to stream $3\times3$ convolutional receptive fields without repeatedly querying external DDR RAM, dramatically cutting off-chip memory bandwidth.
3. **Fixed-Point Quantization**: Translates FP32 weights to INT8 ($Q7$ format), scaling memory requirements by $4\times$ from $\sim 28\text{ MB}$ to $\sim 7\text{ MB}$.

---

## 2. Multi-Stage Execution Plan

1. **Stage 1 (Completed)**: Pretraining, transfer learning, and architecture benchmarking (EfficientNetV2-B0 achieving $92.36\%$ validation accuracy).
2. **Stage 2 (Active)**: Weight distribution analysis, structured pruning, and INT8 post-training quantization.
3. **Stage 3 (Planned)**: Synthesizable Verilog RTL implementation of Zero-Aware MAC tiles and line-buffer controller on Altera/Xilinx FPGA.
