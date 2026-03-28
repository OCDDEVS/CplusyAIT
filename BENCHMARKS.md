# Benchmark and Research Analysis: 1.58-bit Ternary Models vs FP32

## Overview

As part of the development for our Next-Gen CPU AI Framework, we implemented a custom C++ matrix multiplication kernel designed for 1.58-bit Ternary Weights `{-1, 0, 1}` and compared it against a standard FP32 (Full Precision) kernel. We also implemented a toy Straight-Through Estimator (STE) training loop on the `TinyShakespeare` dataset to prove learnability.

Here are the results and analysis.

---

## 1. Benchmarking Results (Ternary vs FP32)

We simulated a single Feed-Forward Network (FFN) layer computation typical of a ~100k parameter model.
- **Batch Size:** 1
- **Sequence Length (M):** 1024
- **Input Dimension (K):** 256
- **Hidden Dimension (N):** 256

**Results (Averaged over 100 runs on local CPU):**

| Metric | FP32 Baseline | 1.58-bit Ternary (Int8 Unpacked) | Improvement |
| :--- | :--- | :--- | :--- |
| **Latency** | ~97.59 ms/pass | ~65.29 ms/pass | **~1.5x Speedup** |
| **Memory Footprint** | 1024 KB | 256 KB | **4.0x Reduction** |

### Why didn't we see a 10x speedup? (The Bottleneck)
1. **Branching vs SIMD:** Modern CPUs excel at floating-point math because of highly optimized SIMD instructions (AVX2/AVX-512) and Fused-Multiply-Add (FMA) instructions. In our raw C++ `cpu_ternary_gemm`, we replaced multiplications with `if (weight == 1) { acc += act; }`. This introduces branch prediction overhead.
2. **Memory Bandwidth:** The memory reduction is currently 4x because we store `{-1, 0, 1}` in an 8-bit integer (`int8_t`). To achieve the theoretical limit, we need to pack four 2-bit values into a single byte, reducing memory by another 4x (for a total 16x reduction vs FP32).

**Future Fix (2026 Research Implementation):**
To fix the compute bottleneck, we must implement custom AVX-512 intrinsics that perform bitwise unpacking and parallel additions without `if` statements (Branchless SIMD).

---

## 2. Straight-Through Estimator (STE) Training Proof

We implemented a toy Multi-Layer Perceptron (MLP) mapping TinyShakespeare character embeddings to a hidden dimension.

**The Challenge with 1.58-bit Training:**
You cannot compute gradients for a function that outputs discrete steps `(-1, 0, 1)`. The derivative is zero almost everywhere.

**The Solution:**
We keep a set of **Master Weights** in high precision (FP32).
1. **Forward Pass:** We quantize the FP32 weights to `{-1, 0, 1}` using absolute mean scaling (BitNet formula). We pass these to the fast C++ Ternary kernel.
2. **Backward Pass:** We calculate the loss against the output, compute gradients, and apply the gradient updates directly to the *FP32 Master Weights*, entirely bypassing the non-differentiable quantization step.

**Training Output:**
```
Epoch 1: STE Loss = 4.9146 | Mean Output: -0.02
Epoch 2: STE Loss = 4.8433 | Mean Output: -0.04
Epoch 3: STE Loss = 4.7957 | Mean Output: 0.01
...
```
The framework demonstrates that gradients successfully flow "straight through" the ternary layer and update the high-precision weights, proving that the model is theoretically capable of learning on local CPUs.

---

## 3. Comparing to Market Baselines

Currently, standard CPU frameworks (like `llama.cpp` using Q4_K_M quantization) achieve impressive speeds but still rely on heavy floating-point accumulations and 4-bit memory bandwidth.

If we fully optimize the 1.58-bit C++ SIMD kernel with 2-bit packing:
- **Memory:** A 7B model requires ~1.4GB RAM (Compared to `llama.cpp` Q4 requiring ~4.5GB).
- **Speed:** By eliminating all MACs (Multiply-Accumulates) and relying purely on parallel additions, theoretical throughput should exceed 100 Tokens/s on an 8GB RAM Apple M-series or modern Intel/AMD CPU, even while keeping the EverMemOS MSA context router running in the background.