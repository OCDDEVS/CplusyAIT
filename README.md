# Next-Gen CPU AI Framework

An experimental, ultra-efficient AI training and inference framework built in Rust and C++. The framework is specifically designed to run multi-million parameter Large Language Models (LLMs) and advanced AI Agents entirely on consumer CPUs with strict memory constraints (e.g., 8GB RAM budgets) while achieving 100+ tokens per second.

This project is built on three core research pillars:
1. Extreme Quantization (1.58-bit Ternary Networks)
2. Memory Sparse Attention (MSA)
3. Biological "Engram" Memory Lifecycles (EverMemOS)

## Architecture

To squeeze every drop of performance out of a standard CPU, the framework employs a hybrid architecture:

*   **The C++ Core (Compute Engine):** Handles all low-level, high-performance tensor operations. We utilize advanced SIMD intrinsics (AVX2) to execute branchless, pure-addition matrix multiplications (Ternary GEMM).
*   **The Rust Orchestrator (Control Plane):** Manages the high-level API, thread pool concurrency (via Rayon), the EverMemOS lifecycle (MemCells/MemScenes), and interfaces with the C++ compute kernels via a zero-cost Foreign Function Interface (FFI).

## Breakthrough Benchmarks: 1.58-bit Ternary AVX2 SIMD

Traditional LLM inference is bottlenecked by Memory Bandwidth and Multiply-Accumulate (MAC) operations. By adopting a 1.58-bit architecture (weights restricted to -1, 0, 1) and writing a custom AVX2 SIMD kernel that packs 4 weights into a single byte (`uint8_t`), we replace floating-point multiplications with parallel integer additions.

### Performance on a simulated ~100k parameter Feed-Forward layer (Batch 1, SeqLen 1024):

| Metric | FP32 Baseline | 1.58-bit Ternary (AVX2 SIMD 2-bit Packed) | Improvement |
| :--- | :--- | :--- | :--- |
| **Latency** | ~95.00 ms/pass | ~7.09 ms/pass | **~13.4x Speedup** |
| **Memory Footprint** | 1024 KB | 64 KB | **16.0x Reduction** |

By packing the memory into 2 bits, a 7B parameter model (typically ~14GB in FP16) compresses down to approximately 875 MB, comfortably fitting within an 8GB CPU RAM budget with ample room for KV caching.

Detailed analysis can be found in `BENCHMARKS.md`.

## Toy Training Loop: Straight-Through Estimator (STE)

To prove that 1.58-bit ternary models can learn locally, this framework includes a toy Multi-Layer Perceptron (MLP) training loop using the TinyShakespeare dataset.

Because a quantized discrete step function (-1, 0, 1) is non-differentiable, the framework employs the Straight-Through Estimator (STE) method:
1.  **Forward Pass:** High-precision (FP32) "Master Weights" are quantized to 1.58-bit and processed through the fast C++ AVX2 kernel.
2.  **Backward Pass:** Gradients are calculated against the mean squared error and applied directly to the high-precision FP32 Master Weights, bypassing the quantization step.

The framework successfully demonstrates the loss lowering across epochs, proving that the model can learn and fine-tune on standard CPU hardware.

## Future Development: MSA and EverMemOS

The next phases of development will wire the C++ AVX2 kernel into the Memory Sparse Attention (MSA) layer and EverMemOS pipeline, allowing the model to:
*   Decouple Working Memory (fast VRAM/Cache routing) from Long-Term Memory (KV cache on RAM/Disk).
*   Execute continuous local learning by compressing raw interactions into "MemCells" and clustering them into "MemScenes" (Semantic Consolidation) in the background.

## Build Instructions

### Prerequisites
*   Rust (Cargo)
*   A C++ compiler supporting C++17 and AVX2 intrinsics (GCC/Clang)

### Running
```bash
cargo run --release
```
This will compile the C++ kernels via `build.rs` and execute the benchmarking suite and STE toy training loop defined in `src/main.rs`.
