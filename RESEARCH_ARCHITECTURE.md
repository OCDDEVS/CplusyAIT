# Research Architecture: Next-Gen AI Framework for CPU-Bound Local LLMs

## Overview
This document outlines an experimental architecture for an ultra-efficient, next-generation AI training and inference framework built in **Rust** and **C++**.

Our primary goal is to achieve **100 tokens per second** on local consumer CPUs with a strict **8GB RAM budget**, while supporting advanced reasoning, continuous learning, and multi-million token context windows.

This framework is built upon three core research pillars:
1. **Extreme Quantization (1.58-bit / Ternary Networks)**
2. **Memory Sparse Attention (MSA) & End-to-End Long-Term Memory**
3. **Biological "Engram" Memory Lifecycles (MemCells / MemScenes)**

---

## 1. Architectural Split: Rust + C++

To squeeze every drop of performance out of a standard CPU while maintaining a robust, memory-safe orchestration layer, the framework employs a hybrid architecture.

### The C++ Core (The Compute Engine)
- **Role:** Handles all low-level, high-performance tensor operations.
- **Why C++?:** Direct access to advanced SIMD intrinsics (AVX2, AVX-512 for x86; NEON for ARM/Apple Silicon). It allows for manual cache-line alignment and aggressive loop unrolling.
- **Key Components:**
  - **Ternary GEMM (General Matrix Multiply):** Custom kernels that replace standard floating-point multiplications with pure additions/subtractions (for 1.58-bit weights).
  - **MSA Router Execution:** Fast vector dot-products for memory routing and Top-K selection.
  - **KV Cache Management:** Direct memory manipulation (mmap) for paging large KV caches in and out of CPU RAM to disk (NVMe).

### The Rust Orchestrator (The Control Plane)
- **Role:** High-level API, memory safety, concurrent execution, and graph compilation.
- **Why Rust?:** Fearless concurrency (Zero-cost async/await) for overlapping compute with I/O (e.g., streaming weights from disk). Memory safety guarantees prevent segmentation faults in the complex memory lifecycles.
- **Key Components:**
  - **Zero-Cost FFI (Foreign Function Interface):** Binds to the C++ compute kernels without overhead.
  - **Memory OS (EverMemOS Implementation):** Manages the lifecycle of MemCells and MemScenes (Episodic Trace Formation -> Semantic Consolidation -> Recollection).
  - **Distributed/Local Orchestration:** Thread pools (via `rayon`) to keep all CPU cores fed with ternary math operations.

---

## 2. Research Pillar 1: Extreme Quantization (1.58-bit LLMs)

**The Bottleneck:** Traditional LLM inference (even 4-bit/8-bit quantized) is bottlenecked by Memory Bandwidth (loading weights from RAM to the CPU cache) and Compute (floating-point MAC - Multiply-Accumulate operations).

**The Solution:** 1.58-bit Ternary Weights (e.g., BitNet b1.58 architecture).
- **Weights:** Restricted to `{-1, 0, 1}`.
- **Math Revolution:** We completely eliminate matrix *multiplication*. A dot product between activations (e.g., 8-bit integers) and ternary weights becomes pure **Addition and Subtraction**.
- **CPU Advantage:** CPUs excel at branching and integer addition. By packing 16 ternary weights into a single 32-bit integer, we reduce the memory bandwidth requirement by a factor of 10-20x compared to FP16.
- **Memory Footprint:** A 7B parameter model, which normally takes ~14GB in FP16, compresses down to **~1.4GB**. This fits comfortably within our 8GB RAM budget, leaving ample room for the KV cache and EverMemOS memory structures.

---

## 3. Research Pillar 2: Memory Sparse Attention (MSA)

Based on recent EverMind research (2026), scaling context to 100M tokens via standard self-attention (quadratic complexity) or RAG (structural disconnect) is impossible on an 8GB CPU limit.

**The Solution:** Decoupled Memory with Native Routing.
- **Hierarchical Storage:**
  - **Working Memory (Router):** Highly compressed feature vectors (Routing Keys) stored in L3 CPU Cache or pinned RAM.
  - **Long-Term Memory (KV Cache):** Full Key-Value tensors stored in system RAM, intelligently paged to fast NVMe storage when exceeding 8GB.
- **End-to-End Differentiable Routing:** The "retrieval" step is internalized. During continuous local learning, the Router is updated via an auxiliary contrastive loss. This allows the model to learn *where* to look without full backpropagation through the entire 100M token KV cache.
- **Document-Level RoPE:** The position counter resets to zero for new documents (MemCells). This allows a model trained on short contexts to seamlessly extrapolate to massive personal knowledge bases on the user's local machine.
- **Memory Interleave:** Instead of a single-shot RAG retrieval, the local CPU orchestrator runs iterative loops—generating document IDs, reading retrieved content, and repeating until sufficient evidence is gathered.

---

## 4. Research Pillar 3: EverMemOS - The Digital Brain Lifecycle

To support true AI Agent capabilities locally without degrading performance, we implement a structured memory operating system.

**Phase I: Episodic Trace Formation (MemCells)**
- As the user interacts with the local LLM, raw chat logs are not kept indefinitely in the context window.
- The Rust orchestrator runs a lightweight background thread to compress interactions into **MemCells**:
  - `E (Episode)`: Narrative summary.
  - `F (Atomic Facts)`: Discrete statements.
  - `P (Foresight)`: Temporary states/plans (e.g., "Remind me tomorrow").

**Phase II: Semantic Consolidation (MemScenes)**
- Continuous local learning: As new MemCells arrive, the CPU runs a fast cosine-similarity check against existing "MemScene" centroids.
- If similar, the MemCell is assimilated. This consolidates user profiles and traits *offline* or during idle CPU cycles, preventing "prompt bloat."

**Phase III: Reconstructive Recollection**
- During generation, the MSA Router selects only the necessary MemScenes. This strictly adheres to the "Necessity and Sufficiency" principle—retrieving only what is needed (minimizing token processing and saving precious CPU cycles/RAM).

---

## 5. Execution Pipeline (How it hits 100 Tokens/s)

1. **User Input:** Rust API receives prompt.
2. **Memory Retrieval:** Rust queries the EverMemOS index; fetches relevant MemCells (paged from disk if needed).
3. **Routing (MSA):** C++ kernel executes fast Top-K vector dot products to select relevant KV blocks from RAM.
4. **Generation (1.58-bit):** Rust orchestrator streams packed ternary weights into the CPU L1/L2 cache. The C++ AVX-512 kernels execute purely additive matrix multiplications against the 8-bit quantized activations.
5. **Background Consolidation:** While generating, a separate Rust thread compresses older context into new MemCells, updating the local vector index without blocking the token stream.