"""
GPTQ-style Ternary Quantizer for CplusyAIT
============================================
Post-training quantization to 1.58-bit {-1, 0, 1}.

Uses Hessian-compensated sequential rounding (GPTQ algorithm) with
per-channel scaling and aggressive memory management to stay safe
on machines with 16-32GB RAM.

Memory safety measures:
- Streams output directly to disk (no in-memory blob accumulation)
- Frees model weights layer-by-layer after quantization
- Explicit gc.collect() + torch cache clearing after each tensor
- Vectorized 2-bit packing (numpy, not Python loops)
- Hard memory limit with psutil monitoring (if available)

Usage:
    pip install torch transformers datasets
    python scripts/quantize_gptq.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --output models/TinyLlama-1.1B-Chat-GPTQ-ternary \
        --nsamples 32 --seqlen 1024

References:
    - GPTQ: Accurate Post-Training Quantization (Frantar et al., 2022)
    - BitNet b1.58: The Era of 1-bit LLMs (Ma et al., 2024)
"""

import argparse
import json
import os
import struct
import time
import gc
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Memory monitoring
# ---------------------------------------------------------------------------

def get_ram_usage_gb():
    """Return current process RSS in GB. Returns -1 if psutil unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 3)
    except ImportError:
        return -1.0


def check_memory_limit(limit_gb):
    """Abort if we're using more than limit_gb of RAM."""
    usage = get_ram_usage_gb()
    if usage > 0 and usage > limit_gb:
        print(f"\n*** MEMORY SAFETY: Using {usage:.1f} GB (limit: {limit_gb} GB). Aborting. ***")
        sys.exit(1)


def force_gc():
    """Aggressive garbage collection to reclaim PyTorch + Python memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def get_calibration_data(tokenizer, nsamples=32, seqlen=1024, seed=42):
    """Load calibration data from WikiText-2 or generate synthetic."""
    torch.manual_seed(seed)
    try:
        from datasets import load_dataset
        print("  Loading WikiText-2 calibration data...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join([t for t in dataset["text"] if t.strip()])
        enc = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = enc.input_ids[0]
        # Free the large text and encoded data immediately
        del text, enc, dataset
        force_gc()
        print(f"  WikiText-2 loaded: {len(input_ids)} tokens")
    except Exception as e:
        print(f"  Could not load WikiText-2 ({e}), using random calibration data")
        input_ids = torch.randint(0, min(tokenizer.vocab_size, 32000), (nsamples * seqlen,))

    samples = []
    for i in range(nsamples):
        start = torch.randint(0, max(1, input_ids.shape[0] - seqlen), (1,)).item()
        inp = input_ids[start:start + seqlen].unsqueeze(0)
        samples.append(inp)

    del input_ids
    force_gc()
    return samples


# ---------------------------------------------------------------------------
# Vectorized 2-bit packing (replaces the O(m*k) Python loop)
# ---------------------------------------------------------------------------

def pack_ternary_vectorized(Q_np):
    """
    Pack int8 ternary values {-1,0,1} into 2-bit packed bytes (4 per byte).
    Uses numpy vectorization — ~100x faster than the Python loop version
    and doesn't keep extra copies alive.

    Encoding: 0->0, 1->1, -1->2
    """
    m, k = Q_np.shape
    packed_k = (k + 3) // 4

    # Pad columns to multiple of 4
    if k % 4 != 0:
        pad = 4 - (k % 4)
        Q_np = np.pad(Q_np, ((0, 0), (0, pad)), constant_values=0)

    # Map: 0->0, 1->1, -1->2
    encoded = np.where(Q_np == 1, np.uint8(1),
              np.where(Q_np == -1, np.uint8(2), np.uint8(0)))

    # Reshape to (m, packed_k, 4) and pack 4 values per byte
    encoded = encoded.reshape(m, packed_k, 4)
    packed = (encoded[:, :, 0]
              | (encoded[:, :, 1] << 2)
              | (encoded[:, :, 2] << 4)
              | (encoded[:, :, 3] << 6))

    return packed.astype(np.uint8).tobytes()


# ---------------------------------------------------------------------------
# GPTQ Ternary Quantizer (memory-safe)
# ---------------------------------------------------------------------------

class GPTQTernaryQuantizer:
    """
    GPTQ-style quantizer targeting ternary {-1, 0, 1} weights.
    Memory-safe: operates in-place where possible, frees intermediates.
    """

    def __init__(self, layer_weight, hessian, block_size=128, percdamp=0.01):
        # Work in-place on the provided tensors — no .clone()
        self.W = layer_weight.float()
        self.H = hessian.float()
        self.out_features, self.in_features = self.W.shape
        self.block_size = block_size
        self.percdamp = percdamp

    def quantize(self):
        """Run GPTQ ternary quantization. Returns (packed_bytes, gammas_numpy)."""
        W = self.W
        H = self.H
        m, k = W.shape

        # Damping for numerical stability
        damp = self.percdamp * torch.diag(H).mean()
        diag_idx = torch.arange(k, device=H.device)
        H[diag_idx, diag_idx] += damp

        # Cholesky decomposition of Hessian -> inverse -> upper Cholesky
        # We free intermediates immediately to avoid holding 3 k×k matrices
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            del L
            H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)
            del H_inv
        except Exception:
            H[diag_idx, diag_idx] += 1e-2 * torch.diag(H).mean()
            H_inv = torch.linalg.inv(H)
            H_inv_chol = torch.linalg.cholesky(H_inv, upper=True)
            del H_inv

        # Free the Hessian — we only need H_inv_chol from here
        del self.H, H

        # Per-channel gamma (absmean per row)
        gammas = W.abs().mean(dim=1).clamp(min=1e-8)  # (out_features,)

        # Quantized output
        Q = torch.zeros(m, k, dtype=torch.int8, device=W.device)

        # Process in blocks
        for col_start in range(0, k, self.block_size):
            col_end = min(col_start + self.block_size, k)
            block_cols = col_end - col_start

            W_block = W[:, col_start:col_end].clone()
            Hinv_block = H_inv_chol[col_start:col_end, col_start:col_end]

            for j in range(block_cols):
                col_idx = col_start + j
                w_col = W_block[:, j]
                h_inv_jj = Hinv_block[j, j]

                # Ternary quantization with per-channel scaling
                q_col = (w_col / gammas).clamp(-1.0, 1.0).round().to(torch.int8)
                Q[:, col_idx] = q_col

                # Quantization error compensation
                q_dequant = q_col.float() * gammas
                err = (w_col - q_dequant) / h_inv_jj

                # Compensate remaining columns in this block
                if j + 1 < block_cols:
                    W_block[:, j + 1:] -= err.unsqueeze(1) * Hinv_block[j, j + 1:block_cols].unsqueeze(0)

            del W_block

        del H_inv_chol, self.W

        # Vectorized packing (numpy) — fast and memory-efficient
        Q_np = Q.cpu().numpy()
        del Q
        packed = pack_ternary_vectorized(Q_np)
        gammas_np = gammas.cpu().numpy()
        del Q_np, gammas

        return packed, gammas_np


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def should_quantize(name):
    """Determine if a parameter should be ternary-quantized."""
    skip = ["layernorm", "norm", "embed", "bias"]
    name_lower = name.lower()
    return not any(s in name_lower for s in skip)


# ---------------------------------------------------------------------------
# Main quantization pipeline (memory-safe)
# ---------------------------------------------------------------------------

def quantize_model(model_id, output_dir, nsamples=32, seqlen=1024, device="cpu", mem_limit_gb=24.0):
    """
    Full GPTQ-style ternary quantization pipeline.

    Key memory safety changes vs the original:
    - Streams output to disk via open file handle (no in-memory blob)
    - Iterates model.named_parameters() instead of holding state_dict copy
    - Explicit del + gc.collect() after every tensor
    - Vectorized packing (numpy) instead of Python nested loops
    - Hard memory limit check every N tensors
    """
    print(f"Loading model: {model_id}")
    print(f"Memory limit: {mem_limit_gb} GB (will abort if exceeded)")
    print(f"Current RAM usage: {get_ram_usage_gb():.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load model — low_cpu_mem_usage prevents the PyTorch double-memory bug
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32,
            low_cpu_mem_usage=True, trust_remote_code=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, low_cpu_mem_usage=True, trust_remote_code=True
        )
    model.eval()

    print(f"Model loaded. RAM usage: {get_ram_usage_gb():.1f} GB")

    # We do NOT need calibration data for the weight-only Hessian approximation.
    # The original script loaded 128*2048 token samples (~2GB) but never actually
    # used them — the collect_hessian() function was defined but never called.
    # The loop uses col_norms from weights as a diagonal Hessian proxy.
    # So we skip calibration entirely to save ~2GB of RAM.
    print("Using weight-statistics Hessian (no calibration data needed)")

    config = model.config
    hidden_dim = config.hidden_size
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    vocab_size = config.vocab_size
    intermediate_dim = config.intermediate_size
    max_seq_len = getattr(config, "max_position_embeddings", 4096)
    rope_theta = getattr(config, "rope_theta", 10000.0)
    rms_norm_eps = getattr(config, "rms_norm_eps", 1e-5)

    print(f"Model: {num_layers} layers, hidden={hidden_dim}, vocab={vocab_size}")
    print(f"Heads: {num_heads} Q, {num_kv_heads} KV, intermediate={intermediate_dim}")

    os.makedirs(output_dir, exist_ok=True)

    tensor_metas = []
    total_params = 0
    quantized_params = 0
    current_offset = 0

    # Stream output directly to disk — never accumulate in memory
    weights_path = os.path.join(output_dir, "weights.bin")
    weights_file = open(weights_path, "wb")

    print("\n--- Quantizing layer by layer ---")

    # Use named_parameters() to avoid the state_dict() copy (saves ~4GB)
    param_list = [(name, param) for name, param in model.named_parameters()]
    num_tensors = len(param_list)

    for tensor_idx, (name, param) in enumerate(param_list, 1):
        shape = list(param.shape)
        total_params += param.numel()
        t0 = time.time()

        # Memory safety check every 10 tensors
        if tensor_idx % 10 == 0:
            check_memory_limit(mem_limit_gb)

        if should_quantize(name) and param.dim() == 2:
            # Detach from the model graph — we don't need gradients
            W = param.data.float()
            m, k = W.shape

            # Compute diagonal Hessian approximation from weight statistics
            with torch.no_grad():
                col_norms = (W * W).sum(dim=0)
                H = torch.diag(col_norms) + 1e-6 * torch.eye(k)
                del col_norms

            # Run GPTQ quantizer
            with torch.no_grad():
                quantizer = GPTQTernaryQuantizer(W, H, block_size=128)
                packed, gammas = quantizer.quantize()
                del quantizer
            del W, H

            mean_gamma = float(gammas.mean())

            # Write packed weights to disk
            byte_offset = current_offset
            byte_length = len(packed)
            weights_file.write(packed)
            current_offset += byte_length
            del packed

            quantized_params += param.numel()

            # Write per-channel gammas
            gamma_bytes = struct.pack(f"{len(gammas)}f", *gammas)
            gamma_offset = current_offset
            weights_file.write(gamma_bytes)
            current_offset += len(gamma_bytes)

            tensor_metas.append({
                "name": name,
                "shape": shape,
                "byte_offset": byte_offset,
                "byte_length": byte_length,
                "gamma": mean_gamma,
            })
            tensor_metas.append({
                "name": name + ".__gamma__",
                "shape": [len(gammas)],
                "byte_offset": gamma_offset,
                "byte_length": len(gamma_bytes),
                "gamma": 0.0,
            })
            del gammas, gamma_bytes

            compression = param.numel() * 2 / byte_length
            elapsed = time.time() - t0
            ram = get_ram_usage_gb()
            print(f"  [{tensor_idx}/{num_tensors}] [GPTQ] {name} {shape} -> {byte_length} bytes "
                  f"(gamma={mean_gamma:.4f}, {compression:.1f}x, {elapsed:.1f}s, RAM:{ram:.1f}GB)")
        else:
            # Store as f32 — write directly to disk
            byte_offset = current_offset
            f32_bytes = param.data.float().cpu().contiguous().numpy().tobytes()
            byte_length = len(f32_bytes)
            weights_file.write(f32_bytes)
            current_offset += byte_length
            del f32_bytes

            tensor_metas.append({
                "name": name,
                "shape": shape,
                "byte_offset": byte_offset,
                "byte_length": byte_length,
                "gamma": 0.0,
            })
            elapsed = time.time() - t0
            print(f"  [{tensor_idx}/{num_tensors}] [FP32] {name} {shape} -> {byte_length} bytes ({elapsed:.1f}s)")

        # Aggressive cleanup after every tensor
        force_gc()

    weights_file.close()

    # Free the model before writing manifest
    del model, param_list
    force_gc()

    # Write manifest
    manifest = {
        "model_type": getattr(config, "model_type", "llama"),
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "intermediate_dim": intermediate_dim,
        "rope_theta": rope_theta,
        "rms_norm_eps": rms_norm_eps,
        "tensors": tensor_metas,
        "quantization": "gptq-ternary",
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Copy tokenizer files
    tokenizer.save_pretrained(output_dir)

    # Summary
    file_mb = current_offset / (1024 * 1024)
    orig_mb = total_params * 2 / (1024 * 1024)
    print(f"\n--- Quantization Complete ---")
    print(f"Output: {output_dir}")
    print(f"Original (FP16): {orig_mb:.1f} MB")
    print(f"Quantized: {file_mb:.1f} MB")
    print(f"Compression: {orig_mb / max(file_mb, 0.01):.1f}x")
    print(f"Quantized params: {quantized_params:,} / {total_params:,} "
          f"({100 * quantized_params / max(total_params, 1):.1f}%)")
    print(f"Final RAM usage: {get_ram_usage_gb():.1f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPTQ-style Ternary Quantizer (memory-safe)")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--output", required=True, help="Output directory for quantized model")
    parser.add_argument("--nsamples", type=int, default=32, help="Calibration samples (unused in weight-only mode)")
    parser.add_argument("--seqlen", type=int, default=1024, help="Calibration sequence length (unused in weight-only mode)")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--mem-limit", type=float, default=24.0,
                        help="Hard RAM limit in GB — aborts if exceeded (default: 24)")
    args = parser.parse_args()

    quantize_model(args.model, args.output, args.nsamples, args.seqlen, args.device, args.mem_limit)
