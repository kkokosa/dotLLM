# GPU Inference — dotLLM

## Overview

dotLLM supports GPU-accelerated inference on NVIDIA GPUs via the CUDA backend (`DotLLM.Cuda`). All transformer operations — embedding lookup, attention, FFN, normalization — execute entirely on the GPU with FP16 precision. The CPU is only involved for tokenization, sampling, and orchestrating kernel launches.

GPU inference targets **10–50× prefill speedup** and **3–10× decode speedup** over the CPU backend.

See [CUDA.md](CUDA.md) for the low-level architecture: PTX loading, P/Invoke declarations, kernel conventions, and build system.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  C# Orchestration (DotLLM.Cuda)                                    │
│  CudaTransformerModel.Forward()                                    │
│  ├── CudaKernels.LaunchXxx()  ──► cuLaunchKernel (PTX kernels)    │
│  ├── CudaGemm.GemmF16()      ──► cublasHgemm (cuBLAS)            │
│  └── cuMemcpyDtoD/HtoD/DtoH  ──► CUDA Driver API                 │
├────────────────────────────────────────────────────────────────────┤
│  NVIDIA System Libraries (installed with driver / CUDA Toolkit)    │
│  ├── libcuda.so / nvcuda.dll    — CUDA Driver API                 │
│  └── libcublas.so / cublas64_*.dll — cuBLAS                       │
├────────────────────────────────────────────────────────────────────┤
│  GPU Hardware                                                       │
│  ├── Tensor Cores (FP16 GEMM)                                     │
│  ├── CUDA Cores (custom kernels)                                   │
│  └── HBM / GDDR (device memory for weights + KV-cache + scratch)  │
└────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| `CudaTransformerModel` | `CudaTransformerModel.cs` | GPU forward pass implementing `IModel` |
| `CudaWeights` | `CudaWeights.cs` | Weight upload from GGUF mmap to GPU |
| `CudaForwardState` | `CudaForwardState.cs` | GPU scratch buffer management |
| `CudaKvCache` | `CudaKvCache.cs` | GPU-resident FP16 KV-cache |
| `CudaQuantizedKvCache` | `CudaQuantizedKvCache.cs` | GPU quantized KV-cache (Q8_0/Q4_0 + FP16 window) |
| `CudaKernels` | `CudaKernels.cs` | PTX module loading + typed kernel launch |
| `CudaGemm` | `CudaGemm.cs` | cuBLAS FP16 GEMM/GEMV wrappers |
| `CudaModule` | `CudaModule.cs` | PTX file loading via `cuModuleLoadData` |

### Relationship to CPU Path

Both `TransformerModel` (CPU) and `CudaTransformerModel` (GPU) implement `IModel`. The `TextGenerator` is backend-agnostic — it calls `model.Forward()` regardless of which backend produced the logits.

The CPU and GPU forward passes are **separate implementations** because:
- CPU uses fused ops, R4-interleaved weight repacking, pre-quantized input reuse, `ComputeThreadPool` dispatch
- GPU uses cuBLAS FP16 GEMM, PTX kernel launches, device memory, CUDA streams
- Abstracting both behind `IKernelRunner` would lose CPU-specific optimizations

## Weight Loading Pipeline

Quantized weights are stored **once** on GPU in their original quantized format. No persistent FP16 copies. Dequantization to FP16 happens on-the-fly into a reusable scratch buffer before each cuBLAS call (see [Weight Strategy](#weight-strategy-on-the-fly-dequantization)).

```
GGUF file (mmap'd on host)
  │
  ├── Quantized weight bytes (Q8_0, Q4_K, Q5_K, Q6_K)
  │     │
  │     └── cuMemcpyHtoD ──► quantized bytes on GPU (single copy)
  │         Prefill: dequant_xxx_f16 kernel → scratch buffer → cuBLAS GEMM
  │         Decode:  quantized GEMV kernel (operates directly, no dequant)
  │
  ├── FP16/F32 weight bytes
  │     │
  │     └── cuMemcpyHtoD ──► FP16 on GPU (single copy, used directly by cuBLAS)
  │
  ├── Norm weights (float[] on CPU)
  │     │
  │     └── upload F32 → convert_f32_to_f16 kernel → FP16 on GPU
  │
  └── Token embeddings (quantized)
        │
        └── cuMemcpyHtoD ──► kept in original format on GPU
            (embedding lookup kernel handles per-row dequant)
```

**Why dequantize on GPU?** Sending smaller quantized data over PCIe and dequantizing on-device is faster than dequantizing on CPU and sending larger FP16 data. GPU dequantization of a full Q4_K layer takes microseconds with massive parallelism.

### VRAM Estimation

| Component | Formula | Example (Llama 3.2 1B, Q8_0) | Example (Llama 3 8B, Q4_K_M) |
|-----------|---------|-------------------------------|-------------------------------|
| Weights (quantized) | `quantized model size` | ~1.1 GB | ~4.9 GB |
| Dequant scratch | `max(numHeads×headDim, intermediateSize) × hiddenSize × 2` | ~32 MB | ~128 MB |
| KV-cache (FP16) | `2 × layers × kvHeads × headDim × maxSeq × 2` | ~16 MB | ~512 MB |
| Activation buffers | `~seqLen × hiddenSize × 10 × 2` | ~1 MB | ~10 MB |
| **Total** | | **~1.15 GB** | **~5.5 GB** |

### Model Fit Table

| VRAM | Max Model (Q8_0) | Max Model (Q4_K_M) | Recommended |
|------|-------------------|---------------------|-------------|
| 2 GB | ~1B params | ~2B params | SmolLM-135M, Llama-3.2-1B |
| 4 GB | ~3B params | ~5B params | Llama-3.2-3B, Phi-3-mini |
| 8 GB | ~6B params | ~12B params | Llama-3.1-8B (Q4_K), Mistral-7B |
| 12 GB | ~10B params | ~18B params | Llama-3.1-8B (Q8_0) |
| 24 GB | ~20B params | ~35B params | Mistral-22B, Llama-3.3-70B (Q4_K, partial) |

## Forward Pass

The GPU forward pass mirrors the CPU path in `TransformerModel.Forward()` but all operations execute on GPU. It branches on `seqLen` for projections:

```
1. Upload tokenIds + positions ──► cuMemcpyHtoD (small H2D)
2. EmbeddingLookup kernel ──► HiddenState [seqLen, hiddenSize] FP16
3. Layer 0 setup: copy hidden→residual, RmsNorm→NormOutput

4. For each transformer layer:
   ── Attention Block ──
   a. Q/K/V projections:
      Prefill (seqLen > 1): dequant → scratch → cublasHgemm
      Decode  (seqLen = 1): quantized GEMV (direct, no dequant)
   b. BiasAdd kernel (if biases)
   c. PerHeadRmsNorm kernel (if QK-norms, Qwen3)
   d. RoPE kernel (in-place on Q, K)
   e. KV-cache update (cuMemcpyDtoD per position)
   f. Attention kernel (scaled dot-product, causal mask, GQA)
   g. O projection (same prefill/decode dispatch as Q/K/V)
   ── Fused: attention residual + FFN norm ──
   h. FusedAddRmsNorm: residual += attnOutput (FP32), NormOutput = rmsnorm(residual)
   ── FFN Block ──
   i. Gate/Up projections (prefill/decode dispatch)
   j. SwiGLU kernel
   k. Down projection
   ── Fused: FFN residual + next attention norm ──
   l. FusedAddRmsNorm (or plain Add for last layer)

5. Final RmsNorm kernel (last token only)
6. LM head projection → FP16 logits
7. ConvertF16ToF32 kernel → FP32 logits
8. cuStreamSynchronize ── single host sync point
9. cuMemcpyDtoH ── logits (FP32) to CPU
10. Return UnmanagedTensor to sampling pipeline
```

### Why FP16?

- cuBLAS FP16 GEMM uses Tensor Cores (Volta+) → ~2× throughput vs FP32
- FP16 activations = half the memory bandwidth → faster decode (bandwidth-bound)
- On pre-Volta GPUs (Pascal): cuBLAS FP16 GEMM falls back to CUDA cores at ~FP32 speed — still correct, just no Tensor Core speedup
- Numerical difference vs CPU (FP32) is minimal: top-k token predictions match

### Synchronization Model

A single CUDA stream executes all operations sequentially on the GPU. **No host–device synchronization occurs during the forward pass** — all kernel launches and cuBLAS calls are asynchronous. The only sync point is `cuStreamSynchronize` before the final logits D2H copy. This minimizes PCIe roundtrips and CPU overhead.

## cuBLAS GEMM/GEMV

cuBLAS provides the most compute-intensive operations:

- **Prefill** (seqLen > 1): `cublasHgemm` — FP16 matrix multiply with Tensor Core acceleration. Compute-bound. Tensor Cores activate automatically when dimensions are multiples of 8.
- **Decode** (seqLen = 1): Custom quantized GEMV kernels — operate directly on quantized weights. Memory-bandwidth-bound.

**Row-major convention**: cuBLAS is natively column-major. To compute `C = A × B` (row-major), we swap operands: `cublasHgemm(B, A) → C`. This is a well-proven trick used by llama.cpp, vLLM, and every CUDA inference engine.

See `CudaGemm.cs` for the implementation.

## Custom Kernels

All custom kernels are compiled to PTX via `nvcc -ptx` and loaded at runtime. See [CUDA.md](CUDA.md) for the kernel catalog, PTX conventions, and build instructions.

| Kernel | Purpose | Block Size | Grid Size |
|--------|---------|------------|-----------|
| `rmsnorm_f16` | RMS normalization | 256 | rows |
| `fused_add_rmsnorm_f16` | Residual add + RMS norm (FP32 add, avoids truncation) | 256 | rows |
| `rope_f16` | Rotary position embedding | 256 | dim pairs |
| `swiglu_f16` | Fused SiLU(gate) × up | 256 | ceil(n/256) |
| `add_f16` | Element-wise addition | 256 | ceil(n/256) |
| `embedding_lookup_*` | Token embedding (F32/F16/Q8_0) | 256 | seqLen |
| `attention_f16` | Scaled dot-product attention | 256 | heads × seqQ |
| `bias_add_f16` | Add bias per token | 256 | ceil(n/256) |
| `per_head_rmsnorm_f16` | QK-norm (Qwen3) | 256 | heads × seqLen |
| `quantized_gemv_*` | Quantized GEMV (Q8_0, Q4_K, Q6_K) | 256 | outputDim |
| `dequant_*_f16` | Dequantize to FP16 (for prefill scratch) | 256 | blocks |
| `convert_*` | FP16↔FP32 conversion | 256 | ceil(n/256) |

## GPU KV-Cache

FP16 device-resident, pre-allocated per layer.

- **Layout**: `[maxSeqLen, numKvHeads × headDim]` per layer, FP16 (2 bytes per element)
- **Update**: `cuMemcpyDtoD` — copy new K/V rows from scratch buffers to cache positions
- **Memory**: `2 × numLayers × numKvHeads × headDim × maxSeqLen × 2` bytes

See [KV_CACHE.md](KV_CACHE.md) for general KV-cache design (KV-cache quantization is implemented; paged attention and prefix caching are future steps).

## Quantization on GPU

Supported GGUF quantization formats for GPU inference:

| Format | Embedding | Dequant→FP16 | Quantized GEMV | Notes |
|--------|-----------|--------------|----------------|-------|
| F32 | Yes | Yes | — | Converted to FP16 on upload |
| F16 | Yes | — | — | Direct upload, used by cuBLAS directly |
| Q8_0 | Yes | Yes | Yes | Best quality quantized format |
| Q4_0 | No | Yes | No | Dequant-only (no custom GEMV kernel) |
| Q5_0 | No | Yes | No | Dequant-only (no custom GEMV kernel) |
| Q4_K | No | Yes | Yes | Good quality-to-size ratio |
| Q5_K | No | Yes | No | Dequant-only (no custom GEMV kernel) |
| Q6_K | No | Yes | Yes | High quality, larger than Q4_K |

**Decode** uses custom quantized GEMV kernels (Q8_0, Q4_K, Q6_K) that operate directly on quantized weights — no dequantization needed. For formats without a custom GEMV kernel, the weight is dequantized on-the-fly into a scratch buffer and cuBLAS GEMV is used.

**Prefill** always dequantizes into a scratch buffer before calling cuBLAS HGEMM. The scratch holds one projection at a time and is reused across all projections.

See [QUANTIZATION.md](QUANTIZATION.md) for block layouts.

## Weight Strategy: On-the-Fly Dequantization

### Problem

Storing quantized weights provides compression (e.g., Q4_K is ~4.5 bits/param), but cuBLAS GEMM for prefill requires FP16 input. A naive approach stores both the quantized copy (for decode GEMV) and a permanent FP16 copy (for cuBLAS). This **doubles** VRAM usage, negating the benefit of quantization:

| Model | Quantized only | Quantized + FP16 copy | Overhead |
|-------|---------------|----------------------|----------|
| Llama-3.2-1B Q8_0 | 1.1 GB | 3.3 GB | +200% |
| Llama-3.1-8B Q4_K_M | 4.9 GB | 21 GB | +330% |

### Industry Approaches

Three strategies exist in the ecosystem:

1. **Fused dequant-in-register (vLLM/Marlin)**: Custom CUDA kernels dequantize INT4→FP16 inside registers using bit-manipulation tricks, feeding directly into Tensor Core `mma` instructions. Dequantized values never touch VRAM. Near-ideal 4× speedup for INT4. *Extremely complex to implement — hand-tuned per quant format, requires Ampere+.*

2. **Fused quantized matmul (llama.cpp MMQ)**: Both operands stay quantized — weights in their original format, activations quantized to Q8_1 on-the-fly. Integer dot-product with scale factors. No FP16 anywhere. *Custom kernel per quant format pair, can't leverage Tensor Cores.*

3. **Temporary dequant buffer (llama.cpp cuBLAS path)**: Dequantize one projection at a time into a reusable scratch buffer, then call standard cuBLAS HGEMM. Scratch is returned to pool after each call. *Simple implementation, leverages cuBLAS Tensor Cores, small fixed VRAM overhead.*

### Our Choice: Approach 3

dotLLM uses the **temporary dequant buffer** approach because:
- All building blocks already exist (dequant PTX kernels + cuBLAS HGEMM wrappers)
- VRAM = quantized weights + one scratch buffer (~32 MB for 1B model)
- cuBLAS provides Tensor Core acceleration on Volta+ GPUs
- Decode path uses custom quantized GEMV (no dequant needed for Q8_0/Q4_K/Q6_K)
- Future: fused quantized GEMM kernels (approach 1 or 2) can replace the prefill path without changing the weight storage strategy

## Attention on GPU

The current GPU attention kernel is **naive** (not flash attention):
- One thread block per (query_token, query_head) pair
- Computes full QK^T score vector, applies causal mask + optional sliding window, softmax, weighted V sum
- GQA support: KV head broadcast via `group_size = num_heads / num_kv_heads`
- FP16 data, FP32 accumulation for numerical stability
- Shared memory for scores + output accumulator

See [ATTENTION.md](ATTENTION.md) for mechanism details. **Flash attention** (tiled, O(N) memory) is a planned future optimization.

## CLI Usage

```bash
# Run on GPU (default: gpu:0)
dotllm run SmolLM-135M --prompt "Hello" --device gpu

# Run on specific GPU
dotllm run model.gguf --prompt "Hello" --device gpu:1

# Chat mode on GPU
dotllm chat model.gguf --device gpu

# CPU (default)
dotllm run model.gguf --prompt "Hello" --device cpu
```

## Prerequisites

- **NVIDIA GPU**: Compute capability 6.1+ (Pascal and newer). Recommended: 7.0+ (Volta) for Tensor Core acceleration.
- **NVIDIA GPU driver**: 525.60+ (CUDA 12.x compatibility).
- **cuBLAS**: Required for GEMM. Installed with CUDA Toolkit.
- **CUDA Toolkit**: Required only for compiling `.cu` → `.ptx` kernels. Pre-compiled PTX files can be distributed.
- **No CMake, no C/C++ compiler** — only `nvcc` for kernel compilation.

See [CUDA.md](CUDA.md) for detailed prerequisites and build instructions.

## Performance Characteristics

| Phase | Bottleneck | GPU Advantage |
|-------|------------|---------------|
| **Prefill** | Compute (GEMM) | Tensor Cores: ~100+ TFLOPS FP16 vs ~1 TFLOPS AVX2 |
| **Decode** | Memory bandwidth | HBM2e: ~1 TB/s vs DDR5: ~90 GB/s |
| **Sampling** | Trivial | Runs on CPU (vocabSize ops, negligible) |

### Expected Throughput Targets

- **Prefill**: 10–50× over CPU (compute-bound, Tensor Core dominated)
- **Decode**: 3–10× over CPU (bandwidth-bound, HBM advantage)
- **Target**: >50 tok/s decode on consumer GPU (RTX 3090/4090) with 7B model

## Hybrid CPU/GPU Inference

When a model doesn't fully fit in VRAM, hybrid mode runs part of the model on GPU and the rest on CPU. The `--gpu-layers N` option specifies how many transformer layers execute on GPU (bottom layers), with the remainder on CPU (top layers). The embedding lookup runs on GPU, while the final RMSNorm and LM head run on CPU. This matches llama.cpp's `--n-gpu-layers` convention.

### Architecture

```
┌─────────────────────────────────────────────────┐
│                    GPU (FP16)                    │
│                                                  │
│  Token IDs ─► Embedding Lookup ─► Hidden (FP16)  │
│                                                  │
│  Layer 0:  RmsNorm → Q/K/V → RoPE → Attn →      │
│            O → Add+RmsNorm → Gate/Up → SwiGLU →  │
│            Down → Add+RmsNorm                     │
│  ...                                             │
│  Layer N-1: (same, but last uses plain Add)      │
│                                                  │
│  HiddenState [seqLen, hiddenSize] FP16           │
├──────────── D2H Transfer (PCIe) ─────────────────┤
│  FP16 → FP32 conversion                         │
│                                                  │
│                    CPU (FP32)                     │
│                                                  │
│  Layer N:  RmsNorm → Q/K/V → RoPE → Attn →      │
│            O → Add → RmsNorm → Gate/Up →         │
│            SwiGLU → Down → Add                    │
│  ...                                             │
│  Layer L-1: (same)                               │
│                                                  │
│  Final RmsNorm → LM Head → Logits [vocabSize]    │
└──────────────────────────────────────────────────┘
```

### CLI Usage

```bash
# Full GPU (default when --device gpu)
dotllm run model.gguf -d gpu -p "Hello"

# Hybrid: 20 of 32 layers on GPU, 12 on CPU
dotllm run model.gguf --gpu-layers 20 -p "Hello"

# CPU only (default, or explicit)
dotllm run model.gguf --gpu-layers 0 -p "Hello"

# Chat mode hybrid
dotllm chat model.gguf --gpu-layers 20
```

Unlike llama.cpp's `-ngl`, the short form is not available (Spectre.Console requires single-character short options). Use the full `--gpu-layers` flag.

### Layer Assignment

Layers are assigned bottom-up: GPU gets layers 0..N-1 (lower/earlier layers), CPU gets layers N..L-1 (upper/later layers). Rationale:

- Lower layers benefit most from GPU compute throughput (large matrix multiplications)
- The LM head (vocabSize × hiddenSize projection) runs on CPU since the hidden state already resides there after CPU layers, avoiding a large vocab-sized D2H transfer
- The embedding lookup runs on GPU (small H2D transfer of token IDs, large embedding table stays in VRAM)

### KV-Cache Split

GPU layers store their KV-cache in FP16 GPU device memory (`CudaKvCache`). CPU layers store their KV-cache in FP32 host memory (`SimpleKvCache`). A `HybridKvCache` routes KV operations by layer index:

- Layers 0..N-1: GPU cache, updated via device-side copies (`UpdateDevice`)
- Layers N..L-1: CPU cache, updated via standard `IKvCache.Update()` with remapped indices

Both caches advance in lockstep — same positions, same sequence length — since the model processes all layers for every token.

### Boundary Transfer

At the GPU/CPU boundary (after the last GPU layer), the hidden state tensor `[seqLen, hiddenSize]` is:

1. Copied D2H as FP16 (2 bytes/element) via `cuMemcpyDtoH`
2. Converted FP16 → FP32 on the CPU host (scalar `Half→float` loop)

Transfer sizes for hiddenSize = 4096:

| Phase | Tokens | FP16 Transfer | Time (PCIe 4.0 x16) |
|-------|--------|---------------|---------------------|
| Decode | 1 | 8 KB | ~0.3 μs |
| Prefill 128 | 128 | 1 MB | ~30 μs |
| Prefill 1024 | 1024 | 8 MB | ~250 μs |
| Prefill 4096 | 4096 | 32 MB | ~1 ms |

The boundary transfer is negligible for decode and small relative to compute for prefill.

### VRAM Estimation

In hybrid mode, VRAM usage is proportional to the number of GPU layers:

- **GPU weights**: Only layers 0..N-1 uploaded (quantized + optional FP16 copies)
- **Token embeddings**: Always on GPU (relatively small)
- **GPU KV-cache**: FP16, only for N layers: `2 × N × numKvHeads × headDim × maxSeqLen × 2 bytes`
- **GPU scratch buffers**: Fixed size regardless of layer count (activation buffers for one layer at a time)
- **Output weights (LM head)**: NOT on GPU — CPU handles final projection

Approximate formula:

```
VRAM ≈ (N / L) × model_weight_bytes + embed_bytes + gpu_kv_cache + gpu_scratch
```

### Edge Cases

| Scenario | Behavior |
|----------|----------|
| `--gpu-layers 0` | Pure CPU, no CUDA initialization (safe on machines without GPU) |
| `--gpu-layers N >= L` | Pure GPU, identical to `--device gpu` |
| `--gpu-layers 1` | Single layer on GPU — minimal GPU acceleration, validates pipeline |
| `--device gpu --gpu-layers N` | Explicit GPU device with partial offload — uses specified GPU ordinal |

### Numerical Precision

GPU layers compute in FP16 (Half precision). At the boundary, the hidden state is converted FP16 → FP32 for CPU layers. This introduces a small quantization error (~5e-4 relative error) compared to pure-FP32 CPU execution, identical to the error in pure-GPU mode. The final logits match pure-GPU output for the GPU layers and pure-CPU output for the CPU layers.

### Implementation

- `HybridTransformerModel` — `IModel` implementation orchestrating split forward pass
- `HybridKvCache` — Routes `IKvCache` operations to `CudaKvCache` or `SimpleKvCache` by layer
- `CudaQuantizedKvCache` — Q8_0/Q4_0 GPU KV-cache with FP16 scratch-buffer attention (see [KV_CACHE.md](KV_CACHE.md))
- `CudaWeights.LoadFromGguf(numGpuLayers)` — Partial weight upload to VRAM

## Future Work

- **Flash Attention**: Replace naive attention with tiled flash attention for O(N) memory and better SM utilization
- **Fused Quantized GEMM**: Custom PTX kernels for Q4_K × FP16 (Marlin-style or MMQ-style) to eliminate per-projection dequant overhead during prefill
- **Multi-GPU** (Step 51): NCCL-based tensor parallelism
- **Fatbin distribution**: Pre-compiled SASS for common architectures to eliminate JIT overhead

See [ROADMAP.md](ROADMAP.md) Phase 4 for the full plan.
