# DotLLM.Metal — Apple Silicon GPU Backend

This is the Metal backend for dotLLM, targeting the unified-memory GPU shipped
with every Apple Silicon Mac (M1, M2, M3, M4 and successors).

## Porting strategy: line-by-line CUDA mirror

Every Metal compute kernel under `native/metal/*.metal` is a **deliberate,
mechanical translation of its CUDA counterpart** in `native/kernels/*.cu`.
No algorithmic changes were introduced during the port:

- Identical reduction structure (warp / SIMD-group then cross-warp).
- Identical quantization-format byte layouts (Q8_0, Q5_0, Q4_K, Q5_K, Q6_K).
- Identical attention masking, RoPE math, and softmax stabilisation.

The working assumption is: **if the CUDA kernel is mathematically correct,
the Metal port is correct as well.** Bugs are inherited (and fixed) jointly
on both backends.

This convention has two practical consequences:

1. The CUDA reference remains the source of truth. Any new model architecture
   or kernel optimisation should land on CUDA first, then be mirrored here.
2. Reading the corresponding `.cu` file alongside any `.metal` file is the
   fastest way to understand what a kernel does.

## Conversion reference

The mechanical CUDA → Metal substitutions used throughout the port (thread
identity attributes, warp reductions, math intrinsics, dispatch strategies,
shared memory, pointer arithmetic on `device` pointers, grid-stride loops,
…) are documented once, in detail, in:

> [`native/metal/README.md`](../../native/metal/README.md)

That document is the canonical reference for anything of the form *"how do
I express this CUDA construct in Metal?"*. It is intentionally not duplicated
here.

If you are about to port a new kernel, read that file first.

---

## GEMM via Metal Performance Shaders

The quantized GEMV kernels cover **decode** (`seqLen = 1`, matrix × vector).
**Prefill** (matrix × matrix) uses **`MPSMatrixMultiplication`** — Apple's tuned
FP16/FP32 matmul from Metal Performance Shaders — exposed via
`dotllm_metal_gemm_f16` / `_f32` and the `Gemm` C# wrapper.

LLM projections (`Y = X · Wᵀ`, with GGUF weights stored `[outputDim, inputDim]`)
all use the same call convention:

```
Gemm.ExecuteF16(ctx, x, w, y,
    m=seqLen, n=outputDim, k=inputDim,
    transposeA=false, transposeB=true, alpha=1, beta=0)
```

Quantized weights must be dequantized to FP16 first (see `MetalWeights`, below).

---

## Phase 3 — MetalWeights

`MetalWeights` is the GPU-side representation of every model parameter
loaded from a GGUF file: per-layer Q/K/V/O and FFN matrices, RMSNorm
scales, optional QK-norm and biases, the embedding table, the final norm,
and the LM head.

The class is structurally analogous to `CudaWeights` in the CUDA backend —
intentionally so, to keep the two backends symmetric — but its
implementation is dramatically simpler thanks to Apple Silicon's unified
memory model.

### Memory model: unified vs. discrete

#### CUDA: discrete VRAM

```
┌──────────────────┐      PCIe       ┌──────────────────┐
│   CPU / Host RAM │ ◄─────────────► │   GPU / VRAM     │
│   ┌────────────┐ │   cudaMemcpy    │  ┌────────────┐  │
│   │ mmap GGUF  │ │ ══════════════► │  │ Weights    │  │
│   └────────────┘ │   (one-shot)    │  └────────────┘  │
└──────────────────┘                 └──────────────────┘
        ▲                                     ▲
   host pointer                       device pointer (nint)
   *ptr works                         *ptr → segfault
```

CUDA must allocate VRAM with `cudaMalloc` and copy every weight from the
mmapped file into device memory once, at load time. The device pointer
returned has **no meaning** in the host process — that is why
`CudaWeights` exposes them as `nint`.

#### Metal: unified RAM

```
┌─────────────────────────────────────────────────────────┐
│                   Unified system RAM                    │
│       (CPU and GPU share the same physical pages)       │
│                                                         │
│   ┌──────────────────────────────────────────────┐      │
│   │   mmap'd GGUF — Llama-7B Q4_K (~4 GB)        │      │
│   └──────────────────────────────────────────────┘      │
│        ▲                                ▲               │
│        │ CPU reads directly             │ GPU reads     │
│        │ (when needed)                  │ directly      │
│                                                         │
│        ──────────── one address ────────────            │
│         (wrapped in MTLBuffer with newBuffer-           │
│          WithBytesNoCopy → zero-copy)                   │
└─────────────────────────────────────────────────────────┘
```

There is **no upload step**. The OS-mapped GGUF pages are the same physical
RAM the GPU reads. Wrapping them in an `MTLBuffer` is metadata-only;
demand-paging is handled by the kernel.

### What `MetalWeights` will do, concretely

1. Open the GGUF file (already memory-mapped by `DotLLM.Models.Gguf`).
2. For each named tensor expected by the architecture
   (`token_embd.weight`, `blk.{i}.attn_q.weight`, `blk.{i}.attn_norm.weight`,
   …), resolve the offset in the mmap and store:
   - the **pointer** (as `nint` — see below),
   - the **`QuantizationType`** (Q4_K, Q5_K, FP16, FP32, …),
   - the **dimensions** (`outputDim`, `inputDim` for projection matrices).
3. Handle architecture-specific optionals:
   - **Tied embedding** — when `output.weight` is absent, fall back to
     `token_embd.weight` (Llama 3.2 1B/3B, small Qwen variants).
   - **QK-norm** — `attn_q_norm.weight` and `attn_k_norm.weight` only
     present in Qwen3 / Cohere; pointers default to `0`.
   - **Biases** — Qwen2 has Q/K/V biases but no O bias; Llama has none.
     Each bias slot is nullable.
4. Compute a `TotalBytes` summary so the loader can warn the user if the
   model is larger than available system RAM (it will still load, but
   page heavily).
5. Implement `IDisposable` and chain to the `GgufFile` lifetime — the
   mmap **must** outlive `MetalWeights`.

### Why `nint`, not `Half*` / `byte*`

Even though Apple Silicon technically allows it, weight pointers are
exposed as `nint` rather than typed `unsafe T*`. Three reasons:

1. **Cross-backend symmetry.** `MetalLayerWeights` and `CudaLayerWeights`
   share the same shape, so future abstractions (`IBackendWeights`,
   diagnostics, per-tensor inspectors) work uniformly.
2. **Opacity signal.** `nint` makes it explicit that this memory is for
   GPU kernels to read; the CPU should not casually index into it. After
   a kernel writes a buffer, GPU caches must be flushed (we do this via
   `waitUntilCompleted` in the bridge today). Typed `Half*` would invite
   `weights.X[0]`-style accesses that work *most* of the time and break
   subtly when a kernel write hasn't completed yet.
3. **Type safety is recovered elsewhere.** A `nint` carries no format
   information by design — the format lives in
   `(QuantizationType, OutputDim, InputDim)` next to the pointer. That
   tuple is the actual schema; a `Half*` would lie about Q4_K data.

### Storage strategy: which formats to keep

For each weight matrix that arrives quantized, there is a design choice:

| Strategy | Decode (GEMV) | Prefill (GEMM) | Memory | Load time |
|----------|---------------|----------------|--------|-----------|
| **A — quantized only** | quantized GEMV kernel | dequant-on-the-fly per call | minimal | instant (mmap) |
| **B — FP16 only** | FP16 GEMV | MPS GEMM | ×3–4 | seconds–minutes (dequant pass) |
| **C — hybrid** | quantized GEMV (kept) | MPS GEMM on FP16 copy | ×1.3–1.5 | medium |

CUDA went with **C**. The recommendation for the first Metal milestone is
to start with **B** (FP16 only) because:

- only one code path through the forward pass,
- exercises `MPSMatrixMultiplication` for every projection,
- correctness can be validated end-to-end before optimising,
- on Apple Silicon the unified memory makes the ×3–4 cost less painful
  than on CUDA (no device VRAM ceiling, only system RAM).

The shape of `MetalLayerWeights` is designed so adding the quantized
fallback later is a non-breaking extension — extra `nint` slots and an
extra `QuantizationType` per matrix.

---

## Phase 4 — MetalForwardState

`MetalForwardState` holds the **pre-allocated scratch buffers** the forward
pass writes into for each token: hidden state, residual, Q/K/V projections,
attention output, FFN intermediates, logits. Where `MetalWeights` is the
read-only side of the memory layout, `MetalForwardState` is the writable
side.

It is the direct counterpart of `CudaForwardState`. And — like
`MetalWeights` — it is dramatically simpler than the CUDA equivalent
**because Apple Silicon's unified memory eliminates every host/device
boundary**. The structure stays identical; only the allocator changes.

### Line-by-line comparison with `CudaForwardState`

The current implementation deliberately mirrors `CudaForwardState`
field-by-field, so the forward pass code can stay symmetric across the two
backends. The differences below are *all* of them — no other line of
behaviour diverges.

| Aspect                       | `CudaForwardState`                       | `MetalForwardState`                                            |
|------------------------------|------------------------------------------|----------------------------------------------------------------|
| **Activation buffers**       | 11 buffers FP16                          | identical 11 buffers FP16                                      |
| **Logits FP16 + FP32**       | yes                                      | yes                                                            |
| **Token IDs / Positions**    | on device, copied from host              | unified RAM — **no H2D copy**                                  |
| **Capacity growth**          | power-of-2 on `seqLen`                   | identical                                                      |
| **Allocator**                | `cuMemAlloc_v2` / `cuMemFree_v2`         | `NativeMemory.AlignedAlloc` / `AlignedFree`                    |
| **Synchronisation**          | single CUDA stream                       | already synchronous via `waitUntilCompleted` in the bridge     |
| **`DequantScratch`**         | required for cuBLAS GEMM                 | depends on the chosen `IWeightLoadStrategy` (see below)        |
| **Reading the logits**       | `cuMemcpyDtoH` → FP32 host               | direct host read (same physical pages)                         |

### Why each row looks the way it does

**Allocator.** Apple Silicon does not need a driver call to obtain
GPU-visible memory: any aligned heap allocation is already accessible to
the GPU. `NativeMemory.AlignedAlloc` with a 64-byte alignment matches the
project's existing convention (`CLAUDE.md`) and gives us back a `nint`
that doubles as a host pointer and a Metal-buffer-wrappable address. This
is the **single substantive difference** in the code: a search-and-replace
on two lines.

**No host-to-device copies.** Token IDs and position indices are passed
into the forward pass as `int[]`. On CUDA they have to be copied into
device memory before any kernel can read them; on Metal the same array's
pinned address is already GPU-visible. We still allocate `TokenIds` and
`Positions` buffers in `MetalForwardState` for layout symmetry, but the
copy step disappears.

**No stream / no inter-kernel sync.** CUDA chains every kernel on a single
stream and only synchronises once per token. The current Metal bridge
calls `waitUntilCompleted` inside every kernel invocation, so each call
is already a synchronisation point from C#'s perspective.
`MetalForwardState` does not need to know about streams at all. (A future
optimisation will batch several kernels into a single
`MTLCommandBuffer` to recover the asynchronous benefits — but that
concerns the bridge, not the forward state.)

**Logits read for free.** The final logits buffer lives in unified RAM,
so the sampler reads it directly after the LM-head kernel completes.
That saves a `cuMemcpyDtoH` of `vocabSize × 4 bytes` (≈ 500 KB for a
128k vocabulary) per generated token.

**`DequantScratch`.** This buffer holds one projection's worth of FP16
weights when the forward pass needs to call `MPSMatrixMultiplication` on a
matrix that is currently stored quantized. Whether it is needed depends
on the loading strategy:

| Loading strategy           | Needs `DequantScratch`? |
|----------------------------|-------------------------|
| `DequantToFp16Strategy`    | **No** — every weight is already FP16 in `MetalWeights` |
| `HybridStrategy`           | **No** — same reason |
| `MmapOnlyStrategy`         | **Yes**, *if* prefill goes through MPS GEMM (the only path that requires FP16 input) |

The pragmatic choice is to **always allocate `DequantScratch`** on Metal,
sized for the largest projection. Strategies that don't need it simply
ignore the buffer, and the RAM cost is bounded (≈ 90 MB for Llama-7B,
negligible against the model itself).

### Why the implementation looks like a copy of `CudaForwardState`

The diff between the two files is essentially:

```diff
- CudaDriverApi.cuMemAlloc_v2(out nint ptr, (nuint)bytes).ThrowOnError();
+ nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);

- CudaDriverApi.cuMemFree_v2(ptr);
+ NativeMemory.AlignedFree((void*)ptr);
```

Everything else — the buffer list, the field names, the
power-of-2 capacity growth, the `EnsureCapacity` re-allocation pattern,
the dispose order — transfers verbatim. Keeping the two files in
lock-step is intentional: when a future change adjusts the buffer set on
one backend (a new model architecture, a fused kernel, a per-head
intermediate), the same change applies mechanically to the other.

The asymmetries that *would* normally exist between a discrete-VRAM
backend and a unified-memory backend (staging buffers, double-buffering,
explicit transfer queues, host-pinned memory pools) all collapse to
nothing here, because there is only one physical memory.
