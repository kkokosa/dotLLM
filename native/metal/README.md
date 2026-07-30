# Metal Kernel Port — CUDA → Metal Mapping Reference

## Context

The goal of this port is simple: get the existing CUDA kernels running on the
Apple Silicon GPU (Metal) as quickly as possible, so they can be tested on a
MacBook without requiring an NVIDIA GPU.

The approach was deliberately conservative — **each kernel is a line-by-line
translation of its CUDA counterpart**, with no algorithmic changes. The
mathematical correctness of the kernels (quantization formats, attention,
reductions) was not re-verified during this port: the assumption is that if the
CUDA kernel is correct, the Metal translation is too.

As a consequence, any pre-existing bug in a CUDA kernel is also present in the
Metal version.
Algorithmic improvements and bug fixes should be done on both backends together
in dedicated PRs.

This document is a reference for the mechanical CUDA → Metal substitutions used
throughout the port.

---

This folder contains the Apple Metal GPU kernels for dotLLM, ported from the
CUDA implementations in `native/kernels/`.

---

## Thread / Grid Identity

| CUDA | Metal attribute | Variable name used here |
|---|---|---|
| `threadIdx.x` | `[[ thread_position_in_threadgroup ]]` | `thread_idx_x` |
| `blockIdx.x` | `[[ threadgroup_position_in_grid ]]` | `block_id_x` |
| `blockDim.x` | `[[ threads_per_threadgroup ]]` | `block_dim_x` |
| `gridDim.x` | `[[ threadgroups_per_grid ]]` | `grid_dim_x` |
| `warpSize` | `[[ threads_per_simdgroup ]]` | `simd_group_width` |
| `thread_position_in_grid` | `[[ thread_position_in_grid ]]` | `idx` (convert/bias kernels) |

---

## Warp / SIMD Reductions

| CUDA | Metal |
|---|---|
| `__shfl_down_sync(0xffffffff, v, delta)` | `simd_shuffle_down(v, delta)` |
| `__reduce_add_sync` / manual warp reduce | `simd_sum(v)` |
| `__syncthreads()` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `__shared__ float buf[N]` | `threadgroup float buf[N]` |

> `simd_sum()` reduces across all lanes in the SIMD-group (32 on Apple Silicon)
> and is the idiomatic replacement for a manual `__shfl_down_sync` loop.

---

## Math & Type Intrinsics

| CUDA | Metal |
|---|---|
| `rsqrtf(x)` | `rsqrt(x)` |
| `__half2float(h)` | `float(h)` |
| `__float2half(f)` | `half(f)` |
| `*reinterpret_cast<const half*>(ptr)` | `*(device const half*)ptr` |
| `(const int8_t*)ptr` | `(device const char*)ptr` |
| `(const uint8_t*)ptr` | `(device const uchar*)ptr` |
| `__launch_bounds__(256)` | (no equivalent — not needed in Metal) |

---

## Dispatch Strategy

### Element-wise kernels (add, multiply, silu, …)

```
// CUDA
kernel<<<gridDim, blockDim>>>(...)   // each thread handles one element

// Metal bridge
[enc dispatchThreads:MTLSizeMake(n, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(tgw, 1, 1)]
```

`dispatchThreads` lets the GPU driver create exactly `n` threads and handles
the boundary automatically — no `if (idx >= n) return` needed in the kernel
(though it is kept for clarity in some kernels).

### Reduction kernels (rmsnorm, per_head_rmsnorm, fused_add_rmsnorm)

```
// CUDA
kernel<<<seq_len, blockDim>>>(...) // one block per token / (token, head)

// Metal bridge
[enc dispatchThreadgroups:MTLSizeMake(nGroups, 1, 1)
    threadsPerThreadgroup:MTLSizeMake(tgSize, 1, 1)]
```

`dispatchThreadgroups` is used here because the unit of work is a whole
threadgroup (one token), not a single thread. This is the direct equivalent
of CUDA's explicit grid/block launch.

---

## Shared (Threadgroup) Memory

```metal
// CUDA
__shared__ float simd_sums[32];

// Metal
threadgroup float simd_sums[32];
```

Threadgroup memory is zero-initialized by default in Metal for variables
declared with an initializer (`= 0.0f`). Without an initializer the value is
undefined, matching CUDA `__shared__` behaviour.

---

## Pointer Arithmetic on Device Pointers

In Metal, all device pointers must carry the `device` address-space qualifier
through every cast. Forgetting it silently produces wrong results.

```metal
// ✅ correct: advance pointer first, then cast and dereference
float d    = float(*(device const half*)block);
float dmin = float(*(device const half*)(block + 2));

// ❌ wrong: arithmetic happens AFTER dereference (adds 2 to the half value)
float dmin = float(*(device const half*)block + 2);
```

---

## Grid-Stride Loops

Both backends use grid-stride loops so a fixed grid can process an arbitrary
number of blocks / superblocks without launching one threadgroup per block.

```cuda
// CUDA — warp-per-block pattern (Q8_0 / Q4_0 / Q5_0)
int warps_per_grid = (gridDim.x * blockDim.x) / BLOCK_SIZE;
for (int b = start; b < total; b += warps_per_grid) { ... }

// CUDA — thread-per-element pattern (Q4_K / Q5_K / Q6_K)
for (int sb = blockIdx.x; sb < total; sb += gridDim.x) { ... }
```

```metal
// Metal — identical structure, different variable names
int warps_per_grid = (grid_dim_x * block_dim_x) / BLOCK_SIZE;
for (int b = start; b < total; b += warps_per_grid) { ... }

for (int sb = block_id_x; sb < total; sb += grid_dim_x) { ... }
```

---

## Known Issues

### Q5_K qs/qh layout mismatch (CUDA + Metal)

The `dequant_q5_k_f16` kernel in both `dequant.cu` and `dequant.metal` has a
bug in the qs/qh access pattern that does not match the CPU scalar reference
(`DequantizeQ5_KScalar`):

```metal
// Current (wrong — direct port of CUDA):
device const uchar* sub_qs = qs + sub * 16;   // should be qs + (sub/2) * 32
device const uchar* sub_qh = qh + sub * 4;    // should share qh[0..31]
int bit = (sub_qh[j/4] >> ((j%4)*2 + (pos&1))) & 1;

// Correct:
uint8_t qs_byte = qs[(sub / 2) * 32 + pos];
int lo4  = (sub % 2 == 0) ? (qs_byte & 0xF) : (qs_byte >> 4);
int bit5 = (qh[pos] >> sub) & 1;
```

Both kernels will be fixed together in a follow-up PR.
The Metal tests for Q5_K are currently marked `[Fact(Skip = ...)]`.
