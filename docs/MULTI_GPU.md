# Multi-GPU & Tensor Parallelism — dotLLM

## Overview

For models exceeding single-GPU memory (70B+ parameters), the engine distributes compute across multiple GPUs. Two strategies are supported: **tensor parallelism (TP)** for splitting operations within a layer, and **pipeline parallelism (PP)** for splitting layers across devices.

## Tensor Parallelism (TP)

The primary multi-GPU strategy. Splits individual tensor operations so each GPU holds a fraction of the model's weights and computes a fraction of each layer.

### Attention Splitting

For TP degree = N:
- **Q/K/V projections**: Each GPU holds `num_heads / N` attention heads. GPU `i` computes attention for heads `[i * heads_per_gpu .. (i+1) * heads_per_gpu)`.
- **Attention output**: Each GPU computes partial attention output for its heads. The output projection (`o_proj`) is row-split — each GPU holds a horizontal slice.
- **All-Reduce**: After the output projection, an **all-reduce** across GPUs sums the partial results to produce the full attention output on every GPU.

For GQA: KV heads are also split across GPUs. With 8 KV heads and TP=4, each GPU gets 2 KV heads.

### FFN Splitting

- **Gate + Up projection**: Column-split. GPU `i` holds columns `[i * cols_per_gpu .. (i+1) * cols_per_gpu)`. Each GPU computes a slice of the intermediate activations.
- **Activation (SiLU/GELU)**: Applied locally to each GPU's slice.
- **Down projection**: Row-split. Each GPU holds rows corresponding to its intermediate slice, producing a partial output.
- **All-Reduce**: After down projection, sum partial outputs across GPUs.

### Communication Pattern

Each transformer layer requires **2 all-reduce operations** (one after attention, one after FFN). All-reduce is the dominant communication cost.

```
Per layer:
  Input (replicated on all GPUs)
  → Q/K/V matmul (local, split heads)
  → Attention (local)
  → Output projection (local, row-split)
  → All-Reduce ← communication
  → Residual add (local)
  → Gate+Up matmul (local, column-split)
  → SiLU (local)
  → Down matmul (local, row-split)
  → All-Reduce ← communication
  → Residual add (local)
```

## Pipeline Parallelism (PP)

Splits layers across GPUs. GPU 0 runs layers 0..L/N-1, GPU 1 runs layers L/N..2L/N-1, etc.

### Communication

Only point-to-point **send/recv** between adjacent pipeline stages (forward the hidden state). Lower bandwidth than TP's all-reduce, but harder to keep all GPUs busy.

### Micro-batching

To keep all pipeline stages busy, split the batch into micro-batches that flow through the pipeline in sequence. While micro-batch 2 is in stage 1, micro-batch 1 is in stage 2. More micro-batches → better utilization but higher memory.

### When to Use

- **TP alone**: Preferred for ≤8 GPUs within a single node (fast NVLink interconnect).
- **PP alone**: Simpler, but low utilization without micro-batching.
- **TP + PP combined**: For very large models across multiple nodes. TP within a node (NVLink), PP across nodes (InfiniBand/ethernet).

## NCCL Integration

All GPU-to-GPU communication uses **NCCL** (NVIDIA Collective Communications Library) via the native C/CUDA library:

### Native API

```c
// Initialize
dotllm_nccl_init(int num_devices, int* device_ids, NcclComm** comms);

// Collective operations
dotllm_nccl_all_reduce(NcclComm* comm, void* sendbuf, void* recvbuf,
                        size_t count, DType dtype, NcclOp op, cudaStream_t stream);

dotllm_nccl_send(NcclComm* comm, void* buf, size_t count, DType dtype,
                  int peer, cudaStream_t stream);

dotllm_nccl_recv(NcclComm* comm, void* buf, size_t count, DType dtype,
                  int peer, cudaStream_t stream);

// Cleanup
dotllm_nccl_destroy(NcclComm** comms, int num);
```

### C# Interop

```csharp
[LibraryImport("dotllm_native")]
internal static partial int dotllm_nccl_all_reduce(
    nint comm, nint sendbuf, nint recvbuf,
    nuint count, DType dtype, NcclOp op, nint stream);
```

## Architectural Requirements

These must be satisfied from day one, even before multi-GPU is implemented:

### 1. Explicit Device Placement

Every tensor carries a `DeviceId` in its metadata. No implicit "current device" global state.

```
TensorMetadata:
  Shape: TensorShape
  DType: DType
  DeviceId: int          // -1 = CPU, 0..N = GPU index
  DataPointer: nint
```

### 2. IBackend Multi-Device Support

```
IBackend:
  DeviceCount → int
  AllocateOnDevice(deviceId, shape, dtype) → ITensor
  CopyBetweenDevices(src, dst) → void
  AllReduce(tensors[], op) → void     // across devices
  Send(tensor, targetDevice) → void
  Recv(sourceDevice, shape, dtype) → ITensor
```

Single-device backend: `DeviceCount = 1`, collective ops are no-ops.

### 3. ParallelismConfig

```
ParallelismConfig:
  TpDegree: int           // Tensor parallelism degree (1 = no TP)
  PpDegree: int           // Pipeline parallelism degree (1 = no PP)
  DeviceMap: int[]         // Layer → device assignment for PP
  TpGroup: int[]          // Device IDs in tensor parallel group
```

The model forward pass is parameterized by this config. When `TpDegree = 1` and `PpDegree = 1`, no parallelism — the default path has zero overhead.

### 4. Device-Local KV-Cache

Each GPU manages its own KV-cache pool for its assigned attention heads. The `PagedKvCacheManager` becomes per-device, with a global coordinator tracking total capacity.

### 5. Weight Distribution

At model load time, weights are sharded according to `ParallelismConfig`:
- TP: Each GPU receives its slice of Q/K/V/O projections and FFN weights.
- PP: Each GPU receives complete weights for its assigned layers.

GGUF memory mapping works per-device: each GPU maps and copies only its weight slice from the memory-mapped file.

## Scaling Limits

| Configuration | Use Case | Interconnect |
|--------------|----------|--------------|
| TP=2-4, PP=1 | 13B-34B models, single node | NVLink preferred |
| TP=8, PP=1 | 70B models, single 8-GPU node | NVLink required |
| TP=8, PP=2 | 70B+ models, 2 nodes | NVLink intra-node, IB inter-node |
| TP=8, PP=4+ | 405B+ models, multi-node | High-bandwidth IB required |

## Implementation Notes

- Start with **TP only** — covers 90% of multi-GPU use cases.
- PP is primarily needed for cross-node scaling (uncommon for initial users).
- NCCL initialization requires all GPUs to be visible to the process (`CUDA_VISIBLE_DEVICES`).
- Test with TP=2 first — catches all the communication bugs with minimal hardware.
- Weight sharding logic must match the exact tensor naming conventions in GGUF metadata.
