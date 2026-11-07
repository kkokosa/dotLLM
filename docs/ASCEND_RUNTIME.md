# Experimental Ascend runtime

The single-card backend implements dense Llama/Qwen2 GGUF inference through C#
model orchestration, AscendCL allocation and CANN ACLNN operators. It connects
`run`, `chat` and `serve` through `--device ascend[:N]`, with independent contiguous
KV per request and synchronous serialized model execution.

The correctness baseline uses **FP32 expanded weights and activations, FP16 KV,
and FP16 RoPE tables**. F32/F16/Q8_0 describe the source GGUF storage; Q8_0 is
expanded at load time and does not execute quantized NPU matrix multiplication.
The initial all-FP16 path exceeded the fixed full-model error budget on short
prompts; the default was changed to preserve FP32 precision. FP16 kernel smoke
coverage remains separate from the default model path.

## Environment and source build

Verified worker: Debian 12 x86_64, Ascend 910B4-64G, CANN/toolkit/ops 8.3.RC1,
g++ 12.2.0, npu-smi 25.5.0, .NET/ASP.NET 10.0.0. The managed development build
uses .NET SDK 10.0.100. No NPU or CANN is needed to compile managed projects.
Other CANN/driver combinations, ARM64, Windows, Native AOT and NuGet delivery
are not verified for this backend.

```bash
# From the repository root; ordinary dotnet builds do not invoke CMake.
dotnet build src/DotLLM.Cli/DotLLM.Cli.csproj -c Release -p:UseAppHost=false \
  --artifacts-path /tmp/dotllm-ascend-managed
dotnet build tests/DotLLM.Tests.Ascend/DotLLM.Tests.Ascend.csproj -c Release \
  -p:UseAppHost=false --artifacts-path /tmp/dotllm-ascend-managed

# Native build and execution require a worker with the CANN toolkit and driver.
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S native/ascend -B /tmp/dotllm-ascend-native -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++
cmake --build /tmp/dotllm-ascend-native --parallel 2
export DOTLLM_ASCEND_LIBRARY=/tmp/dotllm-ascend-native/libdotllm_ascend.so
```

The tested image has an inherited `CXX` pointing at an unavailable compiler;
select an installed compiler explicitly after sourcing CANN. CMake links
`ascendcl`, `nnopbase`, `opapi` and rejects undefined symbols. The custom bridge
is **ABI 3**; the lazy resolver checks all required symbols and rejects an older
or incomplete library. CANN/driver libraries are supplied by the worker.
Alternatively place the bridge alongside the managed executable.

On the tested hardened Linux host, .NET requires
`export DOTNET_EnableWriteXorExecute=0`. The packaged worker uses invariant
globalization and an official ASP.NET runtime whose SHA512 was verified.
Python/NumPy are used only to prepare and run acceptance tests.

## CLI and service

```bash
# Replace the path below with the managed output from your build.
export DOTLLM_DLL=/tmp/dotllm-ascend-managed/bin/DotLLM.Cli/release/DotLLM.Cli.dll
dotnet "$DOTLLM_DLL" run /path/to/model.gguf --device ascend:0 \
  --prompt 'Explain what a key value cache does in a transformer.' \
  --max-tokens 32 --temp 0 --seed 42 --json
dotnet "$DOTLLM_DLL" chat /path/to/model.gguf --device ascend:0
dotnet "$DOTLLM_DLL" serve /path/to/model.gguf --device ascend:0 \
  --host 127.0.0.1 --port 8080 --no-browser

curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true}'
```

`ascend` selects logical ordinal 0; `ascend:N` selects an explicit ordinal.
A container's logical ordinal can differ from the `npu-smi` physical card ID.
Routing occurs before the CPU `gpuLayers=0` branch. Ascend initialization or
computation failure is reported; there is no silent CPU fallback.

Leave `--cache-type-k`/`--cache-type-v` at their inherited defaults: the Ascend
factory supplies FP16 KV independently of those CPU flags. Explicit GPU-layer
offload, paged/quantized KV, cache windows and speculative models are rejected
before model resolution/allocation. Prefix reuse is disabled. Service requests
are serialized; accepting two clients does not implement continuous batching.

## Capability and data flow

Supported model semantics: dense `llama`/`qwen2`, RMSNorm, SwiGLU, full-dimension
unscaled RoPE, and GQA (including MHA/MQA). GGUF row-major matrix mapping, all
projection biases and tied output embeddings have dedicated fixtures. Config
architecture and RoPE pairing must match: interleaved Norm for Llama, NeoX halves
for Qwen2. CANN's tested rotary modes are 1 for interleaved and 0 for NeoX.
Unsupported tensors/quantization types, scaled or per-frequency RoPE, MoE, MLA,
sliding windows, QK norms and other architectures are rejected.

Weights are dequantized in bounded unmanaged staging, checked for finite values
and uploaded once as FP32. Weights, residuals, activations, scores and KV remain
on the NPU. The host prepares token selection and RoPE/mask control metadata;
only the final position's FP32 logits are downloaded for sampling.

| Operation | Default model path |
| --- | --- |
| Linear / output head | FP32 `aclnnMatmul`, transposed stride view of W[N,K], KEEP_DTYPE=0 |
| RMSNorm / residual / bias / SwiGLU | FP32 ACLNN operations |
| RoPE | FP32 rotary with converted FP16 cosine/sine tables |
| Attention | FP32 QK, causal mask, softmax and PV; cached FP16 K/V converted on device |
| KV append | FP32 K/V cast to FP16 and synchronous D2D slice copies |
| Sampling / text / HTTP | Existing managed generator/tokenizer/server |

The reference attention is per-head and materializes scores; it uses repeated
scoped workspace allocations, descriptor construction and synchronization.
It is a correctness baseline, with no FlashAttention, graph replay, NZ packing,
quantized GEMV, tensor parallelism or HCCL. No speed advantage is claimed.

## Contracts and ownership

`AscendSession` owns an explicit context, stream and tensors. Every executing
thread restores the context. Runtime initialization is shared and finalized after
the last session. DotLLM must exclusively own AscendCL initialization in the
process; independent torch_npu/CANN consumers are unsupported.

Managed operators check dtype, shape, byte ranges, foreign sessions, disposal and
output aliasing. Copy operations are synchronous while host spans are pinned.
Native execution drains the stream before releasing workspace/descriptors;
repeatable executors are owned, executed once and destroyed rather than cached.
Same-dtype no-op Cast planning is avoided in the FP32 path because its executor
cannot be marked repeatable on the tested CANN version.

`Forward(tokens, positions, -1, cache)` executes on the model's bound NPU and
returns a non-owning host FP32 view `[1,vocab]`, valid until the next forward or
disposal. Explicit nonnegative device IDs must match the model. Positions append
contiguously; caches belong to exactly one model/context. Each forward commits
cache length only on success; failure aborts its logical contents. Rollback and
reset expose only the committed prefix, and overflow/token errors are explicit.
A native operation failure invalidates the session: dispose and reload it.

Tracked weight memory is `4 × loaded elements`, with tied heads counted once.
FP16 KV uses `2 × layers × maxLength × kvHeads × headDim × 2` bytes. Scratch
accounting includes tracked device buffers and small host staging; it is not a
measurement of driver/runtime workspace peak. Loading reserves headroom before
allocation. Full unload and failed-load tests inspect actual free device memory.

## Reproducible acceptance

```bash
dotnet /tmp/dotllm-ascend-managed/bin/DotLLM.Tests.Ascend/release/DotLLM.Tests.Ascend.dll \
  --reference-self-test
dotnet /tmp/dotllm-ascend-managed/bin/DotLLM.Tests.Ascend/release/DotLLM.Tests.Ascend.dll \
  --device 0
```

The first command runs host assertions only. Hardware mode requires native/CANN
libraries and exits nonzero on missing dependencies; no hardware checks silently
skip. `run-full.sh` verifies the per-file manifest, records environment, compiles
the bridge and runs kernels, models, fixtures, text/KV, cleanup and CLI/API tests.
See its package layout and parameters in `tests/DotLLM.Tests.Ascend`.

Independent goldens are generated by `gguf_reference.py` (exact expanded F32
GGUF) and `numpy_reference.py` (dense FP32 math without DotLLM kernels), using the
same source weights/token IDs. CPU Q8 activation arithmetic is a diagnostic,
not the exact mathematical golden. The model error budget is fixed at
`RMS(error) <= 0.03 + 0.015 × RMS(reference)`, max absolute error <= 0.25 and
cosine >= 0.999. Cache equivalence uses tighter zero-atol RMS/max bounds;
rollback replay must be bit-exact. Reports retain raw and centered errors,
top-1/top-5 and all thresholds. The 2026-10-24 hardware suite passed all 16 stages
with exit code 0; both real models, six storage fixtures and CLI/API checks are
covered in [full results and CPU comparison](https://github.com/Cheongroach/dotLLM/blob/archive/ascend-research-20261024/docs/ASCEND_FULL_RESULTS.md).

The original FP16 Linear bring-up remains documented in [M0 results](https://github.com/Cheongroach/dotLLM/blob/archive/ascend-research-20261024/docs/ASCEND_M0_RESULTS.md).
