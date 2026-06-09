# Vulkan Backend Architecture — dotLLM

## Why a Vulkan Backend

The `DotLLM.Cuda` backend is NVIDIA-only: it P/Invokes the CUDA Driver API
(`libcuda.so` / `nvcuda.dll`) and cuBLAS, and loads PTX text files. On
non-NVIDIA GPUs — AMD Radeon, Intel Arc, Apple Silicon (via MoltenVK),
mobile Adreno/Mali — that path returns nothing.

`DotLLM.Vulkan` exists to cover that gap. It uses the same P/Invoke
philosophy as the CUDA backend — no custom C shared library — but targets
the Vulkan loader (`vulkan-1.dll` / `libvulkan.so.1`) and loads SPIR-V
compute shaders instead of PTX. The architectural parallel is exact:

| | CUDA backend | Vulkan backend |
|---|---|---|
| Native loader | `libcuda.so` / `nvcuda.dll` | `libvulkan.so.1` / `vulkan-1.dll` |
| Shader IR | PTX (text) | SPIR-V (u32 binary) |
| Kernel source | CUDA C++ (`.cu`) | GLSL compute (`.comp`) |
| Compiler | `nvcc -ptx` | `glslc --target-env=vulkan1.2` |
| Module type | `CUmodule` | `VkShaderModule` |
| Launch | `cuLaunchKernel` | `vkCmdDispatch` |
| Vendor reach | NVIDIA only | AMD, NVIDIA, Intel, Apple (MoltenVK), Qualcomm, ARM |

The existing gap-analysis in `docs/CUDA.md` §1 concludes that Vulkan
compute is the only cross-vendor path with full custom-kernel expressivity
and a credible route to Tensor-Core-class throughput via
`VK_NV_cooperative_matrix2` (October 2024). That extension adds
dequantization callbacks specifically for quantized LLM inference and is
implemented by NVIDIA and AMD; Intel support is tracked in the Mesa ANV
driver. Recent benchmarks in llama.cpp's `ggml-vulkan` backend reach
~70–95% of native CUDA throughput on RTX 4090.

## Scope of This PR

**Proof of pipeline only.** This PR establishes the plumbing:

- `DotLLM.Vulkan` project with raw `[LibraryImport("vulkan-1")]` P/Invoke
  — no Silk.NET, no external bindings package. ~15 Vulkan entry points
  covering instance, physical device, logical device, memory, buffer,
  shader module, compute pipeline, descriptor sets, command buffer,
  queue submit.
- Shader build pipeline: `native/vulkan/shaders/*.comp` → `glslc` →
  `native/vulkan/spv/*.spv`, driven by `build.sh` / `build.ps1` and also
  wired into MSBuild (`CompileVulkanShaders` target) so shaders rebuild
  incrementally when sources change and `glslc` is on PATH.
- One working kernel: `add.comp` implements `c[i] = a[i] + b[i]` over
  FP32 buffers.
- `VulkanDevice` — instance creation, physical-device selection
  (discrete > integrated; prefer AMD/NVIDIA over Intel integrated),
  compute queue + command pool, host-visible buffer allocation,
  upload/download.
- `VulkanModule` — loads one `.spv`, creates compute pipelines by
  entry-point name.
- `AddKernel` — wraps `add.comp`; records a one-shot command buffer,
  binds descriptor set, pushes `n` via push constant, dispatches,
  waits.
- Smoke test `VulkanAddKernelTests` — verifies 1024-element addition
  round-trips correctly; skips when no Vulkan loader/device is present.

**Not in scope.** Real LLM kernels, multi-GPU, staging ring, fence-based
pipelining, `VK_NV_cooperative_matrix2`, descriptor-set pooling across
launches, memory-type heuristics beyond HOST_VISIBLE|HOST_COHERENT.

## SPIR-V Compilation Pipeline

End users need only the compiled `.spv` blobs, which ship as MSBuild
`Content` alongside the managed DLL. Shader *authors* need the Vulkan
SDK (https://vulkan.lunarg.com/) for `glslc`. The MSBuild target is
idempotent: if `glslc` is not on PATH it logs a warning and uses the
committed `.spv` files.

```
native/vulkan/shaders/add.comp         (author edits)
            │
            │   glslc --target-env=vulkan1.2 -o add.spv add.comp
            ▼
native/vulkan/spv/add.spv              (checked in — ships to users)
            │
            │   <Content Include> in DotLLM.Vulkan.csproj
            ▼
bin/Debug/net10.0/spv/add.spv          (loaded at runtime)
            │
            │   File.ReadAllBytes → vkCreateShaderModule
            ▼
VkShaderModule handle
            │
            │   vkCreateComputePipelines
            ▼
VkPipeline → vkCmdBindPipeline → vkCmdDispatch
```

The driver compiles SPIR-V to vendor ISA at `vkCreateComputePipelines`
time and caches the result on disk (AMDGPU-PRO cache, Mesa shader cache,
NVIDIA internal blob). First-launch cost is amortized across process
restarts, same as PTX JIT under CUDA.

## P/Invoke Strategy

Identical to `DotLLM.Cuda`:

- `[LibraryImport("vulkan-1")]` with source-generated marshalling.
- `VulkanLibraryResolver` rewrites "vulkan-1" to the correct OS binary
  (`vulkan-1.dll`, `libvulkan.so.1`, `libvulkan.dylib`).
- Handles (`VkInstance`, `VkDevice`, `VkBuffer`, `VkDeviceMemory`, etc.)
  cross the boundary as opaque `nint` — tensor bytes never traverse
  P/Invoke.
- `VkResult` returned as `int`; negative values are errors, zero is
  `VK_SUCCESS`, positive values are non-error status codes
  (`VK_INCOMPLETE`).
- Structs declared `[StructLayout(LayoutKind.Sequential)]`. We only
  declare the fields actually used; extension tails are left as
  `fixed byte` padding (see `VkPhysicalDeviceProperties.limits`).

No Silk.NET dependency. Adding Silk.NET.Vulkan would give us ergonomic
bindings but ~15 MB of transitive DLLs and another layer to audit; the
~15 Vulkan entry points we actually need fit in a single file.

## Target Kernel Catalog

Future sessions port the `DotLLM.Cuda` catalog (see
`docs/CUDA.md` §Kernel Catalog) one kernel at a time. Order is chosen so
each lands a testable end-to-end slice:

| Phase | Kernel | CUDA file | Notes |
|---|---|---|---|
| 1 | `add`           | `add.cu`            | ✅ Done (this PR) |
| 2 | `rmsnorm_f32`   | `rmsnorm_f32.cu`    | Warp reduction → subgroup shuffle (`GL_KHR_shader_subgroup`) |
| 2 | `rope_f32`      | `rope_f32.cu`       | sin/cos lookup; no reduction |
| 2 | `swiglu_f32`    | `swiglu_f32.cu`     | Pointwise, trivial |
| 3 | `embedding_*`   | `embedding.cu`      | Gather; three dtypes (F32/F16/Q8_0) |
| 3 | `bias_add_f32`  | `bias_add_f32.cu`   | Pointwise |
| 3 | `softmax`       | `softmax.cu`        | Two-pass (max, exp-sum), subgroup reduction |
| 4 | `attention_f32` | `attention_f32.cu`  | Per-head QK<sup>T</sup>, softmax, AV |
| 5 | `dequant_q8_0`  | `dequant.cu`        | Per-block 32-element dequant |
| 5 | `dequant_q4_k`  | `dequant.cu`        | K-quant (superblock) |
| 6 | `quantized_gemv_q8_0` | `quantized_gemv.cu` | Decode-path GEMV on quantized weights |
| 6 | `quantized_gemv_q4_k` | `quantized_gemv.cu` | K-quant GEMV |
| 7 | FP16 pipeline   | `*_f16.cu`          | `VK_KHR_16bit_storage` + `VK_KHR_shader_float16_int8` |
| 8 | Cooperative matrix | (new) | `VK_NV_cooperative_matrix2` for Tensor-Core-equivalent GEMM |

Milestone 7 (FP16) is a significant enabler: most Vulkan drivers expose
`shaderFloat16` only through those two extensions. `VK_KHR_16bit_storage`
lets us read/write FP16 in storage buffers; `shader_float16_int8` lets
shaders operate on `float16_t` natively.

Milestone 8 (cooperative matrix) is the cross-vendor equivalent of
NVIDIA's `mma.sync` / Tensor Cores. `VK_NV_cooperative_matrix2` (Oct
2024) is the ticket to ~90% of cuBLAS FP16 GEMM throughput on a 4090,
and AMD's equivalent is on its RDNA3+ driver roadmap.

## GLSL Conventions

All kernels live under `native/vulkan/shaders/*.comp`. Conventions
mirror the CUDA kernels where possible:

- One `.comp` file per kernel; file name matches the CUDA file name
  (e.g. `rmsnorm_f32.comp` ↔ `rmsnorm_f32.cu`).
- `#version 450` for the baseline; bump to `#version 460` when
  cooperative matrix is needed.
- `layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;`
  matches the CUDA block size of 256.
- Storage buffers use `std430` layout and `readonly`/`writeonly`
  qualifiers as appropriate.
- Scalar uniforms (sizes, strides, configuration) are passed through
  push constants (max 128 bytes per pipeline). Larger config goes in
  a uniform buffer.
- Entry point is always `void main()`. The C# `AddKernel` passes
  `"main"` as the entry-point name to match.

## Physical-Device Selection

`VulkanDevice.Create` enumerates all physical devices and scores them:

- Device type: discrete (+1000), integrated (+500), virtual (+100), other (0).
- Vendor: NVIDIA/AMD (+20), Intel (+10), other (+5).

Highest score wins. Tie-breaking is first-enumerated. This prioritizes
a discrete GPU over an integrated one on hybrid laptops, and prefers
AMD/NVIDIA discrete over the (rare) Intel Arc discrete when both are
present. The heuristic is deliberately dumb — real placement policy
(explicit `--device` CLI flag, per-request device hints) is
infrastructure for later.

## Deferred Work

Tracked for future sessions:

1. **Staging buffers for large uploads.** The scaffold allocates
   host-visible buffers directly, which is OK for 1024 floats but will
   burn bandwidth on multi-GB model weights. Real uploads need a
   device-local destination plus a host-visible staging ring and
   `vkCmdCopyBuffer`.
2. **Descriptor-set pool reuse.** `AddKernel` currently allocates a
   descriptor set per launch. Port the per-kernel cache pattern from
   `CudaKernels`.
3. **Fence-based pipelining.** Every launch currently does
   `vkQueueWaitIdle` — synchronous, no overlap with host work.
   Replace with `VkFence` + per-in-flight command-buffer arena.
4. **`IBackend` integration.** `DotLLM.Vulkan` does not yet implement
   `DotLLM.Core.Backends.IBackend`. Hooking it up needs the
   `VulkanTransformerModel` equivalent of `CudaTransformerModel`, which
   in turn needs the attention/rmsnorm/rope kernels.
5. **Model loading.** GGUF weights → Vulkan buffers requires the mmap
   + staging path above.
6. **Validation layers.** `VK_LAYER_KHRONOS_validation` at
   instance-create time gives llama.cpp-style debug output. Opt-in via
   env var (`DOTLLM_VULKAN_VALIDATION=1`) once the SDK-install-check
   story is sorted.

## Building

```bash
# .NET build — compiles C# and (if glslc is on PATH) shaders.
dotnet build src/DotLLM.Vulkan/DotLLM.Vulkan.csproj

# Shader-only rebuild.
./native/vulkan/build.sh          # Linux / macOS / WSL / Git Bash
pwsh ./native/vulkan/build.ps1    # Windows PowerShell

# Run the scaffold test — passes if a Vulkan device is present, else skips.
dotnet test tests/DotLLM.Tests.Unit --filter FullyQualifiedName~Vulkan
```

Opt out on CI without a usable driver: `DOTLLM_SKIP_VULKAN=1`.
