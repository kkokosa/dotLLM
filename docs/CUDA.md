# CUDA Backend Architecture — dotLLM

## GPU Acceleration from .NET: Alternatives Research

Before committing to the native C/CUDA shared library approach (CMake → `libdotllm_native.so` → P/Invoke), we evaluated every viable path to GPU compute from C#/.NET. The goal: determine whether dotLLM can avoid creating a C/C++ library project while maintaining competitive inference performance.

### Evaluated Approaches

**ILGPU** (v1.5.3, ilgpu.net) is the strongest pure-C# option. It JIT-compiles C# kernel methods through .NET IL → SSA IR → CUDA PTX at runtime, entirely in managed code. Sponsored by G-Research, actively maintained, ~1,700 GitHub stars. Key strength: built-in cuBLAS wrapper in `ILGPU.Algorithms` providing FP16 GEMM with automatic Tensor Core usage. Key gaps: no Tensor Core access from custom kernels (no `wmma`/`mma.sync` API), no bfloat16 support, no published LLM workload benchmarks (one third-party benchmark showed ~3.7× slower than native for a physics workload). Flash Attention would be technically possible but impractical without direct Tensor Core access.

**ManagedCuda** (v13.0.64 by Michael Kunz) wraps the CUDA Driver API 1:1 in C# — loads PTX modules, calls cuBLAS, manages device memory. No kernel compilation, just orchestration. Critical issue: **switched to GPLv3/Commercial dual license from CUDA 12 onward**, making it incompatible with dotLLM's GPL v3 license without careful verification, and problematic for any future relicensing. Single-maintainer continuity risk. However, the architectural pattern it uses — P/Invoke against NVIDIA's own driver libraries + PTX loading — is freely replicable.

**ComputeSharp** (v3.2.0 by Sergio Pedri / Microsoft) transpiles C# to HLSL via Roslyn source generators, runs on DirectX 12. Production-proven (Microsoft Store, Paint.NET 5.0) but **fundamentally unsuitable**: Windows-only (DX12 hard dependency, cross-platform explicitly rejected), no CUDA/cuBLAS, no FP16, no tensor cores.

**Silk.NET** (v2.23.0, .NET Foundation) provides low-level Vulkan/OpenCL/DirectX bindings. **No CUDA bindings** (GitHub issue #558 never implemented). Vulkan compute path is viable — `VK_NV_cooperative_matrix2` (Oct 2024) adds dequantization callbacks specifically for quantized LLM inference, and recent benchmarks show Vulkan approaching CUDA parity on RTX 4090. But you'd write GLSL shaders, not C#, and build an entire compute framework — similar effort to CUDA with less ecosystem support.

**NVIDIA provides no official .NET SDK.** Confirmed by NVIDIA staff (March 2025): "None of them are directly provided by or supported by NVIDIA." Historical projects Alea GPU and Hybridizer are both defunct.

**OpenCL** lacks Tensor Core access entirely. **WebGPU** has buffer size limits (256MB–1GB, insufficient for model weights). **No Roslyn source generator compiles C# to PTX.** **.NET 9/10 add zero GPU compute** — `System.Numerics.Tensors` is CPU-only SIMD.

### Conclusion

No pure-C# approach matches native CUDA for LLM inference. However, the ManagedCuda pattern — P/Invoke directly against NVIDIA's system libraries + load PTX text files — achieves full native CUDA performance **without creating any C/C++ shared library**. We adopt this approach with our own minimal P/Invoke declarations (~30 functions), avoiding the ManagedCuda dependency and its GPLv3 license.

### Capability Comparison

| Capability | ILGPU (pure C#) | ManagedCuda + PTX | Own P/Invoke + PTX | Vulkan via Silk.NET |
|---|---|---|---|---|
| **cuBLAS GEMM** | ✅ Built-in wrapper | ✅ Full wrapper | ✅ Direct calls | ❌ No (use coopmat) |
| **Custom GPU kernels** | ✅ C# kernels | ✅ CUDA C → PTX | ✅ CUDA C → PTX | ✅ GLSL → SPIR-V |
| **Tensor Cores (custom)** | ❌ Only via cuBLAS | ✅ Full access | ✅ Full access | ✅ Via cooperative matrix |
| **FP16 / BF16** | FP16 only, no BF16 | ✅ Full | ✅ Full | ✅ FP16 (BF16 varies) |
| **Flash Attention** | ⚠️ Very difficult | ✅ Native quality | ✅ Native quality | ✅ Proven in research |
| **Memory management** | ✅ Full control | ✅ Full control | ✅ Full control | ✅ Full control |
| **Linux support** | ✅ | ✅ | ✅ | ✅ |
| **No C/C++ build system** | ✅ | ✅ (nvcc only) | ✅ (nvcc only) | ✅ (glslc only) |
| **License risk** | NCSA (permissive) | GPLv3/Commercial | None (own code) | MIT (Silk.NET) |
| **Multi-vendor GPU** | ❌ | ❌ NVIDIA only | ❌ NVIDIA only | ✅ Cross-vendor |
| **Perf vs native CUDA** | ~60–80% estimated | ~98–100% | ~98–100% | ~70–95% (improving) |

---

## Chosen Architecture: PTX Loading via CUDA Driver API

dotLLM uses NVIDIA's **CUDA Driver API** (`libcuda.so` / `nvcuda.dll`) and **cuBLAS** (`libcublas.so` / `cublas64_*.dll`) directly via P/Invoke. CUDA kernels are written in `.cu` files, compiled to PTX with a single `nvcc -ptx` command (no CMake, no shared library project), and loaded at runtime. The application is entirely C# — PTX files ship alongside the .NET assemblies as content files.

### How It Works

```
┌─────────────────┐     nvcc -ptx      ┌──────────────┐
│  rmsnorm.cu     │ ──────────────────► │ rmsnorm.ptx  │  (text file, ships with app)
│  rope.cu        │   (one command,     │ rope.ptx     │
│  attention.cu   │    no build system) │ attention.ptx│
│  dequant.cu     │                     │ dequant.ptx  │
└─────────────────┘                     └──────┬───────┘
                                               │ loaded at runtime
┌──────────────────────────────────────────────▼───────────────────┐
│  C# application                                                  │
│                                                                  │
│  [LibraryImport("cuda")]     ← NVIDIA's driver (on system)      │
│  cuModuleLoadData(ptxBytes)  ← loads PTX text into module        │
│  cuModuleGetFunction(module) ← gets kernel handle                │
│  cuLaunchKernel(func, ...)   ← launches on GPU                  │
│                                                                  │
│  [LibraryImport("cublas")]   ← NVIDIA's cuBLAS (on system)      │
│  cublasHgemm(...)            ← Tensor Core FP16 GEMM            │
└──────────────────────────────────────────────────────────────────┘
```

### What Libraries Are Involved

**`libcuda.so` / `nvcuda.dll`** — the CUDA Driver API. Installed with every NVIDIA GPU driver. Provides: device enumeration, context management, memory allocation, PTX module loading, kernel launching, stream management. This is what ManagedCuda wraps, and what we P/Invoke directly.

**`libcublas.so` / `cublas64_*.dll`** — cuBLAS. Installed with the CUDA Toolkit. Provides FP16 GEMM (`cublasHgemm`) with automatic Tensor Core usage when matrix dimensions are multiples of 8 — the single most important operation in LLM inference.

No dotLLM-authored `.so` or `.dll` is ever created.

### PTX: Text-Based GPU Intermediate Representation

PTX (Parallel Thread Execution) is NVIDIA's virtual instruction set — a text-based intermediate representation that the GPU driver JIT-compiles to native SASS instructions for the specific GPU at load time. It is architecturally analogous to SPIR-V for Vulkan or DXIL for DirectX — a shader file, not a compiled binary.

Key properties:
- **Architecture-independent**: the same PTX file runs on any GPU from sm_50 (Maxwell) through sm_90 (Hopper). The driver handles ISA translation.
- **Text-based**: human-readable, diffable, embeddable as string constants or .NET embedded resources.
- **JIT-compiled**: first load incurs ~100–500ms JIT compilation per module. The driver caches compiled kernels across runs (`~/.nv/ComputeCache`).
- **Fatbin option**: `nvcc` can produce fatbin files bundling PTX with pre-compiled SASS for specific architectures, eliminating JIT overhead entirely.

### Comparison with Original Plan (Shared Library Approach)

The original Step 31 plan creates `native/CMakeLists.txt` → builds `libdotllm_native.so`/`.dll` → wraps all CUDA operations behind a flat C API header (`dotllm_native.h`) → P/Invokes through `NativeMethods.cs`.

| Aspect | Shared Library (original) | PTX Loading (chosen) |
|---|---|---|
| **Build system** | CMake project, multi-target | `nvcc -ptx` (one command) |
| **Output artifact** | `libdotllm_native.so/.dll` | `*.ptx` text files |
| **C wrapper layer** | ~30 C functions in header | None — P/Invoke NVIDIA's API directly |
| **Memory/stream mgmt** | Custom C wrappers around cudaMalloc | Direct cuMemAlloc_v2 P/Invoke |
| **cuBLAS access** | Through custom C wrapper | Direct cublasHgemm P/Invoke |
| **Kernel code** | Same `.cu` kernels | Same `.cu` kernels (with `extern "C"`) |
| **Error handling** | Custom error codes | CUDA Driver API error codes (CUresult) |
| **Cross-compilation** | CMake cross-compile + RID packaging | PTX is arch-independent, no cross-compile |
| **CI complexity** | CUDA Toolkit + CMake + native build | CUDA Toolkit + single nvcc command |
| **Runtime dependency** | libdotllm_native + libcuda + libcublas | libcuda + libcublas (system-installed) |

The CUDA kernel code is **identical** in both approaches — the same RMSNorm, RoPE, attention, dequantization kernels. The difference is purely in how they're compiled (shared library vs PTX) and how C# calls them (through a custom C wrapper vs directly through NVIDIA's Driver API).

---

## P/Invoke Layer

### CUDA Driver API Declarations

~25 function declarations against `libcuda.so` / `nvcuda.dll`, covering initialization, device queries, context management, PTX module loading, kernel launching, memory operations, streams, and error handling. Following existing `CpuAffinity.cs` conventions with `[LibraryImport]` source generator.

```csharp
// src/DotLLM.Cuda/Interop/CudaDriverApi.cs
using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

/// <summary>
/// Minimal P/Invoke declarations against NVIDIA's CUDA Driver API.
/// libcuda.so (Linux) / nvcuda.dll (Windows) — installed with GPU driver.
/// All functions return CUresult (int): 0 = CUDA_SUCCESS, non-zero = error.
/// </summary>
internal static partial class CudaDriverApi
{
    // .NET resolves "cuda" to libcuda.so (Linux) / nvcuda.dll (Windows)
    private const string LibName = "cuda";

    // ── Initialization ──────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuInit(uint flags);

    // ── Device ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGet(out int device, int ordinal);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetCount(out int count);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetName(
        [MarshalAs(UnmanagedType.LPArray)] byte[] name, int len, int device);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceTotalMem_v2(out nuint bytes, int device);

    [LibraryImport(LibName)]
    internal static partial int cuDeviceGetAttribute(
        out int value, int attribute, int device);

    // ── Context ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuCtxCreate_v2(out nint ctx, uint flags, int device);

    [LibraryImport(LibName)]
    internal static partial int cuCtxDestroy_v2(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxSetCurrent(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int cuCtxGetCurrent(out nint ctx);

    // ── Module (PTX loading) ────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuModuleLoadData(out nint module, nint ptxImage);

    [LibraryImport(LibName)]
    internal static partial int cuModuleLoadDataEx(
        out nint module, nint ptxImage, uint numOptions,
        nint options, nint optionValues);

    [LibraryImport(LibName)]
    internal static partial int cuModuleGetFunction(
        out nint function, nint module,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    [LibraryImport(LibName)]
    internal static partial int cuModuleUnload(nint module);

    // ── Kernel launch ───────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuLaunchKernel(
        nint function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, nint stream,
        nint kernelParams, nint extra);

    // ── Memory ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuMemAlloc_v2(out nint devicePtr, nuint bytesize);

    [LibraryImport(LibName)]
    [SuppressGCTransition] // trivially short — just cudaFree
    internal static partial int cuMemFree_v2(nint devicePtr);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyHtoD_v2(
        nint dstDevice, nint srcHost, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoH_v2(
        nint dstHost, nint srcDevice, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoD_v2(
        nint dstDevice, nint srcDevice, nuint byteCount);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyHtoDAsync_v2(
        nint dstDevice, nint srcHost, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemcpyDtoHAsync_v2(
        nint dstHost, nint srcDevice, nuint byteCount, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuMemsetD8_v2(nint dstDevice, byte value, nuint n);

    // ── Streams ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuStreamCreate(out nint stream, uint flags);

    [LibraryImport(LibName)]
    internal static partial int cuStreamDestroy_v2(nint stream);

    [LibraryImport(LibName)]
    internal static partial int cuStreamSynchronize(nint stream);

    // ── Error ───────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int cuGetErrorName(int error, out nint str);

    [LibraryImport(LibName)]
    internal static partial int cuGetErrorString(int error, out nint str);
}
```

### cuBLAS Declarations

~6 function declarations for GEMM operations:

```csharp
// src/DotLLM.Cuda/Interop/CublasApi.cs
using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

/// <summary>
/// Minimal cuBLAS P/Invoke. libcublas.so / cublas64_*.dll — from CUDA Toolkit.
/// </summary>
internal static partial class CublasApi
{
    private const string LibName = "cublas";

    [LibraryImport(LibName)]
    internal static partial int cublasCreate_v2(out nint handle);

    [LibraryImport(LibName)]
    internal static partial int cublasDestroy_v2(nint handle);

    [LibraryImport(LibName)]
    internal static partial int cublasSetStream_v2(nint handle, nint stream);

    [LibraryImport(LibName)]
    internal static partial int cublasSetMathMode(nint handle, int mode);
    // CUBLAS_TENSOR_OP_MATH = 1 — enable Tensor Core usage

    // FP16 GEMM — C = alpha * op(A) * op(B) + beta * C, all FP16
    // Tensor Cores used automatically when dims are multiples of 8
    // Row-major trick: compute C^T = B^T @ A^T via swapped args
    [LibraryImport(LibName)]
    internal static partial int cublasHgemm(
        nint handle,
        int transa, int transb,     // cublasOperation_t: 0=N, 1=T, 2=C
        int m, int n, int k,
        in ushort alpha,            // __half passed as ushort
        nint A, int lda,
        nint B, int ldb,
        in ushort beta,
        nint C, int ldc);

    // GemmEx — mixed precision (FP16 input, FP32 accumulate)
    [LibraryImport(LibName)]
    internal static partial int cublasGemmEx(
        nint handle,
        int transa, int transb,
        int m, int n, int k,
        nint alpha,
        nint A, int Atype, int lda,
        nint B, int Btype, int ldb,
        nint beta,
        nint C, int Ctype, int ldc,
        int computeType, int algo);
    // cudaDataType: CUDA_R_16F=2, CUDA_R_32F=0
    // cublasComputeType: CUBLAS_COMPUTE_16F=64, CUBLAS_COMPUTE_32F=68
}
```

### Error Handling

```csharp
// src/DotLLM.Cuda/Interop/CudaException.cs
namespace DotLLM.Cuda.Interop;

public sealed class CudaException : Exception
{
    public int ErrorCode { get; }

    public CudaException(int errorCode, string message)
        : base($"CUDA error {errorCode}: {message}")
    {
        ErrorCode = errorCode;
    }
}

// src/DotLLM.Cuda/Interop/CudaErrorHelper.cs
using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

internal static class CudaErrorHelper
{
    internal static void ThrowOnError(this int result)
    {
        if (result == 0) return; // CUDA_SUCCESS

        string message = "Unknown CUDA error";
        CudaDriverApi.cuGetErrorString(result, out nint strPtr);
        if (strPtr != 0)
            message = Marshal.PtrToStringAnsi(strPtr) ?? message;

        throw new CudaException(result, message);
    }
}
```

---

## PTX Kernel Conventions

All CUDA kernels use `extern "C"` linkage to prevent C++ name mangling, enabling `cuModuleGetFunction` lookup by simple string name:

```cuda
// native/kernels/rmsnorm.cu
#include <cuda_fp16.h>

extern "C" __global__ void rmsnorm_f16(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    const int n,
    const float eps)
{
    // Standard warp-reduction RMS normalization
    // FP16 in/out, FP32 accumulation for numerical stability
    // One block per row, warp shuffle for reduction
}
```

Compiled to PTX:

```bash
nvcc -ptx -arch=compute_80 -o rmsnorm.ptx rmsnorm.cu
```

Or for multi-architecture support (Ampere + Ada + Hopper):

```bash
nvcc -ptx \
    --generate-code arch=compute_80,code=compute_80 \
    --generate-code arch=compute_89,code=compute_89 \
    --generate-code arch=compute_90,code=compute_90 \
    -o rmsnorm.ptx rmsnorm.cu
```

### Kernel Launch from C#

The CUDA Driver API passes kernel arguments as an array of pointers to the actual values:

```csharp
public void LaunchRmsNorm(
    nint input, nint weight, nint output,
    int hiddenSize, float eps,
    uint rows, nint stream)
{
    unsafe
    {
        nint inputArg = input;
        nint weightArg = weight;
        nint outputArg = output;
        int nArg = hiddenSize;
        float epsArg = eps;

        void*[] args = [&inputArg, &weightArg, &outputArg, &nArg, &epsArg];

        fixed (void** argsPtr = args)
        {
            CudaDriverApi.cuLaunchKernel(
                _rmsnormFunc,
                gridDimX: rows, gridDimY: 1, gridDimZ: 1,
                blockDimX: 256, blockDimY: 1, blockDimZ: 1,
                sharedMemBytes: 0,
                stream: stream,
                kernelParams: (nint)argsPtr,
                extra: 0).ThrowOnError();
        }
    }
}
```

### Module Loading

PTX files are loaded once at startup and cached for the lifetime of the process. The CUDA driver caches JIT-compiled SASS in `~/.nv/ComputeCache` across process restarts.

```csharp
// src/DotLLM.Cuda/CudaModule.cs
public sealed class CudaModule : IDisposable
{
    private nint _module;
    private readonly Dictionary<string, nint> _functions = new();

    public static CudaModule LoadFromPtx(string ptxPath)
    {
        byte[] ptxBytes = File.ReadAllBytes(ptxPath);
        var module = new CudaModule();

        unsafe
        {
            fixed (byte* ptxPtr = ptxBytes)
            {
                // Null-terminated — PTX is text, File.ReadAllBytes is fine
                // if the file ends without null, append one
                CudaDriverApi.cuModuleLoadData(out module._module, (nint)ptxPtr)
                    .ThrowOnError();
            }
        }
        return module;
    }

    public nint GetFunction(string name)
    {
        if (!_functions.TryGetValue(name, out nint func))
        {
            CudaDriverApi.cuModuleGetFunction(out func, _module, name)
                .ThrowOnError();
            _functions[name] = func;
        }
        return func;
    }

    public void Dispose()
    {
        if (_module != 0)
        {
            CudaDriverApi.cuModuleUnload(_module);
            _module = 0;
        }
    }
}
```

---

## Build System

### Kernel Compilation

A simple shell script replaces the entire CMake build system:

```bash
#!/bin/bash
# native/build.sh — Compile all .cu kernels to PTX

CUDA_ARCHS="compute_80 compute_89 compute_90"
OUT_DIR="$(dirname "$0")/ptx"
mkdir -p "$OUT_DIR"

for cu_file in "$(dirname "$0")"/kernels/*.cu; do
    base=$(basename "$cu_file" .cu)

    GENCODE_FLAGS=""
    for arch in $CUDA_ARCHS; do
        GENCODE_FLAGS="$GENCODE_FLAGS --generate-code arch=$arch,code=$arch"
    done

    nvcc -ptx $GENCODE_FLAGS \
         --use_fast_math \
         -o "$OUT_DIR/$base.ptx" \
         "$cu_file"

    echo "  $base.cu → $base.ptx"
done
```

### .NET Integration

PTX files are included as content files in the project, copied to output directory:

```xml
<!-- src/DotLLM.Cuda/DotLLM.Cuda.csproj -->
<ItemGroup>
    <Content Include="..\..\native\ptx\*.ptx" Link="ptx\%(Filename)%(Extension)">
        <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </Content>
</ItemGroup>
```

Or embedded as resources for single-file deployment:

```xml
<ItemGroup>
    <EmbeddedResource Include="..\..\native\ptx\*.ptx" Link="ptx\%(Filename)%(Extension)" />
</ItemGroup>
```

### NativeLibrary Resolution

.NET's `NativeLibrary` resolution handles platform differences automatically. For cases where library names differ across platforms, use a resolver:

```csharp
// src/DotLLM.Cuda/Interop/CudaLibraryResolver.cs
using System.Reflection;
using System.Runtime.InteropServices;

namespace DotLLM.Cuda.Interop;

internal static class CudaLibraryResolver
{
    internal static void Register()
    {
        NativeLibrary.SetDllImportResolver(
            typeof(CudaLibraryResolver).Assembly,
            ResolveCudaLibrary);
    }

    private static nint ResolveCudaLibrary(
        string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "cuda")
        {
            // Linux: libcuda.so.1 (symlink from driver install)
            // Windows: nvcuda.dll (in System32)
            string osLib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "nvcuda.dll"
                : "libcuda.so.1";

            if (NativeLibrary.TryLoad(osLib, out nint handle))
                return handle;
        }

        if (libraryName == "cublas")
        {
            // Linux: libcublas.so.XX (versioned, from CUDA Toolkit)
            // Windows: cublas64_XX.dll
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // Try common versions: CUDA 12.x, 13.x
                foreach (var ver in new[] { "12", "11" })
                {
                    if (NativeLibrary.TryLoad($"cublas64_{ver}.dll", out nint h))
                        return h;
                }
            }
            else
            {
                if (NativeLibrary.TryLoad("libcublas.so", out nint h))
                    return h;
            }
        }

        return 0; // fall through to default resolution
    }
}
```

---

## PTX JIT Compilation Overhead

First-time PTX loading incurs ~100–500ms per module as the driver compiles PTX → SASS for the specific GPU. Mitigations:

1. **Driver kernel cache** (`~/.nv/ComputeCache`): enabled by default, persists compiled SASS across process restarts. Second launch is near-instant.
2. **Fatbin compilation**: pre-compile to SASS for target architectures alongside PTX fallback. Zero JIT overhead on matching GPUs:
   ```bash
   nvcc -fatbin \
       --generate-code arch=compute_80,code=sm_80 \
       --generate-code arch=compute_89,code=sm_89 \
       --generate-code arch=compute_90,code=sm_90 \
       --generate-code arch=compute_80,code=compute_80 \
       -o rmsnorm.fatbin rmsnorm.cu
   ```
   Load with `cuModuleLoadData` — same API, auto-selects SASS if available, falls back to PTX.
3. **Startup amortization**: all modules loaded during model initialization (which already takes 1–10s for GGUF mmap + weight upload). PTX JIT adds negligible overhead relative to weight transfer.

---

## Kernel Catalog

All kernels compiled to PTX, loaded via `cuModuleLoadData`, launched via `cuLaunchKernel`:

| Kernel | File | Function Name | Block Size | Grid Size | Shared Mem |
|---|---|---|---|---|---|
| RMS Norm | `rmsnorm.cu` | `rmsnorm_f16` | 256 | rows | Warp reduction |
| RoPE | `rope.cu` | `rope_f16` | 256 | seqLen × numHeads | None |
| SwiGLU | `swiglu.cu` | `swiglu_f16` | 256 | ceil(n/256) | None |
| Add | `add.cu` | `add_f16` | 256 | ceil(n/256) | None |
| Softmax | `softmax.cu` | `softmax_f16` | 256 | rows | Warp reduction |
| Embedding | `embedding.cu` | `embedding_lookup` | 256 | seqLen | None |
| Attention | `attention.cu` | `attention_f16` | 256 | numHeads × batchSize | Per-head scores |
| Bias Add | `bias_add.cu` | `bias_add_f16` | 256 | ceil(n/256) | None |
| Dequant Q8_0 | `dequant.cu` | `dequant_q8_0_f16` | 256 | ceil(blocks/256) | None |
| Dequant Q4_0 | `dequant.cu` | `dequant_q4_0_f16` | 256 | ceil(blocks/256) | None |
| Dequant Q4_K | `dequant.cu` | `dequant_q4_k_f16` | 256 | ceil(superblocks/X) | None |
| Dequant Q5_K | `dequant.cu` | `dequant_q5_k_f16` | 256 | ceil(superblocks/X) | None |
| Dequant Q6_K | `dequant.cu` | `dequant_q6_k_f16` | 256 | ceil(superblocks/X) | None |
| FP16→FP32 | `convert.cu` | `convert_f16_to_f32` | 256 | ceil(n/256) | None |
| FP32→FP16 | `convert.cu` | `convert_f32_to_f16` | 256 | ceil(n/256) | None |

GEMM/GEMV operations use cuBLAS (`cublasHgemm` / `cublasGemmEx`) directly — no custom PTX kernel needed.

---

## cuBLAS Row-Major Convention

cuBLAS uses column-major layout (Fortran convention). Since dotLLM tensors are row-major, we use the standard transposition trick:

To compute `C = A × B` (all row-major), we call `cublasHgemm` with swapped operands:
- Pass `B` as first matrix, `A` as second matrix
- Set `transa = CUBLAS_OP_N`, `transb = CUBLAS_OP_N`
- cuBLAS computes `C_colmajor = B_colmajor × A_colmajor`
- Because `X_colmajor` is equivalent to `X^T_rowmajor`, this yields the correct row-major result

This is well-proven — llama.cpp, vLLM, and every CUDA inference engine uses the same trick.

---

## Prerequisites

### Runtime Requirements

- **NVIDIA GPU**: Compute capability 7.0+ (Volta and newer). Recommended: 8.0+ (Ampere) for best Tensor Core performance.
- **NVIDIA GPU driver**: 525.60+ (for CUDA 12.x compatibility).
- **CUDA Runtime**: not required — the Driver API (`libcuda.so`) is sufficient and ships with the driver.
- **cuBLAS**: required for GEMM. Installed with CUDA Toolkit or available as a standalone redistributable.

### Build Requirements (kernel compilation only)

- **CUDA Toolkit 12.x+**: provides `nvcc` compiler. Only needed to recompile `.cu` → `.ptx` files. Pre-compiled PTX files can be distributed without requiring the toolkit on end-user machines.
- **No CMake, no C/C++ compiler, no build system** beyond `nvcc`.

---

## Future Work

- **Flash Attention**: replace naive attention kernel with tiled flash attention (shared memory, online softmax). Full Tensor Core access via `wmma` intrinsics in PTX.
- **Quantized GEMV**: custom PTX kernels for Q4_K × FP16 decode, avoiding the dequant → FP16 → cuBLAS path. Critical for single-token decode throughput.
- **Multi-stream pipelining** (Step 32): overlap H2D transfer with compute across layers.
- **NCCL integration** (Step 51): multi-GPU tensor parallelism. NCCL is another system library — same P/Invoke pattern, no shared library needed.
- **Fatbin distribution**: ship pre-compiled SASS for common architectures to eliminate JIT overhead.
- **NVRTC runtime compilation**: compile `.cu` source to PTX at application startup using NVIDIA's Runtime Compilation library, eliminating the nvcc build step entirely. NVRTC is available as `libnvrtc.so` / `nvrtc64_*.dll`.
