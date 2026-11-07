using System.Runtime.InteropServices;

namespace DotLLM.Ascend.Interop;

internal static partial class AscendNative
{
    private static readonly object ResolverGate = new();
    private static bool _registered;

    internal static void Register()
    {
        lock (ResolverGate)
        {
            if (_registered) return;
            NativeLibrary.SetDllImportResolver(typeof(AscendNative).Assembly, (name, assembly, path) =>
            {
                if (name != "dotllm_ascend") return 0;
                string? explicitPath = Environment.GetEnvironmentVariable("DOTLLM_ASCEND_LIBRARY");
                string localPath = Path.Combine(AppContext.BaseDirectory, "libdotllm_ascend.so");
                nint handle = !string.IsNullOrWhiteSpace(explicitPath)
                    ? NativeLibrary.Load(Path.GetFullPath(explicitPath))
                    : File.Exists(localPath) ? NativeLibrary.Load(localPath)
                    : NativeLibrary.Load("libdotllm_ascend.so", assembly, path);
                try
                {
                    foreach (string symbol in new[] { "abi_version", "last_error", "linear_f16",
                        "rms_norm_f16", "add_f16", "swiglu_f16", "rope_f16", "attention_f16", "cast", "linear_f32", "rms_norm_f32", "add_f32",
                        "swiglu_f32", "rope_f32", "attention_f32" })
                        if (!NativeLibrary.TryGetExport(handle, "dotllm_ascend_" + symbol, out _))
                            throw new EntryPointNotFoundException($"Ascend bridge is missing dotllm_ascend_{symbol}; rebuild ABI 3.");
                    return handle;
                }
                catch { NativeLibrary.Free(handle); throw; }
            });
            _registered = true;
        }
    }

    [LibraryImport("libascendcl.so")]
    internal static partial int aclInit(nint configPath);
    [LibraryImport("libascendcl.so")]
    internal static partial int aclFinalize();
    [LibraryImport("libascendcl.so")]
    internal static partial int aclrtGetDeviceCount(out uint count);
    [LibraryImport("libascendcl.so")]
    internal static partial int aclrtCreateContext(out nint context, int deviceId);
    [LibraryImport("libascendcl.so")]
    internal static partial int aclrtSetCurrentContext(nint context);
    [LibraryImport("libascendcl.so")]
    internal static partial int aclrtDestroyContext(nint context);
    [LibraryImport("libascendcl.so")]
    internal static partial int aclrtCreateStream(out nint stream);
    [LibraryImport("libascendcl.so")]
    internal static partial int aclrtSynchronizeStream(nint stream);
    [LibraryImport("libascendcl.so")]
    internal static partial int aclrtDestroyStream(nint stream);
    [LibraryImport("libascendcl.so")]
    internal static partial int aclrtMalloc(out nint devicePtr, nuint size, int policy);
    [LibraryImport("libascendcl.so")]
    internal static partial int aclrtFree(nint devicePtr);
    [LibraryImport("libascendcl.so")]
    internal static partial int aclrtMemcpy(nint destination, nuint capacity, nint source, nuint size, int kind);
    [LibraryImport("libascendcl.so")]
    internal static partial nint aclGetRecentErrMsg();

    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_abi_version")]
    internal static partial uint AbiVersion();
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_last_error")]
    internal static partial nint LastError();
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_linear_f16")]
    internal static partial int LinearF16(nint x, nint weight, nint output,
        long rows, long outputFeatures, long inputFeatures, nint stream);

    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_rms_norm_f16")]
    internal static partial int RmsNormF16(nint x, nint weight, nint output,
        long rows, long width, double epsilon, nint stream);
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_add_f16")]
    internal static partial int AddF16(nint left, nint right, nint output,
        long rows, long width, long rightRows, nint stream);
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_swiglu_f16")]
    internal static partial int SwiGluF16(nint gate, nint up, nint output, long count, nint stream);
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_rope_f16")]
    internal static partial int RopeF16(nint x, nint cos, nint sin, nint output,
        long rows, long heads, long headDim, long rotaryDim, int neox, nint stream);
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_attention_f16")]
    internal static partial int AttentionF16(nint query, nint keys, nint values, nint output,
        long rows, long kvRows, long heads, long kvHeads, long headDim, long queryOffset, nint stream);
    [LibraryImport("libascendcl.so")]
    internal static partial int aclrtGetMemInfo(int attribute, out nuint free, out nuint total);


    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_cast")]
    internal static partial int Cast(nint x, nint output, long count, int toFloat32, nint stream);
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_linear_f32")]
    internal static partial int LinearF32(nint x, nint weight, nint output, long rows, long n, long k, nint stream);
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_rms_norm_f32")]
    internal static partial int RmsNormF32(nint x, nint weight, nint output, long rows, long width, double epsilon, nint stream);
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_add_f32")]
    internal static partial int AddF32(nint left, nint right, nint output, long rows, long width, long rightRows, int rightF16, nint stream);
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_swiglu_f32")]
    internal static partial int SwiGluF32(nint gate, nint up, nint output, long count, nint stream);
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_rope_f32")]
    internal static partial int RopeF32(nint x, nint cos, nint sin, nint output, long rows, long heads, long dim, long rotary, int neox, nint stream);
    [LibraryImport("dotllm_ascend", EntryPoint = "dotllm_ascend_attention_f32")]
    internal static partial int AttentionF32(nint query, nint keys, nint values, nint output, long rows, long kvRows, long heads, long kvHeads, long dim, long offset, int kvF16, nint stream);

    internal static void Check(int status, string operation)
    {
        if (status == 0) return;
        string? detail = Marshal.PtrToStringUTF8(aclGetRecentErrMsg());
        throw new AscendException(operation, status, detail);
    }
}
