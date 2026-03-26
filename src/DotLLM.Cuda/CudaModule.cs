using DotLLM.Cuda.Interop;

namespace DotLLM.Cuda;

/// <summary>
/// Loads a PTX text file into a CUDA module and caches kernel function handles.
/// The CUDA driver JIT-compiles PTX → SASS for the current GPU on first load
/// (cached in <c>~/.nv/ComputeCache</c> across process restarts).
/// </summary>
public sealed class CudaModule : IDisposable
{
    private nint _module;
    private readonly Dictionary<string, nint> _functions = new();

    /// <summary>
    /// Loads a PTX module from a file path.
    /// </summary>
    /// <param name="ptxPath">Path to the .ptx file.</param>
    public static CudaModule LoadFromFile(string ptxPath)
    {
        byte[] ptxBytes = File.ReadAllBytes(ptxPath);
        return LoadFromBytes(ptxBytes);
    }

    /// <summary>
    /// Loads a PTX module from a byte array (UTF-8 text with null terminator).
    /// </summary>
    /// <param name="ptxBytes">PTX source bytes. A null terminator is appended if missing.</param>
    public static CudaModule LoadFromBytes(byte[] ptxBytes)
    {
        // Ensure null termination (PTX is text)
        byte[] terminated = ptxBytes;
        if (ptxBytes.Length == 0 || ptxBytes[^1] != 0)
        {
            terminated = new byte[ptxBytes.Length + 1];
            ptxBytes.CopyTo(terminated, 0);
            terminated[^1] = 0;
        }

        var module = new CudaModule();
        unsafe
        {
            fixed (byte* ptxPtr = terminated)
            {
                CudaDriverApi.cuModuleLoadData(out module._module, (nint)ptxPtr)
                    .ThrowOnError();
            }
        }
        return module;
    }

    /// <summary>
    /// Gets a kernel function handle by name. Caches the result for subsequent calls.
    /// </summary>
    /// <param name="name">The <c>extern "C"</c> kernel function name.</param>
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

    /// <inheritdoc/>
    public void Dispose()
    {
        nint module = Interlocked.Exchange(ref _module, 0);
        if (module != 0)
        {
            CudaDriverApi.cuModuleUnload(module);
            _functions.Clear();
        }
    }
}
