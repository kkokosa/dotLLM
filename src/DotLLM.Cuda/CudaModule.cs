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
    /// Resolves the best-matching PTX variant for a kernel given the device compute
    /// capability, then loads it.
    /// </summary>
    /// <remarks>
    /// Looks for arch-tiered variants named <c>&lt;kernel&gt;.sm_&lt;arch&gt;.ptx</c> (e.g.
    /// <c>rmsnorm.sm_80.ptx</c>) alongside the universal <c>&lt;kernel&gt;.ptx</c> built for
    /// <c>compute_61</c>. The highest-arch variant whose architecture is &lt;= the device's
    /// compute capability is selected; if none is present, the plain <c>compute_61</c>
    /// <c>&lt;kernel&gt;.ptx</c> is used as the universal fallback. When no higher-arch
    /// variants are shipped this is a no-op and resolves to the same file as
    /// <see cref="LoadFromFile"/>.
    /// </remarks>
    /// <param name="ptxDir">Directory containing compiled .ptx files.</param>
    /// <param name="baseFileName">Base kernel file name including extension (e.g. <c>rmsnorm.ptx</c>).</param>
    /// <param name="ccMajor">Device compute capability major version.</param>
    /// <param name="ccMinor">Device compute capability minor version.</param>
    public static CudaModule LoadForArch(string ptxDir, string baseFileName, int ccMajor, int ccMinor)
        => LoadFromFile(ResolveArchVariantPath(ptxDir, baseFileName, ccMajor, ccMinor));

    /// <summary>
    /// Selects the best-matching arch-tiered PTX variant path for a kernel without loading it.
    /// Pure file-system logic — exposed for unit testing of variant selection.
    /// </summary>
    /// <remarks>
    /// Given <paramref name="baseFileName"/> = <c>foo.ptx</c>, candidate variants are
    /// <c>foo.sm_&lt;arch&gt;.ptx</c> where <c>&lt;arch&gt;</c> is a two- or three-digit SM number
    /// (e.g. <c>75</c>, <c>80</c>, <c>86</c>, <c>90</c>). The variant with the highest arch value
    /// that does not exceed the device compute capability (<c>ccMajor * 10 + ccMinor</c>) and that
    /// actually exists on disk is returned. If no such variant exists, the plain
    /// <paramref name="baseFileName"/> path (the <c>compute_61</c> universal build) is returned
    /// unchanged.
    /// </remarks>
    /// <param name="ptxDir">Directory containing compiled .ptx files.</param>
    /// <param name="baseFileName">Base kernel file name including extension (e.g. <c>rmsnorm.ptx</c>).</param>
    /// <param name="ccMajor">Device compute capability major version.</param>
    /// <param name="ccMinor">Device compute capability minor version.</param>
    /// <returns>The full path of the variant to load.</returns>
    public static string ResolveArchVariantPath(string ptxDir, string baseFileName, int ccMajor, int ccMinor)
    {
        string basePath = Path.Combine(ptxDir, baseFileName);

        int deviceArch = (ccMajor * 10) + ccMinor;
        if (deviceArch <= 0)
            return basePath;

        // "foo.ptx" -> stem "foo", suffix ".ptx"
        string suffix = Path.GetExtension(baseFileName);              // ".ptx"
        string stem = baseFileName[..^suffix.Length];                 // "foo"

        // Enumerate sibling variants "foo.sm_<arch>.ptx" and pick the highest arch <= deviceArch.
        string prefix = stem + ".sm_";
        int bestArch = -1;
        string bestPath = basePath;

        IEnumerable<string> candidates;
        try
        {
            candidates = Directory.EnumerateFiles(ptxDir, prefix + "*" + suffix);
        }
        catch (DirectoryNotFoundException)
        {
            return basePath;
        }

        foreach (string candidate in candidates)
        {
            string fileName = Path.GetFileName(candidate);
            // Strip "<stem>.sm_" prefix and ".ptx" suffix to isolate the arch token.
            string archToken = fileName[prefix.Length..^suffix.Length];
            if (!int.TryParse(archToken, out int arch))
                continue; // ignore malformed tokens (e.g. "foo.sm_80a.ptx")

            if (arch <= deviceArch && arch > bestArch)
            {
                bestArch = arch;
                bestPath = candidate;
            }
        }

        return bestPath;
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
