using System.Runtime.CompilerServices;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Opt-in diagnostic tracing for Nemotron-H forward pass. All checks are gated on
/// environment variables read once at process startup, so when diagnostics are
/// disabled the overhead is a single predictable branch per layer and no string
/// formatting ever runs. Intended for cross-reference debugging against llama.cpp
/// (see llama.cpp's <c>llama-eval-callback</c> tool for the matching reference side).
/// </summary>
/// <remarks>
/// <para>Environment variables:</para>
/// <list type="bullet">
///   <item><c>DOTLLM_TRACE_LAYERS=1</c> — print a one-liner per layer with
///     last-token hidden state stats (rms, absMax, range, NaN count).</item>
///   <item><c>DOTLLM_TRACE_SSM=1</c> — print intermediate SSM sub-layer stats
///     (ssm_in output, post-conv+SiLU, post-scan, post-D, post-SwiGLU, post-group-RMSNorm, post-ssm_out).</item>
///   <item><c>DOTLLM_TRACE_SSM_LAYER=N</c> — when set alongside TRACE_SSM, print
///     SSM internals only for absolute layer index N (default: all SSM layers).</item>
/// </list>
/// <para>Trace output goes to <c>Console.Error</c>. Format mirrors a compact summary
/// rather than full element dumps so large tensors don't flood the log.</para>
/// </remarks>
internal static class NemotronHDiagnostics
{
    public static readonly bool TraceLayers
        = Environment.GetEnvironmentVariable("DOTLLM_TRACE_LAYERS") == "1";

    public static readonly bool TraceSsm
        = Environment.GetEnvironmentVariable("DOTLLM_TRACE_SSM") == "1";

    private static readonly int TraceSsmLayerOnly
        = int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_TRACE_SSM_LAYER"), out var v) ? v : -1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsSsmTraced(int layer)
        => TraceSsm && (TraceSsmLayerOnly < 0 || TraceSsmLayerOnly == layer);

    /// <summary>
    /// Writes a one-line stats summary (rms, absMax, range, nan count) for the
    /// given float buffer to stderr, prefixed with <paramref name="label"/>.
    /// </summary>
    public static unsafe void DumpStats(string label, float* data, int length)
    {
        if (length == 0)
        {
            Console.Error.WriteLine($"[trace] {label} EMPTY");
            return;
        }

        double sumSq = 0;
        float absMax = 0f;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        int nan = 0;
        for (int i = 0; i < length; i++)
        {
            float v = data[i];
            if (float.IsNaN(v) || float.IsInfinity(v)) { nan++; continue; }
            sumSq += (double)v * v;
            if (v > max) max = v;
            if (v < min) min = v;
            float a = MathF.Abs(v);
            if (a > absMax) absMax = a;
        }
        double rms = Math.Sqrt(sumSq / length);
        Console.Error.WriteLine(
            $"[trace] {label,-40} n={length,7} rms={rms:F4} absMax={absMax:F4} range=[{min:F3},{max:F3}] nan={nan}");
    }
}
