using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Depthwise causal 1-D convolution used by Mamba2 SSM layers (NVIDIA Nemotron-H).
/// Each channel is convolved independently (depthwise) along the time axis with a
/// small kernel (kernel size <c>d_conv</c>, typically 4) followed by a per-channel bias.
/// </summary>
/// <remarks>
/// <para>
/// Memory layout matches llama.cpp / GGUF exactly: time-major row vectors, one row per
/// time step. The caller prepends the cached <c>conv_state</c> (<c>d_conv-1</c> rows)
/// to the new activations <c>xBC</c> (<c>T</c> rows) before calling this kernel, so the
/// combined input has shape <c>[d_conv-1+T, channels]</c> row-major. The kernel writes
/// <c>T</c> output rows, one per current time step.
/// </para>
/// <para>
/// Output formula (per time step <c>t</c> in [0, T) and channel <c>c</c> in [0, channels)):
/// <code>
/// y[t, c] = bias[c] + Σ_{k=0..d_conv-1} input[t+k, c] * weight[k, c]
/// </code>
/// </para>
/// <para>
/// This is a scalar reference implementation — correctness first, SIMD later. The
/// performance of the conv step is dwarfed by the <c>ssm_in</c> GEMM at 3136×17504 Q5_0
/// so vectorisation can wait until that is profiled.
/// </para>
/// </remarks>
public static class Conv1dCausal
{
    /// <summary>
    /// Runs the depthwise causal 1-D convolution.
    /// </summary>
    /// <param name="input">
    /// Concatenated <c>[conv_state | xBC]</c> buffer with shape <c>[d_conv-1+T, channels]</c>,
    /// row-major. Must contain at least <c>(d_conv-1+T) * channels</c> elements.
    /// </param>
    /// <param name="weight">Per-channel conv kernel, shape <c>[d_conv, channels]</c> row-major.</param>
    /// <param name="bias">Per-channel bias, length <c>channels</c>.</param>
    /// <param name="output">Destination buffer, shape <c>[T, channels]</c> row-major.</param>
    /// <param name="dConv">Convolution kernel width (typically 4).</param>
    /// <param name="channels">Number of channels (depthwise width).</param>
    /// <param name="seqLen">Number of output time steps (<c>T</c>).</param>
    [SkipLocalsInit]
    public static void Execute(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int dConv,
        int channels,
        int seqLen)
    {
        if (dConv <= 0) throw new ArgumentOutOfRangeException(nameof(dConv));
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (seqLen < 0) throw new ArgumentOutOfRangeException(nameof(seqLen));

        int inputRows = dConv - 1 + seqLen;
        if (input.Length < (long)inputRows * channels)
            throw new ArgumentException(
                $"input length {input.Length} < (d_conv-1+T)*channels = {(long)inputRows * channels}.",
                nameof(input));
        if (weight.Length < (long)dConv * channels)
            throw new ArgumentException(
                $"weight length {weight.Length} < d_conv*channels = {(long)dConv * channels}.",
                nameof(weight));
        if (bias.Length < channels)
            throw new ArgumentException($"bias length {bias.Length} < channels = {channels}.", nameof(bias));
        if (output.Length < (long)seqLen * channels)
            throw new ArgumentException(
                $"output length {output.Length} < T*channels = {(long)seqLen * channels}.",
                nameof(output));

        ExecuteScalar(input, weight, bias, output, dConv, channels, seqLen);
    }

    /// <summary>
    /// Scalar reference implementation. Kept <c>internal</c> and exposed only so unit tests
    /// can pin a known-good implementation independent of future SIMD variants.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias,
        Span<float> output,
        int dConv,
        int channels,
        int seqLen)
    {
        // Per-time-step outer loop, per-channel inner loop.
        // Inputs and outputs are time-major, so channel loops walk contiguous memory,
        // which matches the depthwise weight layout [k, channels].
        for (int t = 0; t < seqLen; t++)
        {
            Span<float> outRow = output.Slice(t * channels, channels);

            // Initialise with bias.
            bias[..channels].CopyTo(outRow);

            // Accumulate kernel contributions: output[t, c] += sum_k input[t+k, c] * weight[k, c].
            for (int k = 0; k < dConv; k++)
            {
                int inRowStart = (t + k) * channels;
                int wRowStart = k * channels;
                for (int c = 0; c < channels; c++)
                {
                    outRow[c] += input[inRowStart + c] * weight[wRowStart + c];
                }
            }
        }
    }
}
