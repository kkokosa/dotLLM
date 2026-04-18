using System.Runtime.CompilerServices;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Depthwise causal 1-D convolution used by Mamba2 SSM layers (NVIDIA Nemotron-H).
/// Each channel is convolved independently (depthwise) along the time axis with a
/// small kernel (kernel size <c>d_conv</c>, typically 4) followed by a per-channel bias.
/// </summary>
/// <remarks>
/// <para>
/// Memory layout matches llama.cpp / GGUF exactly:
/// </para>
/// <list type="bullet">
///   <item>
///     <description>
///       <b>Input / output</b> are time-major, one row per time step:
///       element <c>(t, c)</c> at flat index <c>t * channels + c</c>.
///       The caller prepends the cached <c>conv_state</c> (<c>d_conv-1</c> rows) to
///       the new activations <c>xBC</c> (<c>T</c> rows), so the combined input has
///       shape <c>[d_conv-1 + T, channels]</c>.
///     </description>
///   </item>
///   <item>
///     <description>
///       <b>Weight</b> follows GGUF's channel-major layout for <c>ssm_conv1d.weight</c>:
///       GGUF shape <c>[d_conv, channels]</c> with <c>ne[0] = d_conv</c> places a single
///       channel's <c>d_conv</c> taps at contiguous addresses — element <c>(k, c)</c>
///       at flat index <c>c * d_conv + k</c>. This matches llama.cpp's
///       <c>ggml_ssm_conv</c> depthwise access pattern and gives each inner loop four
///       sequential loads per channel.
///     </description>
///   </item>
/// </list>
/// <para>
/// Output formula (per time step <c>t</c> in [0, T) and channel <c>c</c> in [0, channels)):
/// <code>
/// y[t, c] = bias[c] + Σ_{k=0..d_conv-1} input[t+k, c] * weight[c * d_conv + k]
/// </code>
/// </para>
/// <para>
/// This is a scalar reference implementation — correctness first, SIMD later. The
/// performance of the conv step is dwarfed by the <c>ssm_in</c> GEMM at 3136×17504 Q5_0,
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
    /// <param name="weight">
    /// Per-channel conv kernel, GGUF shape <c>[d_conv, channels]</c> stored channel-major —
    /// element <c>(k, c)</c> at flat index <c>c * d_conv + k</c>.
    /// </param>
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
        // Channel-major weight layout: w(k, c) at c*dConv + k means each channel's taps
        // are contiguous. Iterate time outer, channel outer, kernel-tap inner so the hot
        // inner loop walks the 4 tap weights of a single channel sequentially.
        for (int t = 0; t < seqLen; t++)
        {
            Span<float> outRow = output.Slice(t * channels, channels);

            for (int c = 0; c < channels; c++)
            {
                int wBase = c * dConv;
                float acc = bias[c];
                // Walk the kernel taps; input[(t+k), c] = input[(t+k)*channels + c].
                for (int k = 0; k < dConv; k++)
                {
                    acc += input[(t + k) * channels + c] * weight[wBase + k];
                }
                outRow[c] = acc;
            }
        }
    }
}
