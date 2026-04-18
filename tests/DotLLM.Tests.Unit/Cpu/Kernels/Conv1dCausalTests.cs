using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// Unit tests for <see cref="Conv1dCausal"/>. Layout reminders:
/// <list type="bullet">
///   <item><description>input / output are time-major: <c>(t, c)</c> at index <c>t*channels + c</c></description></item>
///   <item><description>weight is channel-major (GGUF <c>ssm_conv1d.weight</c>): <c>(k, c)</c> at index <c>c*d_conv + k</c></description></item>
/// </list>
/// </summary>
public sealed class Conv1dCausalTests
{
    /// <summary>Decode path: T=1 means the kernel sees d_conv-1 cached rows plus 1 new row.</summary>
    [Fact]
    public void DecodeSingleToken_MatchesHandComputed()
    {
        // d_conv = 4, channels = 2, T = 1  ->  input shape [4, 2], output shape [1, 2].
        //
        // input rows (time-major):
        //   t=0:  1, 2
        //   t=1:  3, 4
        //   t=2:  5, 6
        //   t=3:  7, 8   <- new activation row xBC
        //
        // weight channel-major: channel 0 taps [0.1, 0.3, 0.5, 0.7], channel 1 taps [0.2, 0.4, 0.6, 0.8]
        //
        // bias: [0.01, 0.02]
        //
        // Expected y[0, c] = bias[c] + sum_k input[k, c] * w(k, c)
        //   c=0: 0.01 + 1*0.1 + 3*0.3 + 5*0.5 + 7*0.7 = 0.01 + 0.1 + 0.9 + 2.5 + 4.9 = 8.41
        //   c=1: 0.02 + 2*0.2 + 4*0.4 + 6*0.6 + 8*0.8 = 0.02 + 0.4 + 1.6 + 3.6 + 6.4 = 12.02

        float[] input =
        [
            1, 2,
            3, 4,
            5, 6,
            7, 8,
        ];
        // Channel-major: channel 0's 4 taps, then channel 1's 4 taps.
        float[] weight =
        [
            0.1f, 0.3f, 0.5f, 0.7f,     // channel 0
            0.2f, 0.4f, 0.6f, 0.8f,     // channel 1
        ];
        float[] bias = [0.01f, 0.02f];
        float[] output = new float[2];

        Conv1dCausal.Execute(input, weight, bias, output, dConv: 4, channels: 2, seqLen: 1);

        Assert.Equal(8.41f, output[0], 1e-5f);
        Assert.Equal(12.02f, output[1], 1e-5f);
    }

    /// <summary>Prefill path: T=3 means we produce 3 output rows from a d_conv-1=3 cached + 3 new = 6 row input.</summary>
    [Fact]
    public void PrefillThreeTokens_CausalityRespected()
    {
        // d_conv = 4, channels = 1, T = 3. input length = 6, weight length = 4.
        // Single channel means channel-major and time-major coincide.
        float[] input = [1, 2, 3, 4, 5, 6];
        float[] weight = [0.25f, 0.25f, 0.25f, 0.25f]; // simple box filter
        float[] bias = [0.5f];
        float[] output = new float[3];

        Conv1dCausal.Execute(input, weight, bias, output, dConv: 4, channels: 1, seqLen: 3);

        // y[0] = 0.5 + 0.25*(1+2+3+4) = 0.5 + 2.5 = 3.0
        // y[1] = 0.5 + 0.25*(2+3+4+5) = 0.5 + 3.5 = 4.0
        // y[2] = 0.5 + 0.25*(3+4+5+6) = 0.5 + 4.5 = 5.0
        Assert.Equal(3.0f, output[0], 1e-5f);
        Assert.Equal(4.0f, output[1], 1e-5f);
        Assert.Equal(5.0f, output[2], 1e-5f);
    }

    /// <summary>Kernel of zeros + non-zero bias ⇒ output is purely the bias broadcast per-channel.</summary>
    [Fact]
    public void ZeroWeight_OutputsBiasOnly()
    {
        const int channels = 5;
        const int dConv = 4;
        const int seqLen = 2;

        float[] input = new float[(dConv - 1 + seqLen) * channels];
        for (int i = 0; i < input.Length; i++) input[i] = i + 1; // ensures inputs are non-zero
        float[] weight = new float[dConv * channels]; // all zero
        float[] bias = [0.1f, 0.2f, 0.3f, 0.4f, 0.5f];
        float[] output = new float[seqLen * channels];

        Conv1dCausal.Execute(input, weight, bias, output, dConv, channels, seqLen);

        for (int t = 0; t < seqLen; t++)
            for (int c = 0; c < channels; c++)
                Assert.Equal(bias[c], output[t * channels + c], 1e-6f);
    }

    /// <summary>
    /// Non-uniform channels: verifies per-channel independence — channel 0 sees one filter,
    /// channel 1 sees another, and each output only touches its own weight column.
    /// </summary>
    [Fact]
    public void PerChannelIndependence()
    {
        // d_conv = 4, channels = 2, T = 2.
        // Input time-major:
        //   t=0: 1, 10
        //   t=1: 2, 20
        //   t=2: 3, 30
        //   t=3: 4, 40
        //   t=4: 5, 50
        float[] input =
        [
            1, 10,
            2, 20,
            3, 30,
            4, 40,
            5, 50,
        ];
        // Channel-major weights:
        //   channel 0: [1, 0, 0, 0]  (first tap — picks input[t, 0])
        //   channel 1: [0, 0, 0, 1]  (last tap — picks input[t+3, 1])
        float[] weight =
        [
            1, 0, 0, 0,   // channel 0
            0, 0, 0, 1,   // channel 1
        ];
        float[] bias = [0f, 0f];
        float[] output = new float[2 * 2];

        Conv1dCausal.Execute(input, weight, bias, output, dConv: 4, channels: 2, seqLen: 2);

        // y[t, 0] = input[t, 0] -> t=0:1, t=1:2
        // y[t, 1] = input[t+3, 1] -> t=0:40, t=1:50
        Assert.Equal(1f, output[0 * 2 + 0], 1e-6f);
        Assert.Equal(40f, output[0 * 2 + 1], 1e-6f);
        Assert.Equal(2f, output[1 * 2 + 0], 1e-6f);
        Assert.Equal(50f, output[1 * 2 + 1], 1e-6f);
    }

    [Fact]
    public void ZeroSeqLen_ProducesNothing()
    {
        // T = 0 is a legal edge case (e.g. empty prefill). Nothing to do, no crashes.
        float[] input = new float[3]; // (d_conv-1 + 0) * channels = 3*1
        float[] weight = new float[4];
        float[] bias = [0f];
        float[] output = [];

        Conv1dCausal.Execute(input, weight, bias, output, dConv: 4, channels: 1, seqLen: 0);
        // No asserts: the contract is "does not throw".
    }
}
