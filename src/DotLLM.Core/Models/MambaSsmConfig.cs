namespace DotLLM.Core.Models;

/// <summary>
/// Mamba2 selective state-space configuration shared by all SSM layers in a hybrid
/// model (e.g. NVIDIA Nemotron-H). All values are sourced from GGUF metadata under
/// the <c>{arch}.ssm.*</c> keys.
/// </summary>
/// <param name="DConv">
/// 1-D causal convolution kernel size (GGUF: <c>{arch}.ssm.conv_kernel</c>). Typically 4.
/// </param>
/// <param name="DInner">
/// SSM inner dimension; the width of the post-projection x/B/C/dt block
/// (GGUF: <c>{arch}.ssm.inner_size</c>).
/// </param>
/// <param name="DState">
/// Per-channel state size N (GGUF: <c>{arch}.ssm.state_size</c>). Typically 128.
/// </param>
/// <param name="NGroup">
/// Number of channel groups sharing B/C and group RMSNorm (GGUF: <c>{arch}.ssm.group_count</c>).
/// </param>
/// <param name="NHead">
/// Number of Mamba2 SSM heads (GGUF: <c>{arch}.ssm.time_step_rank</c>). Each head spans
/// <c>DInner / NHead</c> channels and has a scalar A and D parameter.
/// </param>
public readonly record struct MambaSsmConfig(
    int DConv,
    int DInner,
    int DState,
    int NGroup,
    int NHead)
{
    /// <summary>Channels per Mamba2 head (<c>DInner / NHead</c>).</summary>
    public int HeadDim => DInner / NHead;

    /// <summary>Heads per group (<c>NHead / NGroup</c>); used for repeat-interleave broadcast of B/C.</summary>
    public int HeadsPerGroup => NHead / NGroup;

    /// <summary>
    /// Width of the input projection produced by <c>ssm_in</c>:
    /// <c>2*DInner + 2*NGroup*DState + NHead</c>.
    /// Layout is <c>[z | x | B | C | dt]</c> where <c>z</c> and <c>x</c> are <c>DInner</c>-wide,
    /// <c>B</c> and <c>C</c> are <c>NGroup*DState</c>-wide each, and <c>dt</c> is <c>NHead</c>-wide.
    /// </summary>
    public int InputProjectionDim => 2 * DInner + 2 * NGroup * DState + NHead;

    /// <summary>
    /// Width of the conv1d input <c>xBC</c> after splitting off <c>z</c> from <c>ssm_in</c>:
    /// <c>DInner + 2*NGroup*DState</c>. Equals the channel count of <c>ssm_conv1d.weight</c>.
    /// </summary>
    public int ConvDim => DInner + 2 * NGroup * DState;

    /// <summary>
    /// Per-sequence cached convolution state size in elements:
    /// <c>(DConv-1) * ConvDim</c>. F32.
    /// </summary>
    public int ConvStateElements => (DConv - 1) * ConvDim;

    /// <summary>
    /// Per-sequence cached SSM state size in elements:
    /// <c>NHead * HeadDim * DState</c> = <c>DInner * DState</c>. F32.
    /// </summary>
    public int SsmStateElements => DInner * DState;
}
