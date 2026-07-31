namespace DotLLM.Core.Models;

/// <summary>
/// Mamba-3 architecture configuration (Lahoti et al., ICLR 2026,
/// arXiv 2603.15569). Populated from the HuggingFace <c>config.json</c> of a
/// Mamba-3 checkpoint (e.g. <c>ib-ssm/mamba3-370M-10BT</c>).
/// </summary>
/// <remarks>
/// <para>
/// Mamba-3 is a pure SSM: there is no convolution (Mamba-3's trapezoidal
/// two-input recurrence makes conv1d redundant — see DESIGN_MAMBA_3.md §1.1)
/// and no attention per layer. The HF reference checkpoint also omits the
/// per-layer SwiGLU MLP that the <c>VikramKarLex/mamba3-minimal</c> reference
/// defines — each HF layer is just <c>{pre-norm, mixer}</c> with a residual
/// add. Registered as <c>Architecture.Mamba3</c> (added in PR-4).
/// </para>
/// <para>
/// Carried on <c>ModelConfig.Mamba3Config</c> (added in PR-4) in addition to — not
/// instead of — the shared <see cref="ModelConfig"/> fields: <c>HiddenSize</c>,
/// <c>NumLayers</c>, <c>VocabSize</c>, <c>HeadDim</c>, <c>NormEpsilon</c>,
/// <c>TiedEmbeddings</c> all carry the baseline values there. The fields on
/// this record are Mamba-3 specific and have no natural home on the shared
/// config.
/// </para>
/// <para>
/// Field naming matches the HF JSON keys where practical; values are validated
/// at extractor-level, so downstream code can treat every field as populated
/// with a legal value.
/// </para>
/// </remarks>
public sealed record Mamba3Config
{
    /// <summary>
    /// SSM state dimension <c>N</c>. Must be even (complex-pair RoPE on B/C).
    /// HF key: <c>state_size</c>. Typical value: 128.
    /// </summary>
    public required int StateSize { get; init; }

    /// <summary>
    /// Number of SSM heads (<c>H</c>). HF key: <c>num_heads</c>.
    /// </summary>
    public required int NumHeads { get; init; }

    /// <summary>
    /// Channels per head (<c>P</c>). HF key: <c>head_dim</c>. Note
    /// <c>num_heads * head_dim</c> is the SSM inner width <c>d_inner</c>, which
    /// need not equal <see cref="ModelConfig.HiddenSize"/> — the HF 370M config
    /// has <c>hidden_size=1024</c> but <c>d_inner = 32 * 64 = 2048</c>.
    /// </summary>
    public required int HeadDim { get; init; }

    /// <summary>
    /// Expansion factor for the SSM inner width: <c>d_inner = expand * hidden_size</c>.
    /// Redundant with <see cref="NumHeads"/> * <see cref="HeadDim"/> but stored
    /// explicitly because the HF config stores it and MIMO-variant checkpoints
    /// may legitimately set a value that mismatches the derived product.
    /// HF key: <c>expand</c>. Default: 2.
    /// </summary>
    public required int Expand { get; init; }

    /// <summary>
    /// Number of shared B/C groups (<c>G</c>). <c>1</c> on the known HF
    /// checkpoint. HF key: <c>n_groups</c>.
    /// </summary>
    public required int NumGroups { get; init; }

    /// <summary>
    /// MIMO chunk size for the chunked SSD scan. Must divide the
    /// non-decode-path sequence length. HF key: <c>chunk_size</c>.
    /// Default: 64.
    /// </summary>
    public required int ChunkSize { get; init; }

    /// <summary>
    /// Whether the block uses MIMO (rank-R) B/C projections. When false,
    /// the scan reduces to SISO. HF key: <c>is_mimo</c>.
    /// </summary>
    public required bool IsMimo { get; init; }

    /// <summary>
    /// MIMO rank <c>R</c> (unused when <see cref="IsMimo"/> is false). HF key:
    /// <c>mimo_rank</c>. Default: 4.
    /// </summary>
    public required int MimoRank { get; init; }

    /// <summary>
    /// Floor for <c>-A</c> (i.e. <c>A</c> is clamped to <c>&lt;= -A_floor</c>)
    /// during initialization to avoid zero-decay. Appendix initialization
    /// detail. HF key: <c>A_floor</c>. Default: 1e-4.
    /// </summary>
    public required float AFloor { get; init; }

    /// <summary>
    /// Floor for the initialization of <c>dt</c> (pre-softplus bias).
    /// HF key: <c>dt_init_floor</c>. Default: 1e-4.
    /// </summary>
    public required float DtInitFloor { get; init; }

    /// <summary>
    /// Lower clamp on <c>dt</c> during initialization. HF key: <c>dt_min</c>.
    /// Default: 0.001.
    /// </summary>
    public required float DtMin { get; init; }

    /// <summary>
    /// Upper clamp on <c>dt</c> during initialization. HF key: <c>dt_max</c>.
    /// Default: 0.1.
    /// </summary>
    public required float DtMax { get; init; }

    /// <summary>
    /// Whether to apply the L2-warp regularizer on the logits. HF key:
    /// <c>use_l2warp</c>. Inference-time effect is none (training-only term);
    /// carried for reproducibility.
    /// </summary>
    public required bool UseL2Warp { get; init; }

    /// <summary>
    /// Fraction of the state dimension that carries data-dependent RoPE
    /// rotation. HF key: <c>rope_fraction</c>. Default: 0.5. The
    /// <c>VikramKarLex/mamba3-minimal</c> reference assumes 1.0 (all state
    /// pairs rotated) — any value &lt; 1.0 implies a partial-RoPE variant
    /// that must be honoured during Stage D2 RoPE application.
    /// </summary>
    public required float RopeFraction { get; init; }

    /// <summary>
    /// Whether an output-projection RMSNorm is inserted pre-<c>out_proj</c>.
    /// HF key: <c>is_outproj_norm</c>. Not present on the known HF
    /// checkpoint (<c>false</c>); carried because it's declared in the HF
    /// config schema.
    /// </summary>
    public required bool IsOutProjNorm { get; init; }

    /// <summary>
    /// Whether the pre-norm residual is rescaled by <c>1/sqrt(2*num_layers)</c>
    /// at initialization. HF key: <c>rescale_prenorm_residual</c>.
    /// Training-only; no inference effect.
    /// </summary>
    public required bool RescalePrenormResidual { get; init; }

    /// <summary>
    /// Whether the residual stream is maintained in FP32 regardless of
    /// storage dtype. HF key: <c>residual_in_fp32</c>. In dotLLM the
    /// activation precision is chosen at the backend layer; we record this
    /// flag for reference fidelity.
    /// </summary>
    public required bool ResidualInFp32 { get; init; }

    /// <summary>
    /// SSM inner width — <c>num_heads * head_dim</c>.
    /// </summary>
    public int DInner => NumHeads * HeadDim;

    /// <summary>
    /// Per-block B/C width at the <c>in_proj</c> output:
    /// <c>state_size · num_bc_heads · effective_rank</c>, where
    /// <c>effective_rank = mimo_rank</c> in MIMO and <c>1</c> in SISO.
    /// Matches canonical <c>state-spaces/mamba</c> <c>self.bc_dim</c>.
    /// </summary>
    public int BcDim => StateSize * NumGroups * (IsMimo ? MimoRank : 1);

    /// <summary>
    /// Number of rotary pairs carried through data-RoPE —
    /// <c>int(state_size · rope_fraction) / 2</c> (rounded down to the nearest
    /// even value). <c>rope_fraction = 1.0</c> rotates the full state; <c>0.5</c>
    /// rotates half. For the 370M HF checkpoint (state_size=128, rope_fraction=0.5)
    /// this is <c>32</c>.
    /// </summary>
    public int NumRopeAngles
    {
        get
        {
            int splitSize = (int)(StateSize * RopeFraction);
            if ((splitSize & 1) != 0) splitSize -= 1;
            return splitSize / 2;
        }
    }

    /// <summary>
    /// Width of the <c>in_proj.weight</c> output — the canonical 8-slice
    /// concat <c>[z | x | B | C | dd_dt | dd_A | trap | angles]</c>:
    /// <c>2·d_inner + 2·bc_dim + 3·num_heads + num_rope_angles</c>.
    /// For the 370M HF checkpoint (d_inner=2048, bc_dim=128, num_heads=32,
    /// num_rope_angles=32) this is <c>4480</c>.
    /// </summary>
    public int InputProjectionDim =>
        2 * DInner + 2 * BcDim + 3 * NumHeads + NumRopeAngles;
}
