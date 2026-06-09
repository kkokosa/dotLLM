using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Lora;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Multi-head Latent Attention (MLA) kernel — the DeepSeek-V2/V3 attention
/// mechanism. Runs a full forward pass from hidden states to post-<c>o_proj</c>
/// output using a scalar-first implementation that keeps the projection and
/// attention math self-contained for correctness verification.
/// </summary>
/// <remarks>
/// <para>
/// <b>Data flow (per token, per layer).</b>
/// <list type="number">
///   <item>
///     <b>Q path.</b> If <c>q_lora_rank &gt; 0</c>, compute
///     <c>q_latent = q_a_proj @ hidden</c>, apply <c>q_a_layernorm</c>
///     (RMSNorm), then <c>q = q_b_proj @ q_latent</c>. Otherwise compute
///     <c>q = q_proj @ hidden</c> directly (monolithic Q).
///     Reshape to <c>[num_heads, qk_head_dim]</c> where
///     <c>qk_head_dim = qk_nope_head_dim + qk_rope_head_dim</c>, split into
///     <c>q_nope</c> and <c>q_pe</c> on the last dim.
///   </item>
///   <item>
///     <b>KV path.</b> Compute
///     <c>compressed_kv = kv_a_proj_with_mqa @ hidden</c> of size
///     <c>kv_lora_rank + qk_rope_head_dim</c>. Split: first
///     <c>kv_lora_rank</c> entries are the latent <c>k_nope_latent</c>, next
///     <c>qk_rope_head_dim</c> entries are the shared rope-K
///     (<c>k_pe</c>, broadcast across all heads).
///     Apply <c>kv_a_layernorm</c> (RMSNorm) to <c>k_nope_latent</c>.
///     Expand via <c>kv_b_proj</c> (<c>kv_lora_rank → num_heads *
///     (qk_nope_head_dim + v_head_dim)</c>). Per-head split into
///     <c>k_nope</c> (first <c>qk_nope_head_dim</c>) and <c>v</c> (last
///     <c>v_head_dim</c>).
///   </item>
///   <item>
///     <b>RoPE.</b> Apply rotary embedding (Norm-pair convention: adjacent
///     element pairs) to <c>q_pe</c> per-head and to <c>k_pe</c> once (shared).
///   </item>
///   <item>
///     <b>Attention.</b> For each head h: Q_h = concat(q_nope_h, q_pe_h),
///     K_h = concat(k_nope_h, k_pe_shared). Scaled dot-product with scale
///     <c>1 / sqrt(qk_head_dim)</c>, causal + optional sliding-window mask,
///     softmax, weighted sum over V_h (width <c>v_head_dim</c>).
///   </item>
///   <item>
///     <b>Output.</b> Concatenate all head outputs to
///     <c>[num_heads * v_head_dim]</c>, project with <c>o_proj</c> to
///     <c>hidden_size</c>.
///   </item>
/// </list>
/// </para>
/// <para>
/// <b>Storage convention.</b> All weight matrices are passed as row-major
/// F32 with shape <c>[output_dim, input_dim]</c> (standard HF
/// <c>nn.Linear.weight</c> convention: <c>y = W @ x</c> means
/// <c>y[i] = sum_k W[i, k] * x[k]</c>, so row <c>i</c> of <c>W</c> is
/// contiguous). The <c>kv_b_proj</c> weight stores the per-head
/// <c>[qk_nope_head_dim + v_head_dim]</c> block contiguously for head 0,
/// then head 1, etc.
/// </para>
/// <para>
/// <b>Out of scope.</b> No "absorption" optimisation (precomputing
/// <c>W_q_nope @ W_k_nope^T</c>), no latent KV-cache, no quantised weights,
/// and no YaRN RoPE frequency rescaling (only the YaRN softmax-scale
/// mscale² correction — applied via the optional
/// <c>attnScaleMultiplier</c> parameter). This implementation targets
/// correctness against a Python / HF reference.
/// </para>
/// </remarks>
public static class MlaAttention
{
    /// <summary>
    /// Full MLA forward pass from hidden states to post-<c>o_proj</c> output.
    /// Scalar reference implementation — optimise later.
    /// </summary>
    /// <param name="hidden">Input hidden states [seqLen, hiddenSize], row-major.</param>
    /// <param name="output">Destination [seqLen, hiddenSize], row-major. May alias <paramref name="hidden"/>.</param>
    /// <param name="seqLen">Number of tokens being processed (prefill=prompt length, decode=1).</param>
    /// <param name="positionOffset">
    /// Position offset for causal mask and RoPE. For prefill over a full prompt
    /// starting at position 0 this is 0 and token <c>i</c> sits at position
    /// <c>i</c>. For decode with a cached KV context of length
    /// <c>positionOffset</c>, the single new token sits at position
    /// <c>positionOffset</c> and may attend to all <c>positionOffset + 1</c>
    /// positions.
    /// </param>
    /// <param name="hiddenSize">Model hidden size.</param>
    /// <param name="numHeads">Number of Q attention heads (= num K heads, =
    /// num V heads — MLA is head-parallel on the expanded side).</param>
    /// <param name="qkNopeHeadDim">Non-rope Q·K sub-dimension per head.</param>
    /// <param name="qkRopeHeadDim">Rope Q·K sub-dimension per head (must be even).</param>
    /// <param name="vHeadDim">V head dimension (may differ from qk_head_dim).</param>
    /// <param name="qLoraRank">Q low-rank bottleneck dim; 0 = no factorisation, use <paramref name="qProj"/> instead.</param>
    /// <param name="kvLoraRank">KV low-rank bottleneck dim.</param>
    /// <param name="rmsNormEps">RMSNorm epsilon for <c>q_a_layernorm</c> and <c>kv_a_layernorm</c>.</param>
    /// <param name="ropeCosTable">Pre-computed RoPE cos table [maxSeq, qkRopeHeadDim / 2].</param>
    /// <param name="ropeSinTable">Pre-computed RoPE sin table [maxSeq, qkRopeHeadDim / 2].</param>
    /// <param name="qAProj">Q down-projection weight [qLoraRank, hiddenSize]. Ignored when qLoraRank==0.</param>
    /// <param name="qALayernormWeight">Q LoRA LayerNorm weight [qLoraRank]. Ignored when qLoraRank==0.</param>
    /// <param name="qBProj">Q up-projection weight [numHeads * qkHeadDim, qLoraRank]. Ignored when qLoraRank==0.</param>
    /// <param name="qProj">Monolithic Q projection [numHeads * qkHeadDim, hiddenSize]. Only used when qLoraRank==0.</param>
    /// <param name="kvAProjWithMqa">KV down-projection weight [kvLoraRank + qkRopeHeadDim, hiddenSize].</param>
    /// <param name="kvALayernormWeight">KV LoRA LayerNorm weight [kvLoraRank].</param>
    /// <param name="kvBProj">KV up-projection weight [numHeads * (qkNopeHeadDim + vHeadDim), kvLoraRank].</param>
    /// <param name="oProj">Output projection [hiddenSize, numHeads * vHeadDim].</param>
    /// <param name="attnScaleMultiplier">
    /// Softmax-scale multiplier applied on top of the default
    /// <c>1 / sqrt(qk_head_dim)</c>. Pass <c>1.0f</c> (the default) for the
    /// plain DeepSeek-V2 case. For YaRN context extension, pass
    /// <see cref="DotLLM.Core.Models.MlaConfig.ComputeYarnSoftmaxScaleMultiplier"/>
    /// which returns <c>mscale²</c> per the DeepSeek-V2 YaRN recipe.
    /// </param>
    /// <param name="cachedKNope">
    /// Optional native pointer to a persistent per-layer K_nope buffer of
    /// shape <c>[maxSeqLen, numHeads * qk_nope_head_dim]</c>. When non-zero,
    /// the kernel appends the new <paramref name="seqLen"/> tokens' K_nope
    /// at offset <paramref name="cachedLength"/> and the attention loop
    /// iterates over all <c>cachedLength + seqLen</c> cached positions.
    /// </param>
    /// <param name="cachedV">
    /// Optional native pointer to a persistent per-layer V buffer of shape
    /// <c>[maxSeqLen, numHeads * v_head_dim]</c>. Must be supplied whenever
    /// <paramref name="cachedKNope"/> is supplied.
    /// </param>
    /// <param name="cachedKPe">
    /// Optional native pointer to a persistent per-layer K_pe buffer of
    /// shape <c>[maxSeqLen, qk_rope_head_dim]</c> (single MQA rope-K,
    /// RoPE-already-applied — we cache the post-rotation value). Must be
    /// supplied whenever <paramref name="cachedKNope"/> is supplied.
    /// </param>
    /// <param name="cachedLength">
    /// Number of positions already present in the cache for this layer. The
    /// new tokens sit at <c>[cachedLength, cachedLength + seqLen)</c>; the
    /// attention loop attends over all <c>cachedLength + seqLen</c>
    /// positions. Must equal <paramref name="positionOffset"/> in the typical
    /// autoregressive case — the two are distinct in the signature only to
    /// keep the cache-less call path untouched.
    /// </param>
    /// <param name="loraAdapter">Optional active LoRA adapter for MLA-specific projection deltas.</param>
    /// <param name="loraLayer">Layer index used to resolve adapter weights.</param>
    public static unsafe void Execute(
        ReadOnlySpan<float> hidden,
        Span<float> output,
        int seqLen,
        int positionOffset,
        int hiddenSize,
        int numHeads,
        int qkNopeHeadDim,
        int qkRopeHeadDim,
        int vHeadDim,
        int qLoraRank,
        int kvLoraRank,
        float rmsNormEps,
        ReadOnlySpan<float> ropeCosTable,
        ReadOnlySpan<float> ropeSinTable,
        ReadOnlySpan<float> qAProj,
        ReadOnlySpan<float> qALayernormWeight,
        ReadOnlySpan<float> qBProj,
        ReadOnlySpan<float> qProj,
        ReadOnlySpan<float> kvAProjWithMqa,
        ReadOnlySpan<float> kvALayernormWeight,
        ReadOnlySpan<float> kvBProj,
        ReadOnlySpan<float> oProj,
        float attnScaleMultiplier = 1.0f,
        nint cachedKNope = 0,
        nint cachedV = 0,
        nint cachedKPe = 0,
        int cachedLength = 0,
        ILoraAdapter? loraAdapter = null,
        int loraLayer = -1)
    {
        bool useCache = cachedKNope != 0;
        if (useCache && (cachedV == 0 || cachedKPe == 0))
            throw new ArgumentException(
                "cachedV and cachedKPe must be supplied together with cachedKNope.");

        ValidateArgs(seqLen, hiddenSize, numHeads, qkNopeHeadDim, qkRopeHeadDim, vHeadDim,
                     qLoraRank, kvLoraRank, hidden, output);

        int qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
        int qTotal = numHeads * qkHeadDim;
        int kvBOutputDim = numHeads * (qkNopeHeadDim + vHeadDim);
        float scale = attnScaleMultiplier / MathF.Sqrt(qkHeadDim);

        // Scratch allocations. For PoC we rent managed arrays — the kernel is
        // correctness-first and the hot path will migrate to caller-provided
        // native scratch once wired into the forward pass.
        float[] qBuf = new float[seqLen * qTotal];                             // [S, numHeads * qkHeadDim]
        float[] kNopeBuf = new float[seqLen * numHeads * qkNopeHeadDim];        // [S, numHeads, qkNopeHeadDim]
        float[] kPeBuf = new float[seqLen * qkRopeHeadDim];                     // [S, qkRopeHeadDim] (shared)
        float[] vBuf = new float[seqLen * numHeads * vHeadDim];                 // [S, numHeads, vHeadDim]
        float[] compressedKvBuf = new float[seqLen * (kvLoraRank + qkRopeHeadDim)];
        float[] kvLatentNormBuf = new float[seqLen * kvLoraRank];
        float[] kvBExpanded = new float[seqLen * kvBOutputDim];
        float[] qLatentBuf = qLoraRank > 0 ? new float[seqLen * qLoraRank] : Array.Empty<float>();
        float[] qLatentNormBuf = qLoraRank > 0 ? new float[seqLen * qLoraRank] : Array.Empty<float>();
        float[] attnOutBuf = new float[seqLen * numHeads * vHeadDim];

        // Q projections
        if (qLoraRank > 0)
        {
            for (int t = 0; t < seqLen; t++)
            {
                var hiddenRow = hidden.Slice(t * hiddenSize, hiddenSize);
                // q_latent = q_a_proj @ hidden
                var latent = qLatentBuf.AsSpan(t * qLoraRank, qLoraRank);
                MatVec(qAProj, hiddenRow, latent, qLoraRank, hiddenSize);
            }

            ApplyLoraDelta(loraAdapter, loraLayer, "q_a_proj",
                hidden, qLatentBuf, seqLen, hiddenSize, qLoraRank);

            for (int t = 0; t < seqLen; t++)
            {
                // q_latent_norm = RMSNorm(q_latent, q_a_layernorm)
                var latent = qLatentBuf.AsSpan(t * qLoraRank, qLoraRank);
                var latentNorm = qLatentNormBuf.AsSpan(t * qLoraRank, qLoraRank);
                RmsNormScalar(latent, qALayernormWeight, rmsNormEps, latentNorm);

                // q = q_b_proj @ q_latent_norm
                var qRow = qBuf.AsSpan(t * qTotal, qTotal);
                MatVec(qBProj, latentNorm, qRow, qTotal, qLoraRank);
            }

            ApplyLoraDelta(loraAdapter, loraLayer, "q_b_proj",
                qLatentNormBuf, qBuf, seqLen, qLoraRank, qTotal);
        }
        else
        {
            for (int t = 0; t < seqLen; t++)
            {
                var hiddenRow = hidden.Slice(t * hiddenSize, hiddenSize);
                var qRow = qBuf.AsSpan(t * qTotal, qTotal);
                // q = q_proj @ hidden (monolithic path)
                MatVec(qProj, hiddenRow, qRow, qTotal, hiddenSize);
            }

            ApplyLoraDelta(loraAdapter, loraLayer, "q_proj",
                hidden, qBuf, seqLen, hiddenSize, qTotal);
        }

        // KV down-projection + split
        int compressedKvDim = kvLoraRank + qkRopeHeadDim;
        for (int t = 0; t < seqLen; t++)
        {
            var hiddenRow = hidden.Slice(t * hiddenSize, hiddenSize);
            var compRow = compressedKvBuf.AsSpan(t * compressedKvDim, compressedKvDim);
            MatVec(kvAProjWithMqa, hiddenRow, compRow, compressedKvDim, hiddenSize);
        }

        ApplyLoraDelta(loraAdapter, loraLayer, "kv_a_proj_with_mqa",
            hidden, compressedKvBuf, seqLen, hiddenSize, compressedKvDim);

        for (int t = 0; t < seqLen; t++)
        {
            var compRow = compressedKvBuf.AsSpan(t * compressedKvDim, compressedKvDim);
            // Split: first kvLoraRank = k_nope_latent, next qkRopeHeadDim = k_pe
            var latent = compRow.Slice(0, kvLoraRank);
            var kPe = compRow.Slice(kvLoraRank, qkRopeHeadDim);

            // k_nope_latent = RMSNorm(k_nope_latent, kv_a_layernorm)
            var latentNorm = kvLatentNormBuf.AsSpan(t * kvLoraRank, kvLoraRank);
            RmsNormScalar(latent, kvALayernormWeight, rmsNormEps, latentNorm);

            // kv_b_expanded = kv_b_proj @ latentNorm  (size = numHeads * (qkNope + vHead))
            var expandedRow = kvBExpanded.AsSpan(t * kvBOutputDim, kvBOutputDim);
            MatVec(kvBProj, latentNorm, expandedRow, kvBOutputDim, kvLoraRank);
        }

        ApplyLoraDelta(loraAdapter, loraLayer, "kv_b_proj",
            kvLatentNormBuf, kvBExpanded, seqLen, kvLoraRank, kvBOutputDim);

        for (int t = 0; t < seqLen; t++)
        {
            var compRow = compressedKvBuf.AsSpan(t * compressedKvDim, compressedKvDim);
            var kPe = compRow.Slice(kvLoraRank, qkRopeHeadDim);
            var expandedRow = kvBExpanded.AsSpan(t * kvBOutputDim, kvBOutputDim);

            // Per-head split into kNope [qkNopeHeadDim] and v [vHeadDim]
            int perHead = qkNopeHeadDim + vHeadDim;
            for (int h = 0; h < numHeads; h++)
            {
                var headBlock = expandedRow.Slice(h * perHead, perHead);
                headBlock.Slice(0, qkNopeHeadDim)
                         .CopyTo(kNopeBuf.AsSpan(t * numHeads * qkNopeHeadDim + h * qkNopeHeadDim, qkNopeHeadDim));
                headBlock.Slice(qkNopeHeadDim, vHeadDim)
                         .CopyTo(vBuf.AsSpan(t * numHeads * vHeadDim + h * vHeadDim, vHeadDim));
            }

            // Store k_pe (shared across heads)
            kPe.CopyTo(kPeBuf.AsSpan(t * qkRopeHeadDim, qkRopeHeadDim));
        }

        // Apply RoPE to q_pe portion of Q (per head) and to shared k_pe
        int halfRope = qkRopeHeadDim / 2;
        for (int t = 0; t < seqLen; t++)
        {
            int pos = positionOffset + t;
            var cosRow = ropeCosTable.Slice(pos * halfRope, halfRope);
            var sinRow = ropeSinTable.Slice(pos * halfRope, halfRope);

            // Q: rotate the rope portion for each head
            for (int h = 0; h < numHeads; h++)
            {
                // q_pe_h is at [t, h * qkHeadDim + qkNopeHeadDim .. +qkRopeHeadDim]
                var qPe = qBuf.AsSpan(
                    t * qTotal + h * qkHeadDim + qkNopeHeadDim,
                    qkRopeHeadDim);
                ApplyRopeNormInPlace(qPe, cosRow, sinRow);
            }

            // K shared rope
            var kPe = kPeBuf.AsSpan(t * qkRopeHeadDim, qkRopeHeadDim);
            ApplyRopeNormInPlace(kPe, cosRow, sinRow);
        }

        // If a cache is provided, memcpy the seqLen newly-computed K_nope /
        // V / K_pe rows into the persistent per-layer store at offset
        // `cachedLength`. Subsequent attention reads then see the full
        // history (0..cachedLength + seqLen) via the cache spans built
        // below. The managed scratch arrays (kNopeBuf / vBuf / kPeBuf) are
        // still used as the source; only the *read* side of attention
        // switches to the cache.
        if (useCache)
        {
            int kNopePerTok = numHeads * qkNopeHeadDim;
            int vPerTok = numHeads * vHeadDim;

            var dstKNope = new Span<float>(
                (void*)(cachedKNope + (nint)((long)cachedLength * kNopePerTok * sizeof(float))),
                seqLen * kNopePerTok);
            kNopeBuf.AsSpan(0, seqLen * kNopePerTok).CopyTo(dstKNope);

            var dstV = new Span<float>(
                (void*)(cachedV + (nint)((long)cachedLength * vPerTok * sizeof(float))),
                seqLen * vPerTok);
            vBuf.AsSpan(0, seqLen * vPerTok).CopyTo(dstV);

            var dstKPe = new Span<float>(
                (void*)(cachedKPe + (nint)((long)cachedLength * qkRopeHeadDim * sizeof(float))),
                seqLen * qkRopeHeadDim);
            kPeBuf.AsSpan(0, seqLen * qkRopeHeadDim).CopyTo(dstKPe);
        }

        // Attention per head with causal mask
        // Q_h[t] = concat(q_nope_h[t], q_pe_h[t]) — already adjacent in qBuf
        // K_h[s] = concat(k_nope_h[s], k_pe_shared[s])
        // V_h[s] (width vHeadDim)
        // Score[t, s] = Q_h[t] . K_h[s] * scale
        // Mask: s <= positionOffset + t
        //   Output per head at t: softmax(score[t, :]) . V_h[:]
        //
        // When useCache: read K_nope / V / K_pe from the native cache so the
        // attention loop sees all (cachedLength + seqLen) positions. When
        // not: read from the per-call managed scratch arrays and attend
        // only over seqLen (the historical no-cache PoC path).
        int seqKv = useCache ? cachedLength + seqLen : seqLen;
        int queryPosBase = useCache ? cachedLength : positionOffset;

        ReadOnlySpan<float> kNopeReadAll = useCache
            ? new ReadOnlySpan<float>((void*)cachedKNope, seqKv * numHeads * qkNopeHeadDim)
            : kNopeBuf.AsSpan(0, seqLen * numHeads * qkNopeHeadDim);
        ReadOnlySpan<float> vReadAll = useCache
            ? new ReadOnlySpan<float>((void*)cachedV, seqKv * numHeads * vHeadDim)
            : vBuf.AsSpan(0, seqLen * numHeads * vHeadDim);
        ReadOnlySpan<float> kPeReadAll = useCache
            ? new ReadOnlySpan<float>((void*)cachedKPe, seqKv * qkRopeHeadDim)
            : kPeBuf.AsSpan(0, seqLen * qkRopeHeadDim);

        // Scratch scores reused across all heads.
        float[] scores = new float[seqLen * seqKv];
        for (int h = 0; h < numHeads; h++)
        {

            for (int t = 0; t < seqLen; t++)
            {
                // Build Q vector for head h at query position t
                var qNopeH = qBuf.AsSpan(t * qTotal + h * qkHeadDim, qkNopeHeadDim);
                var qPeH = qBuf.AsSpan(t * qTotal + h * qkHeadDim + qkNopeHeadDim, qkRopeHeadDim);

                // Absolute position of query t in the full causal window.
                int queryPos = queryPosBase + t;

                for (int s = 0; s < seqKv; s++)
                {
                    // Causal mask: s > queryPos → -inf
                    if (s > queryPos)
                    {
                        scores[t * seqKv + s] = float.NegativeInfinity;
                        continue;
                    }

                    // K_h[s] = concat(k_nope_h[s], k_pe_shared[s])
                    var kNopeH = kNopeReadAll.Slice(
                        s * numHeads * qkNopeHeadDim + h * qkNopeHeadDim,
                        qkNopeHeadDim);
                    var kPeS = kPeReadAll.Slice(s * qkRopeHeadDim, qkRopeHeadDim);

                    // Score = Q_nope · K_nope + Q_pe · K_pe_shared — vectorised.
                    float dot = TensorPrimitives.Dot(qNopeH, kNopeH)
                              + TensorPrimitives.Dot(qPeH, kPeS);

                    scores[t * seqKv + s] = dot * scale;
                }

                // Softmax row t
                SoftmaxRowInPlace(scores.AsSpan(), t, seqKv);

                // Weighted sum over V_h — SAXPY via MultiplyAdd
                // (outH = v_h * w + outH).
                var outH = attnOutBuf.AsSpan(t * numHeads * vHeadDim + h * vHeadDim, vHeadDim);
                outH.Clear();
                for (int s = 0; s <= queryPos && s < seqKv; s++)
                {
                    float w = scores[t * seqKv + s];
                    if (w == 0f) continue;
                    var vH = vReadAll.Slice(s * numHeads * vHeadDim + h * vHeadDim, vHeadDim);
                    TensorPrimitives.MultiplyAdd(vH, w, outH, outH);
                }
            }
        }

        // Output projection: o_proj @ attnOut
        int oInputDim = numHeads * vHeadDim;
        for (int t = 0; t < seqLen; t++)
        {
            var attnRow = attnOutBuf.AsSpan(t * oInputDim, oInputDim);
            var outRow = output.Slice(t * hiddenSize, hiddenSize);
            MatVec(oProj, attnRow, outRow, hiddenSize, oInputDim);
        }

        ApplyLoraDelta(loraAdapter, loraLayer, "o_proj",
            attnOutBuf, output, seqLen, oInputDim, hiddenSize);
    }

    private static unsafe void ApplyLoraDelta(
        ILoraAdapter? adapter,
        int layer,
        string projection,
        ReadOnlySpan<float> input,
        Span<float> output,
        int seqLen,
        int inputDim,
        int outputDim)
    {
        if (adapter is null || layer < 0) return;
        var lora = adapter.GetLayerWeights(layer, projection);
        if (lora is not { } w) return;
        if (w.InputDim != inputDim || w.OutputDim != outputDim)
            throw new InvalidOperationException(
                $"LoRA adapter '{adapter.Name}' layer={layer} proj='{projection}' shape "
                + $"({w.InputDim}x{w.OutputDim}) does not match MLA projection "
                + $"({inputDim}x{outputDim}).");

        float scale = adapter.Alpha / adapter.Rank;
        // Phase 4d.6 — opt into the outer-product stage-2 fast path when
        // available (rank=16 + AVX-512). EnsureATransposedF32 is idempotent
        // and cheap on the cached path.
        nint aTransposedHandle = LoraStage2.EnsureATransposedF32(
            adapter as LoraAdapter, layer, projection, in w, adapter.Rank);
        fixed (float* x = input)
        fixed (float* y = output)
        {
            LoraDelta.Apply(x, (void*)w.BHandle, (void*)w.AHandle, y,
                seqLen, inputDim, outputDim, adapter.Rank, scale,
                w.WeightDType, w.WeightDType, aTransposedHandle);
        }
    }

    /// <summary>
    /// Phase B — latent MLA KV-cache + absorbed attention. The production
    /// memory win: stores only <c>c_kv[kv_lora_rank]</c> and
    /// <c>k_pe[qk_rope_head_dim]</c> per token per layer (~7× smaller than
    /// <see cref="Execute"/>'s expanded cache), and recovers per-head K/V
    /// on the fly through the absorbed identity:
    /// <code>
    ///     Q_nope[h] · K_nope[h, s] = Q_nope[h] · (W_UK[h] @ c_kv[s])
    ///                              = (W_UK[h]^T @ Q_nope[h]) · c_kv[s]
    ///                              = Q_latent[h] · c_kv[s]
    /// </code>
    /// and the V path mirrors:
    /// <code>
    ///     out[h] = W_UV[h] @ out_latent[h]
    ///     where out_latent[h] = Σ_s softmax · c_kv[s]
    /// </code>
    /// Per the DeepSeek-V2 paper §2.1.2. This is the structurally-same
    /// algorithm vLLM's MLA backend uses; we keep it scalar for
    /// correctness-first and vectorise later.
    /// </summary>
    /// <remarks>
    /// <b>Correctness note.</b> This method must produce logits that match
    /// <see cref="Execute"/> within <c>1e-3</c> at F32 on the same input +
    /// weights — the only numerical deviation is the order of the identity
    /// <c>(W_UK^T @ Q) · c_kv = Q · (W_UK @ c_kv)</c>, which changes the
    /// summation order of a dot product. Validate against <see cref="Execute"/>
    /// as the oracle on a fresh synthetic fixture before trusting it on
    /// real weights.
    /// </remarks>
    /// <param name="hidden">See <see cref="Execute"/>.</param>
    /// <param name="output">See <see cref="Execute"/>.</param>
    /// <param name="seqLen">See <see cref="Execute"/>.</param>
    /// <param name="positionOffset">See <see cref="Execute"/>.</param>
    /// <param name="hiddenSize">See <see cref="Execute"/>.</param>
    /// <param name="numHeads">See <see cref="Execute"/>.</param>
    /// <param name="qkNopeHeadDim">See <see cref="Execute"/>.</param>
    /// <param name="qkRopeHeadDim">See <see cref="Execute"/>.</param>
    /// <param name="vHeadDim">See <see cref="Execute"/>.</param>
    /// <param name="qLoraRank">See <see cref="Execute"/>.</param>
    /// <param name="kvLoraRank">See <see cref="Execute"/>.</param>
    /// <param name="rmsNormEps">See <see cref="Execute"/>.</param>
    /// <param name="ropeCosTable">See <see cref="Execute"/>.</param>
    /// <param name="ropeSinTable">See <see cref="Execute"/>.</param>
    /// <param name="qAProj">See <see cref="Execute"/>.</param>
    /// <param name="qALayernormWeight">See <see cref="Execute"/>.</param>
    /// <param name="qBProj">See <see cref="Execute"/>.</param>
    /// <param name="qProj">See <see cref="Execute"/>.</param>
    /// <param name="kvAProjWithMqa">See <see cref="Execute"/>.</param>
    /// <param name="kvALayernormWeight">See <see cref="Execute"/>.</param>
    /// <param name="kvBProj">
    /// Same tensor as <see cref="Execute"/>: row-major
    /// <c>[numHeads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]</c>.
    /// The kernel indexes into it directly as <c>W_UK</c> and <c>W_UV</c>
    /// slices; no pre-transpose needed at load time.
    /// </param>
    /// <param name="oProj">See <see cref="Execute"/>.</param>
    /// <param name="cachedLatent">
    /// Native pointer to the persistent per-layer latent cache of shape
    /// <c>[maxSeqLen, kv_lora_rank]</c>. The kernel appends the new
    /// <paramref name="seqLen"/> tokens' latents at offset
    /// <paramref name="cachedLength"/> and attends over all
    /// <c>cachedLength + seqLen</c> cached positions.
    /// </param>
    /// <param name="cachedKPe">
    /// Native pointer to the persistent per-layer shared K_pe buffer of
    /// shape <c>[maxSeqLen, qk_rope_head_dim]</c>. Identical to
    /// <see cref="Execute"/>'s <c>cachedKPe</c>.
    /// </param>
    /// <param name="cachedLength">Positions already in the cache for this layer.</param>
    /// <param name="attnScaleMultiplier">See <see cref="Execute"/>.</param>
    public static unsafe void ExecuteLatent(
        ReadOnlySpan<float> hidden,
        Span<float> output,
        int seqLen,
        int positionOffset,
        int hiddenSize,
        int numHeads,
        int qkNopeHeadDim,
        int qkRopeHeadDim,
        int vHeadDim,
        int qLoraRank,
        int kvLoraRank,
        float rmsNormEps,
        ReadOnlySpan<float> ropeCosTable,
        ReadOnlySpan<float> ropeSinTable,
        ReadOnlySpan<float> qAProj,
        ReadOnlySpan<float> qALayernormWeight,
        ReadOnlySpan<float> qBProj,
        ReadOnlySpan<float> qProj,
        ReadOnlySpan<float> kvAProjWithMqa,
        ReadOnlySpan<float> kvALayernormWeight,
        ReadOnlySpan<float> kvBProj,
        ReadOnlySpan<float> oProj,
        nint cachedLatent,
        nint cachedKPe,
        int cachedLength,
        float attnScaleMultiplier = 1.0f)
    {
        ValidateArgs(seqLen, hiddenSize, numHeads, qkNopeHeadDim, qkRopeHeadDim, vHeadDim,
                     qLoraRank, kvLoraRank, hidden, output);
        if (cachedLatent == 0 || cachedKPe == 0)
            throw new ArgumentException("ExecuteLatent requires non-zero cachedLatent and cachedKPe.");

        int qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
        int qTotal = numHeads * qkHeadDim;
        int perHeadKvBOut = qkNopeHeadDim + vHeadDim;
        float scale = attnScaleMultiplier / MathF.Sqrt(qkHeadDim);

        // Scratch (managed, per call; native persistent scratch is a later
        // optimisation). We deliberately do NOT allocate kNopeBuf/vBuf — the
        // absorbed path never materialises them.
        float[] qBuf = new float[seqLen * qTotal];
        float[] kPeBuf = new float[seqLen * qkRopeHeadDim];            // new K_pe for seqLen
        float[] compressedKvBuf = new float[seqLen * (kvLoraRank + qkRopeHeadDim)];
        float[] kvLatentNormBuf = new float[seqLen * kvLoraRank];      // new latent for seqLen
        float[] qLatentBuf = qLoraRank > 0 ? new float[seqLen * qLoraRank] : Array.Empty<float>();
        float[] qLatentNormBuf = qLoraRank > 0 ? new float[seqLen * qLoraRank] : Array.Empty<float>();
        float[] qAbsorbedBuf = new float[seqLen * numHeads * kvLoraRank]; // Q_latent for seqLen
        float[] attnOutLatentBuf = new float[seqLen * numHeads * kvLoraRank];
        float[] attnOutBuf = new float[seqLen * numHeads * vHeadDim];

        // ── Q projection (identical to Execute) ─────────────────────────
        for (int t = 0; t < seqLen; t++)
        {
            var hiddenRow = hidden.Slice(t * hiddenSize, hiddenSize);
            var qRow = qBuf.AsSpan(t * qTotal, qTotal);

            if (qLoraRank > 0)
            {
                var latent = qLatentBuf.AsSpan(t * qLoraRank, qLoraRank);
                MatVec(qAProj, hiddenRow, latent, qLoraRank, hiddenSize);
                var latentNorm = qLatentNormBuf.AsSpan(t * qLoraRank, qLoraRank);
                RmsNormScalar(latent, qALayernormWeight, rmsNormEps, latentNorm);
                MatVec(qBProj, latentNorm, qRow, qTotal, qLoraRank);
            }
            else
            {
                MatVec(qProj, hiddenRow, qRow, qTotal, hiddenSize);
            }
        }

        // ── KV down-projection + split (identical to Execute) ───────────
        int compressedKvDim = kvLoraRank + qkRopeHeadDim;
        for (int t = 0; t < seqLen; t++)
        {
            var hiddenRow = hidden.Slice(t * hiddenSize, hiddenSize);
            var compRow = compressedKvBuf.AsSpan(t * compressedKvDim, compressedKvDim);
            MatVec(kvAProjWithMqa, hiddenRow, compRow, compressedKvDim, hiddenSize);

            var latent = compRow.Slice(0, kvLoraRank);
            var kPe = compRow.Slice(kvLoraRank, qkRopeHeadDim);

            var latentNorm = kvLatentNormBuf.AsSpan(t * kvLoraRank, kvLoraRank);
            RmsNormScalar(latent, kvALayernormWeight, rmsNormEps, latentNorm);

            kPe.CopyTo(kPeBuf.AsSpan(t * qkRopeHeadDim, qkRopeHeadDim));
            // NOTE: no kv_b_proj expansion — that's the Phase B win.
        }

        // ── RoPE on Q.rope and shared K_pe (identical to Execute) ───────
        int halfRope = qkRopeHeadDim / 2;
        for (int t = 0; t < seqLen; t++)
        {
            int pos = positionOffset + t;
            var cosRow = ropeCosTable.Slice(pos * halfRope, halfRope);
            var sinRow = ropeSinTable.Slice(pos * halfRope, halfRope);

            for (int h = 0; h < numHeads; h++)
            {
                var qPe = qBuf.AsSpan(
                    t * qTotal + h * qkHeadDim + qkNopeHeadDim,
                    qkRopeHeadDim);
                ApplyRopeNormInPlace(qPe, cosRow, sinRow);
            }

            var kPe = kPeBuf.AsSpan(t * qkRopeHeadDim, qkRopeHeadDim);
            ApplyRopeNormInPlace(kPe, cosRow, sinRow);
        }

        // ── Cache write: append latentNorm + k_pe at offset cachedLength ─
        {
            var dstLatent = new Span<float>(
                (void*)(cachedLatent + (nint)((long)cachedLength * kvLoraRank * sizeof(float))),
                seqLen * kvLoraRank);
            kvLatentNormBuf.AsSpan(0, seqLen * kvLoraRank).CopyTo(dstLatent);

            var dstKPe = new Span<float>(
                (void*)(cachedKPe + (nint)((long)cachedLength * qkRopeHeadDim * sizeof(float))),
                seqLen * qkRopeHeadDim);
            kPeBuf.AsSpan(0, seqLen * qkRopeHeadDim).CopyTo(dstKPe);
        }

        // ── Q absorption: Q_latent[h, t][k] = Σ_j W_UK[h][j][k] · Q_nope[h, t][j]
        // W_UK[h][j][k] lives at kvBProj[(h * perHeadKvBOut + j) * kvLoraRank + k].
        // We iterate (h, t) and accumulate into qAbsorbedBuf.
        for (int h = 0; h < numHeads; h++)
        {
            int wUkBaseRow = h * perHeadKvBOut; // rows [wUkBaseRow .. wUkBaseRow + qkNope) are W_UK[h]
            for (int t = 0; t < seqLen; t++)
            {
                var qNopeH = qBuf.AsSpan(t * qTotal + h * qkHeadDim, qkNopeHeadDim);
                var qAbsH = qAbsorbedBuf.AsSpan(t * numHeads * kvLoraRank + h * kvLoraRank, kvLoraRank);
                qAbsH.Clear();
                // SAXPY accumulation: qAbsH += qNopeH[j] * W_UK[h][j] for each j.
                // TensorPrimitives.MultiplyAdd(wRow, qj, qAbsH, qAbsH) vectorises
                // the inner kvLoraRank-wide loop.
                for (int j = 0; j < qkNopeHeadDim; j++)
                {
                    var wRow = kvBProj.Slice((wUkBaseRow + j) * kvLoraRank, kvLoraRank);
                    TensorPrimitives.MultiplyAdd(wRow, qNopeH[j], qAbsH, qAbsH);
                }
            }
        }

        // ── Absorbed attention ──────────────────────────────────────────
        // score[h, t, s] = Q_latent[h, t] · c_kv[s] + Q_pe[h, t] · k_pe[s]
        // softmax over causal mask (s <= cachedLength + t)
        // out_latent[h, t] = Σ_s softmax · c_kv[s]  (shape [kv_lora_rank])
        int seqKv = cachedLength + seqLen;

        ReadOnlySpan<float> latentReadAll =
            new ReadOnlySpan<float>((void*)cachedLatent, seqKv * kvLoraRank);
        ReadOnlySpan<float> kPeReadAll =
            new ReadOnlySpan<float>((void*)cachedKPe, seqKv * qkRopeHeadDim);

        float[] scores = new float[seqLen * seqKv];
        for (int h = 0; h < numHeads; h++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                var qAbsH = qAbsorbedBuf.AsSpan(t * numHeads * kvLoraRank + h * kvLoraRank, kvLoraRank);
                var qPeH = qBuf.AsSpan(t * qTotal + h * qkHeadDim + qkNopeHeadDim, qkRopeHeadDim);

                int queryPos = cachedLength + t;

                for (int s = 0; s < seqKv; s++)
                {
                    if (s > queryPos)
                    {
                        scores[t * seqKv + s] = float.NegativeInfinity;
                        continue;
                    }
                    var cKvS = latentReadAll.Slice(s * kvLoraRank, kvLoraRank);
                    var kPeS = kPeReadAll.Slice(s * qkRopeHeadDim, qkRopeHeadDim);

                    // Absorbed score = Q_latent · c_kv + Q_pe · k_pe — both vectorised.
                    float dot = TensorPrimitives.Dot(qAbsH, cKvS)
                              + TensorPrimitives.Dot(qPeH, kPeS);

                    scores[t * seqKv + s] = dot * scale;
                }

                SoftmaxRowInPlace(scores.AsSpan(), t, seqKv);

                // Weighted sum over latent — SAXPY via MultiplyAdd.
                var outLatentH = attnOutLatentBuf.AsSpan(
                    t * numHeads * kvLoraRank + h * kvLoraRank, kvLoraRank);
                outLatentH.Clear();
                for (int s = 0; s <= queryPos && s < seqKv; s++)
                {
                    float w = scores[t * seqKv + s];
                    if (w == 0f) continue;
                    var cKvS = latentReadAll.Slice(s * kvLoraRank, kvLoraRank);
                    TensorPrimitives.MultiplyAdd(cKvS, w, outLatentH, outLatentH);
                }
            }
        }

        // ── Expand out_latent via W_UV per head ────────────────────────
        // out[h, t][v] = Σ_k W_UV[h][v][k] · out_latent[h, t][k]
        // W_UV[h] rows are kvBProj[(h * perHeadKvBOut + qkNope + v) * kvLoraRank + k].
        for (int h = 0; h < numHeads; h++)
        {
            int wUvBaseRow = h * perHeadKvBOut + qkNopeHeadDim;
            for (int t = 0; t < seqLen; t++)
            {
                var outLatentH = attnOutLatentBuf.AsSpan(
                    t * numHeads * kvLoraRank + h * kvLoraRank, kvLoraRank);
                var outH = attnOutBuf.AsSpan(t * numHeads * vHeadDim + h * vHeadDim, vHeadDim);
                for (int v = 0; v < vHeadDim; v++)
                {
                    var wRow = kvBProj.Slice((wUvBaseRow + v) * kvLoraRank, kvLoraRank);
                    outH[v] = TensorPrimitives.Dot(wRow, outLatentH);
                }
            }
        }

        // ── o_proj (identical to Execute) ───────────────────────────────
        int oInputDim = numHeads * vHeadDim;
        for (int t = 0; t < seqLen; t++)
        {
            var attnRow = attnOutBuf.AsSpan(t * oInputDim, oInputDim);
            var outRow = output.Slice(t * hiddenSize, hiddenSize);
            MatVec(oProj, attnRow, outRow, hiddenSize, oInputDim);
        }
    }

    /// <summary>
    /// Phase C — hybrid dispatch over the Phase B latent KV-cache. The
    /// persistent storage is identical to <see cref="ExecuteLatent"/>
    /// (<c>c_kv</c> + <c>k_pe</c> per token — the ~7× memory win), but the
    /// attention kernel is selected per call based on <paramref name="seqLen"/>:
    /// <list type="bullet">
    ///   <item><b>Prefill</b> (<c>seqLen &gt; 1</c>): expand the latent rows
    ///     (both newly computed and any historically cached) through
    ///     <c>W_UK</c>/<c>W_UV</c> into a local scratch buffer, then run the
    ///     standard per-head 192-dim MHA loop. The seqKv × seqLen attention
    ///     is compute-bound at prefill, where the 192-dim path is cheaper
    ///     than the 576-dim absorbed form.</item>
    ///   <item><b>Decode</b> (<c>seqLen == 1</c>): delegate to
    ///     <see cref="ExecuteLatent"/> — the absorbed 576-dim MQA-style
    ///     loop that reads the compact latent cache directly
    ///     (bandwidth-bound at decode).</item>
    /// </list>
    /// Mirrors vLLM's production MLA backend dispatch.
    /// </summary>
    /// <remarks>
    /// <b>Cache invariant.</b> Regardless of which path executed prefill,
    /// the on-disk cache holds the latent form (<c>c_kv</c> + <c>k_pe</c>).
    /// A subsequent decode step therefore sees the same latents a pure
    /// Phase B prefill would have written, and can run the absorbed
    /// 576-dim kernel over them without re-expansion. Phase A's
    /// expanded-per-head scratch is local-only here — allocated, used for
    /// the prefill attention loop, and discarded.
    /// </remarks>
    /// <param name="hidden">See <see cref="Execute"/>.</param>
    /// <param name="output">See <see cref="Execute"/>.</param>
    /// <param name="seqLen">See <see cref="Execute"/>.</param>
    /// <param name="positionOffset">See <see cref="Execute"/>.</param>
    /// <param name="hiddenSize">See <see cref="Execute"/>.</param>
    /// <param name="numHeads">See <see cref="Execute"/>.</param>
    /// <param name="qkNopeHeadDim">See <see cref="Execute"/>.</param>
    /// <param name="qkRopeHeadDim">See <see cref="Execute"/>.</param>
    /// <param name="vHeadDim">See <see cref="Execute"/>.</param>
    /// <param name="qLoraRank">See <see cref="Execute"/>.</param>
    /// <param name="kvLoraRank">See <see cref="Execute"/>.</param>
    /// <param name="rmsNormEps">See <see cref="Execute"/>.</param>
    /// <param name="ropeCosTable">See <see cref="Execute"/>.</param>
    /// <param name="ropeSinTable">See <see cref="Execute"/>.</param>
    /// <param name="qAProj">See <see cref="Execute"/>.</param>
    /// <param name="qALayernormWeight">See <see cref="Execute"/>.</param>
    /// <param name="qBProj">See <see cref="Execute"/>.</param>
    /// <param name="qProj">See <see cref="Execute"/>.</param>
    /// <param name="kvAProjWithMqa">See <see cref="Execute"/>.</param>
    /// <param name="kvALayernormWeight">See <see cref="Execute"/>.</param>
    /// <param name="kvBProj">See <see cref="ExecuteLatent"/>.</param>
    /// <param name="oProj">See <see cref="Execute"/>.</param>
    /// <param name="cachedLatent">See <see cref="ExecuteLatent"/>.</param>
    /// <param name="cachedKPe">See <see cref="ExecuteLatent"/>.</param>
    /// <param name="cachedLength">See <see cref="ExecuteLatent"/>.</param>
    /// <param name="attnScaleMultiplier">See <see cref="Execute"/>.</param>
    public static unsafe void ExecuteLatentHybrid(
        ReadOnlySpan<float> hidden,
        Span<float> output,
        int seqLen,
        int positionOffset,
        int hiddenSize,
        int numHeads,
        int qkNopeHeadDim,
        int qkRopeHeadDim,
        int vHeadDim,
        int qLoraRank,
        int kvLoraRank,
        float rmsNormEps,
        ReadOnlySpan<float> ropeCosTable,
        ReadOnlySpan<float> ropeSinTable,
        ReadOnlySpan<float> qAProj,
        ReadOnlySpan<float> qALayernormWeight,
        ReadOnlySpan<float> qBProj,
        ReadOnlySpan<float> qProj,
        ReadOnlySpan<float> kvAProjWithMqa,
        ReadOnlySpan<float> kvALayernormWeight,
        ReadOnlySpan<float> kvBProj,
        ReadOnlySpan<float> oProj,
        nint cachedLatent,
        nint cachedKPe,
        int cachedLength,
        float attnScaleMultiplier = 1.0f)
    {
        // Decode (seqLen == 1): absorbed kernel is the bandwidth-optimal
        // choice. Delegate unchanged — the persistent latent cache is
        // consumed directly.
        if (seqLen == 1)
        {
            ExecuteLatent(
                hidden, output, seqLen, positionOffset, hiddenSize, numHeads,
                qkNopeHeadDim, qkRopeHeadDim, vHeadDim, qLoraRank, kvLoraRank,
                rmsNormEps, ropeCosTable, ropeSinTable,
                qAProj, qALayernormWeight, qBProj, qProj,
                kvAProjWithMqa, kvALayernormWeight, kvBProj, oProj,
                cachedLatent, cachedKPe, cachedLength, attnScaleMultiplier);
            return;
        }

        // Prefill (seqLen > 1): expand-then-MHA path.
        ValidateArgs(seqLen, hiddenSize, numHeads, qkNopeHeadDim, qkRopeHeadDim, vHeadDim,
                     qLoraRank, kvLoraRank, hidden, output);
        if (cachedLatent == 0 || cachedKPe == 0)
            throw new ArgumentException(
                "ExecuteLatentHybrid requires non-zero cachedLatent and cachedKPe.");

        int qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
        int qTotal = numHeads * qkHeadDim;
        int perHeadKvBOut = qkNopeHeadDim + vHeadDim;
        int kvBOutputDim = numHeads * perHeadKvBOut;
        float scale = attnScaleMultiplier / MathF.Sqrt(qkHeadDim);

        // Scratch.
        float[] qBuf = new float[seqLen * qTotal];
        float[] kPeBuf = new float[seqLen * qkRopeHeadDim];
        float[] compressedKvBuf = new float[seqLen * (kvLoraRank + qkRopeHeadDim)];
        float[] kvLatentNormBuf = new float[seqLen * kvLoraRank];
        float[] qLatentBuf = qLoraRank > 0 ? new float[seqLen * qLoraRank] : Array.Empty<float>();
        float[] qLatentNormBuf = qLoraRank > 0 ? new float[seqLen * qLoraRank] : Array.Empty<float>();
        float[] attnOutBuf = new float[seqLen * numHeads * vHeadDim];

        // ── Q projection (identical to ExecuteLatent / Execute) ────────
        for (int t = 0; t < seqLen; t++)
        {
            var hiddenRow = hidden.Slice(t * hiddenSize, hiddenSize);
            var qRow = qBuf.AsSpan(t * qTotal, qTotal);

            if (qLoraRank > 0)
            {
                var latent = qLatentBuf.AsSpan(t * qLoraRank, qLoraRank);
                MatVec(qAProj, hiddenRow, latent, qLoraRank, hiddenSize);
                var latentNorm = qLatentNormBuf.AsSpan(t * qLoraRank, qLoraRank);
                RmsNormScalar(latent, qALayernormWeight, rmsNormEps, latentNorm);
                MatVec(qBProj, latentNorm, qRow, qTotal, qLoraRank);
            }
            else
            {
                MatVec(qProj, hiddenRow, qRow, qTotal, hiddenSize);
            }
        }

        // ── KV down-projection + split (identical to ExecuteLatent) ────
        int compressedKvDim = kvLoraRank + qkRopeHeadDim;
        for (int t = 0; t < seqLen; t++)
        {
            var hiddenRow = hidden.Slice(t * hiddenSize, hiddenSize);
            var compRow = compressedKvBuf.AsSpan(t * compressedKvDim, compressedKvDim);
            MatVec(kvAProjWithMqa, hiddenRow, compRow, compressedKvDim, hiddenSize);

            var latent = compRow.Slice(0, kvLoraRank);
            var kPe = compRow.Slice(kvLoraRank, qkRopeHeadDim);

            var latentNorm = kvLatentNormBuf.AsSpan(t * kvLoraRank, kvLoraRank);
            RmsNormScalar(latent, kvALayernormWeight, rmsNormEps, latentNorm);

            kPe.CopyTo(kPeBuf.AsSpan(t * qkRopeHeadDim, qkRopeHeadDim));
        }

        // ── RoPE on Q.rope and shared K_pe (identical to ExecuteLatent) ─
        int halfRope = qkRopeHeadDim / 2;
        for (int t = 0; t < seqLen; t++)
        {
            int pos = positionOffset + t;
            var cosRow = ropeCosTable.Slice(pos * halfRope, halfRope);
            var sinRow = ropeSinTable.Slice(pos * halfRope, halfRope);

            for (int h = 0; h < numHeads; h++)
            {
                var qPe = qBuf.AsSpan(
                    t * qTotal + h * qkHeadDim + qkNopeHeadDim,
                    qkRopeHeadDim);
                ApplyRopeNormInPlace(qPe, cosRow, sinRow);
            }

            var kPe = kPeBuf.AsSpan(t * qkRopeHeadDim, qkRopeHeadDim);
            ApplyRopeNormInPlace(kPe, cosRow, sinRow);
        }

        // ── Cache write: append latentNorm + k_pe at offset cachedLength.
        // Same on-disk layout as ExecuteLatent — a subsequent decode step
        // will consume exactly what a pure-Phase-B prefill would have
        // written.
        {
            var dstLatent = new Span<float>(
                (void*)(cachedLatent + (nint)((long)cachedLength * kvLoraRank * sizeof(float))),
                seqLen * kvLoraRank);
            kvLatentNormBuf.AsSpan(0, seqLen * kvLoraRank).CopyTo(dstLatent);

            var dstKPe = new Span<float>(
                (void*)(cachedKPe + (nint)((long)cachedLength * qkRopeHeadDim * sizeof(float))),
                seqLen * qkRopeHeadDim);
            kPeBuf.AsSpan(0, seqLen * qkRopeHeadDim).CopyTo(dstKPe);
        }

        // ── Expand ALL seqKv latent rows into per-head K_nope/V scratch ─
        // The cache now holds cachedLength + seqLen latent rows. We expand
        // every row through kv_b_proj once so the attention loop below
        // reads the same [seqKv, numHeads*qkNope] / [seqKv, numHeads*vHead]
        // layouts Phase A operates on. The expanded scratch is THROWN AWAY
        // at the end of this call — the persistent cache stays latent.
        int seqKv = cachedLength + seqLen;
        float[] kNopeExpanded = new float[seqKv * numHeads * qkNopeHeadDim];
        float[] vExpanded = new float[seqKv * numHeads * vHeadDim];

        ReadOnlySpan<float> latentReadAll =
            new ReadOnlySpan<float>((void*)cachedLatent, seqKv * kvLoraRank);
        ReadOnlySpan<float> kPeReadAll =
            new ReadOnlySpan<float>((void*)cachedKPe, seqKv * qkRopeHeadDim);

        {
            float[] kvBExpandedRowBuf = new float[kvBOutputDim];
            for (int s = 0; s < seqKv; s++)
            {
                var latentRow = latentReadAll.Slice(s * kvLoraRank, kvLoraRank);
                MatVec(kvBProj, latentRow, kvBExpandedRowBuf, kvBOutputDim, kvLoraRank);

                for (int h = 0; h < numHeads; h++)
                {
                    var headBlock = kvBExpandedRowBuf.AsSpan(h * perHeadKvBOut, perHeadKvBOut);
                    headBlock.Slice(0, qkNopeHeadDim)
                             .CopyTo(kNopeExpanded.AsSpan(
                                 s * numHeads * qkNopeHeadDim + h * qkNopeHeadDim,
                                 qkNopeHeadDim));
                    headBlock.Slice(qkNopeHeadDim, vHeadDim)
                             .CopyTo(vExpanded.AsSpan(
                                 s * numHeads * vHeadDim + h * vHeadDim,
                                 vHeadDim));
                }
            }
        }

        // ── Standard per-head MHA attention on the expanded scratch ─────
        // Identical math to MlaAttention.Execute's attention loop, now
        // reading from the locally-expanded kNopeExpanded / vExpanded
        // instead of Phase A's persistent expanded cache.
        int queryPosBase = cachedLength;
        float[] scores = new float[seqLen * seqKv];
        for (int h = 0; h < numHeads; h++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                var qNopeH = qBuf.AsSpan(t * qTotal + h * qkHeadDim, qkNopeHeadDim);
                var qPeH = qBuf.AsSpan(t * qTotal + h * qkHeadDim + qkNopeHeadDim, qkRopeHeadDim);

                int queryPos = queryPosBase + t;

                for (int s = 0; s < seqKv; s++)
                {
                    if (s > queryPos)
                    {
                        scores[t * seqKv + s] = float.NegativeInfinity;
                        continue;
                    }
                    var kNopeH = kNopeExpanded.AsSpan(
                        s * numHeads * qkNopeHeadDim + h * qkNopeHeadDim, qkNopeHeadDim);
                    var kPeS = kPeReadAll.Slice(s * qkRopeHeadDim, qkRopeHeadDim);

                    float dot = TensorPrimitives.Dot(qNopeH, kNopeH)
                              + TensorPrimitives.Dot(qPeH, kPeS);
                    scores[t * seqKv + s] = dot * scale;
                }

                SoftmaxRowInPlace(scores.AsSpan(), t, seqKv);

                var outH = attnOutBuf.AsSpan(t * numHeads * vHeadDim + h * vHeadDim, vHeadDim);
                outH.Clear();
                for (int s = 0; s <= queryPos && s < seqKv; s++)
                {
                    float w = scores[t * seqKv + s];
                    if (w == 0f) continue;
                    var vH = vExpanded.AsSpan(
                        s * numHeads * vHeadDim + h * vHeadDim, vHeadDim);
                    TensorPrimitives.MultiplyAdd(vH, w, outH, outH);
                }
            }
        }

        // ── o_proj (identical to Execute / ExecuteLatent) ──────────────
        int oInputDim = numHeads * vHeadDim;
        for (int t = 0; t < seqLen; t++)
        {
            var attnRow = attnOutBuf.AsSpan(t * oInputDim, oInputDim);
            var outRow = output.Slice(t * hiddenSize, hiddenSize);
            MatVec(oProj, attnRow, outRow, hiddenSize, oInputDim);
        }
    }

    /// <summary>
    /// Standard <c>y = W @ x</c> matvec. <c>W</c> is row-major with shape
    /// <c>[m, k]</c>, <c>x</c> has length <c>k</c>, <c>y</c> has length <c>m</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MatVec(
        ReadOnlySpan<float> w, ReadOnlySpan<float> x, Span<float> y, int m, int k)
    {
        for (int i = 0; i < m; i++)
            y[i] = TensorPrimitives.Dot(w.Slice(i * k, k), x);
    }

    /// <summary>
    /// Scalar RMSNorm: <c>y[i] = (x[i] / sqrt(mean(x²) + eps)) * weight[i]</c>.
    /// Kept inline here to keep the MLA kernel standalone from the public
    /// <see cref="RmsNorm"/> kernel while we iterate on correctness.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RmsNormScalar(
        ReadOnlySpan<float> input, ReadOnlySpan<float> weight, float epsilon, Span<float> output)
    {
        float sumSq = 0f;
        for (int i = 0; i < input.Length; i++)
            sumSq += input[i] * input[i];
        float rms = MathF.Sqrt(sumSq / input.Length + epsilon);
        float scale = 1.0f / rms;
        for (int i = 0; i < input.Length; i++)
            output[i] = input[i] * scale * weight[i];
    }

    /// <summary>
    /// Applies rotary-pair RoPE in place using the "Norm" (Llama) convention:
    /// element pairs are <c>(v[2i], v[2i+1])</c> and rotate as
    /// <c>v'[2i]   = v[2i]   * cos - v[2i+1] * sin</c>,
    /// <c>v'[2i+1] = v[2i+1] * cos + v[2i]   * sin</c>.
    /// DeepSeek-V2 uses the same paired convention (HF <c>apply_rotary_pos_emb_mla</c>
    /// operates on adjacent pairs via <c>rotate_half_mla</c>). Length must be even.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplyRopeNormInPlace(
        Span<float> vec, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        int half = vec.Length / 2;
        for (int i = 0; i < half; i++)
        {
            float a = vec[2 * i];
            float b = vec[2 * i + 1];
            float c = cos[i];
            float s = sin[i];
            vec[2 * i] = a * c - b * s;
            vec[2 * i + 1] = b * c + a * s;
        }
    }

    /// <summary>
    /// Numerically stable softmax of one row of a [seqLen, seqKv] score matrix,
    /// in place.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SoftmaxRowInPlace(Span<float> scores, int rowIdx, int seqKv)
    {
        var row = scores.Slice(rowIdx * seqKv, seqKv);
        float max = float.NegativeInfinity;
        for (int j = 0; j < row.Length; j++)
            if (row[j] > max) max = row[j];
        float sum = 0f;
        for (int j = 0; j < row.Length; j++)
        {
            float e = MathF.Exp(row[j] - max);
            row[j] = e;
            sum += e;
        }
        float inv = sum > 0f ? 1f / sum : 0f;
        for (int j = 0; j < row.Length; j++)
            row[j] *= inv;
    }

    private static void ValidateArgs(
        int seqLen, int hiddenSize, int numHeads,
        int qkNopeHeadDim, int qkRopeHeadDim, int vHeadDim,
        int qLoraRank, int kvLoraRank,
        ReadOnlySpan<float> hidden, Span<float> output)
    {
        if (seqLen <= 0) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (hiddenSize <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
        if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
        if (qkNopeHeadDim < 0) throw new ArgumentOutOfRangeException(nameof(qkNopeHeadDim));
        if (qkRopeHeadDim <= 0 || qkRopeHeadDim % 2 != 0)
            throw new ArgumentException(
                $"qkRopeHeadDim must be positive and even, got {qkRopeHeadDim}", nameof(qkRopeHeadDim));
        if (vHeadDim <= 0) throw new ArgumentOutOfRangeException(nameof(vHeadDim));
        if (qLoraRank < 0) throw new ArgumentOutOfRangeException(nameof(qLoraRank));
        if (kvLoraRank <= 0) throw new ArgumentOutOfRangeException(nameof(kvLoraRank));
        if (hidden.Length < seqLen * hiddenSize)
            throw new ArgumentException(
                $"hidden has {hidden.Length} elements, need seqLen * hiddenSize = {seqLen * hiddenSize}",
                nameof(hidden));
        if (output.Length < seqLen * hiddenSize)
            throw new ArgumentException(
                $"output has {output.Length} elements, need seqLen * hiddenSize = {seqLen * hiddenSize}",
                nameof(output));
    }
}
