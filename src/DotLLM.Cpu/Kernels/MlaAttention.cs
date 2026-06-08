using System.Numerics.Tensors;
using System.Runtime.CompilerServices;

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
/// no YaRN mscale correction. This implementation targets correctness
/// against a Python / HF reference.
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
    public static void Execute(
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
        ReadOnlySpan<float> oProj)
    {
        ValidateArgs(seqLen, hiddenSize, numHeads, qkNopeHeadDim, qkRopeHeadDim, vHeadDim,
                     qLoraRank, kvLoraRank, hidden, output);

        int qkHeadDim = qkNopeHeadDim + qkRopeHeadDim;
        int qTotal = numHeads * qkHeadDim;
        int kvBOutputDim = numHeads * (qkNopeHeadDim + vHeadDim);
        float scale = 1.0f / MathF.Sqrt(qkHeadDim);

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
        for (int t = 0; t < seqLen; t++)
        {
            var hiddenRow = hidden.Slice(t * hiddenSize, hiddenSize);
            var qRow = qBuf.AsSpan(t * qTotal, qTotal);

            if (qLoraRank > 0)
            {
                // q_latent = q_a_proj @ hidden
                var latent = qLatentBuf.AsSpan(t * qLoraRank, qLoraRank);
                MatVec(qAProj, hiddenRow, latent, qLoraRank, hiddenSize);

                // q_latent_norm = RMSNorm(q_latent, q_a_layernorm)
                var latentNorm = qLatentNormBuf.AsSpan(t * qLoraRank, qLoraRank);
                RmsNormScalar(latent, qALayernormWeight, rmsNormEps, latentNorm);

                // q = q_b_proj @ q_latent_norm
                MatVec(qBProj, latentNorm, qRow, qTotal, qLoraRank);
            }
            else
            {
                // q = q_proj @ hidden (monolithic path)
                MatVec(qProj, hiddenRow, qRow, qTotal, hiddenSize);
            }
        }

        // KV down-projection + split
        int compressedKvDim = kvLoraRank + qkRopeHeadDim;
        for (int t = 0; t < seqLen; t++)
        {
            var hiddenRow = hidden.Slice(t * hiddenSize, hiddenSize);
            var compRow = compressedKvBuf.AsSpan(t * compressedKvDim, compressedKvDim);
            MatVec(kvAProjWithMqa, hiddenRow, compRow, compressedKvDim, hiddenSize);

            // Split: first kvLoraRank = k_nope_latent, next qkRopeHeadDim = k_pe
            var latent = compRow.Slice(0, kvLoraRank);
            var kPe = compRow.Slice(kvLoraRank, qkRopeHeadDim);

            // k_nope_latent = RMSNorm(k_nope_latent, kv_a_layernorm)
            var latentNorm = kvLatentNormBuf.AsSpan(t * kvLoraRank, kvLoraRank);
            RmsNormScalar(latent, kvALayernormWeight, rmsNormEps, latentNorm);

            // kv_b_expanded = kv_b_proj @ latentNorm  (size = numHeads * (qkNope + vHead))
            var expandedRow = kvBExpanded.AsSpan(t * kvBOutputDim, kvBOutputDim);
            MatVec(kvBProj, latentNorm, expandedRow, kvBOutputDim, kvLoraRank);

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

        // Attention per head with causal mask
        // Q_h[t] = concat(q_nope_h[t], q_pe_h[t]) — already adjacent in qBuf
        // K_h[s] = concat(k_nope_h[s], k_pe_shared[s])
        // V_h[s] (width vHeadDim)
        // Score[t, s] = Q_h[t] . K_h[s] * scale
        // Mask: s <= positionOffset + t
        //   Output per head at t: softmax(score[t, :]) . V_h[:]

        // We compute attention with the Q/K/V in place — no cache for this PoC.
        // Layout assumption for self-attention prefill: seqKv = seqLen.
        int seqKv = seqLen;

        // Scratch scores reused across all heads.
        float[] scores = new float[seqLen * seqKv];
        for (int h = 0; h < numHeads; h++)
        {

            for (int t = 0; t < seqLen; t++)
            {
                // Build Q vector for head h at query position t
                var qNopeH = qBuf.AsSpan(t * qTotal + h * qkHeadDim, qkNopeHeadDim);
                var qPeH = qBuf.AsSpan(t * qTotal + h * qkHeadDim + qkNopeHeadDim, qkRopeHeadDim);

                for (int s = 0; s < seqKv; s++)
                {
                    // Causal mask: s > positionOffset + t → -inf
                    if (s > positionOffset + t)
                    {
                        scores[t * seqKv + s] = float.NegativeInfinity;
                        continue;
                    }

                    // K_h[s] = concat(k_nope_h[s], k_pe_shared[s])
                    var kNopeH = kNopeBuf.AsSpan(
                        s * numHeads * qkNopeHeadDim + h * qkNopeHeadDim,
                        qkNopeHeadDim);
                    var kPeS = kPeBuf.AsSpan(s * qkRopeHeadDim, qkRopeHeadDim);

                    float dot = 0f;
                    for (int d = 0; d < qkNopeHeadDim; d++)
                        dot += qNopeH[d] * kNopeH[d];
                    for (int d = 0; d < qkRopeHeadDim; d++)
                        dot += qPeH[d] * kPeS[d];

                    scores[t * seqKv + s] = dot * scale;
                }

                // Softmax row t
                SoftmaxRowInPlace(scores.AsSpan(), t, seqKv);

                // Weighted sum over V_h
                var outH = attnOutBuf.AsSpan(t * numHeads * vHeadDim + h * vHeadDim, vHeadDim);
                outH.Clear();
                for (int s = 0; s <= positionOffset + t && s < seqKv; s++)
                {
                    float w = scores[t * seqKv + s];
                    if (w == 0f) continue;
                    var vH = vBuf.AsSpan(s * numHeads * vHeadDim + h * vHeadDim, vHeadDim);
                    for (int d = 0; d < vHeadDim; d++)
                        outH[d] += w * vH[d];
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
