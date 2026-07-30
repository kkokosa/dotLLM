using System.Runtime.InteropServices;

namespace DotLLM.Metal.Interop;

internal static partial class MetalNative
{
    /// <summary>Element-wise addition (FP32): result[i] = a[i] + b[i]</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_add_f32")]
    internal static unsafe partial int AddF32(
        nint ctx,
        float* a,
        float* b,
        float* result,
        uint length);

    /// <summary>Element-wise addition (FP16): result[i] = a[i] + b[i]</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_add_f16")]
    internal static unsafe partial int AddF16(
        nint ctx,
        ushort* a,
        ushort* b,
        ushort* result,
        uint length);

    /// <summary>Mixed-precision addition: result_f32[i] = a_f32[i] + b_f16[i]</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_add_f32_f16")]
    internal static unsafe partial int AddF32F16(
        nint ctx,
        float*  a,
        ushort* b,
        float*  result,
        uint length);

    /// <summary>Numerically stable softmax (FP16 I/O). One threadgroup per row.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_softmax_f16")]
    internal static unsafe partial int SoftmaxF16(
        nint    ctx,
        ushort* input,
        ushort* output,
        int     rows,
        int     cols);

    /// <summary>SwiGLU (FP32): result[i] = SiLU(gate[i]) * up[i]</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_swiglu_f32")]
    internal static unsafe partial int SwigluF32(
        nint ctx,
        float* gate,
        float* up,
        float* result,
        uint   length);

    /// <summary>SwiGLU (FP16): result[i] = SiLU(gate[i]) * up[i], vectorized half2</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_swiglu_f16")]
    internal static unsafe partial int SwigluF16(
        nint    ctx,
        ushort* gate,
        ushort* up,
        ushort* result,
        uint    length);

    /// <summary>
    /// Rotary Position Embedding (in-place). Direct translation of rope_f32.cu.
    /// ropeType: 0 = norm/interleaved (Llama/Mistral), 1 = neox/split (Qwen/Phi).
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_rope_f32")]
    internal static unsafe partial int RoPEF32(
        nint   ctx,
        float* q,
        float* k,
        int*   positions,
        int    seqLen,
        int    numHeads,
        int    numKvHeads,
        int    headDim,
        int    ropeDim,
        float  theta,
        int    ropeType);

    /// <summary>
    /// Rotary Position Embedding (in-place, FP16). FP32 accumulation internally.
    /// ropeType: 0 = norm/interleaved (Llama/Mistral), 1 = neox/split (Qwen/Phi).
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_rope_f16")]
    internal static unsafe partial int RoPEF16(
        nint    ctx,
        ushort* q,
        ushort* k,
        int*    positions,
        int     seqLen,
        int     numHeads,
        int     numKvHeads,
        int     headDim,
        int     ropeDim,
        float   theta,
        int     ropeType);

    /// <summary>Bias addition (in-place, FP32 output + FP16 bias): output[t, i] += float(bias[i])</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_bias_add_f32")]
    internal static unsafe partial int BiasAddF32(
        nint ctx,
        float*  output,
        ushort* bias,       // FP16
        uint    dim,
        uint    seqLen);

    /// <summary>Bias addition (in-place, FP16 output + FP16 bias): output[t, i] += bias[i]</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_bias_add_f16")]
    internal static unsafe partial int BiasAddF16(
        nint    ctx,
        ushort* output,     // FP16, in-place
        ushort* bias,       // FP16
        uint    dim,
        uint    seqLen);

    /// <summary>
    /// RMS Normalization (FP32): output[t, i] = input[t, i] / rms(input[t]) * weight[i].
    /// Reduction kernel — one threadgroup per token, NOT element-wise.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_rmsnorm_f32")]
    internal static unsafe partial int RmsNormF32(
        nint   ctx,
        float* input,
        float* weight,
        float* output,
        int    n,
        int    seqLen,
        float  eps);

    /// <summary>
    /// RMS Normalization (FP16): same formula as F32, FP16 I/O with FP32 accumulation.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_rmsnorm_f16")]
    internal static unsafe partial int RmsNormF16(
        nint    ctx,
        ushort* input,
        ushort* weight,
        ushort* output,
        int     n,
        int     seqLen,
        float   eps);

    /// <summary>
    /// Per-head RMS Normalization (in-place).
    /// For each (token, head), normalizes head_dim elements: vec = vec / rms(vec) * weight.
    /// One threadgroup per (token × head).
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_per_head_rmsnorm_f32")]
    internal static unsafe partial int PerHeadRmsNormF32(
        nint   ctx,
        float* qk,
        float* weight,
        int    numHeads,
        int    headDim,
        int    seqLen,
        float  eps);

    /// <summary>
    /// Per-head RMS Normalization (FP16, in-place). FP16 I/O with FP32 accumulation.
    /// For each (token, head), normalizes head_dim elements in-place.
    /// One threadgroup per (token × head).
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_per_head_rmsnorm_f16")]
    internal static unsafe partial int PerHeadRmsNormF16(
        nint    ctx,
        ushort* qk,
        ushort* weight,
        int     numHeads,
        int     headDim,
        int     seqLen,
        float   eps);

    /// <summary>
    /// RMS Normalization — FP32 residual input, FP32 weight, FP16 output.
    /// Port of rmsnorm_f32in.cu::rmsnorm_f32in_f16out.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_rmsnorm_f32in_f16out")]
    internal static unsafe partial int RmsNormF32InF16Out(
        nint    ctx,
        float*  input,
        float*  weight,
        ushort* output,
        int     n,
        int     seqLen,
        float   eps);

    /// <summary>
    /// Fused residual-add + RMS normalization (FP16 I/O).
    /// residual is updated in-place (sum written back as f16); output receives the normalized result.
    /// All half buffers passed as ushort* (same bit layout as System.Half).
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_fused_add_rmsnorm_f16")]
    internal static unsafe partial int FusedAddRmsNormF16(
        nint    ctx,
        ushort* residual,
        ushort* x,
        ushort* weight,
        ushort* output,
        int     n,
        int     seqLen,
        float   eps);

    // ── Embedding lookup ──────────────────────────────────────────────────────

    /// <summary>Embedding lookup: FP32 table → FP32 output.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_embedding_f32_f32out")]
    internal static unsafe partial int EmbeddingF32F32Out(
        nint   ctx,
        float* embedTable,
        int*   tokenIds,
        float* output,
        int    vocabSize,
        int    hiddenSize,
        int    seqLen);

    /// <summary>Embedding lookup: FP16 table → FP32 output. embedTable passed as ushort*.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_embedding_f16_f32out")]
    internal static unsafe partial int EmbeddingF16F32Out(
        nint    ctx,
        ushort* embedTable,
        int*    tokenIds,
        float*  output,
        int     vocabSize,
        int     hiddenSize,
        int     seqLen);

    /// <summary>Embedding lookup: Q8_0 quantized table → FP32 output (dequantize on the fly).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_embedding_q8_0_f32out")]
    internal static unsafe partial int EmbeddingQ8_0F32Out(
        nint   ctx,
        byte*  embedTable,
        int*   tokenIds,
        float* output,
        int    vocabSize,
        int    hiddenSize,
        int    seqLen);

    /// <summary>Embedding lookup: FP32 table → FP16 output (cast on the fly).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_embedding_f32_f16out")]
    internal static unsafe partial int EmbeddingF32F16Out(
        nint    ctx,
        float*  embedTable,
        int*    tokenIds,
        ushort* output,
        int     vocabSize,
        int     hiddenSize,
        int     seqLen);

    /// <summary>Embedding lookup: FP16 table → FP16 output (straight copy of half rows).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_embedding_f16_f16out")]
    internal static unsafe partial int EmbeddingF16F16Out(
        nint    ctx,
        ushort* embedTable,
        int*    tokenIds,
        ushort* output,
        int     vocabSize,
        int     hiddenSize,
        int     seqLen);

    /// <summary>Embedding lookup: Q8_0 quantized table → FP16 output (dequantize on the fly).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_embedding_q8_0_f16out")]
    internal static unsafe partial int EmbeddingQ8_0F16Out(
        nint    ctx,
        byte*   embedTable,
        int*    tokenIds,
        ushort* output,
        int     vocabSize,
        int     hiddenSize,
        int     seqLen);

    /// <summary>Embedding lookup: Q6_K quantized table → FP16 output (dequantize on the fly).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_embedding_q6_k_f16out")]
    internal static unsafe partial int EmbeddingQ6_KF16Out(
        nint    ctx,
        byte*   embedTable,
        int*    tokenIds,
        ushort* output,
        int     vocabSize,
        int     hiddenSize,
        int     seqLen);

    /// <summary>FP16 attention using persistent K/V MTLBuffers from the given cache layer.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_attention_f16_kvcache")]
    internal static unsafe partial int AttentionF16WithKvCache(
        nint    ctx,
        nint    cache,
        ushort* q,
        ushort* output,
        int     layer,
        int     seqQ,
        int     numHeads,
        int     numKvHeads,
        int     headDim,
        int     positionOffset,
        int     slidingWindow);

    // ── Attention ─────────────────────────────────────────────────────────────────

    /// <summary>FP16 Q/K/V/output attention. Buffers passed as ushort* (same layout as Half).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_attention_f16")]
    internal static unsafe partial int AttentionF16(
        nint    ctx,
        ushort* q,
        ushort* k,
        ushort* v,
        ushort* output,
        int     seqQ,
        int     seqKv,
        int     numHeads,
        int     numKvHeads,
        int     headDim,
        int     positionOffset,
        int     slidingWindow);

    /// <summary>FP32 Q/K/V/output attention.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_attention_f32")]
    internal static unsafe partial int AttentionF32(
        nint   ctx,
        float* q,
        float* k,
        float* v,
        float* output,
        int    seqQ,
        int    seqKv,
        int    numHeads,
        int    numKvHeads,
        int    headDim,
        int    positionOffset,
        int    slidingWindow);

    // ── KV-cache quantization ─────────────────────────────────────────────────────

    /// <summary>FP16 → Q8_0. src: total_blocks × 32 halves (as ushort). dst: total_blocks × 34 bytes.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_quant_f16_to_q8_0")]
    internal static unsafe partial int QuantF16ToQ8_0(
        nint    ctx,
        ushort* src,
        byte*   dst,
        int     totalBlocks);

    /// <summary>FP16 → Q4_0. src: total_blocks × 32 halves (as ushort). dst: total_blocks × 18 bytes.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_quant_f16_to_q4_0")]
    internal static unsafe partial int QuantF16ToQ4_0(
        nint    ctx,
        ushort* src,
        byte*   dst,
        int     totalBlocks);

    // ── Quantized GEMV ───────────────────────────────────────────────────────────

    /// <summary>
    /// Quantized GEMV: y[i] = dot(dequant(W_q8_0[i,:]), x) for i in [0, n).
    /// weight layout: n × (k/32) blocks × 34 bytes. k must be a multiple of 32.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_quantized_gemv_q8_0_f32in")]
    internal static unsafe partial int QuantizedGemvQ8_0F32In(
        nint   ctx,
        byte*  weight,
        float* x,
        float* y,
        int    n,
        int    k);

    /// <summary>Quantized GEMV Q8_0 (FP16 I/O). weight: n × (k/32) × 34 bytes. k multiple of 32.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_quantized_gemv_q8_0")]
    internal static unsafe partial int QuantizedGemvQ8_0(
        nint    ctx,
        byte*   weight,
        ushort* x,
        ushort* y,
        int     n,
        int     k);

    /// <summary>Quantized GEMV Q5_0 (FP16 I/O). weight: n × (k/32) × 22 bytes. k multiple of 32.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_quantized_gemv_q5_0")]
    internal static unsafe partial int QuantizedGemvQ5_0(
        nint    ctx,
        byte*   weight,
        ushort* x,
        ushort* y,
        int     n,
        int     k);

    /// <summary>Quantized GEMV Q4_K (FP16 I/O). weight: n × (k/256) × 144 bytes. k multiple of 256.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_quantized_gemv_q4_k")]
    internal static unsafe partial int QuantizedGemvQ4_K(
        nint    ctx,
        byte*   weight,
        ushort* x,
        ushort* y,
        int     n,
        int     k);

    /// <summary>Quantized GEMV Q5_K (FP16 I/O). weight: n × (k/256) × 176 bytes. k multiple of 256.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_quantized_gemv_q5_k")]
    internal static unsafe partial int QuantizedGemvQ5_K(
        nint    ctx,
        byte*   weight,
        ushort* x,
        ushort* y,
        int     n,
        int     k);

    /// <summary>Quantized GEMV Q6_K (FP16 I/O). weight: n × (k/256) × 210 bytes. k multiple of 256.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_quantized_gemv_q6_k")]
    internal static unsafe partial int QuantizedGemvQ6_K(
        nint    ctx,
        byte*   weight,
        ushort* x,
        ushort* y,
        int     n,
        int     k);

    // ── Dequantization ────────────────────────────────────────────────────────────

    /// <summary>Dequantize Q8_0 → FP16. src: total_blocks × 34 bytes. dst: total_blocks × 32 halves (as ushort).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_dequant_q8_0_f16")]
    internal static unsafe partial int DequantQ8_0F16(
        nint    ctx,
        byte*   src,
        ushort* dst,
        int     totalBlocks);

    /// <summary>Dequantize Q4_0 → FP16. src: total_blocks × 18 bytes. dst: total_blocks × 32 halves (as ushort).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_dequant_q4_0_f16")]
    internal static unsafe partial int DequantQ4_0F16(
        nint    ctx,
        byte*   src,
        ushort* dst,
        int     totalBlocks);

    /// <summary>Dequantize Q5_0 → FP16. src: total_blocks × 22 bytes. dst: total_blocks × 32 halves (as ushort).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_dequant_q5_0_f16")]
    internal static unsafe partial int DequantQ5_0F16(
        nint    ctx,
        byte*   src,
        ushort* dst,
        int     totalBlocks);

    /// <summary>Dequantize Q4_K → FP16. src: total_superblocks × 144 bytes. dst: total_superblocks × 256 halves (as ushort).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_dequant_q4_k_f16")]
    internal static unsafe partial int DequantQ4_KF16(
        nint    ctx,
        byte*   src,
        ushort* dst,
        int     totalSuperblocks);

    /// <summary>Dequantize Q5_K → FP16. src: total_superblocks × 176 bytes. dst: total_superblocks × 256 halves (as ushort).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_dequant_q5_k_f16")]
    internal static unsafe partial int DequantQ5_KF16(
        nint    ctx,
        byte*   src,
        ushort* dst,
        int     totalSuperblocks);

    /// <summary>Dequantize Q6_K → FP16. src: total_superblocks × 210 bytes. dst: total_superblocks × 256 halves (as ushort).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_dequant_q6_k_f16")]
    internal static unsafe partial int DequantQ6_KF16(
        nint    ctx,
        byte*   src,
        ushort* dst,
        int     totalSuperblocks);

    /// <summary>Converts n float16 values to float32. src is passed as ushort* (same bit layout as Half).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_convert_f16_to_f32")]
    internal static unsafe partial int ConvertF16ToF32(
        nint    ctx,
        ushort* src,
        float*  dst,
        int     n);

    /// <summary>Converts n float32 values to float16. dst is passed as ushort* (same bit layout as Half).</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_convert_f32_to_f16")]
    internal static unsafe partial int ConvertF32ToF16(
        nint    ctx,
        float*  src,
        ushort* dst,
        int     n);

    // ── GEMM (Metal Performance Shaders) ────────────────────────────────────────

    /// <summary>
    /// FP16 GEMM via MPSMatrixMultiplication: C = alpha * op(A) * op(B) + beta * C.
    /// Buffers passed as ushort* (same bit layout as Half).
    /// transposeA / transposeB: pass 1 to transpose, 0 otherwise.
    /// </summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_gemm_f16")]
    internal static unsafe partial int GemmF16(
        nint    ctx,
        ushort* a,
        ushort* b,
        ushort* c,
        int     m,
        int     n,
        int     k,
        int     transposeA,
        int     transposeB,
        float   alpha,
        float   beta);

    /// <summary>FP32 GEMM via MPSMatrixMultiplication: C = alpha * op(A) * op(B) + beta * C.</summary>
    [LibraryImport(LibName, EntryPoint = "dotllm_metal_gemm_f32")]
    internal static unsafe partial int GemmF32(
        nint    ctx,
        float*  a,
        float*  b,
        float*  c,
        int     m,
        int     n,
        int     k,
        int     transposeA,
        int     transposeB,
        float   alpha,
        float   beta);
}
