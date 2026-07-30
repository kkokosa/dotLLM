using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DotLLM.Diagnostics;

/// <summary>
/// Concrete <see cref="ISparseAutoencoder"/> backed by unmanaged-memory weight tensors.
/// </summary>
/// <remarks>
/// <para>
/// Implements the standard SAE forward pass documented in <c>docs/DIAGNOSTICS.md</c>:
/// </para>
/// <code>
/// pre = activation - b_dec        (when ApplyBDecToInput is true; SAELens convention)
/// features = relu(pre @ W_enc + b_enc)        // shape [d_sae]
/// active = top-K features by magnitude
/// reconstruction = (active features @ W_dec rows) + b_dec
/// </code>
/// <para>
/// Weight tensors (<c>W_enc</c> <c>[d_in, d_sae]</c>, <c>b_enc</c> <c>[d_sae]</c>,
/// <c>W_dec</c> <c>[d_sae, d_in]</c>, <c>b_dec</c> <c>[d_in]</c>) are stored in 64-byte-aligned
/// unmanaged memory via <see cref="NativeMemory.AlignedAlloc"/>, mirroring the project's
/// tensor-memory rules (no GC pressure, zero managed copies of weights).
/// </para>
/// <para>
/// The <see cref="Encode"/> method returns a managed <see cref="int"/>[] / <see cref="float"/>[]
/// pair sized to <see cref="TopK"/>; allocations are bounded and the hook only fires when
/// explicitly registered, so per-encode managed allocation is intentional.
/// </para>
/// </remarks>
public sealed unsafe class SparseAutoencoder : ISparseAutoencoder, IDisposable
{
    private float* _wEnc;   // [d_in, d_sae] row-major
    private float* _bEnc;   // [d_sae]
    private float* _wDec;   // [d_sae, d_in] row-major
    private float* _bDec;   // [d_in]
    private bool _disposed;

    /// <inheritdoc/>
    public int HiddenSize { get; }

    /// <inheritdoc/>
    public int FeatureCount { get; }

    /// <summary>
    /// Number of top-magnitude features to return from <see cref="Encode"/>. Clamped to
    /// <see cref="FeatureCount"/>. May be reassigned between encode calls.
    /// </summary>
    public int TopK { get; set; }

    /// <summary>
    /// When <c>true</c>, the decoder bias is subtracted from the activation before encoding —
    /// the SAELens convention for SAEs trained with <c>apply_b_dec_to_input=true</c>.
    /// When <c>false</c>, the encoder sees the raw activation.
    /// </summary>
    public bool ApplyBDecToInput { get; }

    /// <summary>
    /// Constructs an SAE from caller-owned weight spans, copying them into 64-byte-aligned
    /// unmanaged memory. Intended for tests and synthetic fixtures; production callers use
    /// <see cref="SaeLoader"/>.
    /// </summary>
    /// <param name="hiddenSize"><c>d_in</c> — residual-stream width.</param>
    /// <param name="featureCount"><c>d_sae</c> — dictionary size.</param>
    /// <param name="wEnc">Encoder weights, length <c>d_in * d_sae</c>, row-major <c>[d_in, d_sae]</c>.</param>
    /// <param name="bEnc">Encoder bias, length <c>d_sae</c>.</param>
    /// <param name="wDec">Decoder weights, length <c>d_sae * d_in</c>, row-major <c>[d_sae, d_in]</c>.</param>
    /// <param name="bDec">Decoder bias, length <c>d_in</c>.</param>
    /// <param name="topK">Number of top features to return from <see cref="Encode"/>. Defaults to 32.</param>
    /// <param name="applyBDecToInput">Pre-subtract <paramref name="bDec"/> in <see cref="Encode"/>. Defaults to <c>false</c>.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when sizes are non-positive.</exception>
    /// <exception cref="ArgumentException">Thrown when any weight span has the wrong length.</exception>
    public SparseAutoencoder(
        int hiddenSize,
        int featureCount,
        ReadOnlySpan<float> wEnc,
        ReadOnlySpan<float> bEnc,
        ReadOnlySpan<float> wDec,
        ReadOnlySpan<float> bDec,
        int topK = 32,
        bool applyBDecToInput = false)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(hiddenSize, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(featureCount, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(topK, 1);

        if (wEnc.Length != hiddenSize * featureCount)
            throw new ArgumentException(
                $"W_enc must have length d_in*d_sae = {hiddenSize * featureCount}, got {wEnc.Length}.", nameof(wEnc));
        if (bEnc.Length != featureCount)
            throw new ArgumentException(
                $"b_enc must have length d_sae = {featureCount}, got {bEnc.Length}.", nameof(bEnc));
        if (wDec.Length != featureCount * hiddenSize)
            throw new ArgumentException(
                $"W_dec must have length d_sae*d_in = {featureCount * hiddenSize}, got {wDec.Length}.", nameof(wDec));
        if (bDec.Length != hiddenSize)
            throw new ArgumentException(
                $"b_dec must have length d_in = {hiddenSize}, got {bDec.Length}.", nameof(bDec));

        HiddenSize = hiddenSize;
        FeatureCount = featureCount;
        TopK = Math.Min(topK, featureCount);
        ApplyBDecToInput = applyBDecToInput;

        _wEnc = AllocateAndCopy(wEnc);
        _bEnc = AllocateAndCopy(bEnc);
        _wDec = AllocateAndCopy(wDec);
        _bDec = AllocateAndCopy(bDec);
    }

    /// <summary>
    /// Constructs an SAE from caller-owned unmanaged pointers — the SAE takes ownership and
    /// will free them on <see cref="Dispose"/>. Used by <see cref="SaeLoader"/> to avoid the
    /// double allocation of the span-copying constructor.
    /// </summary>
    /// <param name="hiddenSize"><c>d_in</c> — residual-stream width.</param>
    /// <param name="featureCount"><c>d_sae</c> — dictionary size.</param>
    /// <param name="wEnc">Encoder weight pointer, 64-byte aligned, length <c>d_in * d_sae</c>.</param>
    /// <param name="bEnc">Encoder bias pointer, 64-byte aligned, length <c>d_sae</c>.</param>
    /// <param name="wDec">Decoder weight pointer, 64-byte aligned, length <c>d_sae * d_in</c>.</param>
    /// <param name="bDec">Decoder bias pointer, 64-byte aligned, length <c>d_in</c>.</param>
    /// <param name="topK">Number of top features to return from <see cref="Encode"/>.</param>
    /// <param name="applyBDecToInput">Pre-subtract <paramref name="bDec"/> in <see cref="Encode"/>.</param>
    internal SparseAutoencoder(
        int hiddenSize,
        int featureCount,
        nint wEnc,
        nint bEnc,
        nint wDec,
        nint bDec,
        int topK,
        bool applyBDecToInput)
    {
        HiddenSize = hiddenSize;
        FeatureCount = featureCount;
        TopK = Math.Min(topK, featureCount);
        ApplyBDecToInput = applyBDecToInput;
        _wEnc = (float*)wEnc;
        _bEnc = (float*)bEnc;
        _wDec = (float*)wDec;
        _bDec = (float*)bDec;
    }

    private static float* AllocateAndCopy(ReadOnlySpan<float> source)
    {
        nuint bytes = (nuint)(source.Length * sizeof(float));
        var ptr = (float*)NativeMemory.AlignedAlloc(bytes, 64);
        source.CopyTo(new Span<float>(ptr, source.Length));
        return ptr;
    }

    /// <inheritdoc/>
    public (int[] FeatureIndices, float[] FeatureValues) Encode(ReadOnlySpan<float> activation)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (activation.Length != HiddenSize)
            throw new ArgumentException(
                $"activation length must equal HiddenSize ({HiddenSize}), got {activation.Length}.",
                nameof(activation));

        // 1. Optional pre-encoder b_dec subtraction (SAELens apply_b_dec_to_input convention).
        ReadOnlySpan<float> encodeInput;
        Span<float> preBuf = default;
        if (ApplyBDecToInput)
        {
            preBuf = new float[HiddenSize];
            var bDec = new ReadOnlySpan<float>(_bDec, HiddenSize);
            for (int i = 0; i < HiddenSize; i++) preBuf[i] = activation[i] - bDec[i];
            encodeInput = preBuf;
        }
        else
        {
            encodeInput = activation;
        }

        // 2. features = relu(input @ W_enc + b_enc); W_enc is [d_in, d_sae], row-major.
        //    For each output column j: sum_i input[i] * W_enc[i, j] + b_enc[j], then ReLU.
        //    Iterate by row of W_enc so we walk memory contiguously per row.
        var features = new float[FeatureCount];
        Buffer.MemoryCopy(_bEnc, Unsafe.AsPointer(ref features[0]), FeatureCount * sizeof(float), FeatureCount * sizeof(float));

        for (int i = 0; i < HiddenSize; i++)
        {
            float a = encodeInput[i];
            if (a == 0f) continue;
            float* row = _wEnc + (long)i * FeatureCount;
            for (int j = 0; j < FeatureCount; j++)
                features[j] += a * row[j];
        }

        // 3. ReLU in place + count active.
        int activeCount = 0;
        for (int j = 0; j < FeatureCount; j++)
        {
            if (features[j] > 0f) activeCount++;
            else features[j] = 0f;
        }

        // 4. Top-K by magnitude, descending.
        int k = Math.Clamp(TopK, 1, FeatureCount);
        SaeMath.TopK(features, k, out int[] indices, out float[] values);

        // Annotate ActiveCount via out-of-band channel: the hook re-derives it from the
        // returned vectors, but we also store it here as a side-channel via a static helper
        // — instead, the hook recomputes it itself by re-running the dense pass. To keep
        // the interface contract clean, we expose ActiveCount on a richer encode path used
        // internally; ISparseAutoencoder consumers see only the top-K vectors.
        _ = activeCount; // intentionally unused at this surface; SaeHook uses EncodeWithDetails.
        return (indices, values);
    }

    /// <summary>
    /// Like <see cref="Encode"/> but also returns the total active-feature count (post-ReLU
    /// non-zero count across the full dictionary). Used internally by <see cref="SaeHook"/>
    /// to populate <see cref="SaeResult.ActiveFeatureCount"/> without double-encoding.
    /// </summary>
    /// <param name="activation">Input activation, length <see cref="HiddenSize"/>.</param>
    /// <returns>Top-K indices, magnitudes, and total active count across the dictionary.</returns>
    internal (int[] Indices, float[] Values, int ActiveCount) EncodeWithDetails(ReadOnlySpan<float> activation)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (activation.Length != HiddenSize)
            throw new ArgumentException(
                $"activation length must equal HiddenSize ({HiddenSize}), got {activation.Length}.",
                nameof(activation));

        var features = new float[FeatureCount];

        // b_enc bootstrap.
        new ReadOnlySpan<float>(_bEnc, FeatureCount).CopyTo(features);

        // Optional b_dec subtraction.
        Span<float> input;
        if (ApplyBDecToInput)
        {
            input = new float[HiddenSize];
            var bDec = new ReadOnlySpan<float>(_bDec, HiddenSize);
            for (int i = 0; i < HiddenSize; i++) input[i] = activation[i] - bDec[i];
        }
        else
        {
            // Avoid a copy when we won't mutate input.
            input = new float[HiddenSize];
            activation.CopyTo(input);
        }

        for (int i = 0; i < HiddenSize; i++)
        {
            float a = input[i];
            if (a == 0f) continue;
            float* row = _wEnc + (long)i * FeatureCount;
            for (int j = 0; j < FeatureCount; j++)
                features[j] += a * row[j];
        }

        int activeCount = 0;
        for (int j = 0; j < FeatureCount; j++)
        {
            if (features[j] > 0f) activeCount++;
            else features[j] = 0f;
        }

        int k = Math.Clamp(TopK, 1, FeatureCount);
        SaeMath.TopK(features, k, out int[] indices, out float[] values);
        return (indices, values, activeCount);
    }

    /// <inheritdoc/>
    public void Decode(ReadOnlySpan<int> featureIndices, ReadOnlySpan<float> featureValues, Span<float> output)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (featureIndices.Length != featureValues.Length)
            throw new ArgumentException(
                $"featureIndices ({featureIndices.Length}) and featureValues ({featureValues.Length}) must have the same length.");
        if (output.Length != HiddenSize)
            throw new ArgumentException(
                $"output length must equal HiddenSize ({HiddenSize}), got {output.Length}.", nameof(output));

        // Initialize output to b_dec, then add scaled decoder rows for each active feature.
        new ReadOnlySpan<float>(_bDec, HiddenSize).CopyTo(output);

        for (int n = 0; n < featureIndices.Length; n++)
        {
            int idx = featureIndices[n];
            if ((uint)idx >= (uint)FeatureCount)
                throw new ArgumentOutOfRangeException(nameof(featureIndices),
                    $"feature index {idx} is out of range [0, {FeatureCount}).");

            float v = featureValues[n];
            if (v == 0f) continue;

            float* row = _wDec + (long)idx * HiddenSize;
            for (int i = 0; i < HiddenSize; i++)
                output[i] += v * row[i];
        }
    }

    /// <summary>
    /// Computes the L2 norm of (<paramref name="activation"/> − Decode(Encode(<paramref name="activation"/>))).
    /// </summary>
    /// <param name="activation">Original activation. Length must equal <see cref="HiddenSize"/>.</param>
    /// <returns>The reconstruction error (square root of summed squared differences).</returns>
    public float ReconstructionError(ReadOnlySpan<float> activation)
    {
        var (indices, values) = Encode(activation);
        Span<float> reconstruction = new float[HiddenSize];
        Decode(indices, values, reconstruction);
        return SaeMath.L2Distance(activation, reconstruction);
    }

    /// <summary>Releases the unmanaged weight buffers.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_wEnc != null) { NativeMemory.AlignedFree(_wEnc); _wEnc = null; }
        if (_bEnc != null) { NativeMemory.AlignedFree(_bEnc); _bEnc = null; }
        if (_wDec != null) { NativeMemory.AlignedFree(_wDec); _wDec = null; }
        if (_bDec != null) { NativeMemory.AlignedFree(_bDec); _bDec = null; }
        GC.SuppressFinalize(this);
    }

    /// <summary>Finalizer — releases unmanaged memory if <see cref="Dispose"/> was not called.</summary>
    ~SparseAutoencoder() => Dispose();
}
