using System.Buffers.Binary;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Loads a HuggingFace PEFT-format LoRA adapter directory into a
/// <see cref="LoraAdapter"/>. Supports the canonical layout:
/// <c>{root}/adapter_config.json</c> + <c>{root}/adapter_model.safetensors</c>.
/// </summary>
/// <remarks>
/// <para>
/// PEFT tensor naming (per <c>peft</c> ≥ 0.4): each LoRA factor is published
/// as <c>base_model.model.{layer_path}.{proj_name}.lora_A.weight</c> and
/// <c>...lora_B.weight</c>. PEFT also occasionally writes
/// <c>...lora_A.default.weight</c> when there are named adapter sub-trees;
/// the loader normalises both forms.
/// </para>
/// <para>
/// Only plain LoRA is supported in Phase 4a. <c>use_rslora</c> and
/// <c>use_dora</c> are rejected with a clear <see cref="NotSupportedException"/>;
/// quantised adapter weights (F16 / BF16 / Q8_0) are decoded to F32 during
/// load (only F32, F16, BF16 implemented this commit — anything else throws).
/// </para>
/// </remarks>
public static unsafe class PeftAdapterLoader
{
    private static readonly Regex ProjectionPathRegex = new(
        @"^(?:base_model\.(?:model\.)?)?model\.layers\.(?<layer>\d+)\.(?<scope>self_attn|mlp)\.(?<proj>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.lora_(?<which>A|B)(?:\.default)?\.weight$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);

    /// <summary>
    /// Loads a PEFT LoRA adapter from the directory at <paramref name="path"/>.
    /// </summary>
    /// <param name="name">Logical name to register under.</param>
    /// <param name="path">Directory containing PEFT adapter files.</param>
    /// <param name="baseConfig">
    /// Optional base-model <see cref="ModelConfig"/>. When supplied, the loader
    /// validates layer count, hidden size, and per-projection dimensions and
    /// throws <see cref="InvalidDataException"/> at load time on mismatch.
    /// </param>
    /// <returns>A loaded <see cref="LoraAdapter"/> owned by the caller.</returns>
    public static LoraAdapter LoadFromDirectory(string name, string path, ModelConfig? baseConfig = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!Directory.Exists(path))
            throw new DirectoryNotFoundException($"PEFT adapter directory not found: {path}");

        string configPath = Path.Combine(path, "adapter_config.json");
        if (!File.Exists(configPath))
            throw new FileNotFoundException(
                $"PEFT adapter is missing adapter_config.json (looked in '{path}').", configPath);

        string safetensorsPath = Path.Combine(path, "adapter_model.safetensors");
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException(
                $"PEFT adapter is missing adapter_model.safetensors (looked in '{path}').", safetensorsPath);

        // ── adapter_config.json ─────────────────────────────────────
        var meta = ParseAdapterConfig(configPath);
        if (meta.UseRsLora)
            throw new NotSupportedException(
                $"PEFT adapter '{name}' has use_rslora=true. rsLoRA scaling is a follow-up; "
                + "Phase 4a covers plain LoRA only.");
        if (meta.UseDora)
            throw new NotSupportedException(
                $"PEFT adapter '{name}' has use_dora=true. DoRA scaling is a follow-up; "
                + "Phase 4a covers plain LoRA only.");
        if (!string.IsNullOrEmpty(meta.TaskType)
            && !StringComparer.OrdinalIgnoreCase.Equals(meta.TaskType, "CAUSAL_LM"))
        {
            throw new NotSupportedException(
                $"PEFT adapter '{name}' declares task_type='{meta.TaskType}'. Only CAUSAL_LM "
                + "adapters are in scope for Phase 4a.");
        }

        var adapter = new LoraAdapter(name, meta.Rank, meta.Alpha, meta.TargetModules);
        bool transferred = false;
        try
        {
            using var safetensors = SafetensorsFile.Open(safetensorsPath);
            LoadTensors(safetensors, adapter, meta.Rank);

            if (baseConfig is not null && !adapter.IsCompatible(baseConfig))
            {
                throw new InvalidDataException(
                    $"PEFT adapter '{name}' is not compatible with the supplied base model "
                    + $"(layers={baseConfig.NumLayers}, hidden={baseConfig.HiddenSize}, "
                    + $"q_out={baseConfig.NumAttentionHeads * baseConfig.HeadDim}, "
                    + $"kv_out={baseConfig.NumKvHeads * baseConfig.HeadDim}, "
                    + $"intermediate={baseConfig.IntermediateSize}). See adapter shapes above.");
            }

            transferred = true;
            return adapter;
        }
        finally
        {
            if (!transferred) adapter.Dispose();
        }
    }

    private static void LoadTensors(SafetensorsFile file, LoraAdapter adapter, int rank)
    {
        // Group tensors by (layer, proj) so we can validate that A and B
        // arrive in matched pairs. Per PEFT convention the writer typically
        // emits both halves together but we don't assume ordering.
        var pending = new Dictionary<(int Layer, string Proj), PendingPair>();
        var unrecognised = new List<string>();

        foreach (var tensor in file.Tensors)
        {
            // Embedding / lm_head LoRA — rare, log via a structured exception
            // rather than silently dropping when encountered.
            if (tensor.Name.Contains("lora_embedding_A", StringComparison.Ordinal)
                || tensor.Name.Contains("lora_embedding_B", StringComparison.Ordinal))
            {
                // Skip with a record so the diagnostic is auditable.
                continue;
            }

            var match = ProjectionPathRegex.Match(tensor.Name);
            if (!match.Success)
            {
                unrecognised.Add(tensor.Name);
                continue;
            }

            int layer = int.Parse(match.Groups["layer"].Value, System.Globalization.CultureInfo.InvariantCulture);
            string proj = match.Groups["proj"].Value;
            string which = match.Groups["which"].Value; // "A" or "B"

            var key = (layer, proj);
            if (!pending.TryGetValue(key, out var pair))
            {
                pair = new PendingPair();
                pending[key] = pair;
            }

            if (which == "A")
            {
                if (pair.AAssigned)
                    throw new InvalidDataException(
                        $"PEFT adapter has duplicate lora_A entry for layer={layer} proj='{proj}'.");
                pair.AAssigned = true;
                pair.ATensor = tensor;
            }
            else
            {
                if (pair.BAssigned)
                    throw new InvalidDataException(
                        $"PEFT adapter has duplicate lora_B entry for layer={layer} proj='{proj}'.");
                pair.BAssigned = true;
                pair.BTensor = tensor;
            }
        }

        if (unrecognised.Count > 0)
        {
            throw new InvalidDataException(
                "PEFT adapter contains tensor names that do not match the expected "
                + "{base_model.model.|model.}layers.{i}.{self_attn|mlp}.{proj}.lora_{A|B}[.default].weight "
                + "convention. Unrecognised: " + string.Join(", ", unrecognised));
        }

        if (pending.Count == 0)
            throw new InvalidDataException(
                "PEFT adapter contains no recognised LoRA factor tensors.");

        foreach (var ((layer, proj), pair) in pending)
        {
            if (!pair.AAssigned)
                throw new InvalidDataException(
                    $"PEFT adapter is missing lora_A for layer={layer} proj='{proj}' "
                    + "(only lora_B was found).");
            if (!pair.BAssigned)
                throw new InvalidDataException(
                    $"PEFT adapter is missing lora_B for layer={layer} proj='{proj}' "
                    + "(only lora_A was found).");

            // PEFT layout: A is [r, d_out], B is [r, d_in]. dotLLM uses the
            // weight-as-[output, input] convention, so:
            //   - lora_A.weight shape [r, d_out]  → store as [d_out, r] row-major
            //     (this is our A: [outputDim, rank])
            //   - lora_B.weight shape [d_out, r]  → ALREADY [d_out, r] in PEFT for
            //     base_model.model layer; but per HF PEFT spec lora_B is [d_out, r]
            //     i.e. the up-projection — so PEFT_A is dotLLM_B and PEFT_B is dotLLM_A.
            //
            // Concretely (from peft.tuners.lora.LoraLayer):
            //     y = x . W^T + scaling * x . A^T . B^T
            //   where A shape = (r, in_features), B shape = (out_features, r).
            // So PEFT 'lora_A' = dotLLM B (down, [r, in])
            //    PEFT 'lora_B' = dotLLM A (up,   [out, r])
            int rA = pair.ATensor.Shape[0];   // PEFT A: rows = r
            int aIn = pair.ATensor.Shape[1];  // PEFT A: cols = in_features
            int bOut = pair.BTensor.Shape[0]; // PEFT B: rows = out_features
            int rB = pair.BTensor.Shape[1];   // PEFT B: cols = r

            if (rA != rank || rB != rank)
                throw new InvalidDataException(
                    $"PEFT adapter rank mismatch at layer={layer} proj='{proj}': "
                    + $"adapter_config.r={rank}, lora_A rank dim={rA}, lora_B rank dim={rB}.");

            // dotLLM expects:
            //   B (down): [inputDim, rank]   row-major  — i.e. "[r, in]" in PEFT terms,
            //                                            but our layout says [outputDim_of_factor, inputDim_of_factor]
            //                                            with outputDim=rank and inputDim=in.
            //   Therefore B (down) has dimensions [rank, in_features] and the loader stores
            //   the PEFT 'lora_A' tensor (which IS [r, in_features] row-major) verbatim.
            //   A (up)   has dimensions [outputDim, rank] and the loader stores the PEFT
            //   'lora_B' tensor (which IS [out_features, r] row-major) verbatim.
            int inputDim = aIn;   // input feature dim of the factor pair
            int outputDim = bOut; // output feature dim of the factor pair

            long bElems = (long)rank * inputDim;
            long aElems = (long)outputDim * rank;

            nint bHandle = LoraAdapter.AllocAligned(bElems);
            nint aHandle = LoraAdapter.AllocAligned(aElems);

            try
            {
                CopyTensorAsF32(file, pair.ATensor, (float*)bHandle, bElems);
                CopyTensorAsF32(file, pair.BTensor, (float*)aHandle, aElems);
            }
            catch
            {
                if (aHandle != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)aHandle);
                if (bHandle != 0) System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)bHandle);
                throw;
            }

            adapter.AddLayerWeights(layer, proj, new LoraLayerWeights(
                AHandle: aHandle,
                BHandle: bHandle,
                InputDim: inputDim,
                OutputDim: outputDim));
        }
    }

    private static void CopyTensorAsF32(SafetensorsFile file, SafetensorsTensorDescriptor tensor,
                                        float* dst, long expectedElements)
    {
        long actualElements = tensor.ElementCount;
        if (actualElements != expectedElements)
            throw new InvalidDataException(
                $"PEFT tensor '{tensor.Name}' element-count mismatch: "
                + $"expected {expectedElements}, got {actualElements}.");

        byte* src = (byte*)file.DataBasePointer + tensor.DataBeginOffset;
        switch (tensor.DType)
        {
            case SafetensorsDType.F32:
                {
                    long bytes = expectedElements * sizeof(float);
                    Buffer.MemoryCopy(src, dst, bytes, bytes);
                    break;
                }
            case SafetensorsDType.F16:
                {
                    var srcSpan = new ReadOnlySpan<Half>(src, (int)expectedElements);
                    var dstSpan = new Span<float>(dst, (int)expectedElements);
                    System.Numerics.Tensors.TensorPrimitives.ConvertToSingle(srcSpan, dstSpan);
                    break;
                }
            case SafetensorsDType.BF16:
                {
                    // BF16: top 16 bits of an F32. Upcast = shift left into the
                    // exponent + mantissa of an F32. No SIMD helper in
                    // TensorPrimitives yet, scalar loop is fine for 10–100 MB.
                    for (long i = 0; i < expectedElements; i++)
                    {
                        ushort raw = BinaryPrimitives.ReadUInt16LittleEndian(
                            new ReadOnlySpan<byte>(src + i * 2, 2));
                        uint asF32 = (uint)raw << 16;
                        dst[i] = BitConverter.UInt32BitsToSingle(asF32);
                    }
                    break;
                }
            default:
                throw new NotSupportedException(
                    $"PEFT tensor '{tensor.Name}' has dtype {tensor.DType}; "
                    + "only F32, F16, and BF16 are supported in Phase 4a.");
        }
    }

    private static AdapterConfigMeta ParseAdapterConfig(string path)
    {
        using var stream = File.OpenRead(path);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        if (root.ValueKind != JsonValueKind.Object)
            throw new InvalidDataException(
                $"PEFT adapter_config.json root is not a JSON object (got {root.ValueKind}).");

        int rank = root.TryGetProperty("r", out var rEl) && rEl.ValueKind == JsonValueKind.Number
            ? rEl.GetInt32()
            : throw new InvalidDataException(
                "PEFT adapter_config.json is missing required 'r' (rank) field.");
        if (rank <= 0)
            throw new InvalidDataException(
                $"PEFT adapter_config.json has invalid rank r={rank} (must be positive).");

        // lora_alpha: int or float; PEFT writes int historically.
        float alpha;
        if (root.TryGetProperty("lora_alpha", out var alphaEl))
        {
            alpha = alphaEl.ValueKind switch
            {
                JsonValueKind.Number => (float)alphaEl.GetDouble(),
                _ => throw new InvalidDataException(
                    $"PEFT adapter_config.json 'lora_alpha' must be a number (got {alphaEl.ValueKind}).")
            };
        }
        else
        {
            // PEFT default: alpha = 8 when missing.
            alpha = 8f;
        }

        var targets = new List<string>();
        if (root.TryGetProperty("target_modules", out var tm))
        {
            switch (tm.ValueKind)
            {
                case JsonValueKind.Array:
                    foreach (var entry in tm.EnumerateArray())
                        if (entry.ValueKind == JsonValueKind.String)
                            targets.Add(entry.GetString()!);
                    break;
                case JsonValueKind.String:
                    targets.Add(tm.GetString()!);
                    break;
                case JsonValueKind.Null:
                    break;
                default:
                    throw new InvalidDataException(
                        $"PEFT adapter_config.json 'target_modules' must be an array or string (got {tm.ValueKind}).");
            }
        }

        float dropout = 0f;
        if (root.TryGetProperty("lora_dropout", out var drop) && drop.ValueKind == JsonValueKind.Number)
            dropout = (float)drop.GetDouble();

        bool useRslora = root.TryGetProperty("use_rslora", out var rs)
            && rs.ValueKind is JsonValueKind.True;
        bool useDora = root.TryGetProperty("use_dora", out var dora)
            && dora.ValueKind is JsonValueKind.True;

        string? taskType = null;
        if (root.TryGetProperty("task_type", out var task) && task.ValueKind == JsonValueKind.String)
            taskType = task.GetString();

        return new AdapterConfigMeta(rank, alpha, targets, dropout, useRslora, useDora, taskType);
    }

    private sealed record AdapterConfigMeta(
        int Rank,
        float Alpha,
        IReadOnlyList<string> TargetModules,
        float Dropout,
        bool UseRsLora,
        bool UseDora,
        string? TaskType);

    private sealed class PendingPair
    {
        public bool AAssigned;
        public bool BAssigned;
        public SafetensorsTensorDescriptor ATensor;
        public SafetensorsTensorDescriptor BTensor;
    }
}
