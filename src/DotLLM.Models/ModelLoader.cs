using System.Buffers.Binary;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;

namespace DotLLM.Models;

/// <summary>
/// Convenience helper encapsulating the format-open → config-extract → model-load
/// pattern. Single dispatch point for all architecture creation from either
/// GGUF or HuggingFace safetensors on-disk layouts.
/// </summary>
public static class ModelLoader
{
    /// <summary>
    /// Loads a model from a GGUF file path. Opens the file, extracts config,
    /// and creates the appropriate model instance.
    /// </summary>
    /// <param name="path">Path to the GGUF model file.</param>
    /// <param name="threading">Threading configuration. Null defaults to single-threaded.</param>
    /// <returns>The loaded model, GGUF file handle, and model configuration.</returns>
    public static (IModel Model, GgufFile Gguf, ModelConfig Config) LoadFromGguf(
        string path, ThreadingConfig? threading = null)
    {
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = TransformerModel.LoadFromGguf(gguf, config, threading ?? ThreadingConfig.SingleThreaded);
        return (model, gguf, config);
    }

    /// <summary>
    /// Loads a model from a HuggingFace safetensors checkpoint. The directory
    /// containing <paramref name="safetensorsPath"/> is scanned for a
    /// <c>config.json</c> which drives both architecture dispatch and
    /// <see cref="ModelConfig"/> population.
    /// </summary>
    /// <param name="safetensorsPath">Absolute path to a <c>*.safetensors</c> file.</param>
    /// <param name="threading">Threading configuration. Null defaults to single-threaded.</param>
    /// <returns>The loaded model, safetensors file handle, and model configuration.</returns>
    /// <exception cref="FileNotFoundException">
    /// <paramref name="safetensorsPath"/> or an accompanying <c>config.json</c> is missing.
    /// </exception>
    /// <exception cref="InvalidDataException">
    /// <c>config.json</c> is malformed or declares an unsupported architecture.
    /// </exception>
    public static (IModel Model, SafetensorsFile Safetensors, ModelConfig Config) LoadFromSafetensors(
        string safetensorsPath, ThreadingConfig? threading = null)
    {
        if (!File.Exists(safetensorsPath))
            throw new FileNotFoundException(
                $"Safetensors file not found: {safetensorsPath}", safetensorsPath);

        string? directory = Path.GetDirectoryName(safetensorsPath);
        if (directory is null)
            throw new InvalidDataException(
                $"Could not determine directory of safetensors path '{safetensorsPath}'.");
        string configPath = Path.Combine(directory, "config.json");
        if (!File.Exists(configPath))
            throw new FileNotFoundException(
                $"Expected HuggingFace config.json next to '{safetensorsPath}', but '{configPath}' does not exist.",
                configPath);

        // Peek at the architecture so we can dispatch before fully extracting config.
        string configJson = File.ReadAllText(configPath);
        using var doc = JsonDocument.Parse(configJson);
        Architecture arch = HfConfigExtractor.ResolveArchitecture(doc.RootElement);

        var file = SafetensorsFile.Open(safetensorsPath);
        try
        {
            ModelConfig config = HfConfigExtractor.Extract(doc.RootElement);

            IModel model = config.Architecture switch
            {
                Architecture.Llama or Architecture.Mistral or Architecture.Phi or Architecture.Qwen
                    or Architecture.DeepSeekV2 or Architecture.DeepSeekV3
                    or Architecture.Mixtral or Architecture.QwenMoe
                    => TransformerModel.LoadFromSafetensors(file, config, threading ?? ThreadingConfig.SingleThreaded),
                _ => throw new NotSupportedException(
                    $"Safetensors loader does not yet dispatch architecture {config.Architecture}. "
                    + "Supported today: Llama, Mistral, Phi, Qwen, DeepSeekV2, DeepSeekV3, Mixtral, QwenMoe."),
            };

            return (model, file, config);
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Top-level dispatcher that auto-detects GGUF vs safetensors by file
    /// extension, falling back to magic-byte probing when the extension is
    /// ambiguous. Returns an opaque file handle (either
    /// <see cref="GgufFile"/> or <see cref="SafetensorsFile"/>) plus the
    /// loaded model and its config.
    /// </summary>
    /// <remarks>
    /// Callers that need to force a specific format should call
    /// <see cref="LoadFromGguf"/> or <see cref="LoadFromSafetensors"/>
    /// directly — this entry point exists only as a convenience for
    /// generic "given a path, load a model" code paths.
    /// </remarks>
    public static (IModel Model, IDisposable File, ModelConfig Config) Load(
        string path, ThreadingConfig? threading = null)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"Model file not found: {path}", path);

        LoadFormat format = DetectFormat(path);
        switch (format)
        {
            case LoadFormat.Gguf:
            {
                var (model, gguf, config) = LoadFromGguf(path, threading);
                return (model, gguf, config);
            }
            case LoadFormat.Safetensors:
            {
                var (model, st, config) = LoadFromSafetensors(path, threading);
                return (model, st, config);
            }
            default:
                throw new InvalidDataException(
                    $"Cannot determine model format for '{path}'. Expected .gguf or .safetensors.");
        }
    }

    private enum LoadFormat { Unknown, Gguf, Safetensors }

    /// <summary>
    /// Detects the on-disk format of a model file. Extension check first
    /// (fast and unambiguous in practice), then a magic-byte probe for
    /// corner cases like extensionless files in a test harness.
    /// </summary>
    private static LoadFormat DetectFormat(string path)
    {
        string ext = Path.GetExtension(path).ToLowerInvariant();
        if (ext == ".gguf") return LoadFormat.Gguf;
        if (ext == ".safetensors") return LoadFormat.Safetensors;

        // Magic-byte sniff: GGUF starts with ASCII "GGUF" (0x47 0x47 0x55 0x46).
        // Safetensors starts with an 8-byte LE u64 header length — not a magic
        // sequence, but we can sanity-check that the first 8 bytes would be
        // a plausible header length (small but not tiny, not exceeding file size).
        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            Span<byte> buf = stackalloc byte[8];
            int read = fs.Read(buf);
            if (read < 4) return LoadFormat.Unknown;
            if (buf[0] == 0x47 && buf[1] == 0x47 && buf[2] == 0x55 && buf[3] == 0x46)
                return LoadFormat.Gguf;
            if (read == 8)
            {
                ulong headerLen = BinaryPrimitives.ReadUInt64LittleEndian(buf);
                long fileLen = fs.Length;
                // Plausibility: 2 <= headerLen <= fileLen - 8 and headerLen doesn't
                // exceed a few MB (HF headers top out around low MB in practice).
                if (headerLen >= 2 && (long)headerLen + 8 <= fileLen && headerLen < 64 * 1024 * 1024)
                    return LoadFormat.Safetensors;
            }
        }
        catch
        {
            // Fall through to Unknown.
        }
        return LoadFormat.Unknown;
    }
}
