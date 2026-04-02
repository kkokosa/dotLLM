namespace DotLLM.Server;

/// <summary>
/// Server startup configuration parsed from command-line arguments.
/// </summary>
public sealed record ServerOptions
{
    /// <summary>GGUF file path or HuggingFace repo ID.</summary>
    public required string Model { get; init; }

    /// <summary>Quantization filter (e.g., Q8_0, Q4_K_M).</summary>
    public string? Quant { get; init; }

    /// <summary>Compute device: "cpu", "gpu", "gpu:0".</summary>
    public string Device { get; init; } = "cpu";

    /// <summary>Number of GPU layers for hybrid offloading.</summary>
    public int? GpuLayers { get; init; }

    /// <summary>CPU thread count (0 = auto).</summary>
    public int Threads { get; init; }

    /// <summary>Decode thread count (0 = auto).</summary>
    public int DecodeThreads { get; init; }

    /// <summary>Host to bind to.</summary>
    public string Host { get; init; } = "localhost";

    /// <summary>Port to listen on.</summary>
    public int Port { get; init; } = 8080;

    /// <summary>KV-cache key quantization type.</summary>
    public string CacheTypeK { get; init; } = "f32";

    /// <summary>KV-cache value quantization type.</summary>
    public string CacheTypeV { get; init; } = "f32";

    /// <summary>Model display name (derived from file path).</summary>
    public string ModelId { get; init; } = "default";

    /// <summary>
    /// Parses command-line arguments into <see cref="ServerOptions"/>.
    /// </summary>
    public static ServerOptions Parse(string[] args)
    {
        string? model = null;
        string? quant = null;
        string device = "cpu";
        int? gpuLayers = null;
        int threads = 0;
        int decodeThreads = 0;
        string host = "localhost";
        int port = 8080;
        string cacheTypeK = "f32";
        string cacheTypeV = "f32";

        for (int i = 0; i < args.Length; i++)
        {
            string arg = args[i];
            string? next = i + 1 < args.Length ? args[i + 1] : null;

            switch (arg)
            {
                case "--model" or "-m":
                    model = next; i++; break;
                case "--quant" or "-q":
                    quant = next; i++; break;
                case "--device" or "-d":
                    device = next ?? "cpu"; i++; break;
                case "--gpu-layers":
                    gpuLayers = int.Parse(next!); i++; break;
                case "--threads":
                    threads = int.Parse(next!); i++; break;
                case "--decode-threads":
                    decodeThreads = int.Parse(next!); i++; break;
                case "--host":
                    host = next ?? "localhost"; i++; break;
                case "--port" or "-p":
                    port = int.Parse(next!); i++; break;
                case "--cache-type-k":
                    cacheTypeK = next ?? "f32"; i++; break;
                case "--cache-type-v":
                    cacheTypeV = next ?? "f32"; i++; break;
                default:
                    // Positional: treat as model if not set
                    if (model is null && !arg.StartsWith('-'))
                        model = arg;
                    break;
            }
        }

        if (model is null)
        {
            Console.Error.WriteLine("Usage: dotllm-server --model <path-or-repo> [--port 8080] [--device cpu|gpu]");
            Environment.Exit(1);
        }

        string modelId = Path.GetFileNameWithoutExtension(model);
        if (model.Contains('/'))
            modelId = model.Split('/')[^1]; // last segment of repo

        return new ServerOptions
        {
            Model = model,
            Quant = quant,
            Device = device,
            GpuLayers = gpuLayers,
            Threads = threads,
            DecodeThreads = decodeThreads,
            Host = host,
            Port = port,
            CacheTypeK = cacheTypeK,
            CacheTypeV = cacheTypeV,
            ModelId = modelId,
        };
    }
}
