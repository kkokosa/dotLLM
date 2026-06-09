using System.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Non-asserting timing harness for the Vulkan forward pass on SmolLM-135M.
/// Gated by <c>DOTLLM_VULKAN_PERF=1</c> so it does not add run time to the
/// default test sweep; invoked manually from the perf wave.
/// </summary>
/// <remarks>
/// <para>
/// Runs one prefill (≈10 tokens) + N decode steps (default 32) on a warmed-up
/// <see cref="VulkanTransformerModel"/> and prints per-step wall time via
/// <see cref="Stopwatch"/>. The parity test
/// <see cref="VulkanTransformerModelTests.VulkanForward_MatchesCpuReference_OnEightDecodeSteps"/>
/// remains the correctness oracle — this harness only measures latency.
/// </para>
/// <para>
/// Env vars:
/// <list type="bullet">
///   <item><c>DOTLLM_VULKAN_PERF=1</c> — required to run.</item>
///   <item><c>DOTLLM_VULKAN_PERF_DECODE_STEPS</c> — override decode step count (default 32).</item>
///   <item><c>DOTLLM_VULKAN_PERF_WARMUP</c> — warm-up decode steps that are timed but reported separately (default 4).</item>
/// </list>
/// </para>
/// </remarks>
[Collection("SmallModel")]
[Trait("Category", "GPU")]
public class VulkanForwardPerfHarness
{
    private readonly SmallModelFixture _fixture;
    private readonly ITestOutputHelper _output;

    public VulkanForwardPerfHarness(SmallModelFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _output = output;
    }

    [SkippableFact]
    public void MeasureDecodeLatency()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("DOTLLM_VULKAN_PERF") == "1",
            "DOTLLM_VULKAN_PERF=1 not set.");
        Skip.If(
            Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1",
            "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(
            VulkanDevice.IsAvailable(),
            "No Vulkan loader or physical device available on this host.");

        string spvDir = ResolveSpvDir();

        int warmupSteps = ParseEnvInt("DOTLLM_VULKAN_PERF_WARMUP", 4);
        int decodeSteps = ParseEnvInt("DOTLLM_VULKAN_PERF_DECODE_STEPS", 32);

        using var gguf = GgufFile.Open(_fixture.FilePath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        var loadSw = Stopwatch.StartNew();
        using var model = VulkanTransformerModel.LoadFromGguf(gguf, config, spvDir);
        loadSw.Stop();
        _output.WriteLine($"load_ms={loadSw.Elapsed.TotalMilliseconds:F1}");

        int[] prompt = tokenizer.Encode("The capital of France is").ToArray();
        Assert.NotEmpty(prompt);

        using var cache = model.CreateKvCache(maxSeqLen: 256);

        int[] positions = new int[prompt.Length];
        for (int i = 0; i < prompt.Length; i++) positions[i] = i;

        // Prefill
        var prefillSw = Stopwatch.StartNew();
        int nextToken;
        using (var logits = model.Forward(prompt, positions, deviceId: -1, cache))
        {
            prefillSw.Stop();
            nextToken = Argmax(logits);
        }
        _output.WriteLine($"prefill_len={prompt.Length} prefill_ms={prefillSw.Elapsed.TotalMilliseconds:F2}");

        int nextPos = prompt.Length;

        // Warm-up decodes — report separately so JIT / driver shader compile cost
        // does not leak into the steady-state numbers.
        var warmupTotal = 0.0;
        for (int i = 0; i < warmupSteps; i++)
        {
            int[] single = { nextToken };
            int[] pos = { nextPos };
            var sw = Stopwatch.StartNew();
            using (var logits = model.Forward(single, pos, deviceId: -1, cache))
            {
                sw.Stop();
                nextToken = Argmax(logits);
            }
            nextPos++;
            warmupTotal += sw.Elapsed.TotalMilliseconds;
            _output.WriteLine($"warmup[{i}]_ms={sw.Elapsed.TotalMilliseconds:F2}");
        }
        _output.WriteLine($"warmup_avg_ms={(warmupSteps == 0 ? 0.0 : warmupTotal / warmupSteps):F2}");

        // Steady-state decodes.
        double decodeTotal = 0.0;
        double decodeMin = double.PositiveInfinity;
        double decodeMax = 0.0;
        for (int i = 0; i < decodeSteps; i++)
        {
            int[] single = { nextToken };
            int[] pos = { nextPos };
            var sw = Stopwatch.StartNew();
            using (var logits = model.Forward(single, pos, deviceId: -1, cache))
            {
                sw.Stop();
                nextToken = Argmax(logits);
            }
            nextPos++;
            double ms = sw.Elapsed.TotalMilliseconds;
            decodeTotal += ms;
            if (ms < decodeMin) decodeMin = ms;
            if (ms > decodeMax) decodeMax = ms;
            _output.WriteLine($"decode[{i}]_ms={ms:F2}");
        }
        double decodeAvg = decodeSteps == 0 ? 0.0 : decodeTotal / decodeSteps;
        double tokPerSec = decodeAvg > 0 ? 1000.0 / decodeAvg : 0.0;

        _output.WriteLine($"=== summary ===");
        _output.WriteLine($"decode_steps={decodeSteps}");
        _output.WriteLine($"decode_avg_ms={decodeAvg:F2}");
        _output.WriteLine($"decode_min_ms={decodeMin:F2}");
        _output.WriteLine($"decode_max_ms={decodeMax:F2}");
        _output.WriteLine($"decode_tok_per_sec={tokPerSec:F2}");
    }

    private static unsafe int Argmax(ITensor logits)
    {
        int n = logits.Shape[logits.Shape.Rank - 1];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, n);
        int idx = 0;
        float best = span[0];
        for (int i = 1; i < n; i++)
        {
            if (span[i] > best) { best = span[i]; idx = i; }
        }
        return idx;
    }

    private static int ParseEnvInt(string key, int fallback)
    {
        string? v = Environment.GetEnvironmentVariable(key);
        return int.TryParse(v, out int n) && n > 0 ? n : fallback;
    }

    private static string ResolveSpvDir()
    {
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        throw new InvalidOperationException(
            "SPIR-V blobs not found. Run native/vulkan/build.sh (or build.ps1) with the Vulkan SDK installed.");
    }
}
