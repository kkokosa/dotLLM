using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Jobs;
using BenchmarkDotNet.Toolchains.CsProj;
using BenchmarkDotNet.Toolchains.DotNetCli;

namespace DotLLM.Benchmarks;

/// <summary>
/// Builds one BenchmarkDotNet job per requested .NET runtime so a single run can compare
/// runtimes side by side, while keeping BDN's warmup/iteration control and statistics.
/// </summary>
/// <remarks>
/// <para>
/// Runtimes are selected via <c>DOTLLM_BENCH_RUNTIMES</c> (comma-separated target framework
/// monikers, e.g. <c>net10.0,net11.0</c>). When unset, a single job with BDN's default
/// toolchain is used — the host runtime — so behaviour on machines without newer SDKs is
/// unchanged.
/// </para>
/// <para>
/// Jobs are constructed with an explicit <see cref="CsProjCoreToolchain"/> rather than
/// <c>RuntimeMoniker</c>, because BenchmarkDotNet has no moniker for .NET 11 yet. The
/// toolchain API accepts any TFM string, so this keeps working as new versions appear
/// without waiting for BDN to add an enum member.
/// </para>
/// <para>
/// Building a job for a given TFM requires an SDK that can target it. Selecting
/// <c>net11.0</c> on a machine whose <c>global.json</c> resolves to the .NET 10 SDK fails
/// with <c>NETSDK1045</c>.
/// </para>
/// </remarks>
public sealed class MultiRuntimeConfig : ManualConfig
{
    /// <summary>Environment variable holding a comma-separated list of TFMs to benchmark.</summary>
    public const string RuntimesEnvVar = "DOTLLM_BENCH_RUNTIMES";

    // Matches the counts previously carried by [SimpleJob] on the benchmark classes.
    private const int WarmupCount = 2;
    private const int IterationCount = 5;

    public MultiRuntimeConfig()
        : this(Environment.GetEnvironmentVariable(RuntimesEnvVar))
    {
    }

    /// <summary>Exposed for testing; <paramref name="runtimes"/> null/empty selects the host runtime.</summary>
    public MultiRuntimeConfig(string? runtimes)
    {
        // Deliberately no exporters/loggers/column providers here: BenchmarkDotNet merges a
        // [Config] with its default configuration, so adding them again produces "already
        // present" warnings and suppresses exporters selected on the command line
        // (scripts/bench_compare.py relies on --exporters json).
        var baseJob = Job.Default
            .WithWarmupCount(WarmupCount)
            .WithIterationCount(IterationCount);

        var tfms = ParseRuntimes(runtimes);

        if (tfms.Count == 0)
        {
            // No explicit selection: single job on the host runtime, as before.
            AddJob(baseJob);
            return;
        }

        foreach (string tfm in tfms)
        {
            AddJob(baseJob
                .WithToolchain(CsProjCoreToolchain.From(new NetCoreAppSettings(tfm, null, tfm)))
                // The id becomes the job label in BDN's DisplayInfo, which is how
                // scripts/bench_compare.py attributes each result row to a runtime.
                .WithId(tfm));
        }
    }

    /// <summary>
    /// Splits and normalises a runtime list. Accepts <c>net10.0</c> or bare <c>net10</c>;
    /// blank entries are ignored and duplicates collapse, preserving order.
    /// </summary>
    public static List<string> ParseRuntimes(string? runtimes)
    {
        var result = new List<string>();
        if (string.IsNullOrWhiteSpace(runtimes))
            return result;

        foreach (string raw in runtimes.Split(',', StringSplitOptions.RemoveEmptyEntries))
        {
            string tfm = raw.Trim().ToLowerInvariant();
            if (tfm.Length == 0)
                continue;
            // Allow "net10" as shorthand for "net10.0".
            if (!tfm.Contains('.'))
                tfm += ".0";
            if (!result.Contains(tfm))
                result.Add(tfm);
        }
        return result;
    }
}
