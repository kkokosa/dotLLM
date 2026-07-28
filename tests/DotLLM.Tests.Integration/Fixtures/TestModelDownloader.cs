using System.Diagnostics;
using DotLLM.HuggingFace;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Shared helper for downloading GGUF models in test fixtures.
/// Uses a 10-minute HTTP timeout (default 100s is too short for multi-GB models on slow connections).
/// Downloads to <see cref="CacheDirectory"/> and skips if already cached.
/// Shows download progress on stderr so it's visible during <c>dotnet test</c>.
/// </summary>
internal static class TestModelDownloader
{
    /// <summary>Environment variable overriding the test model cache root.</summary>
    internal const string CacheDirEnvVar = "DOTLLM_TEST_CACHE_DIR";

    /// <summary>
    /// Root directory for cached test models: <c>$DOTLLM_TEST_CACHE_DIR</c> when set,
    /// otherwise <c>~/.dotllm/test-cache/</c>. Models are laid out underneath as
    /// <c>{owner}/{repo}/{filename}.gguf</c> — the same layout the CLI uses for
    /// <c>~/.dotllm/models/</c>, so the two are interchangeable.
    /// </summary>
    /// <remarks>
    /// Evaluated per access rather than cached in a static initializer so tests can
    /// exercise the override. Note that <c>~</c> is not expanded by .NET; relative
    /// values are resolved against the current directory via <see cref="Path.GetFullPath(string)"/>.
    /// </remarks>
    internal static string CacheDirectory => ResolveCacheDirectory(
        Environment.GetEnvironmentVariable(CacheDirEnvVar));

    /// <summary>
    /// Resolves the cache root from a raw environment variable value.
    /// Exposed for testing; <paramref name="envValue"/> null/empty/whitespace selects the default.
    /// </summary>
    internal static string ResolveCacheDirectory(string? envValue) =>
        string.IsNullOrWhiteSpace(envValue)
            ? Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                ".dotllm", "test-cache")
            : Path.GetFullPath(envValue);

    /// <summary>
    /// Returns the cached model path if it exists, otherwise downloads from HuggingFace.
    /// </summary>
    public static async Task<string> EnsureModelAsync(string repoId, string filename)
    {
        // Resolve once so the probe and the download target can't disagree if the
        // environment variable changes mid-run.
        string cacheDir = CacheDirectory;
        string cachedPath = Path.Combine(cacheDir, repoId.Replace('/', Path.DirectorySeparatorChar), filename);

        if (File.Exists(cachedPath))
            return cachedPath;

        Console.Error.WriteLine($"[dotLLM] Downloading {filename} from {repoId}...");

        var sw = Stopwatch.StartNew();
        long lastReportedBytes = 0;
        var progress = new Progress<(long bytesDownloaded, long? totalBytes)>(p =>
        {
            // Throttle: report every ~10 MB or when done
            if (p.bytesDownloaded - lastReportedBytes < 10 * 1024 * 1024)
                return;
            lastReportedBytes = p.bytesDownloaded;

            double downloadedMB = p.bytesDownloaded / (1024.0 * 1024.0);
            if (p.totalBytes.HasValue)
            {
                double totalMB = p.totalBytes.Value / (1024.0 * 1024.0);
                int pct = (int)(p.bytesDownloaded * 100 / p.totalBytes.Value);
                Console.Error.WriteLine($"[dotLLM]   {downloadedMB:F0} / {totalMB:F0} MB ({pct}%)");
            }
            else
            {
                Console.Error.WriteLine($"[dotLLM]   {downloadedMB:F0} MB downloaded...");
            }
        });

        using var httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        using var downloader = new HuggingFaceDownloader(httpClient);
        var result = await downloader.DownloadFileAsync(repoId, filename, cacheDir, progress);

        sw.Stop();
        double finalMB = new FileInfo(result).Length / (1024.0 * 1024.0);
        Console.Error.WriteLine($"[dotLLM]   Done: {finalMB:F0} MB in {sw.Elapsed.TotalSeconds:F1}s");

        return result;
    }
}
