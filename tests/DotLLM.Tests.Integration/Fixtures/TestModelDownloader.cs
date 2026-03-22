using System.Diagnostics;
using DotLLM.HuggingFace;

namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Shared helper for downloading GGUF models in test fixtures.
/// Uses a 10-minute HTTP timeout (default 100s is too short for multi-GB models on slow connections).
/// Downloads to <c>~/.dotllm/test-cache/</c> and skips if already cached.
/// Shows download progress on stderr so it's visible during <c>dotnet test</c>.
/// </summary>
internal static class TestModelDownloader
{
    private static readonly string CacheDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".dotllm", "test-cache");

    /// <summary>
    /// Returns the cached model path if it exists, otherwise downloads from HuggingFace.
    /// </summary>
    public static async Task<string> EnsureModelAsync(string repoId, string filename)
    {
        string cachedPath = Path.Combine(CacheDir, repoId.Replace('/', Path.DirectorySeparatorChar), filename);

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
        var result = await downloader.DownloadFileAsync(repoId, filename, CacheDir, progress);

        sw.Stop();
        double finalMB = new FileInfo(result).Length / (1024.0 * 1024.0);
        Console.Error.WriteLine($"[dotLLM]   Done: {finalMB:F0} MB in {sw.Elapsed.TotalSeconds:F1}s");

        return result;
    }
}
