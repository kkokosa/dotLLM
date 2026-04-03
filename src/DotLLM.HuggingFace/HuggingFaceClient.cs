using System.Net.Http.Json;
using System.Web;

namespace DotLLM.HuggingFace;

/// <summary>
/// Client for the HuggingFace Hub API. Searches models, retrieves model info, and lists repository files.
/// Uses raw <see cref="HttpClient"/> — no external dependencies.
/// </summary>
public sealed class HuggingFaceClient : IDisposable
{
    private const string DefaultApiBase = "https://huggingface.co/api";

    private readonly HttpClient _httpClient;
    private readonly bool _ownsClient;
    /// <summary>
    /// Creates a new HuggingFace Hub client.
    /// </summary>
    /// <param name="httpClient">Optional pre-configured <see cref="HttpClient"/>. If null, a new one is created.</param>
    /// <param name="token">Optional HuggingFace API token. Falls back to <c>HF_TOKEN</c> env var.</param>
    public HuggingFaceClient(HttpClient? httpClient = null, string? token = null)
    {
        if (httpClient is not null)
        {
            _httpClient = httpClient;
            _ownsClient = false;
        }
        else
        {
            _httpClient = new HttpClient();
            _ownsClient = true;
        }

        _httpClient.BaseAddress ??= new Uri(DefaultApiBase);

        token ??= Environment.GetEnvironmentVariable("HF_TOKEN");
        if (!string.IsNullOrEmpty(token))
            _httpClient.DefaultRequestHeaders.Authorization =
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);

        _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("dotLLM/0.1");
    }

    /// <summary>
    /// Searches for GGUF models on HuggingFace Hub.
    /// </summary>
    /// <param name="query">Search query string.</param>
    /// <param name="limit">Maximum number of results (default 20).</param>
    /// <param name="sort">Sort field: "downloads", "likes", "lastModified" (default "downloads").</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of matching model infos.</returns>
    public async Task<List<HuggingFaceModelInfo>> SearchModelsAsync(
        string query,
        int limit = 20,
        string sort = "downloads",
        CancellationToken cancellationToken = default)
    {
        var encodedQuery = HttpUtility.UrlEncode(query);
        var url = $"/api/models?search={encodedQuery}&filter=gguf&sort={sort}&direction=-1&limit={limit}";
        var response = await _httpClient.GetAsync(url, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        return await response.Content.ReadFromJsonAsync(HuggingFaceJsonContext.Default.ListHuggingFaceModelInfo, cancellationToken).ConfigureAwait(false)
               ?? [];
    }

    /// <summary>
    /// Gets detailed info for a specific model, including its file list (siblings).
    /// </summary>
    /// <param name="repoId">Repository ID, e.g. "TheBloke/Llama-2-7B-GGUF".</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>Model info with siblings populated.</returns>
    public async Task<HuggingFaceModelInfo> GetModelInfoAsync(
        string repoId,
        CancellationToken cancellationToken = default)
    {
        var url = $"/api/models/{repoId}";
        var response = await _httpClient.GetAsync(url, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        return await response.Content.ReadFromJsonAsync(HuggingFaceJsonContext.Default.HuggingFaceModelInfo, cancellationToken).ConfigureAwait(false)
               ?? throw new InvalidOperationException($"Failed to deserialize model info for '{repoId}'.");
    }

    /// <summary>
    /// Lists all files in a repository via the tree API, which includes accurate file sizes.
    /// </summary>
    /// <param name="repoId">Repository ID.</param>
    /// <param name="revision">Branch or commit (default "main").</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of repository file entries with sizes.</returns>
    public async Task<List<RepoFileEntry>> GetRepoTreeAsync(
        string repoId,
        string revision = "main",
        CancellationToken cancellationToken = default)
    {
        var url = $"/api/models/{repoId}/tree/{revision}";
        var response = await _httpClient.GetAsync(url, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        return await response.Content.ReadFromJsonAsync(HuggingFaceJsonContext.Default.ListRepoFileEntry, cancellationToken).ConfigureAwait(false)
               ?? [];
    }

    /// <summary>
    /// Lists GGUF files in a repository using the tree API (includes file sizes).
    /// </summary>
    /// <param name="repoId">Repository ID.</param>
    /// <param name="cancellationToken">Cancellation token.</param>
    /// <returns>List of GGUF file entries with sizes.</returns>
    public async Task<List<RepoFileEntry>> ListGgufFilesAsync(
        string repoId,
        CancellationToken cancellationToken = default)
    {
        var tree = await GetRepoTreeAsync(repoId, cancellationToken: cancellationToken).ConfigureAwait(false);
        return tree
            .Where(f => f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            .ToList();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsClient)
            _httpClient.Dispose();
    }
}
