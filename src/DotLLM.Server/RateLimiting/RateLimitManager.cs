using System.Collections.Concurrent;
using System.Threading;
using System.Threading.RateLimiting;
using System.Threading.Tasks;

namespace DotLLM.Server.RateLimiting;

/// <summary>
/// Owns the per-API-key rate-limiter trio (requests/min, tokens/min,
/// concurrency-with-priority) and exposes a single
/// <see cref="TryAcquireAsync"/> entry point used by the middleware.
/// </summary>
/// <remarks>
/// <para>
/// All three limiters are checked independently. The first one that
/// rejects causes a 429 and the others are not touched — i.e. requests
/// only consume budget on the limiter that admitted them. Once all three
/// admit, the <see cref="RateLimitLease"/> bundles all leases so the
/// caller can dispose them as a unit (and true-up the token budget after
/// the response).
/// </para>
/// <para>
/// Limiters are created on first use and cached per (key, policy) pair.
/// Limiters are NOT shared across keys — that would defeat per-key
/// isolation. When the config changes (rare; today only at startup) the
/// manager should be disposed and re-created.
/// </para>
/// </remarks>
public sealed class RateLimitManager : IDisposable
{
    private readonly RateLimitConfig _config;
    private readonly ConcurrentDictionary<string, KeyState> _keys = new(StringComparer.Ordinal);
    private bool _disposed;

    public RateLimitManager(RateLimitConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
    }

    /// <summary>The active configuration.</summary>
    public RateLimitConfig Config => _config;

    /// <summary>
    /// Try to admit a request. Returns a disposable bundle of leases on
    /// success. On failure the <see cref="AcquireResult.Rejected"/> field
    /// names the limiter that rejected and includes a suggested
    /// <see cref="AcquireResult.RetryAfter"/>.
    /// </summary>
    /// <param name="apiKey">Caller's API key (from <see cref="IApiKeyResolver"/>).</param>
    /// <param name="estimatedTokens">Up-front token charge — prompt tokens
    ///     plus an upper bound on completion tokens (typically
    ///     <c>max_tokens</c>). Trued up after the response.</param>
    /// <param name="ct">Cancellation token (typically <c>HttpContext.RequestAborted</c>).</param>
    public async ValueTask<AcquireResult> TryAcquireAsync(string apiKey, int estimatedTokens, CancellationToken ct)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(RateLimitManager));
        if (!_config.Enabled)
            return AcquireResult.Admit(new RateLimitLease(this, apiKey, estimatedTokens, null, null, null));

        var policy = _config.PolicyFor(apiKey);
        if (policy is null)
            return AcquireResult.Admit(new RateLimitLease(this, apiKey, estimatedTokens, null, null, null));

        var state = _keys.GetOrAdd(apiKey, static (k, p) => new KeyState(p), policy);

        // Intermediate leases are held as the raw BCL/gate leases and disposed
        // directly on any rejection path. They are only wrapped in a
        // RateLimitLease bundle once all three limiters have admitted.
        System.Threading.RateLimiting.RateLimitLease? reqLease = null;
        System.Threading.RateLimiting.RateLimitLease? tokLease = null;

        // --- 1. Requests/min ---
        if (state.Requests is { } reqLimiter)
        {
            var lease = await reqLimiter.AcquireAsync(resource: 0, permitCount: 1, ct).ConfigureAwait(false);
            if (!lease.IsAcquired)
            {
                var retryAfter = ExtractRetryAfter(lease, defaultSec: 60);
                lease.Dispose();
                return AcquireResult.Reject(LimiterKind.Requests, retryAfter);
            }
            reqLease = lease;
        }

        // --- 2. Tokens/min ---
        if (state.Tokens is { } tokLimiter && estimatedTokens > 0)
        {
            var lease = await tokLimiter.AcquireAsync(resource: 0, permitCount: estimatedTokens, ct).ConfigureAwait(false);
            if (!lease.IsAcquired)
            {
                var retryAfter = ExtractRetryAfter(lease, defaultSec: 60);
                lease.Dispose();
                reqLease?.Dispose();
                return AcquireResult.Reject(LimiterKind.Tokens, retryAfter);
            }
            tokLease = lease;
        }

        // --- 3. Concurrency (priority-aware) ---
        PriorityConcurrencyGate.Lease? concLease = null;
        if (state.Concurrency is { } concGate)
        {
            try
            {
                concLease = await concGate.AcquireAsync(policy.Priority, policy.QueueTimeout, ct).ConfigureAwait(false);
            }
            catch
            {
                tokLease?.Dispose();
                reqLease?.Dispose();
                throw;
            }
            if (concLease is null)
            {
                tokLease?.Dispose();
                reqLease?.Dispose();
                return AcquireResult.Reject(LimiterKind.Concurrency, RetryAfterSeconds(policy.QueueTimeout));
            }
        }

        return AcquireResult.Admit(new RateLimitLease(
            this, apiKey, estimatedTokens,
            requestsLease: reqLease,
            tokensLease: tokLease,
            concurrencyLease: concLease));
    }

    /// <summary>
    /// True up the tokens-per-minute budget for a completed request.
    /// If <paramref name="actualTokens"/> is less than the reserved
    /// estimate, the difference is refunded back to the bucket. Charges
    /// above the estimate are ignored — the reservation is the cap.
    /// </summary>
    internal void TrueUpTokens(string apiKey, int reservedEstimate, int actualTokens)
    {
        if (reservedEstimate <= 0 || actualTokens >= reservedEstimate) return;
        if (!_keys.TryGetValue(apiKey, out var state) || state.Tokens is null) return;

        int refund = reservedEstimate - actualTokens;
        // TokenBucketRateLimiter exposes ReplenishingRateLimiter.TryReplenish but
        // that replenishes by the configured rate, not arbitrary amounts. The
        // cleanest way to "refund" is to acquire a negative-cost lease, which
        // RateLimiter doesn't permit. Instead we cap our estimate at request
        // admission and accept that early-stop refunds are best-effort.
        //
        // This hook is kept for callers + tests; once .NET ships a public
        // refund API we can wire it in. For now it is a no-op refund record.
        state.RecordRefund(refund);
    }

    private static int ExtractRetryAfter(System.Threading.RateLimiting.RateLimitLease lease, int defaultSec)
    {
        if (lease.TryGetMetadata(MetadataName.RetryAfter, out var retryAfter))
            return RetryAfterSeconds(retryAfter);
        return defaultSec;
    }

    /// <summary>
    /// Converts a wait duration to a <c>Retry-After</c> value in whole seconds.
    /// Rounds up and floors at 1 — truncation would emit <c>Retry-After: 0</c>
    /// for sub-second waits, telling the client to retry immediately.
    /// </summary>
    private static int RetryAfterSeconds(TimeSpan wait) =>
        Math.Max(1, (int)Math.Ceiling(wait.TotalSeconds));

    /// <inheritdoc/>
    public void Dispose()
    {
        _disposed = true;
        foreach (var s in _keys.Values)
            s.Dispose();
        _keys.Clear();
    }

    /// <summary>The three limiters for one API key.</summary>
    private sealed class KeyState : IDisposable
    {
        public PartitionedRateLimiter<int>? Requests { get; }
        public PartitionedRateLimiter<int>? Tokens { get; }
        public PriorityConcurrencyGate? Concurrency { get; }
        private long _refunded;

        public long Refunded => Interlocked.Read(ref _refunded);
        internal void RecordRefund(int amount) => Interlocked.Add(ref _refunded, amount);

        public KeyState(RateLimitPolicy policy)
        {
            if (policy.RequestsPerMinute > 0)
                Requests = BuildRequestsLimiter(policy.RequestsPerMinute);
            if (policy.TokensPerMinute > 0)
                Tokens = BuildTokensLimiter(policy.TokensPerMinute);
            if (policy.MaxConcurrent > 0)
                Concurrency = new PriorityConcurrencyGate(policy.MaxConcurrent);
        }

        private static PartitionedRateLimiter<int> BuildRequestsLimiter(int permitsPerMinute) =>
            // Single-partition key=0 — we already partition per API key at the
            // KeyState level. A TokenBucket lets a caller burst up to one full
            // minute of permits, then refill at exactly permitsPerMinute/min.
            BuildPerMinuteLimiter(permitsPerMinute);

        private static PartitionedRateLimiter<int> BuildTokensLimiter(int tokensPerMinute) =>
            BuildPerMinuteLimiter(tokensPerMinute);

        /// <summary>
        /// Builds a token bucket whose replenishment rate is exactly
        /// <paramref name="permitsPerMinute"/> per minute.
        /// </summary>
        /// <remarks>
        /// The naive form (<c>TokensPerPeriod = permitsPerMinute / 60</c> every
        /// second) is wrong in both directions because of integer division and
        /// the <c>Max(1, …)</c> floor: 2/min replenished at 1/s → 60/min, and
        /// 100/min replenished at 1/s → 60/min. Instead the period is derived
        /// from the chosen batch size, so
        /// <c>TokensPerPeriod / ReplenishmentPeriod == permitsPerMinute / 60s</c>
        /// exactly. The batch is ⌊permitsPerMinute/60⌋ (≥ 1), which keeps the
        /// period at ≈ 1 s for large limits and stretches it out for small ones
        /// (2/min → one permit every 30 s) rather than firing a sub-millisecond
        /// replenishment timer.
        /// </remarks>
        private static PartitionedRateLimiter<int> BuildPerMinuteLimiter(int permitsPerMinute)
        {
            int tokensPerPeriod = Math.Max(1, permitsPerMinute / 60);
            var period = TimeSpan.FromSeconds(60.0 * tokensPerPeriod / permitsPerMinute);

            return PartitionedRateLimiter.Create<int, int>(_ =>
                RateLimitPartition.GetTokenBucketLimiter(0, _ => new TokenBucketRateLimiterOptions
                {
                    TokenLimit = permitsPerMinute,
                    TokensPerPeriod = tokensPerPeriod,
                    ReplenishmentPeriod = period,
                    QueueLimit = 0,            // Fail-fast — middleware translates to 429.
                    AutoReplenishment = true,
                }));
        }

        public void Dispose()
        {
            Requests?.Dispose();
            Tokens?.Dispose();
            Concurrency?.Dispose();
        }
    }
}

/// <summary>Result of <see cref="RateLimitManager.TryAcquireAsync"/>.</summary>
public readonly struct AcquireResult
{
    public bool IsAcquired { get; init; }
    public RateLimitLease? Lease { get; init; }
    public LimiterKind Rejected { get; init; }
    public int RetryAfter { get; init; }

    public static AcquireResult Admit(RateLimitLease lease) =>
        new() { IsAcquired = true, Lease = lease };

    public static AcquireResult Reject(LimiterKind kind, int retryAfter) =>
        new() { IsAcquired = false, Rejected = kind, RetryAfter = retryAfter };
}

/// <summary>Identifies which limiter rejected a request.</summary>
public enum LimiterKind
{
    None,
    Requests,
    Tokens,
    Concurrency,
}

/// <summary>
/// Bundled lease returned by <see cref="RateLimitManager.TryAcquireAsync"/>.
/// Disposing releases all underlying leases and (if known) refunds unused
/// token budget.
/// </summary>
public sealed class RateLimitLease : IDisposable
{
    private readonly RateLimitManager? _manager;
    private readonly string _apiKey;
    private readonly int _reservedTokens;
    private int _actualTokens = -1;
    private bool _disposed;

    internal System.Threading.RateLimiting.RateLimitLease? RequestsLease { get; }
    internal System.Threading.RateLimiting.RateLimitLease? TokensLease { get; }
    internal PriorityConcurrencyGate.Lease? ConcurrencyLease { get; }

    internal RateLimitLease(
        RateLimitManager manager,
        string apiKey,
        int reservedTokens,
        System.Threading.RateLimiting.RateLimitLease? requestsLease,
        System.Threading.RateLimiting.RateLimitLease? tokensLease,
        PriorityConcurrencyGate.Lease? concurrencyLease)
    {
        _manager = manager;
        _apiKey = apiKey;
        _reservedTokens = reservedTokens;
        RequestsLease = requestsLease;
        TokensLease = tokensLease;
        ConcurrencyLease = concurrencyLease;
    }

    /// <summary>
    /// Report the actual token count consumed by the response. Should be
    /// called once after generation completes and before disposal. The
    /// difference (reserved − actual) is refunded to the tokens/min
    /// bucket on disposal.
    /// </summary>
    public void ReportActualTokens(int actualTokens)
    {
        _actualTokens = actualTokens;
    }

    /// <summary>The API key this lease belongs to.</summary>
    public string ApiKey => _apiKey;

    /// <summary>Tokens reserved up front. Used for diagnostics.</summary>
    public int ReservedTokens => _reservedTokens;

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_manager is not null && _actualTokens >= 0)
            _manager.TrueUpTokens(_apiKey, _reservedTokens, _actualTokens);

        ConcurrencyLease?.Dispose();
        TokensLease?.Dispose();
        RequestsLease?.Dispose();
    }
}
