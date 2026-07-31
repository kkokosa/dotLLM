using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace DotLLM.Server.RateLimiting;

/// <summary>
/// Per-key concurrency limiter where waiters are ordered by
/// <see cref="RequestPriority"/> instead of FIFO arrival.
/// </summary>
/// <remarks>
/// <para>
/// The .NET BCL <c>ConcurrencyLimiter</c> serves waiters strictly in arrival
/// order. For LLM serving we want a latency-sensitive tier to jump ahead of
/// background traffic when capacity is contested. This gate keeps an internal
/// max-heap (<see cref="PriorityQueue{TElement, TPriority}"/>) of pending
/// <see cref="Waiter"/>s and pops the highest-priority waiter every time a
/// slot frees.
/// </para>
/// <para>
/// Ties between equal priorities are broken by arrival order — FIFO within a
/// single priority class — by composing the priority key with a monotonic
/// sequence number.
/// </para>
/// </remarks>
internal sealed class PriorityConcurrencyGate : IDisposable
{
    private readonly object _lock = new();
    private readonly int _max;
    // .NET PriorityQueue is a min-heap. We negate RequestPriority so the
    // highest-priority waiter has the smallest key. The seqno tiebreaker
    // preserves FIFO within a priority class.
    private readonly PriorityQueue<Waiter, (int negPriority, long seqno)> _queue = new();
    private int _active;
    private long _seqno;
    private bool _disposed;

    public PriorityConcurrencyGate(int maxConcurrent)
    {
        if (maxConcurrent <= 0)
            throw new ArgumentOutOfRangeException(nameof(maxConcurrent));
        _max = maxConcurrent;
    }

    /// <summary>Total slots configured for this gate.</summary>
    public int MaxConcurrent => _max;

    /// <summary>Current number of leases held (excluding queued waiters).</summary>
    public int ActiveCount
    {
        get { lock (_lock) return _active; }
    }

    /// <summary>Current number of waiters parked in the priority queue.</summary>
    public int QueueLength
    {
        get { lock (_lock) return _queue.Count; }
    }

    /// <summary>
    /// Acquire one slot. If capacity is available the lease is granted
    /// synchronously. Otherwise the task completes when either (a) a slot
    /// frees and this waiter has the highest priority (resolves to a
    /// <see cref="Lease"/>), or (b) <paramref name="timeout"/> elapses
    /// (resolves to <c>null</c>), or (c) <paramref name="ct"/> fires
    /// (throws <see cref="OperationCanceledException"/>).
    /// </summary>
    public ValueTask<Lease?> AcquireAsync(RequestPriority priority, TimeSpan timeout, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        Waiter waiter;

        lock (_lock)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(PriorityConcurrencyGate));

            if (_active < _max)
            {
                _active++;
                return new ValueTask<Lease?>(new Lease(this));
            }

            waiter = new Waiter();
            _queue.Enqueue(waiter, (-(int)priority, _seqno++));
        }

        return waiter.WaitAsync(timeout, ct, this);
    }

    private void Release()
    {
        Waiter? next = null;
        lock (_lock)
        {
            // Hand the freed slot to the highest-priority waiter that hasn't
            // already cancelled. Discard cancelled waiters as we go.
            while (_queue.TryDequeue(out var candidate, out _))
            {
                if (candidate.TryReserve())
                {
                    next = candidate;
                    break;
                }
            }

            if (next is null)
            {
                // No takers — give the slot back.
                _active--;
                return;
            }
            // Else: _active stays unchanged — ownership transfers to `next`.
        }

        // Deliver the lease outside the lock so user continuations don't run
        // while we hold it.
        next.Deliver(new Lease(this));
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        Waiter[] toFault;
        lock (_lock)
        {
            _disposed = true;
            toFault = new Waiter[_queue.Count];
            int i = 0;
            while (_queue.TryDequeue(out var w, out _))
                toFault[i++] = w;
        }
        foreach (var w in toFault)
            w.TryFault(new ObjectDisposedException(nameof(PriorityConcurrencyGate)));
    }

    /// <summary>Disposable lease — returns one slot to the gate.</summary>
    public sealed class Lease : IDisposable
    {
        private PriorityConcurrencyGate? _owner;

        internal Lease(PriorityConcurrencyGate owner) => _owner = owner;

        public void Dispose()
        {
            var owner = Interlocked.Exchange(ref _owner, null);
            owner?.Release();
        }
    }

    /// <summary>
    /// A queued waiter. Lifecycle:
    ///   <list type="bullet">
    ///   <item><description>Pending → Reserved (slot owner runs <see cref="Deliver"/>)</description></item>
    ///   <item><description>Pending → Cancelled (timeout/ct fires before reservation)</description></item>
    ///   </list>
    /// </summary>
    private sealed class Waiter
    {
        private readonly TaskCompletionSource<Lease?> _tcs =
            new(TaskCreationOptions.RunContinuationsAsynchronously);
        private int _state; // 0=pending, 1=reserved, 2=cancelled

        public async ValueTask<Lease?> WaitAsync(TimeSpan timeout, CancellationToken ct, PriorityConcurrencyGate gate)
        {
            using var timeoutCts = new CancellationTokenSource(timeout);
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct, timeoutCts.Token);

            await using var _ = linkedCts.Token.Register(static state =>
            {
                var w = (Waiter)state!;
                if (Interlocked.CompareExchange(ref w._state, 2, 0) == 0)
                {
                    // Cancelled before being reserved. Complete with null so
                    // the awaiter can distinguish timeout (returns null) from
                    // external cancellation (re-thrown below).
                    w._tcs.TrySetResult(null);
                }
            }, this).ConfigureAwait(false);

            var result = await _tcs.Task.ConfigureAwait(false);
            if (result is null)
            {
                // External cancellation propagates; timeout returns null.
                ct.ThrowIfCancellationRequested();
            }
            return result;
        }

        /// <summary>
        /// Called from <see cref="Release"/> while holding the gate lock.
        /// Returns <c>true</c> if this waiter has not yet cancelled and is
        /// now reserved (the caller must call <see cref="Deliver"/> outside
        /// the lock).
        /// </summary>
        public bool TryReserve() => Interlocked.CompareExchange(ref _state, 1, 0) == 0;

        /// <summary>Complete the awaiting task with a lease. Must be called once after TryReserve succeeds.</summary>
        public void Deliver(Lease lease) => _tcs.TrySetResult(lease);

        public bool TryFault(Exception ex)
        {
            if (Interlocked.CompareExchange(ref _state, 2, 0) != 0)
                return false;
            return _tcs.TrySetException(ex);
        }
    }
}
