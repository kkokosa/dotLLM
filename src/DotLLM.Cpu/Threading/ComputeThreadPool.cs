using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Cpu.Threading;

/// <summary>
/// A dedicated compute thread pool for CPU inference parallelism.
/// Uses N-1 background workers + caller thread = N total threads.
/// Work is dispatched via <c>delegate*</c> function pointers and stack-allocated context structs
/// for zero GC allocations on the hot path.
/// </summary>
/// <remarks>
/// Supports two dispatch modes:
/// <list type="bullet">
/// <item><see cref="DispatchMode.EventBased"/> — workers wait on <see cref="ManualResetEventSlim"/> (prefill)</item>
/// <item><see cref="DispatchMode.SpinWait"/> — workers spin on a volatile generation counter (decode)</item>
/// </list>
/// Optionally pins workers to specific logical processors for NUMA locality and P-core affinity.
/// </remarks>
public sealed unsafe class ComputeThreadPool : IDisposable
{
    /// <summary>Number of spin iterations before falling back to event wait in spin-wait mode.</summary>
    private const int SpinIterations = 10_000;

    private readonly Thread[] _workers;
    private readonly ManualResetEventSlim[] _workReady;
    private readonly CountdownEvent _completion;
    private readonly int _threadCount;
    private readonly int _decodeThreadCount;
    private readonly int[] _workerCoreAssignment; // maps worker index → logical processor ID (or -1)
    private readonly int _callerCoreId;           // -1 if caller thread should not be pinned

    private volatile bool _shutdown;
    private int _dispatchGeneration;
    private volatile DispatchMode _currentMode;
    private volatile int _activeWorkerCount; // number of active workers (not including caller)

    // Caller-thread pinning state (see PinCallerThread).
    // _callerPinAttempted: 0 = first-dispatch needed, 1 = already attempted (success or skip).
    private int _callerPinAttempted;
    private volatile bool _callerPinSucceeded;

    // Current work item — set by Dispatch before signalling workers
    private nint _context;
    private delegate*<nint, int, int, void> _workFn;
    private volatile int _dispatchThreadCount; // total active threads for current dispatch (workers + caller)

    // Per-worker scratch buffers for attention scores etc.
    private nint[] _workerScratch;
    private int[] _workerScratchSize;
    private readonly object _scratchLock = new();

    /// <summary>Total number of threads (workers + caller).</summary>
    public int ThreadCount => _threadCount;

    /// <summary>
    /// Whether the caller (inference) thread has been successfully pinned to a logical
    /// processor. <c>false</c> when pinning is disabled, unsupported on the current OS,
    /// the caller ran on a ThreadPool thread, or the pin syscall failed. Exposed for
    /// observability and testing.
    /// </summary>
    public bool CallerThreadPinned => _callerPinSucceeded;

    /// <summary>
    /// Creates a compute thread pool with <paramref name="threadCount"/> total threads.
    /// </summary>
    /// <param name="threadCount">Total threads including caller. Must be >= 2.</param>
    public ComputeThreadPool(int threadCount)
        : this(threadCount, topology: null, config: default)
    {
    }

    /// <summary>
    /// Creates a compute thread pool with NUMA topology awareness and threading configuration.
    /// </summary>
    /// <param name="threadCount">Total threads including caller. Must be >= 2.</param>
    /// <param name="topology">Optional NUMA topology for CPU pinning. Null = no pinning.</param>
    /// <param name="config">Threading configuration for decode thread count and pinning options.</param>
    public ComputeThreadPool(int threadCount, NumaTopology? topology, ThreadingConfig config)
    {
        if (threadCount < 2)
            throw new ArgumentOutOfRangeException(nameof(threadCount), "Thread pool requires at least 2 threads.");

        _threadCount = threadCount;
        int workerCount = threadCount - 1;
        _activeWorkerCount = workerCount;

        // Compute decode thread count using the pool's actual threadCount, not the config's EffectiveThreadCount
        // (which may be wrong when using the simple constructor with default config).
        // When topology is null (no NUMA detection), don't apply the memory channel heuristic —
        // use all configured threads. The cap only makes sense with real topology data.
        _decodeThreadCount = config.DecodeThreadCount > 0
            ? Math.Clamp(config.DecodeThreadCount, 2, threadCount)
            : topology is not null
                ? Math.Clamp(topology.MemoryChannelEstimate, 2, threadCount)
                : threadCount;

        // Build core assignment map (and caller core if pinning is enabled).
        (int[] workerAssignment, int callerCore) = BuildCoreAssignment(workerCount, topology, config);
        _workerCoreAssignment = workerAssignment;
        _callerCoreId = config.EnableCallerPinning ? callerCore : -1;

        _workers = new Thread[workerCount];
        _workReady = new ManualResetEventSlim[workerCount];
        _completion = new CountdownEvent(workerCount);
        _workerScratch = new nint[threadCount];
        _workerScratchSize = new int[threadCount];

        _currentMode = DispatchMode.EventBased;
        _dispatchGeneration = 0;

        for (int i = 0; i < workerCount; i++)
        {
            _workReady[i] = new ManualResetEventSlim(false, spinCount: 100);
            int workerIdx = i + 1; // worker 0 is the caller thread
            _workers[i] = new Thread(WorkerLoop)
            {
                IsBackground = true,
                Name = $"dotLLM-compute-{workerIdx}"
            };
            _workers[i].Start(i);
        }
    }

    /// <summary>
    /// Sets the dispatch mode for subsequent <see cref="Dispatch"/> calls.
    /// In <see cref="DispatchMode.SpinWait"/> mode, also reduces active worker count
    /// to the decode thread cap (memory-bandwidth-bound decode doesn't benefit from excess threads).
    /// </summary>
    /// <remarks>
    /// Thread safety: SetDispatchMode is always called before Dispatch, which issues
    /// a full fence via Interlocked.Increment on _dispatchGeneration. The volatile
    /// writes to _currentMode and _activeWorkerCount are therefore safely visible
    /// to worker threads before they process the next dispatch.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SetDispatchMode(DispatchMode mode)
    {
        _currentMode = mode;
        _activeWorkerCount = mode == DispatchMode.SpinWait
            ? Math.Clamp(_decodeThreadCount - 1, 1, _threadCount - 1)
            : _threadCount - 1;
    }

    /// <summary>
    /// Dispatches work to all active threads (caller = thread 0, workers = threads 1..N-1).
    /// The caller blocks until all active threads complete their partition.
    /// <paramref name="context"/> must remain valid until Dispatch returns (stack-allocated is fine).
    /// </summary>
    /// <param name="context">Pointer to context struct on caller's stack.</param>
    /// <param name="fn">Static method matching <c>void(nint ctx, int threadIdx, int threadCount)</c>.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Dispatch(nint context, delegate*<nint, int, int, void> fn)
    {
        // Lazily pin the caller thread on the first dispatch. Doing this in the constructor
        // would be wrong because the inference loop may run on a different thread than the
        // one that built the pool.
        if (_callerPinAttempted == 0)
            PinCallerThread();

        int activeWorkers = _activeWorkerCount;
        int totalActive = activeWorkers + 1; // workers + caller

        _context = context;
        _workFn = fn;
        _dispatchThreadCount = totalActive;

        if (activeWorkers > 0)
            _completion.Reset(activeWorkers);

        // Increment generation counter (wakes spinners)
        Interlocked.Increment(ref _dispatchGeneration);

        // Also set events (wakes any workers that fell through to kernel wait)
        for (int i = 0; i < activeWorkers; i++)
            _workReady[i].Set();

        // Caller executes as thread 0
        fn(context, 0, totalActive);

        // Wait for all active workers to finish
        if (activeWorkers > 0)
            _completion.Wait();
    }

    /// <summary>
    /// Returns a per-worker scratch buffer of at least <paramref name="minBytes"/> bytes (64-byte aligned).
    /// The buffer is lazily allocated and cached — resized only if the current one is too small.
    /// Thread-safe: each thread only accesses its own slot.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public nint GetWorkerScratch(int threadIdx, int minBytes)
    {
        if (_workerScratchSize[threadIdx] >= minBytes)
            return _workerScratch[threadIdx];

        return GrowWorkerScratch(threadIdx, minBytes);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private nint GrowWorkerScratch(int threadIdx, int minBytes)
    {
        if (_workerScratch[threadIdx] != 0)
            NativeMemory.AlignedFree((void*)_workerScratch[threadIdx]);

        var ptr = (nint)NativeMemory.AlignedAlloc((nuint)minBytes, 64);
        _workerScratch[threadIdx] = ptr;
        _workerScratchSize[threadIdx] = minBytes;
        return ptr;
    }

    private void WorkerLoop(object? state)
    {
        int arrayIdx = (int)state!;
        int threadIdx = arrayIdx + 1;

        // Pin thread to assigned core if configured
        if (_workerCoreAssignment[arrayIdx] >= 0)
            CpuAffinity.PinCurrentThread(_workerCoreAssignment[arrayIdx]);

        int lastGeneration = Volatile.Read(ref _dispatchGeneration);

        while (true)
        {
            int previousGen = lastGeneration;

            if (_currentMode == DispatchMode.SpinWait && arrayIdx < _activeWorkerCount)
            {
                // Spin-wait: check generation counter
                bool gotWork = false;
                for (int spin = 0; spin < SpinIterations; spin++)
                {
                    int gen = Volatile.Read(ref _dispatchGeneration);
                    if (gen != lastGeneration)
                    {
                        lastGeneration = gen;
                        gotWork = true;
                        break;
                    }
                    // Intentional Thread.SpinWait(1) over SpinWait struct — avoids
                    // OS yields during the spin budget (struct escalates to Thread.Sleep).
                    Thread.SpinWait(1);
                }

                if (!gotWork)
                {
                    // Fallback to event wait after spin budget exhausted
                    _workReady[arrayIdx].Wait();

                    if (_shutdown) return;

                    _workReady[arrayIdx].Reset();
                    lastGeneration = Volatile.Read(ref _dispatchGeneration);
                }
                else
                {
                    // Dispose increments generation to wake spinners — check before proceeding
                    if (_shutdown) return;

                    // Consume the event if it was also set (avoid stale signal on next iteration)
                    if (_workReady[arrayIdx].IsSet)
                        _workReady[arrayIdx].Reset();
                }
            }
            else
            {
                // Event-based wait (or inactive worker in decode mode)
                _workReady[arrayIdx].Wait();

                if (_shutdown) return;

                _workReady[arrayIdx].Reset();
                lastGeneration = Volatile.Read(ref _dispatchGeneration);
            }

            // Guard against stale event wake-ups: only process if a new dispatch occurred.
            // Race scenario: worker detects gen change via spin, processes, loops back,
            // then gets woken by the stale event from the same Dispatch (set after gen increment).
            if (lastGeneration == previousGen)
                continue;

            // Check if this worker is active for the current dispatch
            if (arrayIdx >= _activeWorkerCount)
                continue;

            try
            {
                _workFn(_context, threadIdx, _dispatchThreadCount);
            }
            catch (Exception ex)
            {
                Environment.FailFast($"dotLLM compute worker {threadIdx} crashed", ex);
            }
            finally
            {
                _completion.Signal();
            }
        }
    }

    /// <summary>
    /// Pins the calling thread to <see cref="_callerCoreId"/>, one-shot and idempotent.
    /// Safe to call multiple times (second and subsequent calls are no-ops). Automatically
    /// invoked on the first <see cref="Dispatch"/> call.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This must run on the thread that drives the inference loop, not on the thread that
    /// constructed the pool — hence the deferred first-dispatch invocation.
    /// </para>
    /// <para>
    /// Pinning is <b>skipped</b> when the calling thread is a <c>ThreadPool</c> thread —
    /// sticking affinity to a pool thread would corrupt the pool for the rest of the process.
    /// In that case the method is a no-op and <see cref="CallerThreadPinned"/> stays <c>false</c>.
    /// </para>
    /// </remarks>
    public void PinCallerThread()
    {
        // One-shot: ensure only the first caller runs the pinning logic.
        if (Interlocked.CompareExchange(ref _callerPinAttempted, 1, 0) != 0)
            return;

        if (_callerCoreId < 0)
            return;

        if (Thread.CurrentThread.IsThreadPoolThread)
        {
            System.Diagnostics.Trace.TraceWarning(
                "ComputeThreadPool: skipping caller-thread pinning — current thread is a ThreadPool thread.");
            return;
        }

        _callerPinSucceeded = CpuAffinity.PinCurrentThread(_callerCoreId);
    }

    /// <summary>
    /// Builds the core assignment array mapping each worker index to a logical processor ID,
    /// plus the logical processor ID reserved for the caller thread (or <c>-1</c> if no
    /// pinning is configured).
    /// </summary>
    private static (int[] Assignment, int CallerCore) BuildCoreAssignment(
        int workerCount, NumaTopology? topology, ThreadingConfig config)
    {
        var assignment = new int[workerCount];
        Array.Fill(assignment, -1); // -1 = no pinning

        if (topology is null)
            return (assignment, -1);

        if (!config.EnableNumaPinning && !config.EnablePCorePinning)
            return (assignment, -1);

        IReadOnlyList<int> candidateCores;
        if (config.EnablePCorePinning && topology.IsHybrid)
            candidateCores = topology.PerformanceCoreIds;
        else if (config.EnableNumaPinning)
        {
            // Round-robin across NUMA nodes
            var cores = new List<int>();
            var nodeQueues = topology.ProcessorsByNumaNode
                .OrderBy(kv => kv.Key)
                .Select(kv => new Queue<int>(kv.Value))
                .ToList();

            while (cores.Count < workerCount + 1)
            {
                bool added = false;
                foreach (var q in nodeQueues)
                {
                    if (q.Count > 0)
                    {
                        cores.Add(q.Dequeue());
                        added = true;
                    }
                }
                if (!added) break;
            }

            candidateCores = cores;
        }
        else
        {
            return (assignment, -1);
        }

        // Reserve candidateCores[0] for the caller thread; workers start at index 1.
        int callerCore = candidateCores.Count > 0 ? candidateCores[0] : -1;
        for (int i = 0; i < workerCount; i++)
        {
            int coreIdx = i + 1; // skip one for caller
            if (coreIdx < candidateCores.Count)
                assignment[i] = candidateCores[coreIdx];
        }

        return (assignment, callerCore);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _shutdown = true;

        // Bump generation to wake any spinners
        Interlocked.Increment(ref _dispatchGeneration);

        // Signal all workers to exit
        for (int i = 0; i < _workers.Length; i++)
            _workReady[i].Set();

        // Join all workers
        for (int i = 0; i < _workers.Length; i++)
            _workers[i].Join();

        // Dispose synchronization primitives
        for (int i = 0; i < _workReady.Length; i++)
            _workReady[i].Dispose();
        _completion.Dispose();

        // Free scratch buffers
        for (int i = 0; i < _workerScratch.Length; i++)
        {
            if (_workerScratch[i] != 0)
            {
                NativeMemory.AlignedFree((void*)_workerScratch[i]);
                _workerScratch[i] = 0;
            }
        }
    }
}
