using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace DotLLM.Cpu.Threading;

/// <summary>
/// A dedicated compute thread pool for CPU inference parallelism.
/// Uses N-1 background workers + caller thread = N total threads.
/// Work is dispatched via <c>delegate*</c> function pointers and stack-allocated context structs
/// for zero GC allocations on the hot path.
/// </summary>
public sealed unsafe class ComputeThreadPool : IDisposable
{
    private readonly Thread[] _workers;
    private readonly ManualResetEventSlim[] _workReady;
    private readonly CountdownEvent _completion;
    private readonly int _threadCount;

    private volatile bool _shutdown;

    // Current work item — set by Dispatch before signalling workers
    private nint _context;
    private delegate*<nint, int, int, void> _workFn;

    // Per-worker scratch buffers for attention scores etc.
    private nint[] _workerScratch;
    private int[] _workerScratchSize;
    private readonly object _scratchLock = new();

    /// <summary>Total number of threads (workers + caller).</summary>
    public int ThreadCount => _threadCount;

    /// <summary>
    /// Creates a compute thread pool with <paramref name="threadCount"/> total threads.
    /// </summary>
    /// <param name="threadCount">Total threads including caller. Must be >= 2.</param>
    public ComputeThreadPool(int threadCount)
    {
        if (threadCount < 2)
            throw new ArgumentOutOfRangeException(nameof(threadCount), "Thread pool requires at least 2 threads.");

        _threadCount = threadCount;
        int workerCount = threadCount - 1;

        _workers = new Thread[workerCount];
        _workReady = new ManualResetEventSlim[workerCount];
        _completion = new CountdownEvent(workerCount);
        _workerScratch = new nint[threadCount];
        _workerScratchSize = new int[threadCount];

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
    /// Dispatches work to all threads (caller = thread 0, workers = threads 1..N-1).
    /// The caller blocks until all threads complete their partition.
    /// <paramref name="context"/> must remain valid until Dispatch returns (stack-allocated is fine).
    /// </summary>
    /// <param name="context">Pointer to context struct on caller's stack.</param>
    /// <param name="fn">Static method matching <c>void(nint ctx, int threadIdx, int threadCount)</c>.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Dispatch(nint context, delegate*<nint, int, int, void> fn)
    {
        _context = context;
        _workFn = fn;

        int workerCount = _threadCount - 1;
        _completion.Reset(workerCount);

        // Signal all workers
        for (int i = 0; i < workerCount; i++)
            _workReady[i].Set();

        // Caller executes as thread 0
        fn(context, 0, _threadCount);

        // Wait for all workers to finish
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

        while (true)
        {
            _workReady[arrayIdx].Wait();

            if (_shutdown)
                return;

            _workReady[arrayIdx].Reset();

            try
            {
                _workFn(_context, threadIdx, _threadCount);
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

    /// <inheritdoc/>
    public void Dispose()
    {
        _shutdown = true;

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
