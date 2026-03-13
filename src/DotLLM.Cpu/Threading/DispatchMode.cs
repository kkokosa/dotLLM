namespace DotLLM.Cpu.Threading;

/// <summary>
/// Controls how worker threads are notified of new work in <see cref="ComputeThreadPool"/>.
/// </summary>
public enum DispatchMode : byte
{
    /// <summary>
    /// Workers wait on <see cref="System.Threading.ManualResetEventSlim"/> (kernel transition).
    /// Best for prefill where dispatch intervals are long relative to kernel transition cost.
    /// </summary>
    EventBased = 0,

    /// <summary>
    /// Workers spin on a volatile generation counter, avoiding kernel transitions.
    /// Falls back to event wait after ~10K iterations to avoid wasting CPU on idle.
    /// Best for decode where dispatches are frequent and short.
    /// </summary>
    SpinWait = 1,
}
