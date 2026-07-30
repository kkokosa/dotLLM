using System.Runtime.CompilerServices;

namespace DotLLM.Core.Diagnostics;

/// <summary>
/// Registry for <see cref="IInferenceHook"/> instances, fired by the forward pass at
/// well-defined <see cref="HookPoint"/> locations.
/// </summary>
/// <remarks>
/// <para>
/// Registration and unregistration are thread-safe (locked). Hot-path methods
/// (<see cref="HasHookAt(HookPoint)"/>, <see cref="Fire"/>) are lock-free and allocation-free; the
/// per-point hook list is replaced atomically on mutation (copy-on-write) so iteration
/// is safe without locking.
/// </para>
/// <para>
/// The intended hot-path usage is a null/flag guard that elides the call entirely when
/// no hooks are registered:
/// <code>
/// if (_hooks is not null &amp;&amp; _hooks.HasHookAt(HookPoint.PostLayer))
///     _hooks.Fire(HookPoint.PostLayer, activation, ctx);
/// </code>
/// </para>
/// </remarks>
public sealed class HookRegistry
{
    private const int HookPointCount = 8;

    // One ordered list of hooks per HookPoint. Indexed by (int)HookPoint.
    // Lists are replaced (not mutated) on register/unregister so Fire/HasHookAt can read
    // them without locking; readers see either the old or the new array atomically.
    private readonly IInferenceHook[]?[] _hooksByPoint = new IInferenceHook[]?[HookPointCount];

    // Mirror flags for O(1) `HasHookAt` without dereferencing the list array.
    // Volatile read in HasHookAt is implicit via field load on .NET; we publish via the
    // assignment after the list update, paired by the lock in mutators.
    private readonly bool[] _hasHookAt = new bool[HookPointCount];

    private readonly object _mutationLock = new();

    /// <summary>
    /// Registers an inference hook at its declared <see cref="IInferenceHook.HookPoint"/>.
    /// Hooks registered at the same point fire in registration order.
    /// </summary>
    /// <param name="hook">The hook to register.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="hook"/> is null.</exception>
    public void Register(IInferenceHook hook)
    {
        ArgumentNullException.ThrowIfNull(hook);
        int idx = (int)hook.HookPoint;

        lock (_mutationLock)
        {
            var current = _hooksByPoint[idx];
            IInferenceHook[] updated;
            if (current is null || current.Length == 0)
            {
                updated = new[] { hook };
            }
            else
            {
                updated = new IInferenceHook[current.Length + 1];
                Array.Copy(current, updated, current.Length);
                updated[current.Length] = hook;
            }

            _hooksByPoint[idx] = updated;
            _hasHookAt[idx] = true;
        }
    }

    /// <summary>
    /// Unregisters a previously-registered hook. Returns <c>true</c> if the hook was found
    /// and removed; <c>false</c> otherwise.
    /// </summary>
    /// <param name="hook">The hook to unregister.</param>
    /// <returns><c>true</c> when removal occurred, <c>false</c> when the hook was not registered.</returns>
    public bool Unregister(IInferenceHook hook)
    {
        ArgumentNullException.ThrowIfNull(hook);
        int idx = (int)hook.HookPoint;

        lock (_mutationLock)
        {
            var current = _hooksByPoint[idx];
            if (current is null || current.Length == 0) return false;

            int found = -1;
            for (int i = 0; i < current.Length; i++)
            {
                if (ReferenceEquals(current[i], hook))
                {
                    found = i;
                    break;
                }
            }

            if (found < 0) return false;

            if (current.Length == 1)
            {
                _hooksByPoint[idx] = null;
                _hasHookAt[idx] = false;
                return true;
            }

            var updated = new IInferenceHook[current.Length - 1];
            if (found > 0) Array.Copy(current, 0, updated, 0, found);
            if (found < current.Length - 1)
                Array.Copy(current, found + 1, updated, found, current.Length - found - 1);

            _hooksByPoint[idx] = updated;
            // _hasHookAt[idx] stays true — there's at least one hook left.
            return true;
        }
    }

    /// <summary>
    /// Returns <c>true</c> if at least one hook is registered at the given point.
    /// O(1) hot-path check intended to gate <see cref="Fire"/> calls.
    /// </summary>
    /// <param name="point">The hook point to query.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool HasHookAt(HookPoint point) => _hasHookAt[(int)point];

    /// <summary>
    /// Returns <c>true</c> if at least one hook is registered at the given point.
    /// The <paramref name="layer"/> argument is accepted for call-site clarity but is not
    /// used for filtering — hooks self-filter by layer via <see cref="HookContext.LayerIndex"/>.
    /// </summary>
    /// <param name="point">The hook point to query.</param>
    /// <param name="layer">Layer index (unused; accepted for documentation at call sites).</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool HasHookAt(HookPoint point, int layer)
    {
        _ = layer;
        return _hasHookAt[(int)point];
    }

    /// <summary>
    /// Fires all hooks registered at <paramref name="point"/> in registration order, threading
    /// any <see cref="HookResult.ReplaceResult"/> output through the buffer for downstream hooks
    /// and downstream computation.
    /// </summary>
    /// <param name="point">The hook point being reached.</param>
    /// <param name="activation">In-place activation buffer. Replacements are copied here.</param>
    /// <param name="context">Per-fire context (layer, position, sequence, step).</param>
    /// <exception cref="InvalidOperationException">
    /// Thrown when a hook returns a <see cref="HookResult.ReplaceResult"/> whose payload length
    /// does not match <paramref name="activation"/>.
    /// </exception>
    public void Fire(HookPoint point, Span<float> activation, in HookContext context)
    {
        var hooks = _hooksByPoint[(int)point];
        if (hooks is null) return;

        for (int i = 0; i < hooks.Length; i++)
        {
            var result = hooks[i].OnActivation(activation, context);
            if (result is HookResult.ReplaceResult replace)
            {
                var replacement = replace.Activation;
                if (replacement.Length != activation.Length)
                {
                    throw new InvalidOperationException(
                        $"Hook at {point} returned a replacement of length {replacement.Length}, " +
                        $"but the activation buffer has length {activation.Length}.");
                }
                replacement.AsSpan().CopyTo(activation);
            }
        }
    }

    /// <summary>
    /// Returns the number of hooks currently registered at <paramref name="point"/>.
    /// Intended for tests and diagnostics; not on the hot path.
    /// </summary>
    /// <param name="point">The hook point to query.</param>
    public int CountAt(HookPoint point)
    {
        var hooks = _hooksByPoint[(int)point];
        return hooks?.Length ?? 0;
    }
}
