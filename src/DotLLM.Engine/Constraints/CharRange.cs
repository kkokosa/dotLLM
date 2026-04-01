namespace DotLLM.Engine.Constraints;

/// <summary>
/// Inclusive character range [<see cref="Lo"/>..<see cref="Hi"/>].
/// Shared by regex and grammar constraint pipelines.
/// </summary>
internal readonly record struct CharRange(char Lo, char Hi)
{
    /// <summary>Returns true if <paramref name="c"/> falls within this range (inclusive).</summary>
    public bool Contains(char c) => c >= Lo && c <= Hi;
}
