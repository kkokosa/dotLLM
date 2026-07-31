using System.Collections.Generic;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Regression tests for cycle detection in <see cref="JinjaEvaluator.Stringify"/>.
/// Without cycle tracking, a self-referencing list or dict produced by the host
/// context (e.g. an embedding loop in the application's data) sent Stringify into
/// an infinite recursive loop until a stack overflow killed the process. See
/// upstream issue #107 item 5.
/// </summary>
public class JinjaEvaluatorCycleTests
{
    /// <summary>
    /// A list that references itself as an element must produce a catchable
    /// <see cref="JinjaException"/> rather than infinite recursion.
    /// </summary>
    [Fact]
    public void Stringify_SelfReferencingList_ThrowsJinjaException()
    {
        var list = new List<object?> { "before" };
        list.Add(list); // self-reference
        list.Add("after");

        var ex = Assert.Throws<JinjaException>(() => JinjaEvaluator.Stringify(list));
        Assert.Contains("Circular reference", ex.Message);
    }

    /// <summary>
    /// A dict that contains itself as a value must produce a catchable
    /// <see cref="JinjaException"/> rather than infinite recursion.
    /// </summary>
    [Fact]
    public void Stringify_SelfReferencingDict_ThrowsJinjaException()
    {
        var dict = new Dictionary<string, object?>
        {
            ["name"] = "outer",
        };
        dict["self"] = dict; // self-reference

        var ex = Assert.Throws<JinjaException>(() => JinjaEvaluator.Stringify(dict));
        Assert.Contains("Circular reference", ex.Message);
        Assert.Contains("dictionary", ex.Message); // container kind named consistently, not abbreviated
    }

    /// <summary>
    /// Mutually referencing list/dict (A contains B, B contains A) — the indirect
    /// cycle must also be caught.
    /// </summary>
    [Fact]
    public void Stringify_MutuallyReferencingListDict_ThrowsJinjaException()
    {
        var list = new List<object?>();
        var dict = new Dictionary<string, object?> { ["list"] = list };
        list.Add(dict);

        var ex = Assert.Throws<JinjaException>(() => JinjaEvaluator.Stringify(list));
        Assert.Contains("Circular reference", ex.Message);
    }

    /// <summary>
    /// Non-cyclic nested data must continue to stringify cleanly — the same
    /// inner list referenced twice (DAG, not cycle) must not be rejected.
    /// </summary>
    [Fact]
    public void Stringify_DagWithSharedInnerList_DoesNotThrow()
    {
        var inner = new List<object?> { "x", "y" };
        var outer = new List<object?> { inner, inner }; // shared but not cyclic

        var result = JinjaEvaluator.Stringify(outer);
        Assert.Contains("['x', 'y']", result);
    }

    /// <summary>
    /// Normal nested structures continue to render correctly.
    /// </summary>
    [Fact]
    public void Stringify_NonCyclicNesting_Works()
    {
        var data = new List<object?>
        {
            "a",
            new List<object?> { "b", "c" },
            new Dictionary<string, object?> { ["k"] = "v" },
        };

        var result = JinjaEvaluator.Stringify(data);
        Assert.Contains("a", result);
        Assert.Contains("b", result);
        Assert.Contains("v", result);
    }
}
