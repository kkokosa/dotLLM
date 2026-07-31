using System.Text;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Tokenizers.ChatTemplates;

/// <summary>
/// Regression tests for the recursion-depth guard added to <see cref="JinjaParser"/>.
/// Without the guard, an adversarial chat template with deep paren / bracket nesting
/// blows the stack with an uncatchable <see cref="StackOverflowException"/> — the
/// model loader simply crashes the process. See upstream issue #107 item 4.
/// </summary>
public class JinjaParserRecursionTests
{
    private static JinjaTemplate Parse(string source)
    {
        var tokens = new JinjaLexer(source).Tokenize();
        return new JinjaParser(tokens).Parse();
    }

    /// <summary>
    /// A template with paren nesting well past the depth limit must raise a catchable
    /// <see cref="JinjaException"/>, not crash the process with a stack overflow.
    /// The chosen depth is well beyond the configured depth limit and enough to
    /// overflow the default thread stack on unfixed code.
    /// </summary>
    [Fact]
    public void DeeplyNestedParens_ThrowsJinjaException_NotStackOverflow()
    {
        const int depth = 5000;
        var sb = new StringBuilder(depth * 2 + 8);
        sb.Append("{{ ");
        for (int i = 0; i < depth; i++) sb.Append('(');
        sb.Append('1');
        for (int i = 0; i < depth; i++) sb.Append(')');
        sb.Append(" }}");

        var ex = Assert.Throws<JinjaException>(() => Parse(sb.ToString()));
        AssertDepthLimitMessage(ex);
    }

    /// <summary>
    /// Nested if blocks (template-body recursion) must also be guarded.
    /// </summary>
    [Fact]
    public void DeeplyNestedIfBlocks_ThrowsJinjaException_NotStackOverflow()
    {
        const int depth = 5000;
        var sb = new StringBuilder(depth * 12);
        for (int i = 0; i < depth; i++) sb.Append("{% if true %}");
        sb.Append("x");
        for (int i = 0; i < depth; i++) sb.Append("{% endif %}");

        var ex = Assert.Throws<JinjaException>(() => Parse(sb.ToString()));
        AssertDepthLimitMessage(ex);
    }

    /// <summary>
    /// Realistic chat templates with normal nesting (a couple of layers) parse fine.
    /// Verifies the guard does not regress legitimate templates.
    /// </summary>
    [Fact]
    public void NormalNesting_ParsesWithoutError()
    {
        const string template =
            "{% for msg in messages %}" +
            "  {% if msg.role == 'user' %}{{ msg.content }}{% endif %}" +
            "{% endfor %}";

        var ast = Parse(template);
        Assert.NotNull(ast);
        Assert.NotEmpty(ast.Nodes);
    }

    /// <summary>
    /// The configured depth limit is high enough that templates with merely
    /// moderate nesting (well within real-world Jinja templates) still parse.
    /// </summary>
    [Fact]
    public void NestingBelowLimit_ParsesWithoutError()
    {
        // Half the configured limit — comfortably inside it, so this must succeed.
        var sb = new StringBuilder();
        sb.Append("{{ ");
        int depth = JinjaParser.MaxRecursionDepth / 2;
        for (int i = 0; i < depth; i++) sb.Append('(');
        sb.Append('1');
        for (int i = 0; i < depth; i++) sb.Append(')');
        sb.Append(" }}");

        var ast = Parse(sb.ToString());
        Assert.NotNull(ast);
        Assert.Single(ast.Nodes);
    }

    /// <summary>
    /// The depth counter must be left consistent after the guard fires: repeatedly parsing
    /// over-nested templates must not accumulate leaked depth that eventually rejects a
    /// perfectly ordinary template.
    /// </summary>
    [Fact]
    public void RepeatedGuardTrips_DoNotLeakDepth()
    {
        var sb = new StringBuilder();
        sb.Append("{{ ");
        int depth = JinjaParser.MaxRecursionDepth + 5;
        for (int i = 0; i < depth; i++) sb.Append('(');
        sb.Append('1');
        for (int i = 0; i < depth; i++) sb.Append(')');
        sb.Append(" }}");
        string overNested = sb.ToString();

        // Far more trips than the depth limit — a one-level-per-trip leak would exceed it.
        for (int i = 0; i < JinjaParser.MaxRecursionDepth * 2; i++)
        {
            Assert.Throws<JinjaException>(() => Parse(overNested));
        }

        var ast = Parse("{{ 1 }}");
        Assert.Single(ast.Nodes);
    }

    /// <summary>
    /// Asserts the failure is the depth guard rather than some other parse error, without
    /// pinning exact wording or casing. The configured limit value is the stable signal —
    /// it is asserted from the constant, so retuning the limit cannot stale the test.
    /// </summary>
    private static void AssertDepthLimitMessage(JinjaException ex)
    {
        Assert.Contains("recursion depth", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains(JinjaParser.MaxRecursionDepth.ToString(), ex.Message, StringComparison.Ordinal);
    }
}
