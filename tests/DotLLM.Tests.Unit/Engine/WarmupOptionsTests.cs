using DotLLM.Engine;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

public sealed class WarmupOptionsTests
{
    [Fact]
    public void Default_IsEnabled_With3Iterations()
    {
        var options = WarmupOptions.Default;

        Assert.True(options.Enabled);
        Assert.Equal(3, options.Iterations);
        Assert.Equal(16, options.MaxTokens);
        Assert.False(string.IsNullOrEmpty(options.DummyPrompt));
    }

    [Fact]
    public void Disabled_IsNotEnabled()
    {
        var options = WarmupOptions.Disabled;

        Assert.False(options.Enabled);
    }

    [Fact]
    public void WithExpression_OverridesProperties()
    {
        var options = WarmupOptions.Default with { Iterations = 5, MaxTokens = 32 };

        Assert.True(options.Enabled);
        Assert.Equal(5, options.Iterations);
        Assert.Equal(32, options.MaxTokens);
    }

    [Fact]
    public void Disabled_PresentsDefaults_ForOtherProperties()
    {
        var options = WarmupOptions.Disabled;

        Assert.Equal(3, options.Iterations);
        Assert.Equal(16, options.MaxTokens);
        Assert.False(string.IsNullOrEmpty(options.DummyPrompt));
    }
}
