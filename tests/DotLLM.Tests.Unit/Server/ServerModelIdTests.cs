using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

public sealed class ServerModelIdTests
{
    [Theory]
    [InlineData("model.gguf", "model")]
    [InlineData("./models/model.gguf", "model")]
    [InlineData("/srv/models/model.Q8_0.gguf", "model.Q8_0")]
    [InlineData(@"C:\models\model.GGUF", "model")]
    [InlineData("owner/model-repo", "model-repo")]
    [InlineData("owner/model.v1", "model.v1")]
    public void Parse_DerivesConsistentModelId(string model, string expected)
    {
        var options = ServerOptions.Parse(["--model", model]);

        Assert.Equal(model, options.Model);
        Assert.Equal(expected, options.ModelId);
    }
}
