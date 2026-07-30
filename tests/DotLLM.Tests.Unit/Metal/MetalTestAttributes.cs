using System.Runtime.InteropServices;

namespace DotLLM.Tests.Unit.Metal;

public sealed class MetalTestFactAttribute : Xunit.FactAttribute
{
    public MetalTestFactAttribute()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ||
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            Skip = "Metal tests require macOS.";
        }
    }
}
