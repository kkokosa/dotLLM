using System.Reflection;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// Serves the embedded web chat UI files (index.html, app.js, app.css).
/// Files are loaded into memory once at startup for zero-allocation serving.
/// </summary>
public static class WebUIEndpoint
{
    private static readonly Dictionary<string, (ReadOnlyMemory<byte> Data, string ContentType)> CachedFiles;

    static WebUIEndpoint()
    {
        var assembly = typeof(WebUIEndpoint).Assembly;
        CachedFiles = new(StringComparer.Ordinal);

        PreloadFile(assembly, "index.html", "text/html; charset=utf-8");
        PreloadFile(assembly, "app.js", "text/javascript; charset=utf-8");
        PreloadFile(assembly, "app.css", "text/css; charset=utf-8");
    }

    private static void PreloadFile(Assembly assembly, string fileName, string contentType)
    {
        var resourceName = $"DotLLM.Server.wwwroot.{fileName}";
        using var stream = assembly.GetManifestResourceStream(resourceName);
        if (stream is null) return;

        var bytes = new byte[stream.Length];
        stream.ReadExactly(bytes);
        CachedFiles[fileName] = (bytes, contentType);
    }

    public static void Map(WebApplication app)
    {
        app.MapGet("/", (HttpContext ctx) => ServeFile(ctx, "index.html"));
        app.MapGet("/app.js", (HttpContext ctx) => ServeFile(ctx, "app.js"));
        app.MapGet("/app.css", (HttpContext ctx) => ServeFile(ctx, "app.css"));
    }

    private static async Task ServeFile(HttpContext ctx, string fileName)
    {
        if (!CachedFiles.TryGetValue(fileName, out var entry))
        {
            ctx.Response.StatusCode = 404;
            return;
        }

        ctx.Response.ContentType = entry.ContentType;
        ctx.Response.Headers.CacheControl = "no-cache";
        ctx.Response.ContentLength = entry.Data.Length;
        await ctx.Response.BodyWriter.WriteAsync(entry.Data, ctx.RequestAborted);
    }
}
