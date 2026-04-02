using System.Reflection;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// Serves the embedded web chat UI files (index.html, app.js, app.css).
/// </summary>
public static class WebUIEndpoint
{
    private static readonly Assembly ServerAssembly = typeof(WebUIEndpoint).Assembly;

    public static void Map(WebApplication app)
    {
        app.MapGet("/", (HttpContext ctx) =>
            ServeEmbeddedFile(ctx, "index.html", "text/html; charset=utf-8"));

        app.MapGet("/app.js", (HttpContext ctx) =>
            ServeEmbeddedFile(ctx, "app.js", "text/javascript; charset=utf-8"));

        app.MapGet("/app.css", (HttpContext ctx) =>
            ServeEmbeddedFile(ctx, "app.css", "text/css; charset=utf-8"));
    }

    private static async Task ServeEmbeddedFile(HttpContext ctx, string fileName, string contentType)
    {
        var resourceName = $"DotLLM.Server.wwwroot.{fileName}";
        var stream = ServerAssembly.GetManifestResourceStream(resourceName);
        if (stream is null)
        {
            ctx.Response.StatusCode = 404;
            return;
        }

        ctx.Response.ContentType = contentType;
        ctx.Response.Headers.CacheControl = "no-cache";
        await using (stream)
        {
            await stream.CopyToAsync(ctx.Response.Body, ctx.RequestAborted);
        }
    }
}
