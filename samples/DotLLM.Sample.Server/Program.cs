using DotLLM.Engine;
using DotLLM.Server;

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: DotLLM.Sample.Server <model.gguf> [--port 8080] [--threads N] [--decode-threads N] [--no-prompt-cache] [--no-warmup]");
    return 1;
}

var options = ServerOptions.Parse(args);

Console.WriteLine($"Loading model: {options.Model}");
var resolvedPath = ServerStartup.ResolveModelPath(options.Model, options.Quant)
    ?? options.Model;

var state = ServerStartup.LoadModel(resolvedPath, options);
var app = ServerStartup.BuildApp(state, [], serveUi: true);

var url = $"http://{options.Host}:{options.Port}";
Console.WriteLine($"Model: {state.Config!.Architecture}, {state.Config.NumLayers} layers");
Console.WriteLine($"Server listening on {url}");
Console.WriteLine("Endpoints: /v1/chat/completions, /v1/completions, /v1/models");

app.Run(url);
state.Dispose();
return 0;
