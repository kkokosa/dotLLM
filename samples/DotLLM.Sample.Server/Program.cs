using DotLLM.Engine;
using DotLLM.Server;

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: DotLLM.Sample.Server <model.gguf> [--port 8080]");
    Console.Error.WriteLine("  model.gguf  Path to a GGUF model file");
    Console.Error.WriteLine("  --port N    Port to listen on (default: 8080)");
    return 1;
}

string modelPath = args[0];
int port = 8080;
for (int i = 1; i < args.Length - 1; i++)
{
    if (args[i] == "--port" && int.TryParse(args[i + 1], out var p))
        port = p;
}

var options = new ServerOptions
{
    Model = modelPath,
    Port = port,
    Warmup = WarmupOptions.Disabled,
};

Console.WriteLine($"Loading model: {modelPath}");
var resolvedPath = ServerStartup.ResolveModelPath(options.Model, options.Quant)
    ?? modelPath;

var state = ServerStartup.LoadModel(resolvedPath, options);
var app = ServerStartup.BuildApp(state, args, serveUi: true);

var url = $"http://{options.Host}:{options.Port}";
Console.WriteLine($"Model: {state.Config!.Architecture}, {state.Config.NumLayers} layers");
Console.WriteLine($"Server listening on {url}");
Console.WriteLine("Endpoints: /v1/chat/completions, /v1/completions, /v1/models");

app.Run(url);
state.Dispose();
return 0;
