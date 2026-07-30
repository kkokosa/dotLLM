using DotLLM.Engine;
using DotLLM.Server;

if (args.Length < 1)
{
    Console.Error.WriteLine("Usage: DotLLM.Sample.Server <model.gguf> [options]");
    Console.Error.WriteLine("  model.gguf  Path to a GGUF model file");
    Console.Error.WriteLine("  --port N    Port to listen on (default: 8080)");
    Console.Error.WriteLine("  --device D  Compute device: cpu, gpu, gpu:0 (default: cpu)");
    Console.Error.WriteLine("  --threads N CPU prefill threads; 0 = auto");
    Console.Error.WriteLine("  --decode-threads N CPU decode threads; 0 = auto");
    return 1;
}

var options = ServerOptions.Parse(args);

Console.WriteLine($"Loading model: {options.Model}");
var resolvedPath = ServerStartup.ResolveModelPath(options.Model, options.Quant)
    ?? options.Model;

var state = ServerStartup.LoadModel(resolvedPath, options);
var app = ServerStartup.BuildApp(state, args, serveUi: true);

var url = $"http://{options.Host}:{options.Port}";
Console.WriteLine($"Model: {state.Config!.Architecture}, {state.Config.NumLayers} layers");
Console.WriteLine($"Server listening on {url}");
Console.WriteLine("Endpoints: /v1/chat/completions, /v1/completions, /v1/models");

app.Run(url);
state.Dispose();
return 0;
