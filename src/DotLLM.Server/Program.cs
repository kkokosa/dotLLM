using DotLLM.Server;

// Parse CLI arguments
var serverOptions = ServerOptions.Parse(args);

// Resolve model path (local file or HuggingFace cached model)
Console.WriteLine($"[dotllm] Resolving model: {serverOptions.Model}");
var resolvedPath = ServerStartup.ResolveModelPath(serverOptions.Model, serverOptions.Quant);
if (resolvedPath is null)
{
    Console.Error.WriteLine("Failed to resolve model path. Provide a .gguf file path or a pre-downloaded HuggingFace repo.");
    Console.Error.WriteLine("Download models via: dotllm model pull <repo>");
    return 1;
}

// Load model and build server state
var state = ServerStartup.LoadModel(resolvedPath, serverOptions);

// Build and run server (API-only, no UI)
var app = ServerStartup.BuildApp(state, args, serveUi: false);

var url = $"http://{serverOptions.Host}:{serverOptions.Port}";
Console.WriteLine($"[dotllm] {state.Config!.Architecture} {state.Config.NumLayers}L/{state.Config.HiddenSize}H | {Path.GetFileName(resolvedPath)}");
Console.WriteLine($"[dotllm] Listening on {url}");
Console.WriteLine($"[dotllm] Endpoints: /v1/chat/completions, /v1/completions, /v1/models, /v1/tokenize, /v1/detokenize");
Console.WriteLine("[dotllm] Single-request mode — requests processed sequentially");

app.Run(url);

// Cleanup
state.Dispose();
return 0;
