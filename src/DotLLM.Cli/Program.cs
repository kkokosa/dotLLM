using System.Runtime.CompilerServices;
using DotLLM.Cli.Commands;
#if DEBUG
using DotLLM.Cli.Commands.Debug;
#endif
using Spectre.Console.Cli;

var app = new CommandApp();

app.Configure(config =>
{
    config.SetApplicationName("dotllm");
    config.SetApplicationVersion("0.1.0");
    // Spectre.Console.Cli's example validation uses reflection that doesn't work under Native AOT
    if (RuntimeFeature.IsDynamicCodeSupported)
        config.ValidateExamples();
    config.Settings.StrictParsing = true;

    config.AddBranch("model", model =>
    {
        model.SetDescription("Manage GGUF models from HuggingFace Hub.");

        model.AddCommand<ModelSearchCommand>("search")
            .WithDescription("Search HuggingFace for GGUF models.")
            .WithExample("model", "search", "llama", "--limit", "5");

        model.AddCommand<ModelPullCommand>("pull")
            .WithDescription("Download a GGUF model from HuggingFace.")
            .WithExample("model", "pull", "TheBloke/Llama-2-7B-GGUF", "--file", "llama-2-7b.Q4_K_M.gguf");

        model.AddCommand<ModelListCommand>("list")
            .WithDescription("List locally downloaded models.");

        model.AddCommand<ModelInfoCommand>("info")
            .WithDescription("Show details for a HuggingFace model repository.")
            .WithExample("model", "info", "TheBloke/Llama-2-7B-GGUF");
    });

    config.AddCommand<RunCommand>("run")
        .WithDescription("Run greedy text generation on a GGUF model.")
        .WithExample("run", "QuantFactory/SmolLM-135M-GGUF", "--prompt", "The capital of France is", "--max-tokens", "64");

    config.AddCommand<ChatCommand>("chat")
        .WithDescription("Interactive multi-turn chat with a GGUF model.")
        .WithExample("chat", "QuantFactory/SmolLM-135M-GGUF", "--system", "You are a helpful assistant.");

    config.AddCommand<ServeCommand>("serve")
        .WithDescription("Launch API server with built-in web chat UI.")
        .WithExample("serve", "Qwen/Qwen3-0.6B-GGUF", "--port", "8080");

#if DEBUG
    config.AddBranch("debug", debug =>
    {
        debug.SetDescription("Debug/diagnostic commands (Debug build only).");

        debug.AddCommand<DebugGgufHeaderCommand>("gguf-header")
            .WithDescription("Parse and display the GGUF file header.")
            .WithExample("debug", "gguf-header", "model.gguf");

        debug.AddCommand<DebugGgufMetadataCommand>("gguf-metadata")
            .WithDescription("List all GGUF metadata key-value pairs.")
            .WithExample("debug", "gguf-metadata", "model.gguf");

        debug.AddCommand<DebugGgufTensorsCommand>("gguf-tensors")
            .WithDescription("List all tensor descriptors from a GGUF file.")
            .WithExample("debug", "gguf-tensors", "model.gguf");

        debug.AddCommand<DebugGgufConfigCommand>("gguf-config")
            .WithDescription("Extract and display ModelConfig from GGUF metadata.")
            .WithExample("debug", "gguf-config", "model.gguf");

        debug.AddCommand<DebugForwardPassCommand>("forward-pass")
            .WithDescription("Run a single forward pass and display logit diagnostics.")
            .WithExample("debug", "forward-pass", "model.gguf", "--prompt", "Hello");

        debug.AddCommand<DebugTokenizeCommand>("tokenize")
            .WithDescription("Tokenize text and verify round-trip decode.")
            .WithExample("debug", "tokenize", "model.gguf", "--text", "Hello world");

        debug.AddCommand<DebugEmbedLookupCommand>("embed-lookup")
            .WithDescription("Inspect the embedding vector for a single token.")
            .WithExample("debug", "embed-lookup", "model.gguf", "--token-id", "1");
    });
#endif
});

return app.Run(args);
