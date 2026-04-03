using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Executes warm-up inference passes to trigger JIT compilation and CUDA kernel loading.
/// </summary>
public static class WarmupRunner
{
    /// <summary>
    /// Runs warm-up inference passes using the given generator and tokenizer.
    /// Each iteration exercises the full inference pipeline: tokenize → prefill → decode → sample.
    /// </summary>
    /// <param name="generator">The text generator to warm up.</param>
    /// <param name="tokenizer">The tokenizer (used for logging prompt token count).</param>
    /// <param name="options">Warm-up configuration. If null, uses <see cref="WarmupOptions.Default"/>.</param>
    public static void Run(TextGenerator generator, ITokenizer tokenizer, WarmupOptions? options = null)
    {
        options ??= WarmupOptions.Default;
        if (!options.Enabled || options.Iterations <= 0)
            return;

        int promptTokens = tokenizer.Encode(options.DummyPrompt).Length;
        Console.WriteLine($"[dotllm] Warming up ({options.Iterations} iterations, " +
                          $"{promptTokens} prompt tokens, {options.MaxTokens} max gen tokens)...");

        long totalStart = Stopwatch.GetTimestamp();

        var inferenceOptions = new InferenceOptions
        {
            MaxTokens = options.MaxTokens,
            Temperature = 0f,
        };

        for (int i = 0; i < options.Iterations; i++)
        {
            long iterStart = Stopwatch.GetTimestamp();
            generator.Generate(options.DummyPrompt, inferenceOptions);
            double iterMs = Stopwatch.GetElapsedTime(iterStart).TotalMilliseconds;
            Console.WriteLine($"[dotllm]   Iteration {i + 1}/{options.Iterations}: {iterMs:F0}ms");
        }

        double totalMs = Stopwatch.GetElapsedTime(totalStart).TotalMilliseconds;
        Console.WriteLine($"[dotllm] Warm-up complete in {totalMs:F0}ms");
    }
}
