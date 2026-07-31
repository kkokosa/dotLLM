using System.Runtime.CompilerServices;
using System.Text;
using DotLLM.Engine;
using DotLLM.Tokenizers;
using Microsoft.Extensions.AI;
using ChatMessage = Microsoft.Extensions.AI.ChatMessage;
using EngineToolCall = DotLLM.Tokenizers.ToolCall;

namespace DotLLM.Extensions.AI;

/// <summary>
/// A <see cref="IChatClient"/> implementation backed by the dotLLM inference engine.
/// This makes dotLLM a native, in-process backend for the Microsoft Agent Framework
/// and the broader <c>Microsoft.Extensions.AI</c> ecosystem: wrap a model and pass
/// the client to <c>chatClient.CreateAIAgent(...)</c>.
/// </summary>
/// <remarks>
/// The underlying <see cref="TextGenerator"/> is stateful and single-request; calls
/// are serialized through an internal gate (matching the dotLLM server). Streaming a
/// response holds the gate for the duration of the enumeration.
/// </remarks>
public sealed class DotLLMChatClient : IChatClient
{
    private readonly TextGenerator _generator;
    private readonly IChatTemplate _chatTemplate;
    private readonly IToolCallParser? _toolCallParser;
    private readonly string _modelId;
    private readonly ChatClientMetadata _metadata;
    private readonly SemaphoreSlim _gate = new(1, 1);

    /// <summary>Creates a chat client over a loaded dotLLM model.</summary>
    /// <param name="generator">The engine text generator wired to the loaded model.</param>
    /// <param name="chatTemplate">The model's chat template (renders messages to a prompt).</param>
    /// <param name="modelId">Model id reported in responses and metadata.</param>
    /// <param name="toolCallParser">
    /// Optional model-specific tool-call parser. When supplied and the request carries
    /// tools, generated tool calls are surfaced as <see cref="FunctionCallContent"/>.
    /// </param>
    public DotLLMChatClient(
        TextGenerator generator,
        IChatTemplate chatTemplate,
        string modelId = "dotllm",
        IToolCallParser? toolCallParser = null)
    {
        ArgumentNullException.ThrowIfNull(generator);
        ArgumentNullException.ThrowIfNull(chatTemplate);
        _generator = generator;
        _chatTemplate = chatTemplate;
        _modelId = modelId;
        _toolCallParser = toolCallParser;
        _metadata = new ChatClientMetadata("dotLLM", providerUri: null, defaultModelId: modelId);
    }

    /// <inheritdoc/>
    public async Task<ChatResponse> GetResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var (prompt, inferenceOptions, tools) = Prepare(messages, options);

        InferenceResponse result;
        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // TextGenerator.Generate is synchronous and runs for the whole completion, so it
            // is offloaded rather than run inline: an async API must not block its caller's
            // thread (an ASP.NET request thread, a UI thread) for seconds at a time. The
            // token therefore only gates scheduling — it cannot abort a running generation.
            // Use GetStreamingResponseAsync when mid-generation cancellation is required.
            result = await Task.Run(
                () => _generator.Generate(prompt, inferenceOptions), cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            _gate.Release();
        }

        string text = result.Text;
        EngineToolCall[]? toolCalls = null;
        var finishReason = result.FinishReason;
        if (_toolCallParser is not null && tools is { Length: > 0 })
        {
            var enriched = ToolCallDetector.DetectToolCalls(result, _toolCallParser);
            text = enriched.Text;
            toolCalls = enriched.ToolCalls;
            finishReason = enriched.FinishReason;
        }

        var responseMessage = new ChatMessage(
            ChatRole.Assistant, ChatClientMapping.ToResponseContents(text, toolCalls));

        return new ChatResponse(responseMessage)
        {
            ResponseId = Guid.NewGuid().ToString("N"),
            ModelId = options?.ModelId ?? _modelId,
            FinishReason = ChatClientMapping.ToChatFinishReason(finishReason),
            Usage = ChatClientMapping.ToUsageDetails(result.PromptTokenCount, result.GeneratedTokenCount),
        };
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<ChatMessage> messages,
        ChatOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var (prompt, inferenceOptions, tools) = Prepare(messages, options);

        string responseId = Guid.NewGuid().ToString("N");
        string modelId = options?.ModelId ?? _modelId;

        var accumulated = new StringBuilder();
        var finishReason = FinishReason.Length;

        // Tool calls are only recognisable once the full completion is available, so when
        // detection is active the text is buffered instead of streamed. Streaming it would
        // leak the model's raw tool-call syntax to the caller, and would make the coalesced
        // stream (ChatResponseUpdate.ToChatResponse) disagree with GetResponseAsync, which
        // drops the text whenever tool calls are present.
        bool detectToolCalls = _toolCallParser is not null && tools is { Length: > 0 };

        await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            await foreach (var token in _generator
                .GenerateStreamingTokensAsync(prompt, inferenceOptions, cancellationToken).ConfigureAwait(false))
            {
                if (token.Text.Length > 0)
                {
                    accumulated.Append(token.Text);
                    if (!detectToolCalls)
                    {
                        yield return new ChatResponseUpdate
                        {
                            Role = ChatRole.Assistant,
                            Contents = [new TextContent(token.Text)],
                            ResponseId = responseId,
                            MessageId = responseId,
                            ModelId = modelId,
                        };
                    }
                }

                if (token.FinishReason.HasValue)
                    finishReason = token.FinishReason.Value;
            }
        }
        finally
        {
            _gate.Release();
        }

        // Post-generation tool-call detection (mirrors the server streaming endpoints).
        string text = accumulated.ToString();
        EngineToolCall[]? toolCalls = null;
        if (detectToolCalls)
        {
            toolCalls = _toolCallParser!.TryParse(text);
            if (toolCalls is { Length: > 0 })
                finishReason = FinishReason.ToolCalls;
        }

        // Buffered path: the final update carries the whole message (text or tool calls),
        // built by the same mapping the non-streaming path uses. Streamed path: the text has
        // already been delivered, so the final update carries only the finish reason.
        List<AIContent> finalContents = detectToolCalls
            ? ChatClientMapping.ToResponseContents(text, toolCalls)
            : [];

        yield return new ChatResponseUpdate
        {
            Role = ChatRole.Assistant,
            Contents = finalContents,
            ResponseId = responseId,
            MessageId = responseId,
            ModelId = modelId,
            FinishReason = ChatClientMapping.ToChatFinishReason(finishReason),
        };
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        if (serviceKey is not null)
            return null;
        if (serviceType == typeof(ChatClientMetadata))
            return _metadata;
        return serviceType.IsInstanceOfType(this) ? this : null;
    }

    /// <inheritdoc/>
    public void Dispose() => _gate.Dispose();

    private (string Prompt, DotLLM.Core.Configuration.InferenceOptions Options, ToolDefinition[]? Tools) Prepare(
        IEnumerable<ChatMessage> messages, ChatOptions? options)
    {
        var engineMessages = ChatClientMapping.ToEngineMessages(messages);
        var tools = ChatClientMapping.ToToolDefinitions(options);
        var inferenceOptions = ChatClientMapping.ToInferenceOptions(options);
        string prompt = _chatTemplate.Apply(
            engineMessages, new ChatTemplateOptions { AddGenerationPrompt = true, Tools = tools });
        return (prompt, inferenceOptions, tools);
    }
}
