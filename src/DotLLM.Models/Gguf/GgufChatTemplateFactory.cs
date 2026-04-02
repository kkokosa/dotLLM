using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.ToolCallParsers;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Bridge between GGUF metadata and the Jinja2 chat template engine.
/// Creates a <see cref="JinjaChatTemplate"/> from model metadata and tokenizer info.
/// </summary>
public static class GgufChatTemplateFactory
{
    /// <summary>
    /// Tries to create a <see cref="JinjaChatTemplate"/> from GGUF metadata.
    /// Returns null if no chat template is present in the metadata.
    /// </summary>
    /// <param name="metadata">GGUF metadata containing the template string and token info.</param>
    /// <param name="tokenizer">Tokenizer for resolving BOS/EOS token strings.</param>
    public static JinjaChatTemplate? TryCreate(GgufMetadata metadata, ITokenizer tokenizer)
    {
        string? template = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        if (string.IsNullOrEmpty(template))
            return null;

        string bosToken = tokenizer.DecodeToken(tokenizer.BosTokenId);
        string eosToken = tokenizer.DecodeToken(tokenizer.EosTokenId);

        return new JinjaChatTemplate(template, bosToken, eosToken);
    }

    /// <summary>
    /// Tries to create a <see cref="JinjaChatTemplate"/> from a ModelConfig.
    /// Returns null if no chat template is present in the config.
    /// </summary>
    /// <param name="config">Model configuration with chat template string.</param>
    /// <param name="bosToken">BOS token string.</param>
    /// <param name="eosToken">EOS token string.</param>
    public static JinjaChatTemplate? TryCreate(ModelConfig config, string bosToken, string eosToken)
    {
        if (string.IsNullOrEmpty(config.ChatTemplate))
            return null;

        return new JinjaChatTemplate(config.ChatTemplate, bosToken, eosToken);
    }

    /// <summary>
    /// Creates a tool call parser appropriate for the model based on architecture
    /// and chat template content.
    /// </summary>
    /// <param name="metadata">GGUF metadata for template heuristics.</param>
    /// <param name="architecture">Model architecture.</param>
    /// <returns>A tool call parser for this model.</returns>
    public static IToolCallParser CreateToolCallParser(GgufMetadata metadata, Architecture architecture)
    {
        string? template = metadata.GetStringOrDefault("tokenizer.chat_template", null!);
        return ToolCallParserFactory.Create(architecture, template);
    }

    /// <summary>
    /// Creates a tool call parser appropriate for the model based on architecture
    /// and chat template content.
    /// </summary>
    /// <param name="config">Model configuration.</param>
    /// <returns>A tool call parser for this model.</returns>
    public static IToolCallParser CreateToolCallParser(ModelConfig config)
        => ToolCallParserFactory.Create(config.Architecture, config.ChatTemplate);
}
