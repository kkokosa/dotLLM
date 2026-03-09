using DotLLM.Core.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;

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
}
