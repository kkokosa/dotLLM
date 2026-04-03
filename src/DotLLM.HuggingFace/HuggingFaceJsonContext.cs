using System.Text.Json.Serialization;

namespace DotLLM.HuggingFace;

/// <summary>
/// Source-generated JSON serializer context for HuggingFace API types.
/// Enables Native AOT compilation by eliminating reflection-based deserialization.
/// </summary>
[JsonSerializable(typeof(List<HuggingFaceModelInfo>))]
[JsonSerializable(typeof(HuggingFaceModelInfo))]
[JsonSerializable(typeof(List<RepoFileEntry>))]
[JsonSourceGenerationOptions(PropertyNameCaseInsensitive = true)]
internal partial class HuggingFaceJsonContext : JsonSerializerContext;
