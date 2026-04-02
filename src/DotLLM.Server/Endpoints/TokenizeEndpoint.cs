using DotLLM.Server.Models;
using DotLLM.Tokenizers;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// POST /v1/tokenize and /v1/detokenize — extension endpoints for token inspection.
/// </summary>
public static class TokenizeEndpoint
{
    public static void Map(WebApplication app)
    {
        app.MapPost("/v1/tokenize", (TokenizeRequest request, ITokenizer tokenizer) =>
        {
            int[] tokens = tokenizer.Encode(request.Text);
            string[] tokenStrings = tokens.Select(t => tokenizer.DecodeToken(t)).ToArray();
            return new TokenizeResponse
            {
                Tokens = tokens,
                TokenStrings = tokenStrings,
                Count = tokens.Length,
            };
        });

        app.MapPost("/v1/detokenize", (DetokenizeRequest request, ITokenizer tokenizer) =>
        {
            string text = tokenizer.Decode(request.Tokens);
            return new DetokenizeResponse { Text = text };
        });
    }
}
