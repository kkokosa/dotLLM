using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// POST /v1/tokenize and /v1/detokenize — extension endpoints for token inspection.
/// </summary>
public static class TokenizeEndpoint
{
    public static void Map(WebApplication app)
    {
        app.MapPost("/v1/tokenize", (TokenizeRequest request, ServerState state) =>
        {
            if (state.Tokenizer is not { } tokenizer)
                return Results.StatusCode(503);

            int[] tokens = tokenizer.Encode(request.Text);
            string[] tokenStrings = tokens.Select(t => tokenizer.DecodeToken(t)).ToArray();
            return Results.Ok(new TokenizeResponse
            {
                Tokens = tokens,
                TokenStrings = tokenStrings,
                Count = tokens.Length,
            });
        });

        app.MapPost("/v1/detokenize", (DetokenizeRequest request, ServerState state) =>
        {
            if (state.Tokenizer is not { } tokenizer)
                return Results.StatusCode(503);

            string text = tokenizer.Decode(request.Tokens);
            return Results.Ok(new DetokenizeResponse { Text = text });
        });
    }
}
