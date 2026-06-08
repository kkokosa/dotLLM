using DotLLM.Engine.PromptCache;
using DotLLM.Server.Models;

namespace DotLLM.Server.Endpoints;

/// <summary>
/// Cross-request prompt cache administration (Step 37).
/// <list type="bullet">
///   <item><c>GET    /v1/prompt-cache</c> — stats for the active trie.</item>
///   <item><c>GET    /v1/prompt-cache/{id}</c> — inspect a named prefix.</item>
///   <item><c>POST   /v1/prompt-cache/{id}</c> — pre-warm and pin a named prefix.</item>
///   <item><c>DELETE /v1/prompt-cache/{id}</c> — unpin a named prefix.</item>
///   <item><c>DELETE /v1/prompt-cache</c> — drop every entry from the trie.</item>
/// </list>
/// </summary>
public static class PromptCacheEndpoint
{
    public static void Map(WebApplication app)
    {
        app.MapGet("/v1/prompt-cache", (ServerState state) =>
        {
            var mgr = state.PrefixTrieManager;
            if (mgr is null)
                return Results.Ok(EmptyStats());

            var stats = mgr.GetStats();
            return Results.Ok(new PromptCacheStatsResponse
            {
                Enabled = stats.Enabled,
                BlockSize = stats.BlockSize,
                Nodes = stats.NodeCount,
                HitTokens = stats.HitTokens,
                MissTokens = stats.MissTokens,
                Lookups = stats.Lookups,
                Hits = stats.Hits,
                Misses = stats.Misses,
                EvictionRefusals = stats.EvictionRefusals,
                FreeBlocks = stats.FreeBlocks,
                TotalBlocks = stats.TotalBlocks,
            });
        });

        app.MapGet("/v1/prompt-cache/{id}", (string id, ServerState state) =>
        {
            var mgr = state.PrefixTrieManager;
            if (mgr is null) return Results.NotFound(new ErrorResponse { Error = "Prefix cache disabled." });

            var info = mgr.InspectNamedPrefix(id);
            return info is null
                ? Results.NotFound(new ErrorResponse { Error = $"prefix_id '{id}' not registered." })
                : Results.Ok(new PromptCacheResponse
                {
                    PrefixId = info.Value.PrefixId,
                    Tokens = info.Value.Tokens,
                    Blocks = info.Value.Blocks,
                    Status = "registered",
                });
        });

        app.MapPost("/v1/prompt-cache/{id}", async (string id, PromptCacheRegisterRequest request,
            ServerState state, HttpContext ctx) =>
        {
            if (!state.IsReady || state.Generator is null || state.Tokenizer is null)
                return Results.StatusCode(503);

            var mgr = state.PrefixTrieManager;
            if (mgr is null)
                return Results.BadRequest(new ErrorResponse { Error = "Paged KV-cache + prefix sharing is not enabled on this server." });

            int[] tokens = request.TokenIds ?? state.Tokenizer.Encode(request.Prompt ?? string.Empty);
            if (tokens.Length == 0)
                return Results.BadRequest(new ErrorResponse { Error = "Either 'prompt' or 'token_ids' must be supplied with at least one token." });

            // Pre-warm the trie by running the prompt through the generator once with
            // MaxTokens=0 (no generation). The KV-cache is captured by RecordCompletion
            // when the generator-managed cache is disposed.
            await state.ExecuteAsync(async () =>
            {
                var opts = new DotLLM.Core.Configuration.InferenceOptions
                {
                    Temperature = 0f,
                    MaxTokens = 1,
                };
                var prompt = request.Prompt ?? state.Tokenizer.Decode(tokens);
                state.Generator.Generate(prompt, opts);
                await Task.CompletedTask;
            }, ctx.RequestAborted);

            int matched;
            try
            {
                matched = mgr.RegisterNamedPrefix(id, tokens);
            }
            catch (InvalidOperationException)
            {
                return Results.Conflict(new ErrorResponse { Error = $"prefix_id '{id}' is already registered." });
            }

            if (matched == 0)
                return Results.UnprocessableEntity(new ErrorResponse
                {
                    Error = "Prompt did not produce a cacheable prefix (too short, or generation skipped the prefix cache).",
                });

            return Results.Ok(new PromptCacheResponse
            {
                PrefixId = id,
                Tokens = matched,
                Blocks = matched / mgr.Trie.BlockSize,
                Status = "registered",
            });
        });

        app.MapDelete("/v1/prompt-cache/{id}", (string id, ServerState state) =>
        {
            var mgr = state.PrefixTrieManager;
            if (mgr is null) return Results.NotFound(new ErrorResponse { Error = "Prefix cache disabled." });

            return mgr.UnpinNamedPrefix(id)
                ? Results.Ok(new StatusResponse { Status = "unregistered" })
                : Results.NotFound(new ErrorResponse { Error = $"prefix_id '{id}' not registered." });
        });

        app.MapDelete("/v1/prompt-cache", (ServerState state) =>
        {
            state.PrefixTrieManager?.Trie.Clear();
            return Results.Ok(new StatusResponse { Status = "cleared" });
        });
    }

    private static PromptCacheStatsResponse EmptyStats() => new()
    {
        Enabled = false,
        BlockSize = 0,
        Nodes = 0,
        HitTokens = 0,
        MissTokens = 0,
        Lookups = 0,
        Hits = 0,
        Misses = 0,
        EvictionRefusals = 0,
        FreeBlocks = 0,
        TotalBlocks = 0,
    };
}
