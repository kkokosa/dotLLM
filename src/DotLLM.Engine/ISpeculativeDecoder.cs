using DotLLM.Core.Attention;
using DotLLM.Core.Constraints;
using DotLLM.Core.Models;

namespace DotLLM.Engine;

/// <summary>
/// Speculative decoding: a draft model proposes tokens and the target model verifies them in a single forward pass.
/// </summary>
public interface ISpeculativeDecoder
{
    /// <summary>
    /// Drafts candidate tokens with the draft model and verifies them with the target model.
    /// </summary>
    /// <param name="targetModel">The full (target) model for verification.</param>
    /// <param name="draftModel">The smaller (draft) model for fast token proposals.</param>
    /// <param name="kvCacheTarget">KV-cache for the target model.</param>
    /// <param name="kvCacheDraft">KV-cache for the draft model.</param>
    /// <param name="constraint">Optional decoding constraint (rolled back on rejection).</param>
    /// <param name="numCandidates">Number of draft tokens to propose per step.</param>
    /// <returns>The accepted tokens and count of how many were accepted.</returns>
    (int[] Tokens, int AcceptedCount) DraftAndVerify(
        IModel targetModel,
        IModel draftModel,
        IKvCache kvCacheTarget,
        IKvCache kvCacheDraft,
        IDecodingConstraint? constraint,
        int numCandidates);
}
