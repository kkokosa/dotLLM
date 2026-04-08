# Speculative Decoding — dotLLM

## Overview

A small **draft model** proposes K candidate tokens; the larger **target model** verifies them in a single forward pass. Achieves 2-3× speedup while producing output **exactly equal** to the target model's distribution.

## Algorithm

### Draft Phase
Draft model generates K tokens autoregressively (K typically 3-5):
```
for i in 1..K:
  draft_logits = draft_model.forward(token)
  q[i] = softmax(draft_logits)   // draft probability
  token[i] = sample(q[i])
```

### Verification Phase
Target model processes all K candidates in one batched forward pass:
```
target_logits[1..K] = target_model.forward([token[1], ..., token[K]])
p[i] = softmax(target_logits[i])  // target probability for each position
```

### Acceptance (Modified Rejection Sampling)
Accept tokens left-to-right:
```
for i in 1..K:
  r = random_uniform(0, 1)
  if r < min(1, p[i][token[i]] / q[i][token[i]]):
    accept token[i]
  else:
    // Reject: sample corrected token from adjusted distribution
    corrected = sample(normalize(max(0, p[i] - q[i])))
    output corrected, discard tokens[i+1..K]
    break

if all K accepted:
  // Bonus: sample one more token from p[K+1]
  bonus = sample(p[K+1])
  output bonus
```

**Key property**: This scheme guarantees the output distribution is **exactly** the target model's distribution, not an approximation.

## ISpeculativeDecoder Interface

```
ISpeculativeDecoder:
  DraftAndVerify(targetModel, draftModel, kvCacheTarget, kvCacheDraft,
                 constraint, numCandidates) → AcceptedTokens
```

## Draft Model Options

| Type | Description | Trade-off |
|------|-------------|-----------|
| Separate small model | e.g., Llama 1B drafting for Llama 70B | Must share vocab. Extra memory. Best acceptance rate. |
| Layer subset | First N layers of target model | No extra params. Lower acceptance rate. |
| Speculative head | Small MLP trained alongside target | Minimal overhead. Model-specific. |

Draft and target **must share vocabulary and tokenizer** — the acceptance scheme requires comparing probabilities over the same token space.

## KV-Cache Rollback

Speculated tokens that are rejected need their KV-cache entries invalidated:
- Draft model KV-cache: roll back to pre-speculation position.
- Target model KV-cache: only keep entries for accepted tokens.
- With PagedAttention: simply update the sequence length counter in the block table (blocks are reused, data overwritten on next append).

## Constraint Interaction

When constrained decoding is active:
1. **Before speculation**: Clone the constraint state via `IDecodingConstraint.Clone()`.
2. **During drafting**: Each draft token advances the cloned constraint state. Draft must respect the constraint mask (invalid tokens excluded from draft sampling).
3. **On rejection**: Restore constraint state from the clone at the rejection point.
4. **On corrected token**: Advance constraint from the restored state.

## Performance Characteristics

- Speedup scales with **acceptance rate** (how often draft matches target).
- Acceptance rate depends on: model similarity, task difficulty, temperature.
- Typical: 60-80% acceptance at low temperature → 2-3× effective speedup.
- Higher temperature → lower acceptance → less benefit.
- K (candidates per iteration): diminishing returns past ~5. Optimal K depends on acceptance rate.

## Vocabulary Compatibility

Draft and target models must share the same base tokenizer. A small vocabulary size difference (up to 128 tokens) is tolerated — matching llama.cpp's `SPEC_VOCAB_MAX_SIZE_DIFFERENCE`. The extra tokens are typically padding/reserved IDs that never appear in normal generation.

When vocab sizes differ, probability comparison uses the shared range (`Math.Min(targetVocab, draftVocab)`). Tokens beyond the draft's vocab can only be produced by the target model (as corrected or bonus tokens).

| Compatibility | Condition | Status |
|---------------|-----------|--------|
| Exact match | `targetVocab == draftVocab` | Best — no clamping needed |
| Close match | `abs(diff) <= 128` | Supported — shared range comparison |
| Incompatible | `abs(diff) > 128` | Rejected — different tokenizer family |

## When NOT to Use

- Very short generations (overhead of draft exceeds benefit)
- Very high temperature (low acceptance rate)
- No suitable draft model available
- Memory-constrained (draft model requires additional memory)

## Future Considerations

### Universal Assisted Generation (UAG)

HuggingFace Transformers v4.46.0 introduced UAG, which enables speculative decoding across model families with **different tokenizers**. The approach: draft tokens are decoded to text, re-tokenized with the target tokenizer, and aligned via longest common subsequence. This removes the vocabulary matching requirement entirely but adds tokenization overhead per speculation step. Reported speedups: 1.5-2× across model families. See [HuggingFace blog](https://huggingface.co/blog/universal_assisted_generation).

### Layer-Subset Drafting

Use the first N layers of the target model itself as a draft — no separate model needed. Lower acceptance rate than a dedicated draft model, but zero extra memory and guaranteed vocabulary compatibility.