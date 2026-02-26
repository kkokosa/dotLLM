# Sampling Pipeline — dotLLM

## Composable ISamplerStep Chain

The sampler pipeline is a sequence of `ISamplerStep` operations applied to raw logits before final token selection. Steps are ordered and extensible.

```
ISamplerStep:
  Apply(Span<float> logits, SamplerContext ctx) → void
```

## Default Pipeline Order

### 1. Logit Bias (`LogitBiasStep`)
Per-token additive bias from request `logit_bias` map: `logits[token_id] += bias`.
OpenAI API compatible: `{token_id: float_value}`.

### 2. Constraint Mask (`ConstraintMaskStep`)
Apply `IDecodingConstraint.GetAllowedTokens()` mask if structured output active.
Invalid tokens → `-∞`. See [CONSTRAINED_DECODING.md](CONSTRAINED_DECODING.md).

### 3. Repetition Penalties (`RepetitionPenaltyStep`)
Three configurable modes (can combine):

- **Repetition penalty** (multiplicative): For tokens in history, `logit = logit > 0 ? logit/penalty : logit*penalty`. Common in open models.
- **Frequency penalty** (additive, proportional): `logit -= freq_penalty × count(token)`. OpenAI API.
- **Presence penalty** (additive, binary): `logit -= presence_penalty × (count > 0 ? 1 : 0)`. OpenAI API.

Operates over a configurable lookback window of recent tokens.

### 4. Temperature (`TemperatureStep`)
`logits /= temperature`. T=0 → greedy (argmax). T=1 → unmodified. T>1 → more random.

### 5. Top-K (`TopKStep`)
Keep only K highest-probability tokens. Set rest to `-∞`.

### 6. Top-P / Nucleus (`TopPStep`)
Sort by probability descending. Keep smallest set where cumulative probability ≥ P.

### 7. Min-P (`MinPStep`)
Keep tokens with `probability ≥ min_p × max_probability`. More stable than top-p across distributions.

### 8. Categorical Sample (`CategoricalSampleStep`)
Convert logits to probabilities (softmax), sample. Argmax if temperature was 0.

## Custom Logit Processors

Users can inject arbitrary processing at any pipeline position:

```
ILogitProcessor:
  Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext ctx) → void
```

Use cases: classifier-free guidance, contrastive decoding, custom penalty schemes.

## Beam Search

Alternative to sampling. Maintains N candidate beams:

1. Each step: expand each beam by top-M tokens → N×M candidates.
2. Score by cumulative log-probability with length normalization.
3. Keep top N beams.
4. Stop when all beams hit EOS or max length.
5. Return top-K completed sequences by normalized score.

**KV-cache**: Beams sharing prefix use copy-on-write (PagedAttention COW blocks).
**Constraints**: Each beam clones its `IDecodingConstraint` state at branch points.
**Configured via**: `n` parameter in API request (n > 1 triggers beam search).

## Stop Conditions — IStopCondition

Multiple conditions active simultaneously. First match wins.

```
IStopCondition:
  ShouldStop(tokenId, generatedTokens, decodedText) → StopResult

StopResult: Continue | Stop | StopInclude
```

### Built-in Conditions

- **EOS token** — Always active. Model's end-of-sequence token.
- **Max tokens** — Hard limit on generated tokens.
- **Stop strings** — Text patterns that terminate generation (e.g., `"\n\nHuman:"`, `"END"`). Rolling buffer of decoded text, check suffix matches. Stop string excluded from output.
- **Stop token sequences** — Token ID sequences (avoids tokenization ambiguity).
- **Custom predicate** — Arbitrary `IStopCondition` implementation.

OpenAI API: `stop: ["str1", "str2"]` maps to stop string conditions.