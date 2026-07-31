# Perplexity harness — design

**Issue:** jamesburton/dotLLM#231 · **Upstream driver:** kkokosa/dotLLM#416 · **Date:** 2026-07-30

## Problem

Upstream #416's investigation log ends on a blocker: *"Every remaining lever changes numerics and
none of them can currently be evaluated."* Scale-granularity repacking (G=64/128, estimated
1.31–1.56×) and every other numerics-changing CPU lever are gated on an evaluation harness that
does not exist upstream.

`upstream/main` has only `samples/DotLLM.Sample.Logprobs/Program.cs`. Our `dev` has perplexity
scoring in ten files — but entirely as **duplicated private per-test helpers**. `StableLogProb` is
copy-pasted verbatim; the same English corpus string appears in multiple files; two different
scoring strategies have diverged without a shared definition. None of it is reachable as a gate for
kernel work.

So this is consolidation plus extension, not greenfield — but the consolidation half must be done
from `dev`, because that is where the helpers live. See Sequencing.

## The one real axis

The existing helpers diverged into two shapes for a single reason: **whether the backend's
`Forward` returns logits for every position or only the last one.**

- CPU returns `[seqLen, vocab]`, so one teacher-forced prefill scores every next-token NLL — O(n).
  (`BitNetAccuracyTests.Cpu_Perplexity_OnFixedPassage_IsSane`)
- CUDA returns only the final row, so each target needs its own growing-prefix re-prefill — O(n²),
  which is why those helpers carry a stride to keep the sweep brisk.
  (`CudaFlashPrefillForwardHarness.PrefillGrowingPrefixPerplexity`, and its twin in
  `CudaG3PrefillForwardHarness`)

Everything else about them is identical. Modelling that axis explicitly — `ReturnsAllRows` on
`IPerplexityModel` — is what lets one evaluator replace both without changing any number either
currently produces.

## Components

**`DotLLM.Core.Evaluation`** (abstractions only, so `main` and `dev` can both reference the
contract without dragging in the implementation):

- `IPerplexityModel` — `VocabSize`, `MaxContextLength`, `ReturnsAllRows`, `Forward(tokens, positions)`.
  Deliberately narrower than `IModel`: perplexity needs no sampling, no KV-cache lifetime
  management, no streaming, and binding to the full interface would prevent scoring a bare backend
  or a test double.
- `PerplexityMode` — `TeacherForced` | `SlidingWindow`.
- `PerplexityOptions(Mode, ContextLength, Stride, MaxTokens)`.
- `PerplexityResult(Perplexity, MeanNegativeLogLikelihood, ScoredTokens, WindowCount)`.

**`PerplexityEvaluator`** — strategy selected from `ReturnsAllRows`, not from the caller.

**Corpus handling** — streamed and tokenized in chunks.

**CLI verb** — context length, stride, corpus path, token cap.

## Two modes, two purposes

`TeacherForced` preserves the in-tree "G1 precedent" methodology exactly. It is **ratio-oriented**:
the load-bearing signal is the OFF/ON perplexity ratio on identical tokens under a <1% gate, not
the absolute value. Not comparable to published figures, and must not be presented as if it were.

`SlidingWindow` is new and **absolute-value oriented**: windows of `ContextLength` advanced by
`Stride`, scoring only tokens beyond the carried-over prefix so every scored token has full-length
context. Matches llama.cpp's `--perplexity` methodology so figures compare directly to published
numbers.

## Memory constraint (from planned Track D)

On Strix Halo's UMA, a large VRAM carve-out leaves host RAM scarce, and perplexity is the workload
most punished by it — a long sequence of full-context prefills rather than a single load. A harness
that mmaps weights host-side while the backend separately uploads them pays for the model twice
against an already-halved budget.

Therefore, as a design constraint rather than a later optimisation:

- **The evaluator never loads weights.** It takes an already-constructed `IPerplexityModel`.
- **The corpus is streamed and tokenized in chunks**, never materialized whole.

Both cost nothing now and keep Track D a pure optimisation rather than a rewrite.

## Verification

1. **Consolidation is behaviour-preserving.** Migrated tests must produce *numerically identical*
   results to their previous private helpers. A changed number means the consolidation altered
   semantics — this is the primary regression gate, and it is the reason `TeacherForced` is
   preserved verbatim rather than "improved" in passing.
2. **`SlidingWindow` is genuinely comparable.** Validated against a published llama.cpp perplexity
   figure on matching model, corpus, context length and stride, within a stated tolerance. Without
   this the word "comparable" is unearned, and the harness would give upstream false confidence on
   exactly the numerics-changing decisions it is meant to gate.
3. **`ScoredTokens` is reported and checked.** A perplexity over a different token count is a
   different measurement; cross-run comparison requires it to match.

## Sequencing

Built from `main` in a worktree (`issue/231-perplexity-harness`), PR targets upstream `main`, then
merges to `dev`.

Because the ten duplicated helpers exist only on `dev`, the split is:

- **On `main` / this PR:** the contract, the evaluator, corpus handling, CLI verb, and validation
  of `SlidingWindow` against llama.cpp. Self-contained and upstream-contributable.
- **On `dev`, follow-up:** migrate the ten existing helpers onto the harness, gated on producing
  identical numbers.

## Out of scope

Track D (iGPU optimisation of the harness under a large VRAM carve-out, including eliminating
host+device double-loading) is deliberately deferred until this and #233 land. The constraints
above exist so D is optimisation, not rework.
