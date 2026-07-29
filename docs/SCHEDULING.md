# Scheduling & Continuous Batching — dotLLM

## Overview

The scheduler operates at **iteration granularity**, not request granularity. This enables continuous batching: as sequences finish, new ones are admitted immediately, keeping the hardware batch always full.

## IScheduler Interface

```
IScheduler:
  Enqueue(request: InferenceRequest) → Task<InferenceResponse>
  RunLoop(cancellation) → Task    // Main scheduling loop
  GetMetrics() → SchedulerMetrics
```

Concrete implementation: `ContinuousBatchScheduler` (step-driven, exposed via `IBatchScheduler`) wrapped by `ContinuousBatchSchedulerService` (async, exposed via `IScheduler`).

## Iteration-Level Scheduling

Each `ContinuousBatchScheduler.Step()` call:

1. **Sweep cancelled sequences** — caller-side `CancellationToken` may have flipped state; release their KV-cache.
2. **Admit** new sequences up to `MaxActiveSequences` and (when paged) sufficient free blocks. Admission allocates the KV-cache, consults the optional `ISchedulerPrefixCache` for reuse, and transitions to `Prefilling`. **Actual prefill work happens in step 3** — admission is purely a slot/cache assignment.
3. **Build a batch** containing one entry per active sequence that needs a forward pass this iteration:
   - `Prefilling` sequences contribute their next prefill chunk, sized by `MaxPrefillTokensPerStep` (0 = unlimited, single shot).
   - `Decoding` sequences contribute their last sampled token.
   When the batch has ≥2 entries, the scheduler calls `IModel.ForwardBatch(requests, deviceId)` — a single dispatch that backends can fuse into one batched kernel (the default interface implementation falls back to a per-sequence `Forward` loop). For a 1-entry batch, the scheduler calls `Forward` directly to avoid the batch-allocation overhead and keep single-tenant decode latency unchanged from `TextGenerator`.
4. **Process forward results**: for prefilling sequences that just consumed their final chunk, sample the first token and transition to `Decoding`; for decoding sequences, sample the next token. Apply stop conditions (EOS, max-tokens) and transition to `Completed` when fired.
5. **Sweep completed/cancelled** active entries — build their `InferenceResponse`, release the KV-cache, complete the task.

```
while (!cancelled):
  SweepCancelled()
  Admit(pendingQueue, MaxActiveSequences, prefixCache?)
  batch = BuildBatch(active)           # mix of prefill chunks and decode tokens
  results = batch.Count >= 2 ? Model.ForwardBatch(batch) : Model.Forward(batch[0])
  for entry in batch:
    ProcessResult(entry, results[i])   # sample, advance constraint, check stops
  SweepCompleted()
```

## Chunked Prefill

`MaxPrefillTokensPerStep` controls how many prompt tokens a single Step iteration may push through the model in aggregate. When non-zero, a prompt longer than the cap is split across multiple Step iterations: the sequence stays in `Prefilling` state until its `PrefilledTokens == PromptLength`, advancing one chunk per Step. **Decode tokens of already-decoding sequences keep running every step** regardless of the prefill budget — this is the head-of-line-blocking property that lets a 4096-token user prompt land without freezing every other concurrent chat session.

The trade-off: a very small chunk size raises per-step overhead (lots of small kernel dispatches); a very large chunk size lets one long prompt dominate the GPU for several steps before decode catches up. Production setups tune chunk size against expected prompt-length distribution and decode-batch size.

## Kernel-Batched Forward (`IModel.ForwardBatch`)

`IModel.ForwardBatch(IReadOnlyList<SequenceForwardRequest>, int deviceId)` is the seam for true batched compute across sequences:

```csharp
readonly record struct SequenceForwardRequest
{
    public required ReadOnlyMemory<int> TokenIds { get; init; }   // 1 (decode) or N (prefill chunk)
    public required ReadOnlyMemory<int> Positions { get; init; }
    public required IKvCache KvCache { get; init; }                // independent per sequence
    public ILoraAdapter? Adapter { get; init; }
}
```

The default interface implementation loops over `Forward` per request — backends pay the per-sequence kernel-dispatch overhead until they override with a fused implementation. Override candidates:

- **CPU**: bundle N sequences into a single GEMM (currently N GEMVs) for matmul-bound layers; attention stays per-sequence because each has its own K/V.
- **CUDA / Vulkan**: launch one merged kernel with a per-sequence offset table — same Q·K·V GEMM, separate attention scratch regions.

The acceptance test (`FourConcurrentSchedulerTests`) drives 4 distinct prompts concurrently through the scheduler and verifies each gets its own per-request response — the API contract is in place even when the underlying backend is still using the per-sequence-loop fallback.

## Prefill/Decode Separation

Different compute characteristics:
- **Prefill**: Process N prompt tokens. Compute-bound (GEMM). High arithmetic intensity.
- **Decode**: Process 1 token per sequence. Memory-bandwidth-bound (GEMV). Low arithmetic intensity.

The scheduler can separate these into micro-batches within one iteration for optimal utilization. Prefill benefits from large batch GEMM; decode benefits from batching many sequences together.

## Request Priority

Each request carries a priority level (from API key or explicit parameter):

| Level | Behavior |
|-------|----------|
| `critical` | Never preempted, admitted first |
| `high` | Preempts `normal` and `low` |
| `normal` | Default |
| `low` | Preempted first, admitted last |

Priority affects:
- **Queue ordering**: Higher priority → admitted sooner.
- **Preemption**: When memory scarce, lower-priority sequences preempted first.

## Preemption

When KV-cache memory is exhausted and high-priority requests arrive:

1. Select lowest-priority active sequences.
2. **Swap out**: Save their KV-cache blocks to CPU memory (or mark for recomputation).
3. Free GPU KV blocks for the new request.
4. Later: when capacity returns, **swap in**: restore KV blocks and resume.

Swap options:
- **Recompute**: Discard KV, re-prefill when resuming. Simple, no CPU memory needed.
- **CPU offload**: Copy KV blocks to CPU memory. Faster resume but uses CPU RAM.

## Sequence State Machine

```
QUEUED → PREFILLING → DECODING → COMPLETED
                ↕                    ↓
           PREEMPTED ←──────── (memory pressure)
```

## Scheduling Policies

The `IScheduler` interface allows different policies:
- **FCFS with priority**: Default. Priority queue ordered by (priority, arrival_time).
- **Shortest-job-first**: Estimate remaining tokens, prioritize short generations.
- **Fair-share**: Balance token throughput across API keys/users.
