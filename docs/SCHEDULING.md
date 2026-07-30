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

## Iteration-Level Scheduling

Each scheduler iteration:

1. **Check completions**: Sequences hitting EOS/max tokens/stop conditions → evict, free KV blocks.
2. **Admit new requests**: Fill freed capacity from the priority queue.
3. **Prefill**: For newly admitted sequences, process full prompt tokens (batch prefill).
4. **Decode**: For all active sequences, generate one token each (batched decode).

```
while (!cancelled):
  completed = batch.RemoveCompleted()
  FreeKvBlocks(completed)
  NotifyClients(completed)

  admitted = AdmitFromQueue(available_kv_blocks)
  RunPrefill(admitted)

  tokens = RunDecode(batch.ActiveSequences)
  ApplySamplerPipeline(tokens)
  CheckStopConditions(tokens)
  StreamTokensToClients(tokens)
```

## Prefill/Decode Separation

Different compute characteristics:
- **Prefill**: Process N prompt tokens. Compute-bound (GEMM). High arithmetic intensity.
- **Decode**: Process 1 token per sequence. Memory-bandwidth-bound (GEMV). Low arithmetic intensity.

The scheduler can separate these into micro-batches within one iteration for optimal utilization. Prefill benefits from large batch GEMM; decode benefits from batching many sequences together.

## Request Priority

Each request carries a `RequestPriority` enum (`Critical`, `High`, `Normal` (default), `Low`) on `InferenceRequest.Priority`. The scheduler's admission queue is a `PriorityQueue` keyed by `(-(int)Priority, submissionOrder)`, so higher priorities are dequeued before lower priorities and FIFO holds within a tier.

| Level | Admission behavior | Preemption (Step 59, pending) |
|-------|--------------------|-------------------------------|
| `Critical` | Admitted first, ahead of all other tiers | Never preempted |
| `High` | Admitted ahead of `Normal` and `Low` | Preempts `Normal` and `Low` |
| `Normal` | Default tier | — |
| `Low` | Admitted last | Preempted first |

**Admission ordering shipped (Phase 9 Step 59 first piece).** Verified by 4 unit tests in `ContinuousBatchSchedulerTests`:
- `Priority_HighAfterLowQueue_AdmittedFirst` — High submitted after queued Lows admits first.
- `Priority_SameTier_DrainsInSubmissionOrder` — FIFO within tier.
- `Priority_InferenceRequest_DefaultsToNormal` — guards default.
- `Priority_CriticalBeatsHighBeatsNormalBeatsLow` — strict tier ordering.

**Preemption (the second half of Step 59) still pending** — see § Preemption below. The `_preemptionCount` field and `BatchSchedulerMetrics.PreemptionCount` are wired through but always read 0 today.

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