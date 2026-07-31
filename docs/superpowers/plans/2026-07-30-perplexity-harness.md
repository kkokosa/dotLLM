# Perplexity Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a shared perplexity harness with a llama.cpp-comparable sliding-window mode, so numerics-changing kernel work (ours and upstream's) has an evaluation gate.

**Architecture:** Abstractions in `DotLLM.Core.Evaluation` (already landed, commit `7af01b73`); a single `PerplexityEvaluator` that selects its execution strategy from `IPerplexityModel.ReturnsAllRows` rather than from the caller; a streaming corpus reader; a `TransformerModel` adapter; and a `dotllm perplexity` CLI verb.

**Tech Stack:** .NET 10, C#, xUnit (+ `Xunit.SkippableFact`), Spectre.Console `CommandApp` for CLI.

## Global Constraints

- Branch `issue/231-perplexity-harness`, worktree `.claude/worktrees/ppl-harness`, based on `main`. PR targets upstream `main`.
- File-scoped namespaces. `<Nullable>enable</Nullable>`. XML doc comments on all public APIs.
- Commit messages include `(#231)` and end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- **The evaluator never loads weights.** It accepts an already-constructed `IPerplexityModel`. Non-negotiable: it is the Track D constraint.
- **The corpus is streamed and tokenized in chunks**, never materialized whole.
- `TeacherForced` mode must be preserved verbatim in behaviour. It is the "G1 precedent" methodology that existing quality gates depend on; changing any number it produces defeats the migration gate.
- No `System.Linq` on scoring hot paths; no managed allocation per scored token.

---

## Methodology note: what "llama.cpp-comparable" means precisely

llama.cpp's `perplexity` walks the corpus in chunks of `n_ctx` and scores **only the second half** of each chunk, so every scored token has at least `n_ctx/2` tokens of preceding context. The first half of the very first chunk is never scored.

Generalised to `(ContextLength L, Stride S)`:

- Window `w` covers absolute token range `[w*S, w*S + L)`.
- Scored targets are absolute indices `t` in `[w*S + L - S, w*S + L)`.
- Each scored token therefore has `L - S` tokens of context, and consecutive windows tile the scored tokens exactly — no gaps, no double-counting.
- **`S = L/2` reproduces llama.cpp's default exactly.**

A forward pass over window `[s, s+L)` returns rows `0..L-1`, where row `i` holds the distribution predicting the token at absolute index `s+i+1`. So scoring absolute target `t` reads row `t-s-1`. This requires `L - S >= 1`.

Tokens before the first window's scored range are never scored, matching llama.cpp. This is a known small bias and is documented rather than silently "fixed", because fixing it would break comparability.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/DotLLM.Core/Evaluation/IPerplexityModel.cs` | **Landed.** Model contract. |
| `src/DotLLM.Core/Evaluation/PerplexityResult.cs` | **Landed.** `PerplexityMode`, `PerplexityOptions`, `PerplexityResult`. |
| `src/DotLLM.Engine/Evaluation/LogProb.cs` | Numerically stable log-softmax of a single logit row. |
| `src/DotLLM.Engine/Evaluation/PerplexityEvaluator.cs` | Both modes; strategy chosen from `ReturnsAllRows`. |
| `src/DotLLM.Engine/Evaluation/CorpusReader.cs` | Streaming corpus → token chunks. |
| `src/DotLLM.Models/Evaluation/TransformerPerplexityModel.cs` | `IPerplexityModel` over `TransformerModel`. |
| `src/DotLLM.Cli/Commands/PerplexityCommand.cs` | `dotllm perplexity` verb. |
| `tests/DotLLM.Tests.Unit/Evaluation/FakePerplexityModel.cs` | Deterministic test double. |
| `tests/DotLLM.Tests.Unit/Evaluation/LogProbTests.cs` | Log-softmax correctness. |
| `tests/DotLLM.Tests.Unit/Evaluation/PerplexityEvaluatorTests.cs` | Both modes, both backend shapes. |
| `tests/DotLLM.Tests.Unit/Evaluation/CorpusReaderTests.cs` | Streaming/chunking. |
| `tests/DotLLM.Tests.Integration/Evaluation/PerplexityComparabilityTests.cs` | llama.cpp figure validation. |

---

### Task 1: Stable log-probability

**Files:**
- Create: `src/DotLLM.Engine/Evaluation/LogProb.cs`
- Test: `tests/DotLLM.Tests.Unit/Evaluation/LogProbTests.cs`

**Interfaces:**
- Consumes: nothing.
- Produces: `static double LogProb.OfTarget(ReadOnlySpan<float> logits, int target)` — log-softmax value at `target`, in nats.

- [ ] **Step 1: Write the failing test**

```csharp
using DotLLM.Engine.Evaluation;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

public sealed class LogProbTests
{
    [Fact]
    public void OfTarget_UniformLogits_IsNegativeLogVocab()
    {
        // Four equal logits => p = 1/4 for each => log p = -log 4.
        var logits = new float[] { 2.5f, 2.5f, 2.5f, 2.5f };
        double actual = LogProb.OfTarget(logits, target: 2);
        Assert.Equal(-Math.Log(4.0), actual, 12);
    }

    [Fact]
    public void OfTarget_IsShiftInvariant()
    {
        var a = new float[] { 1f, 2f, 3f };
        var b = new float[] { 1001f, 1002f, 1003f };
        Assert.Equal(LogProb.OfTarget(a, 1), LogProb.OfTarget(b, 1), 12);
    }

    [Fact]
    public void OfTarget_LargeLogits_DoesNotOverflow()
    {
        // Naive exp() would overflow to infinity here; the max-shift must prevent it.
        var logits = new float[] { 800f, 900f, 1000f };
        double actual = LogProb.OfTarget(logits, target: 2);
        Assert.True(double.IsFinite(actual));
        Assert.Equal(0.0, actual, 6);   // target dominates => p ~ 1 => log p ~ 0
    }

    [Fact]
    public void OfTarget_SumOfProbabilitiesIsOne()
    {
        var logits = new float[] { -1.5f, 0.25f, 3f, 0.5f };
        double sum = 0;
        for (int i = 0; i < logits.Length; i++) sum += Math.Exp(LogProb.OfTarget(logits, i));
        Assert.Equal(1.0, sum, 10);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~LogProbTests"`
Expected: FAIL — `LogProb` does not exist (compile error).

- [ ] **Step 3: Write minimal implementation**

```csharp
namespace DotLLM.Engine.Evaluation;

/// <summary>Numerically stable log-softmax over a single row of logits.</summary>
public static class LogProb
{
    /// <summary>
    /// Returns <c>log P(target)</c> in nats under a softmax over <paramref name="logits"/>.
    /// </summary>
    /// <remarks>
    /// Uses the max-shift identity <c>log softmax(x)_t = (x_t - m) - log sum_j exp(x_j - m)</c>
    /// with <c>m = max(x)</c>, so no <c>exp</c> argument is ever positive and overflow is
    /// impossible. Accumulates in <see cref="double"/>: a vocab of 128k float32 terms loses
    /// meaningful precision in float32, and perplexity differences between near-identical runs
    /// are exactly what this harness exists to resolve.
    /// </remarks>
    public static double OfTarget(ReadOnlySpan<float> logits, int target)
    {
        if ((uint)target >= (uint)logits.Length)
            throw new ArgumentOutOfRangeException(nameof(target));

        float max = logits[0];
        for (int j = 1; j < logits.Length; j++)
            if (logits[j] > max) max = logits[j];

        double sumExp = 0;
        for (int j = 0; j < logits.Length; j++)
            sumExp += Math.Exp(logits[j] - max);

        return (logits[target] - max) - Math.Log(sumExp);
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~LogProbTests"`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Engine/Evaluation/LogProb.cs tests/DotLLM.Tests.Unit/Evaluation/LogProbTests.cs
git commit -m "feat(eval): numerically stable log-softmax for perplexity scoring (#231)"
```

---

### Task 2: Test double

**Files:**
- Create: `tests/DotLLM.Tests.Unit/Evaluation/FakePerplexityModel.cs`

**Interfaces:**
- Consumes: `IPerplexityModel` from `DotLLM.Core.Evaluation`.
- Produces: `FakePerplexityModel(int vocabSize, int maxContextLength, bool returnsAllRows, Func<int,int,float[]> rowFactory)` with `IReadOnlyList<int[]> ForwardCalls` recording every window passed to `Forward`.

The recording matters as much as the logits: Tasks 3–5 assert on *which windows were evaluated*, which is how window tiling and the O(n) vs O(n²) strategy split get tested without a real model.

- [ ] **Step 1: Write the implementation** (a test double, so no test-first cycle)

```csharp
using DotLLM.Core.Evaluation;
using DotLLM.Core.Tensors;

namespace DotLLM.Tests.Unit.Evaluation;

/// <summary>
/// Deterministic <see cref="IPerplexityModel"/> for evaluator tests. Records every window it is
/// asked to score so tests can assert on window tiling, not just on the resulting number.
/// </summary>
internal sealed class FakePerplexityModel : IPerplexityModel, IDisposable
{
    private readonly Func<int, int, float[]> _rowFactory;   // (absolutePosition, vocabSize) => logits
    private readonly List<int[]> _forwardCalls = new();
    private readonly List<UnmanagedTensor> _issued = new();

    public FakePerplexityModel(
        int vocabSize, int maxContextLength, bool returnsAllRows,
        Func<int, int, float[]> rowFactory)
    {
        VocabSize = vocabSize;
        MaxContextLength = maxContextLength;
        ReturnsAllRows = returnsAllRows;
        _rowFactory = rowFactory;
    }

    public int VocabSize { get; }
    public int MaxContextLength { get; }
    public bool ReturnsAllRows { get; }

    /// <summary>Token windows passed to <see cref="Forward"/>, in call order.</summary>
    public IReadOnlyList<int[]> ForwardCalls => _forwardCalls;

    public ITensor Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions)
    {
        _forwardCalls.Add(tokens.ToArray());

        int rows = ReturnsAllRows ? tokens.Length : 1;
        int firstRow = ReturnsAllRows ? 0 : tokens.Length - 1;
        var tensor = UnmanagedTensor.Allocate(new TensorShape(rows, VocabSize), DType.F32);
        unsafe
        {
            var dest = new Span<float>((void*)tensor.DataPointer, rows * VocabSize);
            for (int r = 0; r < rows; r++)
                _rowFactory(positions[firstRow + r], VocabSize).CopyTo(dest[(r * VocabSize)..]);
        }
        _issued.Add(tensor);
        return tensor;
    }

    /// <summary>Uniform logits: every target scores exactly -log(vocabSize).</summary>
    public static Func<int, int, float[]> Uniform => (_, vocab) => new float[vocab];

    public void Dispose()
    {
        foreach (var t in _issued) t.Dispose();
        _issued.Clear();
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `dotnet build tests/DotLLM.Tests.Unit`
Expected: build succeeds. If `UnmanagedTensor.Allocate` differs in signature, adjust to the actual factory in `src/DotLLM.Core/Tensors/UnmanagedTensor.cs` — do not change the tensor type used.

- [ ] **Step 3: Commit**

```bash
git add tests/DotLLM.Tests.Unit/Evaluation/FakePerplexityModel.cs
git commit -m "test(eval): deterministic IPerplexityModel double recording forward windows (#231)"
```

---

### Task 3: `TeacherForced` mode, all-rows backend

**Files:**
- Create: `src/DotLLM.Engine/Evaluation/PerplexityEvaluator.cs`
- Test: `tests/DotLLM.Tests.Unit/Evaluation/PerplexityEvaluatorTests.cs`

**Interfaces:**
- Consumes: `LogProb.OfTarget`, `IPerplexityModel`, `PerplexityOptions`, `PerplexityResult`, `FakePerplexityModel`.
- Produces: `PerplexityResult PerplexityEvaluator.Evaluate(IPerplexityModel model, ReadOnlySpan<int> tokens, PerplexityOptions options)`.

- [ ] **Step 1: Write the failing test**

```csharp
using DotLLM.Core.Evaluation;
using DotLLM.Engine.Evaluation;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

public sealed class PerplexityEvaluatorTests
{
    private static readonly int[] Tokens = Enumerable.Range(0, 32).ToArray();

    [Fact]
    public void TeacherForced_AllRowsBackend_UniformLogitsGivesVocabSizePerplexity()
    {
        // Uniform logits => P(any target) = 1/vocab => perplexity == vocab, exactly.
        using var model = new FakePerplexityModel(
            vocabSize: 7, maxContextLength: 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        var result = PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, ContextLength: 32, Stride: 32));

        Assert.Equal(7.0, result.Perplexity, 9);
        Assert.Equal(31, result.ScoredTokens);   // n-1 targets from one pass
    }

    [Fact]
    public void TeacherForced_AllRowsBackend_UsesASingleForwardPass()
    {
        using var model = new FakePerplexityModel(7, 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32));

        // The whole point of ReturnsAllRows: one pass scores every target.
        Assert.Single(model.ForwardCalls);
        Assert.Equal(32, model.ForwardCalls[0].Length);
    }

    [Fact]
    public void MeanNll_AndPerplexity_AreConsistent()
    {
        using var model = new FakePerplexityModel(7, 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        var result = PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32));

        Assert.Equal(result.Perplexity, Math.Exp(result.MeanNegativeLogLikelihood), 9);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~PerplexityEvaluatorTests"`
Expected: FAIL — `PerplexityEvaluator` does not exist.

- [ ] **Step 3: Write minimal implementation**

```csharp
using DotLLM.Core.Evaluation;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine.Evaluation;

/// <summary>
/// Computes perplexity over a token sequence using an <see cref="IPerplexityModel"/>.
/// </summary>
/// <remarks>
/// The evaluator never loads weights — callers pass an already-constructed model. On unified-memory
/// parts a large VRAM carve-out leaves host RAM scarce, and perplexity (a long run of full-context
/// prefills) is the workload most punished by holding a second host-side copy of the weights.
/// </remarks>
public static class PerplexityEvaluator
{
    /// <summary>Scores <paramref name="tokens"/> and returns the aggregate result.</summary>
    public static PerplexityResult Evaluate(
        IPerplexityModel model, ReadOnlySpan<int> tokens, PerplexityOptions options)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (tokens.Length < 2)
            throw new ArgumentException("At least two tokens are required to score one target.", nameof(tokens));

        int context = Math.Min(options.ContextLength, model.MaxContextLength);
        if (context < 2)
            throw new ArgumentException("Context length must be at least 2.", nameof(options));

        return options.Mode switch
        {
            PerplexityMode.TeacherForced => EvaluateTeacherForced(model, tokens, context),
            _ => throw new NotSupportedException($"Mode {options.Mode} is not implemented yet."),
        };
    }

    private static unsafe PerplexityResult EvaluateTeacherForced(
        IPerplexityModel model, ReadOnlySpan<int> tokens, int context)
    {
        int length = Math.Min(tokens.Length, context);
        Span<int> positions = length <= 512 ? stackalloc int[length] : new int[length];
        for (int i = 0; i < length; i++) positions[i] = i;

        double sumNll = 0;
        int scored = 0;

        using ITensor logits = model.Forward(tokens[..length], positions);
        int vocab = model.VocabSize;
        // Row i predicts token i+1, so the final row has no target within the window.
        for (int i = 0; i < length - 1; i++)
        {
            var row = new ReadOnlySpan<float>((void*)(logits.DataPointer + (nint)i * vocab * sizeof(float)), vocab);
            sumNll += -LogProb.OfTarget(row, tokens[i + 1]);
            scored++;
        }

        double meanNll = sumNll / scored;
        return new PerplexityResult(Math.Exp(meanNll), meanNll, scored, WindowCount: 1);
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~PerplexityEvaluatorTests"`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Engine/Evaluation/PerplexityEvaluator.cs tests/DotLLM.Tests.Unit/Evaluation/PerplexityEvaluatorTests.cs
git commit -m "feat(eval): PerplexityEvaluator teacher-forced mode for all-rows backends (#231)"
```

---

### Task 4: `TeacherForced` mode, last-row-only backend

**Files:**
- Modify: `src/DotLLM.Engine/Evaluation/PerplexityEvaluator.cs`
- Test: `tests/DotLLM.Tests.Unit/Evaluation/PerplexityEvaluatorTests.cs`

**Interfaces:**
- Consumes: everything from Task 3.
- Produces: no new public surface. `Evaluate` now honours `ReturnsAllRows == false` via growing-prefix re-prefill.

This is the O(n²) path the CUDA harnesses need. It must produce the **same number** as the all-rows path on the same tokens — that equivalence is what makes the migration gate meaningful.

- [ ] **Step 1: Write the failing test**

```csharp
    [Fact]
    public void TeacherForced_LastRowOnlyBackend_MatchesAllRowsBackendExactly()
    {
        // Position-dependent but deterministic logits, so a wrong row/position mapping shows up.
        static float[] Rows(int position, int vocab)
        {
            var row = new float[vocab];
            for (int j = 0; j < vocab; j++) row[j] = (float)Math.Sin((position + 1) * (j + 1) * 0.37);
            return row;
        }

        using var allRows = new FakePerplexityModel(7, 64, returnsAllRows: true, Rows);
        using var lastRow = new FakePerplexityModel(7, 64, returnsAllRows: false, Rows);
        var options = new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32);

        var a = PerplexityEvaluator.Evaluate(allRows, Tokens, options);
        var b = PerplexityEvaluator.Evaluate(lastRow, Tokens, options);

        Assert.Equal(a.Perplexity, b.Perplexity, 9);
        Assert.Equal(a.ScoredTokens, b.ScoredTokens);
    }

    [Fact]
    public void TeacherForced_LastRowOnlyBackend_ReprefixesGrowingWindows()
    {
        using var model = new FakePerplexityModel(7, 64, returnsAllRows: false, FakePerplexityModel.Uniform);

        PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.TeacherForced, 32, 32));

        // One forward per scored target, each one token longer than the last.
        Assert.Equal(31, model.ForwardCalls.Count);
        for (int i = 0; i < model.ForwardCalls.Count; i++)
            Assert.Equal(i + 1, model.ForwardCalls[i].Length);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~PerplexityEvaluatorTests"`
Expected: FAIL — last-row-only path reads row 0 as if it were the full grid, so perplexity mismatches (or an index error).

- [ ] **Step 3: Write minimal implementation**

Replace `EvaluateTeacherForced` with a dispatcher plus the two strategies:

```csharp
    private static PerplexityResult EvaluateTeacherForced(
        IPerplexityModel model, ReadOnlySpan<int> tokens, int context)
        => model.ReturnsAllRows
            ? TeacherForcedSinglePass(model, tokens, context)
            : TeacherForcedGrowingPrefix(model, tokens, context);

    // Backend returns only the final row, so each target needs its own prefill over the growing
    // prefix. O(n^2) in forward passes — unavoidable, and the reason the CUDA harnesses that
    // originated this methodology carry a stride.
    private static unsafe PerplexityResult TeacherForcedGrowingPrefix(
        IPerplexityModel model, ReadOnlySpan<int> tokens, int context)
    {
        int length = Math.Min(tokens.Length, context);
        int vocab = model.VocabSize;
        var positions = new int[length];
        for (int i = 0; i < length; i++) positions[i] = i;

        double sumNll = 0;
        int scored = 0;
        for (int prefix = 1; prefix < length; prefix++)
        {
            using ITensor logits = model.Forward(tokens[..prefix], positions.AsSpan(0, prefix));
            var row = new ReadOnlySpan<float>((void*)logits.DataPointer, vocab);
            sumNll += -LogProb.OfTarget(row, tokens[prefix]);
            scored++;
        }

        double meanNll = sumNll / scored;
        return new PerplexityResult(Math.Exp(meanNll), meanNll, scored, WindowCount: scored);
    }
```

Rename the Task 3 body to `TeacherForcedSinglePass` (signature unchanged).

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~PerplexityEvaluatorTests"`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Engine/Evaluation/PerplexityEvaluator.cs tests/DotLLM.Tests.Unit/Evaluation/PerplexityEvaluatorTests.cs
git commit -m "feat(eval): growing-prefix teacher-forced path for last-row-only backends (#231)"
```

---

### Task 5: `SlidingWindow` mode

**Files:**
- Modify: `src/DotLLM.Engine/Evaluation/PerplexityEvaluator.cs`
- Test: `tests/DotLLM.Tests.Unit/Evaluation/PerplexityEvaluatorTests.cs`

**Interfaces:**
- Consumes: everything above.
- Produces: no new public surface. `PerplexityMode.SlidingWindow` becomes functional.

Implement exactly the tiling defined in the methodology note at the top of this plan. Re-read it before writing code.

- [ ] **Step 1: Write the failing test**

```csharp
    [Fact]
    public void SlidingWindow_TilesScoredTokensWithoutGapsOrOverlap()
    {
        var tokens = Enumerable.Range(0, 40).ToArray();
        using var model = new FakePerplexityModel(7, 64, returnsAllRows: true, FakePerplexityModel.Uniform);

        // L=16, S=8 => windows start at 0, 8, 16, 24; each scores its last 8 targets.
        var result = PerplexityEvaluator.Evaluate(
            model, tokens, new PerplexityOptions(PerplexityMode.SlidingWindow, ContextLength: 16, Stride: 8));

        Assert.Equal(4, result.WindowCount);
        Assert.Equal(32, result.ScoredTokens);      // 4 windows x 8 targets
        Assert.Equal(7.0, result.Perplexity, 9);    // uniform logits

        Assert.Equal(4, model.ForwardCalls.Count);
        Assert.All(model.ForwardCalls, w => Assert.Equal(16, w.Length));
        Assert.Equal(0, model.ForwardCalls[0][0]);
        Assert.Equal(8, model.ForwardCalls[1][0]);
        Assert.Equal(16, model.ForwardCalls[2][0]);
        Assert.Equal(24, model.ForwardCalls[3][0]);
    }

    [Fact]
    public void SlidingWindow_ScoresEachTargetAtItsTrueAbsolutePosition()
    {
        // Logits keyed to absolute position: a window that restarts positions at zero scores
        // different values and fails this.
        static float[] Rows(int position, int vocab)
        {
            var row = new float[vocab];
            row[position % vocab] = 10f;
            return row;
        }

        var tokens = new int[40];
        for (int i = 0; i < tokens.Length; i++) tokens[i] = (i + 1) % 7;   // target == predicted argmax

        using var model = new FakePerplexityModel(7, 64, returnsAllRows: true, Rows);
        var result = PerplexityEvaluator.Evaluate(
            model, tokens, new PerplexityOptions(PerplexityMode.SlidingWindow, 16, 8));

        // Row i of window [s, s+L) sits at absolute position s+i and predicts token s+i+1,
        // whose id is (s+i+1)%7 -- exactly the argmax. So NLL is near zero throughout.
        Assert.True(result.MeanNegativeLogLikelihood < 0.01,
            $"expected confident predictions, got mean NLL {result.MeanNegativeLogLikelihood}");
    }

    [Fact]
    public void SlidingWindow_RejectsStrideNotSmallerThanContext()
    {
        using var model = new FakePerplexityModel(7, 64, returnsAllRows: true, FakePerplexityModel.Uniform);
        Assert.Throws<ArgumentException>(() => PerplexityEvaluator.Evaluate(
            model, Tokens, new PerplexityOptions(PerplexityMode.SlidingWindow, ContextLength: 16, Stride: 16)));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~PerplexityEvaluatorTests"`
Expected: FAIL — `NotSupportedException` from the `Evaluate` switch.

- [ ] **Step 3: Write minimal implementation**

Add the `SlidingWindow` arm to the `Evaluate` switch, then:

```csharp
    // llama.cpp `--perplexity` tiling. Window w covers [w*S, w*S + L) and scores absolute targets
    // in [w*S + L - S, w*S + L), so every scored token carries L-S tokens of context and the
    // scored ranges tile the corpus exactly. S = L/2 reproduces llama.cpp's default.
    // Targets before the first window's scored range are never scored -- llama.cpp skips them too,
    // and "fixing" that would break comparability.
    private static unsafe PerplexityResult EvaluateSlidingWindow(
        IPerplexityModel model, ReadOnlySpan<int> tokens, int context, int stride)
    {
        if (stride < 1 || stride >= context)
            throw new ArgumentException(
                $"Stride must be in [1, {context - 1}] for a context of {context}; each scored token needs at least one token of context.",
                nameof(stride));

        int vocab = model.VocabSize;
        var positions = new int[context];
        double sumNll = 0;
        int scored = 0, windows = 0;

        for (int start = 0; start + context <= tokens.Length; start += stride)
        {
            for (int i = 0; i < context; i++) positions[i] = start + i;

            using ITensor logits = model.Forward(tokens.Slice(start, context), positions);
            windows++;

            // Absolute targets [start + context - stride, start + context); row for target t is t-start-1.
            for (int t = start + context - stride; t < start + context; t++)
            {
                int row = t - start - 1;
                var span = new ReadOnlySpan<float>(
                    (void*)(logits.DataPointer + (nint)row * vocab * sizeof(float)), vocab);
                sumNll += -LogProb.OfTarget(span, tokens[t]);
                scored++;
            }
        }

        if (scored == 0)
            throw new ArgumentException(
                $"Corpus of {tokens.Length} tokens is shorter than one context window of {context}.", nameof(tokens));

        double meanNll = sumNll / scored;
        return new PerplexityResult(Math.Exp(meanNll), meanNll, scored, windows);
    }
```

Note: the last-row-only backend is **not** supported in `SlidingWindow` — throw `NotSupportedException` with a message pointing at `TeacherForced`, since re-prefilling per target inside a sliding window is both O(n²) and redundant with the growing-prefix mode.

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~PerplexityEvaluatorTests"`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Engine/Evaluation/PerplexityEvaluator.cs tests/DotLLM.Tests.Unit/Evaluation/PerplexityEvaluatorTests.cs
git commit -m "feat(eval): llama.cpp-comparable sliding-window perplexity mode (#231)"
```

---

### Task 6: Streaming corpus reader

**Files:**
- Create: `src/DotLLM.Engine/Evaluation/CorpusReader.cs`
- Test: `tests/DotLLM.Tests.Unit/Evaluation/CorpusReaderTests.cs`

**Interfaces:**
- Consumes: `DotLLM.Tokenizers` `ITokenizer` (`int[] Encode(string)`).
- Produces: `static IEnumerable<int> CorpusReader.StreamTokens(TextReader reader, ITokenizer tokenizer, int maxTokens = 0, int charChunkSize = 65536)`.

Streaming is a spec constraint, not an optimisation: `wiki.test.raw` is ~1.3 MB and its token array is ~340k ints, and Track D's premise is that host RAM is the scarce resource.

- [ ] **Step 1: Write the failing test**

```csharp
using DotLLM.Engine.Evaluation;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

public sealed class CorpusReaderTests
{
    // One token per whitespace-separated word; ids are word lengths, so order is checkable.
    private sealed class WordTokenizer : ITokenizer
    {
        public int[] Encode(string text) =>
            text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(w => w.Length).ToArray();
        public string Decode(ReadOnlySpan<int> ids) => throw new NotSupportedException();
    }

    [Fact]
    public void StreamTokens_ProducesTokensInOrder()
    {
        using var reader = new StringReader("a bb ccc dddd");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer()).ToArray();
        Assert.Equal(new[] { 1, 2, 3, 4 }, tokens);
    }

    [Fact]
    public void StreamTokens_HonoursMaxTokens()
    {
        using var reader = new StringReader("a bb ccc dddd eeeee");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer(), maxTokens: 3).ToArray();
        Assert.Equal(new[] { 1, 2, 3 }, tokens);
    }

    [Fact]
    public void StreamTokens_DoesNotSplitTokensAcrossChunkBoundaries()
    {
        // A tiny chunk size forces the boundary case: "ccc" must not become "c" + "cc".
        using var reader = new StringReader("a bb ccc dddd eeeee ffffff");
        var tokens = CorpusReader.StreamTokens(reader, new WordTokenizer(), maxTokens: 0, charChunkSize: 4).ToArray();
        Assert.Equal(new[] { 1, 2, 3, 4, 5, 6 }, tokens);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~CorpusReaderTests"`
Expected: FAIL — `CorpusReader` does not exist. If `ITokenizer`'s real shape differs, adjust the stub to match `src/DotLLM.Tokenizers/` and keep the assertions.

- [ ] **Step 3: Write minimal implementation**

```csharp
using System.Text;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Evaluation;

/// <summary>Streams a text corpus into tokens without materializing the whole file or token array.</summary>
public static class CorpusReader
{
    /// <summary>
    /// Reads <paramref name="reader"/> in character chunks, tokenizes each chunk, and yields token
    /// ids in order, stopping after <paramref name="maxTokens"/> (0 = unbounded).
    /// </summary>
    /// <remarks>
    /// Chunks are cut at the last whitespace so a token is never split across a boundary; the
    /// remainder is carried into the next chunk. The final chunk is flushed whole.
    /// </remarks>
    public static IEnumerable<int> StreamTokens(
        TextReader reader, ITokenizer tokenizer, int maxTokens = 0, int charChunkSize = 65536)
    {
        ArgumentNullException.ThrowIfNull(reader);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (charChunkSize < 1) throw new ArgumentOutOfRangeException(nameof(charChunkSize));

        var buffer = new char[charChunkSize];
        var carry = new StringBuilder();
        int emitted = 0;

        while (true)
        {
            int read = reader.Read(buffer, 0, buffer.Length);
            if (read == 0) break;

            carry.Append(buffer, 0, read);
            string pending = carry.ToString();

            int cut = pending.LastIndexOf(' ');
            if (cut < 0) continue;   // no safe split point yet; accumulate

            string ready = pending[..cut];
            carry.Clear();
            carry.Append(pending[(cut + 1)..]);

            foreach (int id in tokenizer.Encode(ready))
            {
                yield return id;
                if (maxTokens > 0 && ++emitted >= maxTokens) yield break;
            }
        }

        if (carry.Length > 0)
        {
            foreach (int id in tokenizer.Encode(carry.ToString()))
            {
                yield return id;
                if (maxTokens > 0 && ++emitted >= maxTokens) yield break;
            }
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test tests/DotLLM.Tests.Unit --filter "FullyQualifiedName~CorpusReaderTests"`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Engine/Evaluation/CorpusReader.cs tests/DotLLM.Tests.Unit/Evaluation/CorpusReaderTests.cs
git commit -m "feat(eval): streaming corpus tokenizer for perplexity runs (#231)"
```

---

### Task 7: `TransformerModel` adapter

**Files:**
- Create: `src/DotLLM.Models/Evaluation/TransformerPerplexityModel.cs`

**Interfaces:**
- Consumes: `TransformerModel` (`Config.VocabSize`, `Config.MaxSequenceLength`, `ITensor Forward(ReadOnlySpan<int>, ReadOnlySpan<int>, int deviceId)`), `IPerplexityModel`.
- Produces: `TransformerPerplexityModel(TransformerModel model, int deviceId = -1)` implementing `IPerplexityModel`.

- [ ] **Step 1: Write the implementation**

`TransformerModel.Forward` is documented as returning `[seqLen, vocab]` for all input positions, so `ReturnsAllRows` is `true`.

```csharp
using DotLLM.Core.Evaluation;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;

namespace DotLLM.Models.Evaluation;

/// <summary>Adapts <see cref="TransformerModel"/> to <see cref="IPerplexityModel"/>.</summary>
/// <remarks>
/// Holds a borrowed reference: the adapter does not own the model and does not dispose it, so the
/// caller keeps a single resident copy of the weights. This is the whole point of the evaluator
/// taking a constructed model rather than a path.
/// </remarks>
public sealed class TransformerPerplexityModel : IPerplexityModel
{
    private readonly TransformerModel _model;
    private readonly int _deviceId;

    /// <param name="model">An already-loaded model. Not owned; not disposed by this adapter.</param>
    /// <param name="deviceId">Device for the forward pass; <c>-1</c> is CPU.</param>
    public TransformerPerplexityModel(TransformerModel model, int deviceId = -1)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _deviceId = deviceId;
    }

    /// <inheritdoc/>
    public int VocabSize => _model.Config.VocabSize;

    /// <inheritdoc/>
    public int MaxContextLength => _model.Config.MaxSequenceLength;

    /// <inheritdoc/>
    public bool ReturnsAllRows => true;

    /// <inheritdoc/>
    public ITensor Forward(ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions)
        => _model.Forward(tokens, positions, _deviceId);
}
```

- [ ] **Step 2: Verify it compiles**

Run: `dotnet build src/DotLLM.Models`
Expected: build succeeds.

- [ ] **Step 3: Commit**

```bash
git add src/DotLLM.Models/Evaluation/TransformerPerplexityModel.cs
git commit -m "feat(eval): IPerplexityModel adapter over TransformerModel (#231)"
```

---

### Task 8: `dotllm perplexity` CLI verb

**Files:**
- Create: `src/DotLLM.Cli/Commands/PerplexityCommand.cs`
- Modify: `src/DotLLM.Cli/Program.cs` (register alongside `run`/`chat`/`serve`)

**Interfaces:**
- Consumes: `PerplexityEvaluator.Evaluate`, `CorpusReader.StreamTokens`, `TransformerPerplexityModel`, `GgufFileResolver` (follow `RunCommand`'s resolution pattern exactly).
- Produces: CLI verb `dotllm perplexity <model> --corpus <path> [--context N] [--stride N] [--max-tokens N] [--mode sliding-window|teacher-forced]`.

- [ ] **Step 1: Write the command**

Follow `RunCommand`'s `AsyncCommand<Settings>` shape. Defaults: `--context 512`, `--stride 256` (i.e. `L/2`, llama.cpp's default), `--mode sliding-window`, `--max-tokens 0`.

Output must report `Perplexity`, `MeanNegativeLogLikelihood`, `ScoredTokens`, `WindowCount`, plus the effective context and stride — a perplexity figure without its token count and window geometry is not comparable to anything, which is the failure this harness exists to prevent.

- [ ] **Step 2: Register the verb in `Program.cs`**

```csharp
config.AddCommand<PerplexityCommand>("perplexity")
    .WithDescription("Compute perplexity over a text corpus.")
    .WithExample("perplexity", "QuantFactory/SmolLM-135M-GGUF", "--corpus", "wiki.test.raw", "--context", "512", "--stride", "256");
```

- [ ] **Step 3: Verify end-to-end on a small real model**

Run: `dotnet run --project src/DotLLM.Cli -- perplexity QuantFactory/SmolLM-135M-GGUF --corpus <path to a few KB of English text> --context 256 --stride 128 --max-tokens 2048`
Expected: a finite perplexity in a plausible range for a 135M model on English prose (roughly 15–60), with `ScoredTokens` and `WindowCount` consistent with the tiling — `WindowCount == floor((tokens - context)/stride) + 1` and `ScoredTokens == WindowCount * stride`.

- [ ] **Step 4: Commit**

```bash
git add src/DotLLM.Cli/Commands/PerplexityCommand.cs src/DotLLM.Cli/Program.cs
git commit -m "feat(cli): dotllm perplexity verb (#231)"
```

---

### Task 9: llama.cpp comparability validation

**Files:**
- Create: `tests/DotLLM.Tests.Integration/Evaluation/PerplexityComparabilityTests.cs`

**Interfaces:**
- Consumes: all of the above.
- Produces: the acceptance evidence for the spec's second verification requirement.

Without this the word "comparable" is unearned, and the harness would give upstream false confidence on exactly the numerics-changing decisions it is meant to gate.

- [ ] **Step 1: Produce the reference figure**

Run llama.cpp's own perplexity on a fixed model + corpus, recording the exact build, model file, quantization, context and stride:

```bash
llama-perplexity -m <model.gguf> -f wiki.test.raw -c 512
```

Record the reported perplexity, chunk count, and token count in the test file as constants with a comment naming the llama.cpp build hash. **Do not** paste a figure from a blog post — the model file and quantization must match ours exactly.

- [ ] **Step 2: Write the test**

`[SkippableFact]`, skipped when the corpus or GGUF is unavailable (follow the `DOTLLM_BITNET_GGUF` early-return pattern already used in the integration suite). Assert our perplexity is within a stated relative tolerance of the recorded llama.cpp figure, and that `ScoredTokens` matches llama.cpp's token count.

Start with a 1% tolerance. If it fails, **do not widen the tolerance to make it pass** — a real discrepancy means the tiling or the scored-range definition differs, and that is the bug this task exists to catch. Investigate the window geometry first.

- [ ] **Step 3: Run it**

Run: `dotnet test tests/DotLLM.Tests.Integration --filter "FullyQualifiedName~PerplexityComparabilityTests"`
Expected: PASS, or a documented discrepancy with the geometry investigated.

- [ ] **Step 4: Commit and open the PR**

```bash
git add tests/DotLLM.Tests.Integration/Evaluation/PerplexityComparabilityTests.cs
git commit -m "test(eval): validate sliding-window perplexity against llama.cpp reference (#231)"
```

Then open the PR against upstream `main` referencing `Closes #231` and linking kkokosa/dotLLM#416.

---

## Follow-up (separate, on `dev`)

Migrate the ten existing per-test perplexity helpers onto this harness, gated on producing **numerically identical** results. Not part of this PR: those helpers exist only on `dev`, and this PR targets upstream `main`. File as its own issue once this lands.

## Self-Review

**Spec coverage:** contract (landed) · evaluator both modes (Tasks 3–5) · corpus streaming (Task 6) · adapter (Task 7) · CLI verb (Task 8) · comparability validation (Task 9) · never-loads-weights constraint (Task 7 adapter is borrowed-reference; evaluator signature takes a model) · streaming constraint (Task 6). Migration of existing helpers is explicitly deferred with a reason.

**Placeholders:** none. Task 8's command body follows an existing in-repo pattern rather than inventing one, and its acceptance is a concrete arithmetic check on window geometry.

**Type consistency:** `LogProb.OfTarget`, `PerplexityEvaluator.Evaluate`, `CorpusReader.StreamTokens`, `TransformerPerplexityModel` are used with identical signatures everywhere they appear. `TeacherForcedSinglePass`/`TeacherForcedGrowingPrefix` naming is fixed in Task 4.
