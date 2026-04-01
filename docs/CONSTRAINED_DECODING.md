# Constrained Decoding — dotLLM

## Overview

Constrained decoding guarantees generated output conforms to a structure (JSON schema, regex, grammar) by masking invalid token logits to `-∞` at each decode step.

## Mechanism

1. Constraint compiled into FSM (finite state machine) or PDA (pushdown automaton) at request time.
2. Each step: automaton state → set of valid tokens → bit mask over vocabulary.
3. Mask applied to logits before temperature/top-k/top-p.
4. After sampling: automaton advances to next state.
5. Output is **mathematically guaranteed** to conform.

## IDecodingConstraint Interface

```
IDecodingConstraint:
  Advance(tokenId) → void           // Update state after token sampled
  GetAllowedTokens() → TokenMask    // Bit mask for current state
  IsComplete() → bool               // Constraint fully satisfied
  Clone() → IDecodingConstraint      // Snapshot for speculative rollback
  Reset() → void                     // Return to initial state
```

`TokenMask`: Compact bit vector over vocabulary (128K vocab = 16KB). Applied via vectorized masked-fill.

## Constraint Types (Priority Order)

### 1. JSON Mode

Guarantees syntactically valid JSON. FSM tracks parser state: in-object, in-array, in-string, expecting-key, expecting-value, etc. Modest state count.

**Implementation** (`JsonConstraint`): `JsonCharParser` struct — a 22-state FSM with `InlineArray(64)` nesting stack (~600 bytes). Tracks `JsonParserState` (Start, ObjectOpen, InString, InNumber, InLiteral, Done, etc.) plus nesting context (Object/Array), literal progress, and string escape/unicode state. State key: `HashCode.Combine(state, depthBucket, topContext, literalKind, literalIndex)`. Cache: `Dictionary<int, TokenMask>`.

### 2. JSON Schema

Guarantees output matches specific schema: required fields, types, enum values, nested structures. Schema compiled into automaton enforcing structural constraints. **Highest value** for tool calling and structured APIs.

**Implementation** (`JsonSchemaConstraint`): Layers `SchemaTracker` struct (~1.3KB) on top of `JsonCharParser`. `SchemaCompiler` compiles JSON Schema to flat `SchemaNode[]` array + `PropertyNameTrie[]` for key/enum validation. Tracker uses `InlineArray` stacks for node indices, emitted-property bitmasks, key char buffer, and array indices. Composite state key: `(parserKey, nodeIdx, emittedProps, triePos)`. Cache: `Dictionary<SchemaStateKey, TokenMask>` with full eviction at capacity.

### 3. Regex

Regular expression compiled to DFA. Each state: compute which tokens extend current partial match. Use cases: dates, phone numbers, enums, identifiers.

**Implementation** (`RegexConstraint`): Four-stage compilation pipeline:

1. **`RegexParser`** — Recursive descent parser. Pattern string → AST (`Literal`, `CharClass`, `Concat`, `Alternation`, `Repeat`). Supports: `[a-z]`, `[^...]`, `\d`/`\w`/`\s`, `.`, `*`/`+`/`?`/`{n,m}`, `|`, `(...)`. Rejects backreferences, lookahead/lookbehind, lazy quantifiers with clear error messages. Implicit anchoring — entire output must match.

2. **`NfaBuilder`** — Thompson construction. AST → NFA with epsilon transitions. `{n,m}` bounded repetition unrolled as n mandatory + (m−n) optional copies. Each fragment has single start/accept state.

3. **`DfaBuilder`** — Subset construction (powerset) + Hopcroft minimization. **Equivalence class compression**: collects character-range boundaries from all NFA transitions, partitions the 65536 UTF-16 code unit space into classes where all chars in a class behave identically. `CharToClass[65536]` (64KB) maps each char to a class index (typically 10–50 classes). Transition table is `StateCount × ClassCount` ints instead of `StateCount × 65536`. Max 10,000 DFA states guard against state explosion.

4. **`DfaSimulator`** — Zero-alloc struct holding a single `int` (current DFA state) + reference to shared immutable `CompiledDfa`. `TryAdvance(char)`: two array lookups (`CharToClass[c]` → `Transitions[state * ClassCount + classId]`). Struct copy for vocabulary scan = copying one `int`.

Cache key: DFA state ID (a single `int`). After visiting all DFA states (typically <50 for practical patterns), the cache is fully populated and `GetAllowedTokens()` is a pure dictionary lookup — no vocabulary scan.

**Supported regex subset** (DFA-compatible):

| Construct | Examples | Notes |
|-----------|----------|-------|
| Literals | `abc` | Concatenation of literal chars |
| Character classes | `[a-z]`, `[^0-9]` | Including negation |
| Predefined classes | `\d`, `\w`, `\s`, `.` | Expanded to char ranges |
| Quantifiers | `*`, `+`, `?`, `{n}`, `{n,m}` | Greedy only |
| Alternation | `a\|b` | |
| Grouping | `(abc)`, `(?:abc)` | Non-capturing only |
| Escapes | `\\`, `\.`, `\n`, `\t` | Standard |

Not supported: backreferences (`\1`), lookahead/lookbehind (`(?=...)`), lazy quantifiers (`*?`), anchors (`^`/`$`).

### 4. Context-Free Grammar (CFG)

GBNF-like notation (similar to llama.cpp). Pushdown automaton. Most general — constrains to programming languages, XML, custom DSLs.

**Implementation** (`GrammarConstraint`): Three-stage pipeline:

1. **`GbnfParser`** — Recursive descent parser for GBNF notation. Grammar text → AST (`Literal`, `CharClass`, `RuleRef`, `Sequence`, `Alternation`, `Repeat`). Handles `#` comments, multi-line rules, `\n`/`\t`/`\\`/`\"` escapes. Validates all rule references exist. Detects direct left recursion at parse time.

2. **`CompiledGrammar`** — Flattens AST rules into a linear `GrammarPosition[]` array. Each position is one of: `Terminal` (match char/class), `RuleCall` (push return point, jump to target), `RuleReturn` (pop stack), `Alternation` (list of branch entry points), `Join` (no-op convergence point). Fragment-based compilation returns `(entry, tail)` tuples ensuring alternation/repeat nodes are positioned as entry points for correct PDA traversal. Rule calls push return positions onto an `InlineArray(64)` stack.

3. **`PdaSimulator`** — Zero-alloc struct (~268 bytes): current position index + `ReturnStack` (`InlineArray(64)` of `int`) + stack depth + accepted flag. `TryAdvance(char)` resolves epsilon transitions (rule calls, returns, joins, alternation exploration) internally before matching a terminal character. Alternation is explored via a bounded worklist (`ExploreStack`, `InlineArray(64)` of `ExploreState`) — clones the PDA state for each branch, commits to the first branch where the character matches (ordered-choice / PEG semantics). `CanAccept()` explores epsilon paths to determine if EOS should be allowed.

Cache key: `GrammarStateKey(position, depthBucket, topOfStack, acceptedFlag)`. Cache: `Dictionary<GrammarStateKey, TokenMask>` with full eviction at 4096 entries.

**GBNF syntax**:

```
rule-name ::= definition
"literal"           — string literal
[a-z]               — character class
[^a-z]              — negated character class
( ... )             — grouping
|                   — alternation (ordered choice)
* + ?               — repetition
rule-name           — rule reference
# comment           — line comment
```

## Key Implementation Challenges

### Token-Level Masking
Tokens span multiple characters. Must check if any valid continuation exists *through* the full token string, not just the first character. Requires pre-computing per-state token masks.

### Token Mask Precomputation
For each automaton state, pre-compute allowed token IDs. Cache masks indexed by state.
- **JSON mode**: modest state count (~20 parser states × depth/context buckets). Fully cached after a few tokens.
- **Regex**: state count = DFA states (typically 5–50). Cache key is a single `int`. Fully cached very quickly.
- **JSON Schema**: composite key (parser state × schema node × emitted props × trie position). Lazy computation + eviction at capacity.
- **Grammar (CFG)**: composite key (position × depth × top-of-stack). Lazy computation + eviction at 4096 entries.

### Token Mask Application
`TokenMaskApplier` sets disallowed tokens to `-∞` using AVX2-vectorized masking:
- 64-token blocks: if all allowed (`~0L`), skip entirely; if all disallowed (`0L`), `Fill(-∞)`.
- 8-token sub-blocks: `ExpandByteMask` → `VBLENDVPS` selects between source logit and `-∞`.
- Scalar fallback for non-AVX2 systems.

Cost: O(vocabSize / 64) iterations with fast-path skipping for fully-allowed/disallowed blocks. For a 128K vocabulary, this is 2048 long comparisons — negligible compared to model forward pass.

### Speculative Decoding Interaction
Draft model must respect constraints. Each speculated token advances automaton. On rejection: `Clone()` state before speculation, restore on rollback.

### Continuous Batching Interaction
Each sequence may have different constraint (or none). Per-sequence masks applied as batched masked-fill on logit tensor.

## Performance Overhead

Constrained decoding adds per-token overhead at three points: mask computation, mask application, and automaton advance. The key question is how this compares to the model forward pass that dominates inference time.

### Cost Model Per Decode Step

| Phase | Operation | Cost |
|-------|-----------|------|
| **Mask computation** | On cache miss: iterate full vocabulary, clone simulator struct, simulate token text char-by-char. On cache hit: dictionary lookup. | Miss: O(V × L̄) where V = vocab size, L̄ = avg token length. Hit: O(1). |
| **Mask application** | AVX2 masked-fill over logit vector. | O(V / 64) with fast-path skipping. ~2K iterations for 128K vocab. |
| **Advance** | Decode token text, feed chars to simulator. | O(L) where L = token text length. Negligible. |

### Mask Computation: The Dominant Overhead

The expensive operation is `BuildAndCacheMask` — a full vocabulary scan that clones the automaton state and simulates each token's decoded text character by character. The per-token clone cost varies dramatically:

| Constraint | Simulator struct | Clone cost | Typical cache entries | Cache warm-up |
|-----------|-----------------|------------|----------------------|---------------|
| **Regex** | `DfaSimulator` (~8 bytes: 1 `int` + ref) | Copying one `int` | 5–50 (= DFA states) | 1–5 tokens |
| **JSON mode** | `JsonCharParser` (~600 bytes) | ~600 byte memcpy | ~40–100 | 5–20 tokens |
| **JSON Schema** | `JsonCharParser` + `SchemaTracker` (~1.9KB) | ~1.9KB memcpy | Hundreds–thousands | Slow warm-up, evictions |
| **Grammar** | `PdaSimulator` (~268 bytes) + alternation exploration | ~268 byte memcpy × branches | Hundreds | Moderate warm-up |

For a 128K vocabulary, one cache-miss mask computation performs 128K clone+simulate operations. At ~600 bytes per clone (JSON mode), that's ~73MB of memcpy per mask build — done once per unique state, then cached.

### Steady-State Throughput Impact

Once the mask cache is warm (all reachable states have cached masks), the per-token overhead is:
1. Dictionary lookup for cached mask: **~50–200ns**
2. AVX2 masked-fill over 128K floats: **~5–15μs**
3. Automaton advance (a few chars): **<100ns**

**Total steady-state overhead: ~5–15μs per token.**

For context, a single decode step (one token) on CPU takes:
- SmolLM-135M Q8_0: ~2–5ms → overhead is **~0.1–0.5%**
- Llama-3.2-1B Q4_K_M: ~15–40ms → overhead is **<0.1%**
- Llama-3.2-3B Q4_K_M: ~40–100ms → overhead is **<0.05%**

**Conclusion**: In steady state, constrained decoding overhead is negligible — well under 1% of decode latency for any model larger than ~100M parameters.

### Cold-Start / Cache-Miss Impact

The first time each automaton state is encountered, a full vocabulary scan is required. This adds:
- **Regex**: ~1–5ms per miss (very light clone). With ~10–30 DFA states, total cold-start cost is ~10–100ms spread across the first few tokens.
- **JSON mode**: ~5–20ms per miss. With ~40–80 effective states, ~200–1000ms total spread across early tokens.
- **JSON Schema**: ~10–50ms per miss (heavier clone). Many unique state keys mean frequent misses, especially with complex schemas. Can dominate decode time for the first ~20–50 tokens.
- **Grammar**: ~5–20ms per miss. Cache misses are more frequent than regex (deeper state space) but less frequent than complex schemas.

For short outputs (< 20 tokens), cache-miss overhead can be significant — adding 50–200% to total generation time. For longer outputs (100+ tokens), the amortized cost approaches the steady-state negligible overhead.

### Prefill Impact

Constrained decoding applies only to the **last logit** of the prefill output (the first generated token). One mask computation + one mask application. Negligible compared to the prefill forward pass.

### Memory Overhead

| Component | Size | Lifetime |
|-----------|------|----------|
| `TokenMask` per cached state | V / 8 bytes (16KB for 128K vocab) | Shared across clones, persists until Reset |
| `CompiledDfa` (regex) | `StateCount × ClassCount × 4` + 64KB `CharToClass` | Per-request, shared |
| `CompiledGrammar` | `Positions.Length × ~32 bytes` | Per-request, shared |
| Mask cache dictionary | 16KB × number of states | Shared across clones |

For regex with 30 states: ~480KB mask cache + ~65KB DFA tables = ~545KB.
For JSON mode with 80 states: ~1.3MB mask cache.
For complex schema: up to ~64MB mask cache at 4096 entries (before eviction).

## Key Files

| File | Purpose |
|------|---------|
| `Core/Constraints/IDecodingConstraint.cs` | Interface |
| `Core/Constraints/TokenMask.cs` | Bit vector over vocabulary |
| `Engine/Constraints/TokenMaskApplier.cs` | AVX2/scalar masked-fill |
| `Engine/Constraints/JsonConstraint.cs` | JSON mode constraint |
| `Engine/Constraints/JsonCharParser.cs` | 22-state JSON FSM |
| `Engine/Constraints/JsonSchemaConstraint.cs` | JSON Schema constraint |
| `Engine/Constraints/Schema/` | SchemaCompiler, SchemaTracker, PropertyNameTrie |
| `Engine/Constraints/RegexConstraint.cs` | Regex constraint |
| `Engine/Constraints/Regex/` | RegexParser, NfaBuilder, DfaBuilder, CompiledDfa, DfaSimulator |
| `Engine/Constraints/GrammarConstraint.cs` | Grammar constraint |
| `Engine/Constraints/Grammar/` | GbnfParser, CompiledGrammar, PdaSimulator |
| `Engine/TextGenerator.cs` | Integration point (ResponseFormat switch) |

## CLI Usage

```bash
# JSON mode — valid JSON
dotllm run model.gguf -p "..." --response-format json_object

# JSON Schema — schema-constrained JSON
dotllm run model.gguf -p "..." --response-format json_schema --schema @schema.json

# Regex — output matches pattern
dotllm run model.gguf -p "..." --response-format regex --pattern "\d{4}-\d{2}-\d{2}"

# Grammar — output conforms to GBNF grammar
dotllm run model.gguf -p "..." --response-format grammar --grammar @grammar.gbnf
dotllm run model.gguf -p "..." --response-format grammar --grammar 'root ::= "yes" | "no"'
```

`--schema` and `--grammar` accept inline strings or `@file` paths.

## Reference Implementations

- **llama.cpp** — GBNF grammar support
- **Outlines** (Python) — FSM-based structured generation
- **guidance** (Microsoft) — Interleaved generation and control
- **XGrammar** — Optimized grammar-based constrained decoding (vLLM/MLC-LLM)
