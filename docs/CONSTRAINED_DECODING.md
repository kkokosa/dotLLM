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

### 2. JSON Schema
Guarantees output matches specific schema: required fields, types, enum values, nested structures. Schema compiled into automaton enforcing structural constraints. **Highest value** for tool calling and structured APIs.

### 3. Regex
Regular expression compiled to DFA. Each state: compute which tokens extend current partial match. Use cases: dates, phone numbers, enums, identifiers.

### 4. Context-Free Grammar (CFG)
GBNF-like notation (similar to llama.cpp). Pushdown automaton. Most general — constrains to programming languages, XML, custom DSLs.

## Key Implementation Challenges

### Token-Level Masking
Tokens span multiple characters. Must check if any valid continuation exists *through* the full token string, not just the first character. Requires pre-computing per-state token masks.

### Token Mask Precomputation
For each FSM state, pre-compute allowed token IDs. Cache masks indexed by state.
- Regex/JSON mode: feasible (modest state count)
- Complex JSON schemas: lazy computation + LRU cache

### Speculative Decoding Interaction
Draft model must respect constraints. Each speculated token advances automaton. On rejection: `Clone()` state before speculation, restore on rollback.

### Continuous Batching Interaction
Each sequence may have different constraint (or none). Per-sequence masks applied as batched masked-fill on logit tensor.

## Reference Implementations

- **llama.cpp** — GBNF grammar support
- **Outlines** (Python) — FSM-based structured generation
- **guidance** (Microsoft) — Interleaved generation and control
- **XGrammar** — Optimized grammar-based constrained decoding (vLLM/MLC-LLM)