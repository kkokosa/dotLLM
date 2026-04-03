# KV-Cache — dotLLM

## Purpose

The KV-cache stores previously computed key and value vectors for all layers, avoiding O(n²) recomputation during autoregressive generation. At each decode step, only the new token's K/V are computed and appended.

## Memory Consumption

Llama 3 8B, FP16, 2048 tokens:
```
2 (K+V) × 32 layers × 8 KV heads × 128 head_dim × 2048 tokens × 2 bytes
= ~1 GB
```
Scales linearly with sequence length and batch size. Dominant memory consumer in production.

## Simple KV-Cache (Phase 1)

Pre-allocated contiguous buffer per sequence:
```
K_cache[layer][kv_head][max_seq_len][head_dim]  — FP16
V_cache[layer][kv_head][max_seq_len][head_dim]  — FP16
```
Simple indexing: `K_cache[layer][head][pos] = new_K`. Wastes memory for short sequences.

## Paged KV-Cache — PagedAttention

Inspired by OS virtual memory paging.

### Design
- Divide cache into fixed-size **blocks** of B tokens (B = 16 or 32).
- **Block table** per sequence: maps logical positions to physical blocks (page table).
- **Free pool**: blocks allocated on demand, returned on completion.
- Memory waste: <4% (vs ~60% for static pre-allocation).

### Operations
- **Allocate**: When sequence needs more blocks, pop from free pool.
- **Free**: On sequence completion/eviction, return all blocks to pool.
- **Copy-on-write**: For beam search — beams share prefix blocks (ref-counted). On divergence, copy the shared block.
- **Fork**: For prompt caching — new sequence references existing prefix blocks.

### Attention Integration (v1: Staging Buffer)

Current implementation uses a staging buffer approach: `GetKeysRef()`/`GetValuesRef()` gather block data into a pre-allocated contiguous buffer before each attention call. This avoids modifying attention kernels. The gather cost is O(seqLen × headDim) — negligible compared to attention's O(seqLen² × headDim).

Future optimization: specialized paged attention kernels that read blocks directly via block table indirection, eliminating the staging copy.

### Key Classes

- `KvBlockPool` — Fixed-size pool of KV blocks. Per-layer contiguous storage, 64-byte aligned. Free list with `Interlocked` ref counting for thread safety.
- `KvBlockTable` — Per-sequence mapping from logical positions to physical block IDs. Supports `Fork` (shared prefix) and `EnsureWritable` (copy-on-write).
- `PagedKvCache : IKvCache` — Paged cache with staging-buffer gather. Drop-in replacement for `SimpleKvCache`.
- `PagedKvCacheFactory` — Creates `PagedKvCache` instances backed by a shared `KvBlockPool`.

### CLI

```
# CLI (run/chat): paged is opt-in
dotllm chat model.gguf --paged

# Server (serve): paged is on by default, opt-out with --no-paged
dotllm serve model.gguf                 # paged KV-cache active
dotllm serve model.gguf --no-paged      # simple KV-cache

# API-only server (no web UI)
dotllm serve model.gguf --no-ui
```

### Limitations (v1)

- CPU `PagedKvCache` only. CUDA and hybrid models fall back to their native KV-cache.
- Not compatible with quantized KV-cache (`--cache-type-k`/`--cache-type-v`). Falls back to `QuantizedKvCache` with a warning.
- Staging buffer means `GetKeysRef`/`GetValuesRef` results are only valid until the next call (shared buffer).

## KV-Cache Quantization

Compress cached K/V to extend context capacity:

| Format | CPU (vs FP32) | GPU (vs FP16) | Quality Impact |
|--------|---------------|---------------|----------------|
| FP32 (CPU) / FP16 (GPU) | 1× (baseline) | 1× (baseline) | None |
| Q8_0 | 3.76× | 1.88× | Minimal |
| Q4_0 | 7.11× | 3.56× | Small (for older tokens) |

### Implementation

**Dual-region storage** with quantize-on-evict:
1. **Quantized buffer** (append-only): Q8_0/Q4_0 blocks for positions outside the window
2. **Full-precision ring buffer**: FP32 (CPU) or FP16 (GPU) for the most recent W tokens

On each new token write, the oldest window entry is quantized and appended to the quantized buffer. Separate key/value quantization types are supported (e.g., Q8_0 keys + Q4_0 values).

### Key Classes
- `QuantizedKvCache` — CPU implementation in `DotLLM.Engine`
- `CudaQuantizedKvCache` — GPU implementation in `DotLLM.Cuda`
- `IQuantizedKvCache` — Interface extending `IKvCache` for quantized access
- `KvCacheConfig { KeyDType, ValueDType, MixedPrecisionWindowSize }` — Configuration

### Attention Integration
- **CPU**: Per-tile dequantization inside tiled attention kernel (`Attention.ExecuteTiledQuantizedHead`). Phase 1 processes quantized region with on-the-fly dequant, Phase 2 reads FP32 window directly.
- **GPU**: Scratch-buffer approach — dequant quantized region + ring-ordered window copy into temporary FP16 buffer, then standard attention kernel.

### CLI
```
--cache-type-k q8_0    # key quantization (f32, q8_0, q4_0)
--cache-type-v q4_0    # value quantization (f32, q8_0, q4_0)
--cache-window 64      # recent tokens in full precision (0 = all quantized)
```

Orthogonal to weight quantization — Q4_K_M model can use Q8_0 KV-cache.

## Simple Prompt Caching (Step 54)

Live KV-cache reuse for multi-turn conversations. No paged attention required — works with `SimpleKvCache`.

### Design: Live Reuse (not Snapshots)

After each generation, the KV-cache is transferred to a `PrefixCache` instead of being disposed. On the next call:

1. `PrefixCache.FindMatch(promptTokenIds)` scans entries (max 1–4) with element-wise prefix comparison (`MemoryExtensions.CommonPrefixLength`, SIMD-vectorized).
2. On hit: `SimpleKvCache.SetCurrentLength(matchedTokens)` truncates visible length. Suffix tokens are prefilled at positions `[matchedLen..promptLen)`.
3. On miss: fresh KV-cache allocated, full prefill as usual.

No data copying on cache hit. The same KV-cache object persists across calls.

### Key Classes

- `PrefixCache` — LRU cache with configurable max entries. Owns cached KV-cache instances.
- `PrefixCacheEntry` — Token sequence + live KV-cache + LRU timestamp.
- `SimpleKvCache.SetCurrentLength(int)` — Truncates visible length for prefix reuse.

### Multi-turn Chat Pattern

Each turn's prompt = previous prompt + assistant response + new user message. The stored token sequence (prompt + generated) shares a prefix with the new prompt. Typical cache hit rate: near 100%.

### CLI

```
--no-prompt-cache      # Disable (enabled by default in chat/serve)
--prompt-cache-size 1  # Max cached sessions (1 for chat, 4 for serve)
```

### Scope

- CPU `SimpleKvCache` only. QuantizedKvCache and GPU caches fall back to no caching.
- Cache cleared on model swap, `/clear` command (CLI), or `POST /v1/cache/clear` (server).

### Stats

`InferenceTimings.CachedTokenCount` reports how many prompt tokens were served from cache. Displayed in CLI output, API `timings.cached_tokens`, and Chat UI stats bar.

## Advanced Prompt Caching / Prefix Sharing (Future — Step 36+)

Requires paged KV-cache. Enables cross-request prefix sharing (e.g., shared system prompts across users).

### Problem
Many requests share the same system prompt (e.g., all chat requests in a deployment).
Recomputing KV-cache for the shared prefix is wasteful.

### Solution: Prefix Trie
- Maintain a **trie** of recently computed prompt prefixes, keyed by token sequences.
- On new request: walk the trie matching the prompt's token sequence.
- If match found: share the cached KV blocks (read-only), only prefill the new suffix.

### Implementation
- Shared blocks use **reference counting**. Freed when all referencing sequences complete.
- **LRU eviction** when memory scarce. Frequently used prefixes (system prompts) stay cached.
- **Explicit registration**: Server API accepts `prefix_id` for deterministic caching.

### Integration with PagedAttention
The prefix trie stores references to physical KV blocks. New sequences get their own block table with shared prefix entries pointing to existing blocks, plus new blocks for the suffix. Copy-on-write if modification needed (rare — KV cache is append-only).