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

### Attention Integration
Attention kernels must handle non-contiguous K/V:
```
For each position in the sequence:
  block_idx = block_table[seq][pos / block_size]
  offset = pos % block_size
  K = cache_blocks[block_idx][offset]
```
Paged Flash Attention kernels handle this indirection natively.

## KV-Cache Quantization

Compress cached K/V to extend context capacity:

| Format | Compression | Quality Impact |
|--------|-------------|----------------|
| FP16 | 1× (baseline) | None |
| FP8 (E4M3) | 2× | Minimal (native on Hopper+) |
| INT8 | 2× | Small (per-head scales) |
| INT4 | 4× | Moderate (for older tokens) |

### Mixed Precision Cache
- Recent tokens (within window W): FP16 (high attention weight, quality-sensitive)
- Older tokens: INT8 or INT4 (low attention weight, less quality-sensitive)
- Configurable window size and quantization thresholds.

Configured via `KvCacheConfig { DType, MixedPrecisionWindowSize }`.
Orthogonal to weight quantization — Q4_K_M model can use FP8 cache.

## Prompt Caching / Automatic Prefix Sharing

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