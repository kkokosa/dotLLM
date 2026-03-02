# SentencePieceEncoding — Implementation Notes

> Implementation file: `SentencePieceEncoding.cs`
> Used by: Llama 1/2, Mistral, TinyLlama, SmolLM, and any model with `tokenizer.ggml.model = "llama"` or `"mistral"` in GGUF metadata.

---

## What is SentencePiece BPE?

SentencePiece is a language-independent tokenization library developed by Google. It treats the input as a raw character stream (rather than pre-splitting on whitespace) and represents word boundaries with a special marker character `▁` (U+2581, LOWER ONE EIGHTH BLOCK). Every word-initial position is prefixed with `▁`, so `"hello world"` is normalized to `"▁hello▁world"` before tokenization.

The BPE (Byte-Pair Encoding) variant of SentencePiece builds its vocabulary greedily: starting from individual characters, pairs of adjacent tokens are merged into a new token according to a **score** (log-probability of the merged token in the training corpus). Higher score = more frequent merge = applied first. The result is a compact set of sub-word tokens that balances coverage and compression.

---

## Encode Pipeline

`Encode(string text)` runs four steps:

### Step 1 — Normalization

```
"hello world" → "▁hello▁world"
```

- Every ASCII space is replaced by `▁` (`U+2581`).
- If `_addBosSpace` is `true` (the default for GGUF-loaded models) **and** the normalized text does not already begin with `▁`, a `▁` is prepended. This matches the SentencePiece library default, where the very first word also gets the space-marker prefix.

```csharp
string normalized = text.Replace(' ', SpaceMarker);
if (_addBosSpace && (normalized.Length == 0 || normalized[0] != SpaceMarker))
    normalized = SpaceMarker + normalized;
```

The purpose: after decoding, stripping the single leading space recovers the original text exactly. Encoding `"hello"` and `" hello"` must produce the same tokens — the normalization makes them identical.

### Step 2 — BuildInitialSymbols

The normalized string is segmented into an initial sequence of **symbols**, one per Unicode code point. A `Symbol` is a node in a mutable doubly-linked list (stored as a flat array for cache locality):

```csharp
internal struct Symbol
{
    public int Prev;     // index of previous live symbol; -1 = head
    public int Next;     // index of next live symbol; -1 = tail
    public int TokenId;
    public bool Deleted;
}
```

For each code point the method tries a **trie lookup** first. If the character (or ▁, or any multi-char sequence) is in the vocabulary, it gets its token ID directly. If not, the code falls through to **byte fallback**: the code point is encoded to its UTF-8 byte sequence and each byte is looked up in `_byteToTokenId[256]` (a pre-built table mapping byte value → `<0xNN>` token ID). If even a byte has no `<0xNN>` entry, `_unkId` (the `<unk>` token) is emitted rather than silently dropping the byte.

```csharp
// Direct vocab hit:
if (_vocabTrie.TryMatchLongest(cpSpan, ...) && ml == charLen)
    symbols[count++] = new Symbol { TokenId = tokenId, ... };
else
{
    // Byte fallback:
    int byteLen = Encoding.UTF8.GetBytes(cpSpan, utf8);
    for (int b = 0; b < byteLen; b++)
    {
        int byteId = _byteToTokenId[utf8[b]];
        int effectiveId = byteId >= 0 ? byteId : _unkId;
        symbols[count++] = new Symbol { TokenId = effectiveId, ... };
    }
}
```

The `Prev`/`Next` fields are initialized as a simple 0…N-1 chain; the last live symbol has `Next = -1`.

The symbol array is **rented from `ArrayPool<Symbol>.Shared`** — no heap allocation for typical inputs.

### Step 3 — BPE Merge Loop

This is the core of the SentencePiece algorithm. Starting from the initial symbol sequence, we repeatedly find the highest-priority adjacent pair (bigram) and merge it.

**Priority queue**: `PriorityQueue<BgramEntry, (float negScore, int leftIdx)>`. The .NET `PriorityQueue<T, P>` is a min-heap, so we negate the score to turn it into an effective max-heap. On ties (equal score), `leftIdx` is the tiebreaker — lower position is preferred, matching llama.cpp behaviour.

**TryEnqueueBigram** checks whether concatenating the text of two adjacent symbols produces a token that exists in the vocabulary. The concatenated string is built on the **stack** (`stackalloc char[256]` for lengths ≤ 256 chars) to avoid any heap allocation:

```csharp
Span<char> buf = totalLen <= 256 ? stackalloc char[256] : new char[totalLen];
// copy left and right into buf, then:
_vocabTrie.TryMatchLongest(concat, out int mergedId, out float score, out int ml)
```

The trie returns both the merged token ID and its score. If a full-length match is found, a `BgramEntry` is enqueued with:
- `Left` / `Right`: current indices of the two symbols
- `MergedId`: token ID of the merged result
- `ExpectedLeft` / `ExpectedRight`: the token IDs the symbols hold **right now**

**RunMergeLoop** dequeues entries and applies them:

```
while queue not empty:
    entry = dequeue()
    if left.Deleted or right.Deleted: skip (stale)
    if left.Next != entry.Right:      skip (no longer adjacent)
    if left.TokenId != entry.ExpectedLeft
    or right.TokenId != entry.ExpectedRight: skip (symbol was re-merged)

    left.TokenId = entry.MergedId
    right.Deleted = true
    relink: left.Next = right.Next; right.Next.Prev = left

    enqueue (left.Prev, left) and (left, left.Next) as new candidates
```

The stale-entry check is critical. Because the priority queue is not updated when symbols change, old entries accumulate. The `ExpectedLeft`/`ExpectedRight` fields detect the case where a symbol was the left side of a different merge (changing its `TokenId`) without being marked `Deleted`.

### Step 4 — CollectTokenIds

A single pass over the symbol array skips `Deleted` entries and collects the surviving `TokenId` values into a new `int[]`.

---

## Decode Pipeline

`Decode(ReadOnlySpan<int> tokenIds)` reconstructs text from a sequence of token IDs.

SentencePiece token strings fall into two categories:

1. **Normal tokens** — strings like `"▁hello"`, `"world"`, `"ing"`. The `▁` marker is replaced by a space.
2. **Byte tokens** — strings like `"<0xC3>"`, `"<0xA9>"`. These represent raw UTF-8 bytes for characters not in the vocabulary (e.g., emoji, rare Unicode).

The decoder handles them differently: byte tokens are accumulated in a pooled `byte[]` buffer. When a non-byte token is encountered (or at the end of the sequence), the buffer is flushed by decoding it as UTF-8. This guarantees that multi-byte UTF-8 sequences (which may span several consecutive byte tokens) are decoded atomically.

```
foreach token:
    if IsByteToken(token):
        byteBuffer[byteCount++] = parsed_byte
    else:
        FlushByteBuffer(sb)      // decode accumulated bytes as UTF-8
        sb.Append(token with ▁→' ')

FlushByteBuffer(sb)              // flush any trailing bytes
```

**Leading space strip**: if `_addBosSpace` is true, the encoder prepended `▁` to the input, which decodes to a leading space. This space is stripped from the result, so `Decode(Encode("hello")) == "hello"`.

`DecodeToken(int tokenId)` handles the single-token case: byte tokens are returned as their Latin-1 single-byte string (for streaming display); normal tokens have `▁` replaced by space.

---

## Data Structures

| Field | Type | Purpose |
|-------|------|---------|
| `_idToToken` | `string[]` | Vocabulary: token ID → string representation |
| `_byteToTokenId` | `int[256]` | Byte value → token ID for `<0xNN>` tokens; -1 = no token |
| `_vocabTrie` | `Trie` | Prefix-match trie for O(L) vocab lookup; stores token ID and score per leaf |
| `_addBosSpace` | `bool` | Whether to prepend `▁` (true for all standard SentencePiece models) |
| `_unkId` | `int` | Index of `<unk>` token; emitted when a byte has no `<0xNN>` fallback |

The `Trie` stores per-node: `Dictionary<char, TrieNode> Children`, `int TokenId`, `float Score`. `TryMatchLongest` does an O(L) scan returning the longest match found.

---

## Memory Allocation

| Site | Allocation | Notes |
|------|-----------|-------|
| Symbol array | `ArrayPool<Symbol>.Shared.Rent(normalized.Length)` | Returned in `finally` |
| Bigram concat buffer | `stackalloc char[256]` (≤ 256 chars) or `new char[N]` | Stack for almost all tokens |
| Byte decode buffer | `ArrayPool<byte>.Shared.Rent(16)` | Returned after decode |
| Result `int[]` | `new int[survivingCount]` | Unavoidable output allocation |
| Normalized string | `string.Replace(...)` + optional concat | One-time per encode call |

The priority queue itself allocates internally (it's a managed heap structure), but this is bounded by the number of possible bigrams (O(N) where N is the symbol count).

---

## Known Limitations

- No support for SentencePiece unigram models (only BPE is implemented).
- `tokenTypes` parameter is accepted but not used — type flags (normal/unknown/control/byte/user-defined) are not yet applied to modify encoding behaviour.
