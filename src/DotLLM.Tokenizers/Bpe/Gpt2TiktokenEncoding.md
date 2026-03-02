# Gpt2TiktokenEncoding — Implementation Notes

> Implementation file: `Gpt2TiktokenEncoding.cs`
> Used by: Llama 3, GPT-4 (cl100k_base), and any model with `tokenizer.ggml.model = "gpt2"` or `"llama3"` in GGUF metadata.

---

## What is tiktoken-style BPE?

tiktoken (OpenAI's tokenizer library) uses a **byte-level BPE** strategy. Rather than operating on Unicode code points, every input byte is first mapped to a designated Unicode character. Merges are then applied to these byte-representative characters using an explicit **rank table** (merge list): the merge with the lowest rank number is applied first.

The two key differences from SentencePiece BPE:

| Property | SentencePiece | tiktoken / GPT-2 |
|----------|--------------|-----------------|
| Input unit | Unicode code points | UTF-8 bytes (via byte→Unicode encoding) |
| Merge priority | Float score (higher = first) | Integer rank (lower = first) |
| Word boundary | `▁` prepended to words | Regex pre-tokenization (splits into segments) |
| Scores in vocab | Yes (log-prob) | No (all zeros) |

---

## The GPT-2 Byte-to-Unicode Encoding

### Why it exists

UTF-8 bytes include many control characters (0x00–0x1F, 0x7F–0x9F) that are not printable and cannot safely appear in JSON or text files. GPT-2 solves this by mapping every possible byte value (0–255) to a printable Unicode character before building the BPE vocabulary. This lets every token string in the vocabulary consist entirely of printable characters.

### The mapping

The mapping (`byte_encoder` in the original GPT-2 Python code) is defined as follows:

1. **Bytes 33–126** (printable ASCII, `!` through `~`) → same code point (identity mapping).
2. **Bytes 161–172 and 174–255** (Latin-1 supplement, excluding 173 = soft hyphen) → same code point.
3. **All remaining bytes** (0–32, 127–160, 173) — the 66 bytes that include control characters, space, DEL, and soft hyphen — are mapped to the range **U+0100 through U+0141** in order.

So:
- Byte `0x00` → U+0100 (`Ā`)
- Byte `0x20` (space) → U+0120 (`Ġ`)
- Byte `0x7F` (DEL) → U+0140 (`ŀ`)

This means the character `Ġ` in a tiktoken token string represents a space byte, `ĉ` represents a tab, etc.

### Implementation: two static tables

`BuildGpt2ByteToUnicode()` produces `char[256]`: index = byte value, value = Unicode char.

`BuildGpt2UnicodeToByteTable()` inverts the forward table. It finds the maximum code point produced (U+0144 = 324), allocates `short[325]`, and fills it as: index = Unicode char ordinal, value = byte (0–255) or -1 if not a GPT-2-encoded byte. Using `short[]` instead of `int[]` halves the table size (650 bytes vs 1300 bytes) and keeps it in L1 cache.

Both tables are `static readonly` — computed once per process, shared across all tokenizer instances.

---

## Construction

The constructor receives:
- `string[] tokens` — vocabulary; index = token ID, value = token string in GPT-2 encoding (e.g., `"Ġhello"` for `" hello"`).
- `string[] merges` — merge table; each entry is `"A B"` where `A` and `B` are token strings. Entry at index 0 has rank 0 (applied first).
- `int[]? tokenTypes` — accepted but not currently used.

**Building the trie**: all tokens are inserted into `_vocabTrie` with score 0 (scores are irrelevant for rank-based merging).

**Building `_mergeRanks`**: this is the key performance-critical data structure. The naïve approach would look up `"A" + " " + "B"` in a `Dictionary<string, int>` per bigram check — allocating a string on every call in the hot encode path. Instead:

1. A temporary `Dictionary<string, int> tokenToId` reverse-maps token strings to IDs (one pass at init).
2. Each merge string `"A B"` is split at the space separator; `A` and `B` are looked up in `tokenToId` to get `idA` and `idB`.
3. The entry `(idA, idB) → rank` is stored in `Dictionary<(int, int), int> _mergeRanks`.

At encode time, the bigram check becomes a value-type tuple lookup — zero allocation:

```csharp
_mergeRanks.TryGetValue((symbols[leftIdx].TokenId, symbols[rightIdx].TokenId), out int rank)
```

---

## Encode Pipeline

`Encode(string text)` runs three logical steps:

### Step 1 — Convert input to GPT-2 byte-level representation

```csharp
byte[] utf8Bytes = Encoding.UTF8.GetBytes(text);
// rent a char[] from ArrayPool to avoid allocation:
for (int i = 0; i < utf8Bytes.Length; i++)
    gpt2Chars[i] = Gpt2ByteToUnicode[utf8Bytes[i]];
string gpt2Text = new string(gpt2Chars, 0, utf8Bytes.Length);
```

Every UTF-8 byte of the input is mapped through `Gpt2ByteToUnicode`. The resulting `gpt2Text` string is the input as the BPE algorithm sees it — a sequence of GPT-2-encoded "byte characters". A space in the original text becomes `Ġ` (U+0120); a multi-byte character like `é` (UTF-8: `0xC3 0xA9`) becomes two GPT-2 chars `Ã©`.

> **Missing: regex pre-tokenization.** Real tiktoken models (Llama 3, GPT-4) split `gpt2Text` into segments using a model-specific regex before applying BPE independently to each segment. This ensures BPE cannot merge across word boundaries — e.g., `"dog."` and `"dog "` produce different tokens. The current implementation skips this step and treats the entire `gpt2Text` as one segment. For short inputs and SentencePiece models this is harmless, but for true tiktoken models it produces incorrect tokenization. Tracked as a TODO; will be implemented using `tokenizer.ggml.pre`.

### Step 2 — BuildInitialSymbols

Identical in structure to the SentencePiece version. The `gpt2Text` is segmented code-point by code-point. For each code point:

- If it exists as a direct token in `_vocabTrie`, emit that token ID.
- Otherwise, fall back to UTF-8 bytes of the code point → look up in `_byteToTokenId` → emit `<0xNN>` token ID, or `_unkId` if even the byte token is absent.

Because `gpt2Text` already represents every original byte as a single GPT-2 Unicode character, each character should be a direct vocab hit (the vocabulary contains one token per byte). The byte-fallback path only activates for surrogate pairs or code points not covered by the GPT-2 alphabet.

### Step 3 — BPE Merge Loop (rank-based)

The merge loop is identical in structure to `SentencePieceEncoding.RunMergeLoop`, with one key difference: the priority key is `(int rank, int leftIdx)` instead of `(float negScore, int leftIdx)`. Lower rank = higher priority (applied first).

**TryEnqueueBigram**:

```csharp
// 1. Check merge rank — zero allocation
if (!_mergeRanks.TryGetValue((leftTokenId, rightTokenId), out int rank)) return;

// 2. Resolve merged token ID via trie (stack-allocated concat)
Span<char> buf = totalLen <= 256 ? stackalloc char[256] : new char[totalLen];
// ... copy left+right into buf ...
_vocabTrie.TryMatchLongest(concat, out int mergedId, ...)
```

Two-phase check: first the rank dictionary tells us if this bigram is a valid merge at all (O(1), zero alloc). Only if it is, we do the trie lookup to resolve the merged token ID (needed because the trie stores the canonical merged token, while the rank dict only stores the priority).

**Stale entry detection** is identical: `ExpectedLeft`/`ExpectedRight` fields in `BgramEntry` guard against applying an outdated merge after one of the symbols has already been consumed by a higher-priority merge.

---

## Decode Pipeline

`Decode(ReadOnlySpan<int> tokenIds)` is the inverse of the byte-encoding step.

Each token string in the vocabulary is a sequence of GPT-2-encoded characters — each character represents exactly one byte. To recover the original text:

1. For each token ID, look up its string in `_idToToken`.
2. For each character in the token string, look up its byte value in `Gpt2UnicodeToByteTable`.
3. Collect all bytes into a pooled `byte[]` buffer.
4. At the end, decode the buffer as UTF-8.

```csharp
foreach (int id in tokenIds)
{
    string token = _idToToken[id];
    foreach (char c in token)
    {
        short b = Gpt2UnicodeToByteTable[(int)c];
        if (b >= 0) buf[count++] = (byte)b;
    }
}
return Encoding.UTF8.GetString(buf, 0, count);
```

The buffer is initially sized at `tokenIds.Length * 8` (generous upper bound: 8 bytes per token) and doubles if needed. It is rented from `ArrayPool<byte>.Shared` and returned after the `GetString` call.

`DecodeToken(int tokenId)` handles the single-token case with an inline byte array (no `ArrayPool` — token strings are short).

---

## Data Structures

| Field | Type | Purpose |
|-------|------|---------|
| `_idToToken` | `string[]` | Vocabulary: token ID → GPT-2-encoded string |
| `_byteToTokenId` | `int[256]` | Byte value → `<0xNN>` token ID; -1 = absent |
| `_vocabTrie` | `Trie` | Prefix-match trie; all scores are 0 (rank-based, not score-based) |
| `_mergeRanks` | `Dictionary<(int,int),int>` | (leftId, rightId) → merge rank; value-type key, zero alloc per lookup |
| `_unkId` | `int` | Index of `<unk>` token; emitted for unmapped bytes |
| `Gpt2ByteToUnicode` | `static char[256]` | Byte → GPT-2 Unicode char; shared, computed once |
| `Gpt2UnicodeToByteTable` | `static short[325]` | GPT-2 Unicode char → byte; -1 = not a byte char |

---

## Memory Allocation

| Site | Allocation | Notes |
|------|-----------|-------|
| UTF-8 byte array | `Encoding.UTF8.GetBytes(text)` | Managed; proportional to input length |
| GPT-2 char array | `ArrayPool<char>.Shared.Rent(utf8Bytes.Length)` | Returned immediately after `new string(...)` |
| Symbol array | `ArrayPool<Symbol>.Shared.Rent(gpt2Text.Length * 2)` | Returned in `finally` |
| Bigram concat buffer | `stackalloc char[256]` (≤ 256 chars) | Stack; avoids heap alloc for virtually all tokens |
| Decode byte buffer | `ArrayPool<byte>.Shared.Rent(tokenIds.Length * 8)` | Returned after `GetString` |
| Result `int[]` | `new int[survivingCount]` | Unavoidable output allocation |

---

## Known Limitations

- **Regex pre-tokenization is not implemented.** This is the most significant functional gap vs. the reference tiktoken library. Without it, BPE merges can cross word/punctuation boundaries, producing token sequences that differ from the reference for any input longer than a single word. Tracked as a TODO in `Encode`.
- `tokenTypes` parameter is accepted but not used.
