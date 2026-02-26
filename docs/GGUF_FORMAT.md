# GGUF Format Reference — dotLLM

## Binary Layout

```
┌─────────────────────────────────────────────┐
│ HEADER                                       │
│  Magic: 0x46554747 ("GGUF" little-endian)    │
│  Version: uint32 (currently 3)               │
│  Tensor count: uint64                        │
│  Metadata KV count: uint64                   │
├─────────────────────────────────────────────┤
│ METADATA KEY-VALUE PAIRS                     │
│  Repeated metadata_kv_count times:           │
│    Key: gguf_string (uint64 len + bytes)     │
│    Value type: uint32 (enum)                 │
│    Value: type-dependent                     │
├─────────────────────────────────────────────┤
│ TENSOR INFO ENTRIES                          │
│  Repeated tensor_count times:                │
│    Name: gguf_string                         │
│    N dimensions: uint32                      │
│    Dimensions: uint64[n_dims]                │
│    Type: uint32 (quantization type enum)     │
│    Offset: uint64 (from start of data)       │
├─────────────────────────────────────────────┤
│ PADDING (to alignment boundary)              │
├─────────────────────────────────────────────┤
│ TENSOR DATA                                  │
│  Contiguous blob, alignment-padded per tensor│
└─────────────────────────────────────────────┘
```

## Value Types

| Enum | Type | Size |
|------|------|------|
| 0 | UINT8 | 1 |
| 1 | INT8 | 1 |
| 2 | UINT16 | 2 |
| 3 | INT16 | 2 |
| 4 | UINT32 | 4 |
| 5 | INT32 | 4 |
| 6 | FLOAT32 | 4 |
| 7 | BOOL | 1 |
| 8 | STRING | uint64 len + bytes |
| 9 | ARRAY | uint32 type + uint64 len + elements |
| 10 | UINT64 | 8 |
| 11 | INT64 | 8 |
| 12 | FLOAT64 | 8 |

## Key Metadata Fields

### Architecture Config

| Key | Type | Example |
|-----|------|---------|
| `general.architecture` | string | `"llama"` |
| `general.name` | string | `"Llama 3 8B"` |
| `general.file_type` | uint32 | `15` (Q4_K_M) |
| `{arch}.block_count` | uint32 | `32` |
| `{arch}.context_length` | uint32 | `8192` |
| `{arch}.embedding_length` | uint32 | `4096` |
| `{arch}.feed_forward_length` | uint32 | `14336` |
| `{arch}.attention.head_count` | uint32 | `32` |
| `{arch}.attention.head_count_kv` | uint32 | `8` |
| `{arch}.attention.layer_norm_rms_epsilon` | float32 | `1e-5` |
| `{arch}.rope.freq_base` | float32 | `500000.0` |
| `{arch}.rope.dimension_count` | uint32 | `128` |
| `{arch}.rope.scaling.type` | string | `"yarn"` |
| `{arch}.rope.scaling.factor` | float32 | `4.0` |

### Tokenizer Metadata

| Key | Type | Description |
|-----|------|-------------|
| `tokenizer.ggml.model` | string | `"llama"` (SentencePiece) or `"gpt2"` (BPE) |
| `tokenizer.ggml.tokens` | string[] | Vocabulary |
| `tokenizer.ggml.scores` | float32[] | Token scores (SentencePiece) |
| `tokenizer.ggml.token_type` | int32[] | Token types |
| `tokenizer.ggml.merges` | string[] | BPE merge rules |
| `tokenizer.ggml.bos_token_id` | uint32 | BOS token |
| `tokenizer.ggml.eos_token_id` | uint32 | EOS token |
| `tokenizer.chat_template` | string | Jinja2 template |

## Tensor Naming Convention

```
token_embd.weight                    — Token embedding
blk.{i}.attn_norm.weight             — Pre-attention RMSNorm
blk.{i}.attn_q.weight                — Q projection
blk.{i}.attn_k.weight                — K projection
blk.{i}.attn_v.weight                — V projection
blk.{i}.attn_output.weight           — Attention output
blk.{i}.ffn_norm.weight              — Pre-FFN RMSNorm
blk.{i}.ffn_gate.weight              — FFN gate (SwiGLU)
blk.{i}.ffn_up.weight                — FFN up
blk.{i}.ffn_down.weight              — FFN down
output_norm.weight                   — Final RMSNorm
output.weight                        — LM head
```

## Quantization Type Enum

| Enum | Name | Bits/Wt | Block Size |
|------|------|---------|------------|
| 0 | F32 | 32 | 1 |
| 1 | F16 | 16 | 1 |
| 2 | Q4_0 | 4.5 | 32 |
| 3 | Q4_1 | 5.0 | 32 |
| 6 | Q5_0 | 5.5 | 32 |
| 7 | Q5_1 | 6.0 | 32 |
| 8 | Q8_0 | 8.5 | 32 |
| 12 | Q4_K | 4.5 | 256 |
| 13 | Q5_K | 5.5 | 256 |
| 14 | Q6_K | 6.6 | 256 |

## Parsing Implementation

1. Read header: validate magic `0x46554747`, extract version/counts.
2. Parse metadata into `Dictionary<string, GgufMetadataValue>`.
3. Parse tensor info into `List<GgufTensorInfo>` (name, shape, type, offset).
4. Calculate `data_section_start` = current position rounded up to alignment (default 32, overridable via `general.alignment`).
5. Memory-map from `data_section_start` to EOF: `MemoryMappedFile.CreateFromFile`.
6. Tensor data pointer = `mmap_base + tensor.offset`. No copying.

### Tensors >2GB

`MemoryMappedViewAccessor` has 2GB limit. Use:
```csharp
byte* ptr;
viewHandle.AcquirePointer(ref ptr);
// ptr + tensor.offset gives direct access
```

### ModelConfig Extraction

```
arch = metadata["general.architecture"]
ModelConfig.HiddenSize = metadata[$"{arch}.embedding_length"]
ModelConfig.NumLayers = metadata[$"{arch}.block_count"]
ModelConfig.NumAttentionHeads = metadata[$"{arch}.attention.head_count"]
ModelConfig.NumKvHeads = metadata[$"{arch}.attention.head_count_kv"] ?? NumAttentionHeads
ModelConfig.RoPETheta = metadata[$"{arch}.rope.freq_base"] ?? 10000.0
```

## Reference

- Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Parser: llama.cpp `ggml/src/gguf.cpp` → `gguf_init_from_file`