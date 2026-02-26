# Quantization — dotLLM

## Implementation Priority

1. **FP16** — Baseline
2. **Q8_0** — Simplest quantized, minimal quality loss
3. **Q4_0** — Basic 4-bit
4. **Q4_K_M** — Most popular, best quality/size tradeoff
5. **Q5_K_M, Q6_K** — Higher quality K-quants
6. **GPTQ/AWQ** — GPU-native (future)

## Block Layouts

### Q8_0 (8.5 bits/weight)

```
struct block_q8_0 {          // 34 bytes, 32 values
    half d;                  // scale factor
    int8_t qs[32];           // quantized values (-127..127)
};
Dequantize: val[i] = d * qs[i]
```

### Q4_0 (4.5 bits/weight)

```
struct block_q4_0 {          // 18 bytes, 32 values
    half d;                  // scale factor
    uint8_t qs[16];          // 32 × 4-bit packed into 16 bytes
};
Unpack: lo = qs[j] & 0x0F, hi = qs[j] >> 4
Dequantize: val[i] = d * (nibble - 8)
```

### Q4_K (4.5 bits/weight, super-block)

```
struct block_q4_K {          // 144 bytes, 256 values (8 sub-blocks of 32)
    half d;                  // super-block scale
    half dmin;               // super-block minimum
    uint8_t scales[12];      // 8 × (6-bit scale + 6-bit min)
    uint8_t qs[128];         // 256 × 4-bit values
};
For sub-block j: val = d * scale_j * nibble - dmin * min_j
```

Q4_K_M = mixed file: attention layers Q6_K, FFN layers Q4_K.

### Q6_K (6.6 bits/weight)

```
struct block_q6_K {          // 210 bytes, 256 values
    uint8_t ql[128];         // low 4 bits
    uint8_t qh[64];          // high 2 bits
    int8_t scales[16];       // INT8 sub-block scales
    half d;                  // super-block scale
};
```

### Q5_K (5.5 bits/weight)

```
struct block_q5_K {          // 176 bytes, 256 values
    half d, dmin;
    uint8_t scales[12];
    uint8_t qh[32];          // 5th bit
    uint8_t qs[128];         // low 4 bits
};
```

## Kernel Types

Each quantization format needs two kernels:

### Dequantize Kernel
Converts block → FP32. For layers needing full precision activations.

### Vec_Dot Kernel (Fused)
Dot product directly on quantized data. Faster than dequant+dot because avoids FP32 expansion.

**CPU SIMD (Q8_0 × Q8_0)**:
```
prod = Avx2.MultiplyAddAdjacent(qs_a, qs_b)  // INT8→INT16
acc = Avx2.MultiplyAddAdjacent(prod, ones)     // INT16→INT32
sum += ConvertToFloat(acc) * (da * db)
```

**CPU SIMD (Q4_0 × Q8_0)**:
Unpack nibbles to INT8, subtract offset, then integer multiply-accumulate.

## Mixed Quantization

GGUF files can have different types per tensor. Dispatch to correct kernel based on tensor metadata. Never assume uniform quantization.

## Performance Notes

- Vec_dot dominant for decode (GEMV). Dequant+BLAS may win for prefill (GEMM).
- GPU: custom CUDA kernels dequantize in shared memory, use tensor cores. Ref: llama.cpp `ggml-cuda/mmq.cu`.
- Block alignment awkward for SIMD — handle tail elements carefully.
