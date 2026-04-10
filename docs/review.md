  Revised CUDA Kernel Review — mapped to your learning

  What's Aligned with What You Learned

  PTX architecture — nailed it

  Your learning: PTX is a portable IR, analogous to LLVM IR. dotLLM compiles to PTX with compute_61 and loads via cuModuleLoadData — the driver JIT-compiles to SASS at runtime. This is exactly right.
  Forward-compatible, no native shared lib needed.

  Thread hierarchy & indexing — correct everywhere

  Your learning: i = blockIdx.x * blockDim.x + threadIdx.x. Every element-wise kernel (add, SwiGLU, convert, bias_add) uses exactly this pattern. Row-per-block kernels (RMSNorm, softmax) use blockIdx.x as
  the row index with grid-stride loops inside. Block sizes are 256 everywhere — a good multiple of 32.

  Block/grid sizing — good

  Your learning: always multiples of 32, 128/256/512 typical. All kernels use BlockSize = 256. Grid sizes are computed correctly: ceil(n / 256) for element-wise, rows for per-row, seqQ * numHeads for
  attention.

  Warps & warp reductions — textbook

  Your learning: warp = 32 threads in lockstep, the true hardware execution unit. The two-level reduction pattern in RMSNorm/softmax/GEMV is exactly the canonical approach: warp shuffle → warp_sums[32]
  shared mem → first warp reduces → broadcast. 0xFFFFFFFF mask for full warp participation is correct.

  Shared memory usage — correct pattern

  Your learning: fast on-SM scratchpad, __shared__, same SRAM as L1. Used appropriately in reductions (warp_sums), attention (scores + output accumulator), and broadcasting scalar results (rms_inv,
  shared_max).

  __syncthreads() — correctly placed

  Your learning: block barrier, required when threads share data via shared memory. Every kernel that writes to shared memory has a __syncthreads() before reads. The fused_add_rmsnorm has the pattern right:
   write warp_sums → sync → read in first warp → sync → broadcast → sync → use.

  Host↔device separation — respected

  Your learning: CPU and GPU memory are separate, minimize transfers. The C# side uses nint handles for GPU memory, never copies tensor data across the P/Invoke boundary. cuLaunchKernel is asynchronous as
  expected.

  Library-first for matmul — yes

  Your learning: don't hand-write matmul, use cuBLAS. CudaGemm.cs delegates to cublasHgemm / cublasGemmEx for dense GEMM. Custom kernels are only for the novel parts (quantized GEMV, RMSNorm, RoPE,
  attention) — exactly the right split.

  Tensor Cores — enabled

  Your learning: specialized hardware for matrix multiply, accessible via cuBLAS. CublasApi.cs sets CUBLAS_TENSOR_OP_MATH and uses cublasGemmEx with FP16→FP32 accumulation. Tensor Cores are being used for
  the dense linear layers.

  ---
  Where the Kernels Violate What You Learned

  1. Attention kernel ignores memory hierarchy rules

  Your learning: memory hierarchy: registers (~1 cycle) → shared (~5 cycles) → global (~400-800 cycles)

  attention.cu:60-83 — The Q·K dot product reads q_vec[d] and k_vec[d] from global memory inside a tight inner loop, for every KV position:

  for (int tkv = threadIdx.x; tkv < seq_kv; tkv += blockDim.x)
  {
      for (int d = 0; d < head_dim; d++)
          score += __half2float(q_vec[d]) * __half2float(k_vec[d]);

  q_vec is the same for all tkv iterations — it should be loaded into registers or shared memory once. Instead it's re-read seq_kv times from global memory. That's seq_kv × head_dim × 2 bytes of redundant
  global reads per thread.

  Your learning: stage reused data through shared memory (the tiling pattern). The Q vector is a perfect candidate — it's small (head_dim = 128 halfs = 256 bytes) and reused across every KV position.

  2. Attention: serial reduction = warp divergence + wasted parallelism

  Your learning: GPUs hide memory stalls by having many warps ready to run

  attention.cu:88-98 — Thread 0 serially scans all seq_kv scores while 255 threads sit idle:

  if (threadIdx.x == 0)
  {
      for (int tkv = 0; tkv < seq_kv; tkv++)
          if (smem[tkv] > max_score) max_score = smem[tkv];
  }

  You already know the parallel reduction pattern — you use it correctly in softmax.cu. The attention kernel should use the same warp-shuffle + cross-warp reduction instead of this serial scan.

  3. Attention shared memory scales with sequence length

  Your learning: shared memory is per-block, limited (~48-228 KB)

  CudaKernels.cs:470:
  uint sharedBytes = (uint)((seqKv + headDim) * sizeof(float));

  At seq_kv=32K: (32768 + 128) × 4 = ~128 KB — exceeds the default 48 KB limit on most GPUs (up to 228 KB with cudaFuncSetAttribute on Ampere+, which isn't called). The kernel will fail to launch silently
  for long contexts.

  4. Dequant kernels — anti-pattern for coalescing

  Your learning: when threads in a warp access consecutive addresses, hardware merges into one transaction

  dequant.cu:18-28 — One thread per Q8_0 block. Thread 0 writes to dst[0..31], thread 1 writes to dst[32..63], etc. Within each thread's serial loop:

  for (int j = 0; j < Q8_0_BLOCK_SIZE; j++)
      out[j] = __float2half(d * (float)qs[j]);

  At any given instruction, thread 0 writes dst[0], thread 1 writes dst[32], thread 2 writes dst[64]... — stride-32 access pattern. This is the opposite of coalesced. Adjacent threads should write adjacent
  addresses.

  Fix: One warp per Q8_0 block — thread lane writes element lane. All 32 threads write consecutive addresses → one coalesced 64-byte transaction instead of 32 separate ones.

  5. Quantized GEMV — register pressure concerns

  Your learning: local variables live in registers; overflow spills to slow "local memory". High register use lowers occupancy

  quantized_gemv.cu (Q4_K, line 82-131): Each thread accumulates across entire superblocks with many live variables: d, dmin, sc0, m0, sc1, m1, scale0, min0, scale1, min1, x_even, x_odd, byte_val, acc, plus
   loop counters. Q6_K is even worse with q1-q4, s0-s3.

  Without __launch_bounds__, the compiler doesn't know the intended occupancy and may allocate too many registers, reducing active warps. Your learning: occupancy = active warps / max warps, governed by
  whichever resource runs out first.

  6. --use_fast_math contradicts precision-first philosophy

  Your learning: this wasn't explicitly in your notes, but it connects to numerical correctness.

  --use_fast_math replaces expf → __expf (approximate), rsqrtf → approximate, sinf/cosf → approximate. These are used in:
  - Softmax (expf) — affects probability distribution
  - RMSNorm (rsqrtf) — affects normalization
  - RoPE (sinf, cosf, powf) — affects position encoding

  For LLM inference where you're trying to match HuggingFace reference outputs, this introduces non-trivial numerical drift, especially accumulated over 32+ layers.

  7. No vectorized memory access anywhere

  Your learning: coalesce global memory accesses + general optimization

  Even when access is coalesced (element-wise kernels), every load/store is a single half (2 bytes). GPUs issue 32-byte, 64-byte, or 128-byte transactions. Loading one half per thread in a warp = 64 bytes =
   one transaction, which is fine for coalescing — but you could process 2× the data by using half2 packed operations, or 8× with float4 loads. This matters most in bandwidth-bound kernels like add,
  convert, SwiGLU, and embedding lookup.

  8. RoPE — redundant computation between Q and K

  Your learning: minimize redundant work

  rope.cu:26-93 — When idx < total_k_pairs (which is common since num_kv_heads ≤ num_heads), the same thread computes powf, cosf, sinf twice for the same (position, pair) — once for Q and once for K. These
  transcendentals are expensive (~20 cycles each). The freq/cos/sin could be computed once and reused.

  ---
  Priority Summary

  ┌─────┬──────────────────────────────────────┬───────────────────────────────────┬──────────────────────────────────────────┐
  │  #  │                Issue                 │       Your learning concept       │                  Impact                  │
  ├─────┼──────────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────┤
  │ 1   │ Attention re-reads Q from global mem │ Tiling / shared memory staging    │ High — seq_kv × head_dim redundant reads │
  ├─────┼──────────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────┤
  │ 2   │ Attention serial max/sum             │ Parallel reduction                │ High — 255/256 threads idle              │
  ├─────┼──────────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────┤
  │ 3   │ Attention shared mem limit           │ Shared memory is finite per block │ High — breaks at long contexts           │
  ├─────┼──────────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────┤
  │ 4   │ Dequant stride-32 writes             │ Memory coalescing                 │ High — 32× more transactions than needed │
  ├─────┼──────────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────┤
  │ 5   │ No __launch_bounds__                 │ Occupancy / register pressure     │ Medium — compiler can't optimize         │
  ├─────┼──────────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────┤
  │ 6   │ --use_fast_math global               │ Numerical precision               │ Medium — accumulated drift over layers   │
  ├─────┼──────────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────┤
  │ 7   │ No half2 vectorization               │ Coalescing / throughput           │ Medium — 2× bandwidth opportunity        │
  ├─────┼──────────────────────────────────────┼───────────────────────────────────┼──────────────────────────────────────────┤
  │ 8   │ RoPE redundant transcendentals       │ Minimize redundant work           │ Low-medium — powf+sinf+cosf × 2          │
  └─────┴──────────────────────────────────────┴───────────────────────────────────┴──────────────────────────────────────────┘

  The foundation is solid — correct thread hierarchy, proper reductions where used, cuBLAS for dense ops, proper host/device separation. The main gaps are applying the tiling/staging pattern to attention,
  fixing coalescing in dequant, and the --use_fast_math precision tradeoff.

  Pre-Open-Source Code Review — dotLLM

  1. CRITICAL BUGS (fix before going public)

  1a. Missing regex pre-tokenization for tiktoken/Llama3

  src/DotLLM.Tokenizers/Bpe/Gpt2TiktokenEncoding.cs:132
  // TODO: implement regex pre-tokenization using tokenizer.ggml.pre pattern.
  This is a correctness-breaking bug. Tiktoken-based models (Llama 3, GPT-2, etc.) require regex-based word splitting before BPE merges. Without it, the tokenizer produces wrong token sequences, which means
   wrong model output. Anyone loading a Llama 3 GGUF will get garbage. This is the #1 thing people will try.

  1b. TopK sampler keeps more than K tokens on ties

  src/DotLLM.Engine/Samplers/TopKSampler.cs:35-41

  Sorts the entire array (wasteful), then uses strict < comparison against the K-th largest value. If multiple tokens share that value, more than K survive. Example: logits [1, 2, 2, 2, 3] with K=2 →
  threshold=2 → keeps 4 tokens. Should use a selection algorithm and keep exactly K, or document the behavior explicitly.

  1c. CategoricalSampler falls back to vocabSize - 1

  src/DotLLM.Engine/Samplers/CategoricalSampler.cs:40-41
  // TODO: Return last non-masked token instead of vocab end
  return vocabSize - 1;
  When cumulative probability doesn't reach the random threshold (floating-point edge case), it returns the last token in the vocabulary. This could be a padding token, producing nonsensical output. Should
  return the highest-probability unmasked token.

  1d. Stop string removes entire token instead of suffix

  src/DotLLM.Engine/TextGenerator.cs:112
  // TODO: Trim matched suffix only, not entire token (see PR #24 review)
  If generated text is "Hello, world<|im_end|>" and the stop sequence is "<|im_end|>", but the last token decoded was "ld<|im_end|>", the current code may either include the stop sequence or trim too
  aggressively. The comment acknowledges this is known.

  ---
  2. EMBARRASSING "NAIVE" IMPLEMENTATIONS (will be the first thing people scrutinize)

  2a. CUDA attention kernel — serial reduction, unbounded shared memory

  native/kernels/attention.cu:88-98 — Thread 0 serially scans seq_kv scores while 255 threads are idle. You already have parallel reduction in softmax.cu — attention doesn't use it.

  CudaKernels.cs:470 — Shared memory = (seqKv + headDim) * 4 bytes. At 8K context: 32 KB (ok). At 32K: 128 KB (exceeds default limit). At 128K: 512 KB (impossible). No guard or fallback. The kernel will
  silently fail to launch.

  The kernel comment says "Naive" but it's the only CUDA attention implementation. People will benchmark this immediately.

  2b. SchemaTracker cloning — 160MB per cache miss   [FIXED — Wave 7 (#109)]

  src/DotLLM.Engine/Constraints/Schema/JsonSchemaConstraint.cs:131-133
  // TODO: Perf — SchemaTracker is ~1.3KB due to InlineArray stacks.
  // Copying it for each of the 128K vocab tokens (~160MB per cache miss)
  This TODO is honest, but it means structured JSON output with schema constraints will be noticeably slow. Constrained decoding is a headline feature — people will test it.
  Fix: first-char bucketing in BuildAndCacheMask skips entire buckets whose first character the tracker rejects, eliminating >95% of clones at typical schema states.

  2c. No scheduler implementation

  src/DotLLM.Engine/IScheduler.cs — Interface only, no concrete implementation. The server uses SemaphoreSlim(1, 1) to serialize all requests (ServerState.cs:19). This means the server can only handle one
  request at a time. No continuous batching, no preemption, no priority scheduling. The interface exists but nothing implements it.

  2d. No speculative decoding implementation

  src/DotLLM.Engine/ISpeculativeDecoder.cs — Interface only, 30 lines, no implementation.

  2e. No LoRA support

  Not mentioned in code, only in docs and CLAUDE.md.

  2f. No SafeTensors support

  Only GGUF is implemented.

  Recommendation for 2c-2f: Either remove the interfaces/docs or clearly label them as "planned" in a roadmap. Empty interfaces with no implementation look like vaporware.

  ---
  3. CORRECTNESS ISSUES

  3a. GGUF alignment not validated as power-of-2

  src/DotLLM.Models/Gguf/GgufFile.cs:90,158

  alignment = metadata.GetUInt32OrDefault("general.alignment", 32);
  // ...later:
  long mask = alignment - 1; // assumes power-of-2!
  If a corrupted/adversarial GGUF has alignment=3, the mask computation is wrong. Add BitOperations.IsPow2(alignment) check.

  3b. GGUF tensor offset not validated against file size

  src/DotLLM.Models/Gguf/GgufReader.cs:103 — Reads ulong offset for tensor data but never checks it's within the memory-mapped region. Corrupted file → out-of-bounds access → segfault.

  3c. SchemaTracker anyOf overapproximation

  src/DotLLM.Engine/Constraints/Schema/SchemaTracker.cs:249-256

  // TODO: This is an overapproximation — after the first character
  // disambiguates the branch, nested constraints are not enforced.
  For complex schemas with anyOf, the constraint allows tokens that should be rejected. This means "guaranteed valid JSON/schema" (a claimed feature) isn't actually guaranteed for schemas with anyOf.

  3d. DFA simulator doesn't handle supplementary Unicode

  src/DotLLM.Engine/Constraints/Regex/DfaSimulator.cs:53 — _dfa.CharToClass[c] indexes by char (16-bit). Supplementary plane characters (emoji, CJK Extension B) arrive as surrogate pairs and would index out
   of bounds or match wrong classes.

  3e. DeepSeek architecture listed but not implemented

  src/DotLLM.Models/Gguf/GgufModelConfigExtractor.cs:78 — The Architecture.DeepSeek case exists in config extraction, but the model forward pass doesn't support MLA. Will throw at runtime. Either remove or
  clearly mark as unsupported.

  3f. --use_fast_math in CUDA builds

  native/build.sh:27 — Approximate expf, rsqrtf, sinf/cosf across all kernels. Affects numerical accuracy of softmax, RMSNorm, RoPE. May produce visibly different output vs reference implementations.

  ---
  4. MEMORY SAFETY & RESOURCE LEAKS

  4a. GgufFile partial initialization leak

  src/DotLLM.Models/Gguf/GgufFile.cs:103-119 — If exception occurs after AcquirePointer() but before the constructor completes, the local basePointer reference can be lost. The catch block tries to clean up
   but the pattern is fragile.

  4b. TransformerWeights partial allocation leak

  src/DotLLM.Models/Architectures/TransformerWeights.cs:338-346 — If loading one weight fails midway through LoadLayer, previously allocated float arrays are abandoned. No try-finally wrapping the
  allocation sequence.

  4c. Fused QKV split — no dimension validation

  src/DotLLM.Models/Architectures/TransformerWeights.cs:295-307 — Splits a fused QKV tensor by computing pointer offsets, but never validates that the tensor's output dimension equals qDim + 2 * kvDim.
  Wrong GGUF metadata → pointer arithmetic past allocated memory.

  4d. Integer overflow in tensor size cast

  src/DotLLM.Models/Architectures/TransformerWeights.cs:449 — int size = (int)desc.Shape.ElementCount — unchecked cast from long. A tensor with >2B elements wraps to negative.

  ---
  5. PERFORMANCE ISSUES (will show in benchmarks)

  5a. CPU TopK sorts entire vocab   [FIXED — Wave 7 (#109)]

  src/DotLLM.Engine/Samplers/TopKSampler.cs:35 — Array.Sort on full vocab (128K+ floats) when only the top K (typically 40-100) are needed. Should use nth_element / partial sort — an order of magnitude
  faster.
  Fix: hand-rolled size-K binary min-heap replaces Array.Sort. O(N log K), stack-resident scratch via stackalloc for K ≤ 512, no ArrayPool rental on the common path.

  5b. CUDA dequant kernels — stride-32 writes (anti-coalesced)

  native/kernels/dequant.cu — One thread per quantization block. Adjacent threads write 32+ elements apart in memory. 32× more memory transactions than needed. Should use one warp per block for perfect
  coalescing.

  5c. No half2 vectorization in any CUDA kernel

  All FP16 element-wise kernels (add, SwiGLU, convert, bias_add) process one half per thread. Using half2 packed operations would double throughput.

  5d. CUDA RoPE: redundant powf + sinf + cosf per thread

  native/kernels/rope.cu:34,69 — Same (position, pair) computes expensive transcendentals twice (once for Q, once for K). ~60 extra cycles per thread.

  5e. CUDA softmax: 3 passes over global memory

  native/kernels/softmax.cu — Pass 1: find max, Pass 2: exp + sum, Pass 3: re-read input, re-compute exp, normalize. Pass 3 could use stored exp values from shared memory.

  5f. CPU Q5_0 dequant — scalar only, no AVX2   [FIXED — Wave 7 (#109)]

  src/DotLLM.Cpu/Kernels/Dequantize.cs:154-202 — Every other quant format has AVX2 paths. Q5_0 is scalar-only.
  Fix: DequantizeQ5_0Avx2 reuses MatMulQ5_0.ExtractQ5HighBits (vpshufb broadcast + bit masking) to unpack low/high nibbles plus the 5th bit in one SIMD pass, matching the Q8_0 AVX2 pipeline shape.

  5g. JsonSchemaConstraint cache eviction = full flush   [FIXED — Wave 7 (#109)]

  src/DotLLM.Engine/Constraints/Schema/JsonSchemaConstraint.cs:109-110
  if (_maskCache.Count >= _maxCacheEntries)
      _maskCache.Clear();
  On overflow, discards the entire cache instead of LRU. High miss rates for complex schemas.
  Fix: private LruCache<SchemaStateKey, TokenMask> (LinkedList + Dictionary) evicts the least-recently-used entry instead of the whole cache.

  ---
  6. SERVER & API ISSUES

  6a. Single-request concurrency

  src/DotLLM.Server/ServerState.cs:19 — SemaphoreSlim(1, 1) means the server is single-threaded for inference. This is architecturally correct given no scheduler exists, but should be documented
  prominently. People will expect concurrent requests.

  6b. No request validation / size limits

  src/DotLLM.Server/Endpoints/ChatCompletionEndpoint.cs — No validation on:
  - messages array size (could be 10,000 messages)
  - max_tokens upper bound (could request 1M tokens)
  - Total prompt length vs model's MaxSequenceLength
  - Empty messages array

  6c. Streaming leaks prompt in response

  ChatCompletionEndpoint.cs:244 — Prompt = prompt is included in the final SSE chunk. The full expanded prompt (with system prompt, chat template, etc.) is sent back to the client. This may not be desired —
   OpenAI's API doesn't do this.

  6d. Default temperature mismatch

  RequestConverter.cs:47 — Default temperature is 0.7f when request doesn't specify. ServerState.cs:116 — SamplingDefaults has Temperature = 0.0f. These disagree. The request converter ignores
  SamplingDefaults entirely — it's never consulted.

  6e. LINQ allocations in request conversion

  RequestConverter.cs:17-25 — dtos.Select(...).ToArray() allocates closures and intermediate arrays for every request. In a hot path. Should be a simple loop with pre-allocated array.

  ---
  7. CODE QUALITY / OPEN-SOURCE READINESS

  7a. Jinja template engine — no recursion limit

  src/DotLLM.Tokenizers/ChatTemplates/JinjaParser.cs — Recursive descent parser with no stack depth guard. Malicious chat template = StackOverflowException.

  7b. Jinja circular reference → infinite loop

  src/DotLLM.Tokenizers/ChatTemplates/JinjaEvaluator.cs — If context contains self-referencing data, Stringify loops forever.

  7c. ComputeThreadPool caller thread not pinned   [FIXED — Wave 7 (#109)]

  src/DotLLM.Cpu/Threading/ComputeThreadPool.cs:286
  // TODO: The caller thread (thread 0 = Forward() thread) is not pinned.
  // If scheduled on an E-core, pinned P-core workers idle at the barrier.
  On Intel hybrid CPUs (Alder Lake+), this causes P-cores to idle waiting for an E-core caller. Measurable perf impact.
  Fix: new PinCallerThread() method invoked lazily on first Dispatch, gated by ThreadingConfig.EnableCallerPinning (default true, auto-no-op when no other pinning is configured). Skips gracefully on ThreadPool threads to avoid corrupting the pool.

  7d. MXCSR rounding mode assumption

  src/DotLLM.Cpu/Kernels/MatMulKQuants.cs:69-72 — Assumes vcvtps2dq uses round-to-nearest, which depends on MXCSR. Other libraries or runtime code could change MXCSR.

  7e. Special token scanning is O(n × m)   [FIXED — Wave 7 (#109)]

  src/DotLLM.Tokenizers/Bpe/BpeTokenizer.cs:196-213 — For each position in input text, checks all special tokens. With 100+ special tokens and long texts, this is measurably slow. Should use Aho-Corasick or
   a trie.
  Fix: EncodeWithSpecialTokens and FindNextSpecialToken both call the existing Trie.TryMatchLongest (O(L) longest-prefix match per position). Preserves the "longest match wins" semantics of the previous sorted-by-descending-length scan.

  ---
  8. WHAT'S EXCELLENT (highlight these in README/docs)

  ┌──────────────────────────────┬────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │             Area             │  Quality   │                                                    Why                                                     │
  ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Tensor memory management     │ Excellent  │ 64-byte aligned NativeMemory, thread-safe disposal via Interlocked.Exchange, finalizer safety net          │
  ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ CPU SIMD kernels             │ Excellent  │ AVX-512 + AVX2 + scalar fallbacks everywhere, [SkipLocalsInit], [AggressiveInlining], proper FMA detection │
  ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Quantization correctness     │ Excellent  │ Q4_K/Q5_K/Q6_K scale extraction matches GGUF spec exactly                                                  │
  ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Warp reductions (CUDA)       │ Correct    │ Standard two-level shuffle pattern, properly used in RMSNorm/softmax/GEMV                                  │
  ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ PTX architecture             │ Elegant    │ No native shared library, forward-compatible, driver JIT handles GPU targeting                             │
  ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Fused Add+RMSNorm            │ Smart      │ Computing RMS from FP32 sum before truncating to FP16 residual — real precision win                        │
  ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Online softmax tiling (CPU)  │ Solid      │ Handles sequences exceeding L2 without O(n²) memory                                                        │
  ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ P/Invoke interop             │ Clean      │ [LibraryImport], stackalloc for kernel args, zero GC on launch path                                        │
  ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Constrained decoding design  │ Ambitious  │ FSM + PDA + regex DFA with Hopcroft minimization — well-engineered                                         │
  ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ TokenMask AVX2 masking       │ Nice touch │ Vectorized bit-to-float mask expansion for logit masking                                                   │
  ├──────────────────────────────┼────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ ModelConfig parameterization │ Good       │ Single record handles Llama/Mistral/Phi/Qwen via config, not inheritance                                   │
  └──────────────────────────────┴────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  ---
  Priority Fix List

  ┌──────────┬───────────────────────────────────────────┬──────────┬─────────┐
  │ Priority │                   Issue                   │  Effort  │ Section │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P0       │ Tiktoken regex pre-tokenization missing   │ 2-3 days │ 1a      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P0       │ CUDA attention shared mem limit (crashes) │ 1 day    │ 2a      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P1       │ CategoricalSampler fallback token         │ 1 hour   │ 1c      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P1       │ TopK tie-breaking behavior                │ 2 hours  │ 1b      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P1       │ GGUF tensor offset bounds check           │ 1 hour   │ 3b      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P1       │ GGUF alignment power-of-2 check           │ 30 min   │ 3a      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P1       │ Remove or label unimplemented interfaces  │ 2 hours  │ 2c-2f   │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P1       │ --use_fast_math → selective per-kernel    │ 2 hours  │ 3f      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P1       │ Server prompt leak in streaming           │ 15 min   │ 6c      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P1       │ Default temperature mismatch              │ 15 min   │ 6d      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P2       │ DeepSeek architecture stub                │ 30 min   │ 3e      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P2       │ Request validation in server              │ 2 hours  │ 6b      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P2       │ Stop string suffix trimming               │ 3 hours  │ 1d      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P2       │ Fused QKV dimension validation            │ 1 hour   │ 4c      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P2       │ Jinja recursion depth limit               │ 1 hour   │ 7a      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P3       │ CUDA dequant coalescing                   │ 4 hours  │ 5b      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P3       │ TopK partial sort                         │ 2 hours  │ 5a      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P3       │ half2 vectorization                       │ 1-2 days │ 5c      │
  ├──────────┼───────────────────────────────────────────┼──────────┼─────────┤
  │ P3       │ Schema cache LRU eviction                 │ 3 hours  │ 5g      │
  └──────────┴───────────────────────────────────────────┴──────────┴─────────┘

  Additions from Server/Diagnostics/Tests Review

  CRITICAL SECURITY (add to P0)

  Path traversal in /v1/models/inspect
  src/DotLLM.Server/Endpoints/ModelInspectEndpoint.cs:13
  app.MapGet("/v1/models/inspect", (string path) =>
  {
      if (string.IsNullOrEmpty(path) || !File.Exists(path))
          return Results.BadRequest(...);
      using var gguf = GgufFile.Open(path);  // path is user-controlled!
  Anyone can probe your filesystem: /v1/models/inspect?path=/etc/passwd. Need path normalization + allowlist, or remove the endpoint.

  No authentication on any endpoint — /v1/models/load lets anyone load arbitrary GGUF files, /v1/config lets anyone change sampling params. Either add API key auth or prominently document "development only,
   use reverse proxy."

  CORS wide open
  src/DotLLM.Server/ServerStartup.cs:175-177
  p.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader()
  Enables CSRF from any website.

  EMPTY / STUB MODULES

  ┌─────────────────────────────────────────┬───────────────────────────────────────────────────────────────┬────────────────────────────────────────────────┐
  │                 Module                  │                             State                             │                     Notes                      │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ DotLLM.Diagnostics/                     │ 1 interface file (ISparseAutoencoder.cs, 26 lines)            │ No hooks, no logit lens, no activation capture │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ DotLLM.Telemetry/                       │ Empty — only .csproj, zero C# files                           │ No metrics, no tracing, no OpenTelemetry       │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ samples/DotLLM.Sample.Server/           │ 7 lines, Console.WriteLine("dotLLM Server Sample") equivalent │ Doesn't start a server                         │
  ├─────────────────────────────────────────┼───────────────────────────────────────────────────────────────┼────────────────────────────────────────────────┤
  │ samples/DotLLM.Sample.Interpretability/ │ 1 line: Console.WriteLine(...)                                │ Completely non-functional                      │
  └─────────────────────────────────────────┴───────────────────────────────────────────────────────────────┴────────────────────────────────────────────────┘

  These will be the first things people click on. Empty samples with promising names are worse than no samples.

  MISSING SERVER TESTS

  Zero endpoint tests. No integration tests for:
  - /v1/chat/completions (the main endpoint people will use)
  - Streaming SSE format compliance
  - Error responses
  - Malformed input handling
  - Model loading/swapping

  Engine unit tests are excellent (samplers, constraints, kernels all well-covered). The gap is entirely on the server layer.

  ---
  Updated Priority Fix List

  ┌──────────┬─────────────────────────────────────────────────────────┬──────────────┐
  │ Priority │                          Issue                          │   Section    │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P0       │ Path traversal in /v1/models/inspect                    │ Security     │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P0       │ Tiktoken regex pre-tokenization missing                 │ Correctness  │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P0       │ CUDA attention shared mem limit (silent crash)          │ CUDA         │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ Remove or label empty modules (Diagnostics, Telemetry)  │ Presentation │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ Delete or fix broken samples (Server, Interpretability) │ Presentation │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ CORS too permissive — document or restrict              │ Security     │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ Add note about no auth (or add API key)                 │ Security     │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ CategoricalSampler fallback token                       │ Correctness  │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ TopK tie-breaking                                       │ Correctness  │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ GGUF tensor offset bounds check                         │ Safety       │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ GGUF alignment power-of-2 validation                    │ Safety       │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ --use_fast_math → selective                             │ Precision    │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ Streaming prompt leak (Prompt = prompt)                 │ Privacy      │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ Default temperature mismatch (0.7 vs 0.0)               │ Consistency  │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ CudaWeights.cs:263 missing .ThrowOnError()              │ CUDA         │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P1       │ DeepSeek architecture stub → remove or mark             │ Honesty      │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P2       │ Request validation (max_tokens bounds, etc.)            │ Server       │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P2       │ Stop string suffix trimming                             │ Correctness  │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P2       │ Fused QKV dimension validation                          │ Safety       │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P2       │ Jinja recursion depth limit                             │ DoS          │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P2       │ Server endpoint tests                                   │ Testing      │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P3       │ CUDA dequant coalescing, half2, RoPE optimization       │ Performance  │
  ├──────────┼─────────────────────────────────────────────────────────┼──────────────┤
  │ P3       │ TopK partial sort, schema cache LRU                     │ Performance  │
  └──────────┴─────────────────────────────────────────────────────────┴──────────────┘

  The two biggest embarrassment categories for open-source:
  1. Security — the path traversal and no-auth will be called out immediately
  2. Empty modules with big names — Diagnostics, Telemetry, and the two broken samples suggest vaporware

  The engine core, CPU SIMD, CUDA kernels, quantization, and constrained decoding are genuinely impressive and will hold up well to scrutiny. The issues are at the edges.

    Core Abstractions — Additional Findings

  BUG: ByteCount overflow for quantized tensors (P1)

  src/DotLLM.Core/Tensors/TensorRef.cs and UnmanagedTensor.cs:34:
  public long ByteCount => _elementCount * DType.SizeInBytes;
  For quantized types (Q4_0, Q4_1, etc.), DType.SizeInBytes = 0. So ByteCount = 0 for any quantized tensor. If this is ever used for allocation sizing, you get a zero-byte allocation followed by writes →
  heap corruption. UnmanagedTensor.Allocate has the same pattern at line 62.

  DESIGN: TokenMask is a mutable struct (footgun)

  src/DotLLM.Core/Constraints/TokenMask.cs — Allow() and Disallow() mutate the struct in-place. If someone passes a TokenMask by value (which C# does by default for structs), modifications are lost on the
  copy. No Clone() method exists. This will confuse contributors.

  DESIGN: IKernelRunner is incomplete

  src/DotLLM.Core/Backends/IKernelRunner.cs — Only 6 operations (MatMul, RmsNorm, SiLU, Softmax, Add, Multiply). Missing: RoPE, Embedding lookup, Attention, SwiGLU gating, all quantization ops. The actual
  implementations bypass this interface entirely, which raises the question: why does it exist? Either complete it or remove it.

  DESIGN: MappedTensor and TensorView are identical

  Two classes with identical code (non-owning views with no-op Dispose). Should consolidate.

  DESIGN: ModelConfig has no validation

  HeadDim is required but is redundant with HiddenSize / NumAttentionHeads. No validation that NumKvHeads divides NumAttentionHeads. No validation that RoPEConfig is only set when PositionEncodingType ==
  RoPE. A ModelConfig.Validate() method or builder would prevent silent misconfiguration.

  DESIGN: RoPEConfig mixes algorithm-specific params

  YaRN-specific fields (AttnFactor, BetaFast, BetaSlow) sit alongside NTK fields. No validation that you don't set YaRN params when using NTK scaling. Should be a discriminated union per scaling type.

  ---
  Complete Updated Summary

  All 6 agents have reported. Here's the consolidated view:

  ┌──────────────────────────┬──────────────────────────────────────────────────────────────────────┐
  │           Area           │                               Verdict                                │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ CPU SIMD kernels         │ Excellent — AVX-512/AVX2/scalar, proper annotations, fused ops       │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Quantization (CPU)       │ Excellent — all K-quant formats, matches GGUF spec                   │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ CUDA PTX architecture    │ Excellent — elegant, forward-compatible, no native lib               │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ CUDA P/Invoke interop    │ Very good — stackalloc, proper error checking (1 miss)               │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Constrained decoding     │ Very good — FSM + PDA + DFA with Hopcroft minimization               │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Tensor memory management │ Very good — 64-byte aligned, RAII, thread-safe disposal              │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Sampling pipeline        │ Good — composable, a few correctness edge cases                      │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ CUDA kernels             │ Good for v1 — correct but naive attention, no vectorization          │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ GGUF loader              │ Good — needs bounds checking on tensor offsets                       │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Tokenizer                │ Broken for tiktoken — missing regex pre-tokenization                 │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Server                   │ Needs work — security gaps, no validation, no auth                   │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Core abstractions        │ Needs cleanup — ByteCount bug, incomplete IKernelRunner, duplication │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Diagnostics              │ Stub only                                                            │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Telemetry                │ Empty                                                                │
  ├──────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Samples                  │ 2 good, 2 broken                                                     │
  └──────────────────────────┴──────────────────────────────────────────────────────────────────────┘

❯ ok, next steo - review all inference pipelines (esp. attention), list what's implemented on cpu and/or gpu, list which quantization (fused or not) versions are implemented. See what's used or unused, any duplication? Provide a detailed description and the list

  Inference Pipeline Analysis — CPU vs GPU

  1. Forward Pass: Operation-by-Operation Comparison

  Per-Layer Flow (both backends follow the same structure)

  ┌──────┬───────────────────────────────┬────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
  │ Step │           Operation           │                                CPU                                 │                                 GPU                                 │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 1    │ Embedding Lookup              │ F32/F16/Q8_0/Q4_K/Q5_K/Q6_K → FP32                                 │ F32/F16/Q8_0 → FP16 kernel                                          │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 2    │ Copy hidden → residual        │ FP32 memcpy                                                        │ FP16 D2D async copy                                                 │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 3    │ Attention RMSNorm             │ Separate RmsNorm.Execute() or fused FusedOps.RmsNormQuantize()     │ LaunchRmsNorm() (separate)                                          │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 4a   │ Q Projection                  │ Quantized GEMV (decode) or GEMM (prefill), input pre-quantized     │ Project(): cuBLAS HGEMM (prefill) or quantized GEMV kernel (decode) │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 4b   │ K Projection                  │ Same, reuses pre-quantized input if compatible family              │ Same dispatch; no input reuse (GPU always FP16)                     │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 4c   │ V Projection                  │ Same, reuses pre-quantized input if compatible family              │ Same dispatch                                                       │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 5    │ Optional Biases               │ AddBias() FP32                                                     │ LaunchBiasAdd() FP16                                                │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 6    │ Optional QK-Norm              │ Per-head RMSNorm (FP32)                                            │ LaunchPerHeadRmsNorm() FP16                                         │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 7    │ RoPE                          │ Pre-computed cos/sin tables, AVX2 vectorized                       │ In-kernel powf/sinf/cosf per thread (no table)                      │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 8    │ KV Cache Update               │ kvCache.Update() (memcpy rows)                                     │ CudaKvCache.UpdateDevice() (D2D async copy)                         │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 9    │ Attention                     │ Head-parallel, tiled online softmax, quantized KV dequant-per-tile │ Naive: 1 block per (query, head), serial max/sum                    │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 10   │ O Projection                  │ Same GEMV/GEMM as Q/K/V                                            │ Same Project() dispatch                                             │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 11   │ Attention residual + FFN norm │ Separate Add + RmsNorm                                             │ Fused LaunchFusedAddRmsNorm() — single kernel                       │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 12   │ Gate Projection               │ GEMV/GEMM                                                          │ Project()                                                           │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 13   │ Up Projection                 │ GEMV/GEMM                                                          │ Project()                                                           │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 14   │ SwiGLU                        │ Tiled (256-float L1 tiles), FP32                                   │ Single kernel, FP16                                                 │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 15   │ Down Projection               │ GEMV/GEMM                                                          │ Project()                                                           │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 16   │ FFN residual + next attn norm │ Separate Add + RmsNorm                                             │ Fused LaunchFusedAddRmsNorm()                                       │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 17   │ Final RMSNorm                 │ RmsNorm.Execute() (all tokens)                                     │ LaunchRmsNorm() (last token only)                                   │
  ├──────┼───────────────────────────────┼────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ 18   │ LM Head                       │ GEMV on last hidden                                                │ Project() → LaunchConvertF16ToF32() → D2H copy                      │
  └──────┴───────────────────────────────┴────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────┘

  ---
  2. Key Architectural Differences

  Precision

  ┌─────────────────┬─────────────────┬─────────────────────────────────────────────────────────────┐
  │                 │       CPU       │                             GPU                             │
  ├─────────────────┼─────────────────┼─────────────────────────────────────────────────────────────┤
  │ Activations     │ FP32 throughout │ FP16 throughout                                             │
  ├─────────────────┼─────────────────┼─────────────────────────────────────────────────────────────┤
  │ Residual stream │ FP32            │ FP16 (but fused add+norm uses FP32 accumulation internally) │
  ├─────────────────┼─────────────────┼─────────────────────────────────────────────────────────────┤
  │ Accumulation    │ FP32            │ FP32 (in CUDA kernels via __half2float)                     │
  ├─────────────────┼─────────────────┼─────────────────────────────────────────────────────────────┤
  │ Logits output   │ FP32 native     │ FP16 → convert to FP32 → D2H                                │
  └─────────────────┴─────────────────┴─────────────────────────────────────────────────────────────┘

  Projection Dispatch (Project())

  CPU (TransformerModel.cs:235-240):
  - Decode (seqLen=1): Fused Q/K/V decode → single thread pool dispatch for 3 GEMVs
  - Prefill (seqLen>1): GEMM with pre-quantized input reuse across Q/K/V if same quant family
  - Pre-quantization is the key CPU optimization: quantize hidden once, reuse for up to 3 projections

  GPU (CudaTransformerModel.cs:317-349):
  - Decode (seqLen=1): Quantized GEMV kernel (Q8_0, Q4_K, Q6_K) or cuBLAS GEMV if no quantized kernel
  - Prefill (seqLen>1): Dequant to FP16 scratch → cuBLAS HGEMM (Tensor Cores)
  - No input pre-quantization reuse (GPU operates in FP16; quantized weights are dequantized or used directly)

  Fused Operations

  ┌──────────────────────────────┬──────────────────────────────────────┬───────────────────────────────────────────────────────┐
  │            Fusion            │                 CPU                  │                          GPU                          │
  ├──────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ RMSNorm + Quantize           │ Yes (Q8_0, Q8_1, Q8_K) — single pass │ No (separate RMSNorm, no CPU-side quantization)       │
  ├──────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Residual Add + RMSNorm       │ No (separate kernels)                │ Yes (fused_add_rmsnorm.cu) — FP32 accumulation inside │
  ├──────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ SwiGLU (gate × sigmoid × up) │ Yes — L1-tiled (256 floats)          │ Yes — single kernel                                   │
  ├──────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Q/K/V Decode (3 GEMVs)       │ Yes — single pool dispatch           │ No (3 separate Project() calls)                       │
  ├──────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Gate/Up Decode (2 GEMVs)     │ Yes — single pool dispatch           │ No (2 separate Project() calls)                       │
  ├──────────────────────────────┼──────────────────────────────────────┼───────────────────────────────────────────────────────┤
  │ Weight Repacking (R4)        │ Yes — interleaved for TLB/prefetch   │ No (GPU doesn't need repacking)                       │
  └──────────────────────────────┴──────────────────────────────────────┴───────────────────────────────────────────────────────┘

  RoPE

  ┌───────────────────────┬──────────────────────────────────────────┬──────────────────────────────────────────────┐
  │                       │                   CPU                    │                     GPU                      │
  ├───────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Frequency computation │ Pre-computed cos/sin table at model load │ powf(theta, ...) per thread per forward pass │
  ├───────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Vectorization         │ AVX2: 4 or 8 pairs per iteration         │ 1 pair per thread, no half2                  │
  ├───────────────────────┼──────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ Layout support        │ Norm + NeoX                              │ Norm + NeoX                                  │
  └───────────────────────┴──────────────────────────────────────────┴──────────────────────────────────────────────┘

  Attention

  ┌───────────────────┬─────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────┐
  │                   │                       CPU                       │                              GPU                               │
  ├───────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Parallelism       │ Head-parallel via ComputeThreadPool             │ 1 block per (query_token, head)                                │
  ├───────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Small sequences   │ Materialized score matrix, SIMD softmax         │ Same naive kernel for all sizes                                │
  ├───────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Large sequences   │ Tiled online softmax (O(tile) memory)           │ Unbounded shared memory (O(seq_kv) — crashes at ~12K)          │
  ├───────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Quantized KV      │ Per-tile dequant (Q8_0/Q4_0), online softmax    │ Dequant full quantized region to scratch first, then attention │
  ├───────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Softmax           │ Fast approximate exp for attention (acceptable) │ expf (or __expf with --use_fast_math)                          │
  ├───────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Max/Sum reduction │ SIMD parallel                                   │ Serial (thread 0 scans all scores)                             │
  ├───────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ GQA               │ Yes (groupSize = numHeads / numKvHeads)         │ Yes (same mapping)                                             │
  ├───────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Sliding window    │ Not in attention kernel (handled at KV level)   │ Yes (in attention kernel)                                      │
  ├───────────────────┼─────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────┤
  │ Causal mask       │ Yes                                             │ Yes                                                            │
  └───────────────────┴─────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────┘

  ---
  3. Quantization Support Matrix

  Weight Formats Supported Per Backend

  ┌────────┬───────────────┬─────────────────────────┬───────────────┬──────────────┬───────────────────────┬────────────────┐
  │ Format │  CPU Dequant  │        CPU GEMV         │   CPU GEMM    │ CUDA Dequant │       CUDA GEMV       │   CUDA GEMM    │
  ├────────┼───────────────┼─────────────────────────┼───────────────┼──────────────┼───────────────────────┼────────────────┤
  │ F32    │ passthrough   │ TensorPrimitives.Dot    │ -             │ -            │ cuBLAS GEMV           │ cuBLAS HGEMM   │
  ├────────┼───────────────┼─────────────────────────┼───────────────┼──────────────┼───────────────────────┼────────────────┤
  │ F16    │ convert       │ -                       │ -             │ passthrough  │ cuBLAS GEMV           │ cuBLAS HGEMM   │
  ├────────┼───────────────┼─────────────────────────┼───────────────┼──────────────┼───────────────────────┼────────────────┤
  │ Q8_0   │ AVX2 + scalar │ AVX2/512 VecDot         │ AVX2/512 GEMM │ PTX kernel   │ PTX GEMV kernel       │ dequant→cuBLAS │
  ├────────┼───────────────┼─────────────────────────┼───────────────┼──────────────┼───────────────────────┼────────────────┤
  │ Q5_0   │ scalar only   │ AVX2 VecDot (Q5_0×Q8_1) │ AVX2 GEMM     │ PTX kernel   │ NONE (dequant→cuBLAS) │ dequant→cuBLAS │
  ├────────┼───────────────┼─────────────────────────┼───────────────┼──────────────┼───────────────────────┼────────────────┤
  │ Q4_0   │ scalar only   │ NONE                    │ NONE          │ PTX kernel   │ NONE (dequant→cuBLAS) │ dequant→cuBLAS │
  ├────────┼───────────────┼─────────────────────────┼───────────────┼──────────────┼───────────────────────┼────────────────┤
  │ Q4_K   │ AVX2 + scalar │ AVX2 VecDot (Q4_K×Q8_K) │ via VecDot    │ PTX kernel   │ PTX GEMV kernel       │ dequant→cuBLAS │
  ├────────┼───────────────┼─────────────────────────┼───────────────┼──────────────┼───────────────────────┼────────────────┤
  │ Q5_K   │ AVX2 + scalar │ AVX2 VecDot (Q5_K×Q8_K) │ via VecDot    │ PTX kernel   │ NONE (dequant→cuBLAS) │ dequant→cuBLAS │
  ├────────┼───────────────┼─────────────────────────┼───────────────┼──────────────┼───────────────────────┼────────────────┤
  │ Q6_K   │ AVX2 + scalar │ AVX2 VecDot (Q6_K×Q8_K) │ via VecDot    │ PTX kernel   │ PTX GEMV kernel       │ dequant→cuBLAS │
  └────────┴───────────────┴─────────────────────────┴───────────────┴──────────────┴───────────────────────┴────────────────┘

  Input Pre-Quantization (CPU only)

  ┌──────────────────┬─────────────────────────────────────────────────┬──────────────────┐
  │  Weight Family   │               Input Quantized To                │     Used By      │
  ├──────────────────┼─────────────────────────────────────────────────┼──────────────────┤
  │ Q8_0             │ Q8_0 (half scale, 32-elem)                      │ Q8_0 VecDot      │
  ├──────────────────┼─────────────────────────────────────────────────┼──────────────────┤
  │ Q5_0             │ Q8_1 (half d + half s, 32-elem with block sums) │ Q5_0×Q8_1 VecDot │
  ├──────────────────┼─────────────────────────────────────────────────┼──────────────────┤
  │ Q4_K, Q5_K, Q6_K │ Q8_K (float32 scale, 256-elem with bsums)       │ K-quant VecDot   │
  └──────────────────┴─────────────────────────────────────────────────┴──────────────────┘

  GPU has no input pre-quantization — operates directly in FP16.

  KV-Cache Quantization

  ┌─────────────────────────────────┬─────┬─────────────────────────────┐
  │             Format              │ CPU │             GPU             │
  ├─────────────────────────────────┼─────┼─────────────────────────────┤
  │ FP32 (SimpleKvCache)            │ Yes │ N/A (GPU uses FP16)         │
  ├─────────────────────────────────┼─────┼─────────────────────────────┤
  │ FP16 (CudaKvCache)              │ N/A │ Yes                         │
  ├─────────────────────────────────┼─────┼─────────────────────────────┤
  │ Q8_0 (quantize-on-evict)        │ Yes │ Yes (PTX quant_f16_to_q8_0) │
  ├─────────────────────────────────┼─────┼─────────────────────────────┤
  │ Q4_0 (quantize-on-evict)        │ Yes │ Yes (PTX quant_f16_to_q4_0) │
  ├─────────────────────────────────┼─────┼─────────────────────────────┤
  │ Dual-region (quant + FP window) │ Yes │ Yes                         │
  └─────────────────────────────────┴─────┴─────────────────────────────┘

  ---
  4. Unused / Duplicated Code

  Duplicated Between CPU and GPU

  ┌───────────────────────────────────┬─────────────────────────────┬───────────────────────────────┬───────────────────────────────────────────┐
  │               Item                │             CPU             │              GPU              │                   Notes                   │
  ├───────────────────────────────────┼─────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
  │ Q8_0 quantization                 │ QuantizedKvCache.cs:319-348 │ quant_kv.cu:16-59             │ Same algorithm, different language        │
  ├───────────────────────────────────┼─────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
  │ Q4_0 quantization                 │ QuantizedKvCache.cs:352-385 │ quant_kv.cu:66-109            │ Same algorithm                            │
  ├───────────────────────────────────┼─────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
  │ Q8_0 dequant                      │ Dequantize.cs (AVX2)        │ dequant.cu (PTX)              │ CPU has AVX2 path GPU lacks vectorization │
  ├───────────────────────────────────┼─────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
  │ Scale extraction (Q4_K/Q5_K/Q6_K) │ DequantizeKQuants.cs        │ dequant.cu, quantized_gemv.cu │ Identical bit manipulation logic          │
  ├───────────────────────────────────┼─────────────────────────────┼───────────────────────────────┼───────────────────────────────────────────┤
  │ Attention                         │ Tiled online softmax        │ Naive single-pass             │ CPU is significantly more sophisticated   │
  └───────────────────────────────────┴─────────────────────────────┴───────────────────────────────┴───────────────────────────────────────────┘

  FP16 vs FP32 Kernel Duplication (GPU)

  Every CUDA kernel has both FP16 and FP32 variants:

  ┌─────────────────┬─────────────────────────┬─────────────────────────┬─────────────────────────────────────────────┐
  │     Kernel      │        FP16 file        │        FP32 file        │                    Used?                    │
  ├─────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ RMSNorm         │ rmsnorm.cu              │ rmsnorm_f32.cu          │ Both used (FP32 for HybridTransformerModel) │
  ├─────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ Attention       │ attention.cu            │ attention_f32.cu        │ Both used                                   │
  ├─────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ RoPE            │ rope.cu                 │ rope_f32.cu             │ Both used                                   │
  ├─────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ SwiGLU          │ swiglu.cu               │ swiglu_f32.cu           │ Both used                                   │
  ├─────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ Add             │ add.cu                  │ add_f32.cu              │ Both used                                   │
  ├─────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ BiasAdd         │ bias_add.cu             │ bias_add_f32.cu         │ Both used                                   │
  ├─────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ PerHeadRmsNorm  │ per_head_rmsnorm.cu     │ per_head_rmsnorm_f32.cu │ Both used                                   │
  ├─────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ Embedding       │ embedding.cu            │ embedding_f32out.cu     │ Both used                                   │
  ├─────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ RmsNorm F32→F16 │ rmsnorm_f32in.cu        │ -                       │ Used by hybrid model                        │
  ├─────────────────┼─────────────────────────┼─────────────────────────┼─────────────────────────────────────────────┤
  │ GEMV F32 input  │ quantized_gemv_f32in.cu │ -                       │ Used by hybrid model                        │
  └─────────────────┴─────────────────────────┴─────────────────────────┴─────────────────────────────────────────────┘

  The FP32 variants exist for the HybridTransformerModel which runs some layers on GPU with FP32 residual stream. Not dead code — all are referenced from CudaKernels.cs.

  Potentially Unused

  ┌─────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │          Item           │                                                                                   Status                                                                                    │
  ├─────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ IKernelRunner interface │ 6 operations defined, none of them called by the actual forward pass. Both CPU and GPU forward passes call kernels directly. Dead interface.                                │
  ├─────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ IAttentionMechanism     │ Not used in either forward pass. Attention is called directly.                                                                                                              │
  ├─────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Q4_0 CPU dequant        │ Dequantize.cs has it, but there's no Q4_0 GEMV on CPU. Weights are repacked to Q8_0 at load time. Only used if someone dequantizes Q4_0 embeddings.                         │
  ├─────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Softmax.Execute()       │ CPU attention uses Softmax.ExecuteFast() (approximate). The exact softmax exists but is only called from tests. The GPU softmax.cu kernel is launched from                  │
  │ standalone              │ CudaKernels.LaunchSoftmax() which is... not called in the GPU forward pass — softmax is baked into attention.cu. Potentially dead GPU kernel.                               │
  └─────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Missing on GPU (present on CPU)

  ┌────────────────────────────────────┬────────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────┐
  │              Feature               │                      CPU                       │                                  GPU                                   │
  ├────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Fused RMSNorm+Quantize             │ Yes                                            │ No (would eliminate one kernel launch + one DRAM pass)                 │
  ├────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Fused Q/K/V decode dispatch        │ Yes (single pool call)                         │ No (3 separate launches)                                               │
  ├────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Fused Gate/Up decode dispatch      │ Yes                                            │ No (2 separate launches)                                               │
  ├────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Pre-quantized input reuse          │ Yes                                            │ N/A (GPU doesn't quantize activations)                                 │
  ├────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Q5_0 quantized GEMV                │ Yes (AVX2)                                     │ No (falls back to dequant → cuBLAS)                                    │
  ├────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Q5_K quantized GEMV                │ Yes (AVX2)                                     │ No (falls back to dequant → cuBLAS)                                    │
  ├────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Tiled online softmax               │ Yes                                            │ No                                                                     │
  ├────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Per-tile KV dequant in attention   │ Yes (avoids materializing full dequantized KV) │ No (dequants full region to scratch first via PrepareAttentionScratch) │
  ├────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ RoPE pre-computed tables           │ Yes                                            │ No (computes in-kernel)                                                │
  ├────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Weight repacking (R4 interleaving) │ Yes                                            │ N/A                                                                    │
  ├────────────────────────────────────┼────────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────┤
  │ Q5_0 dequant AVX2                  │ No (scalar only)                               │ Has PTX kernel                                                         │
  └────────────────────────────────────┴────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────┘

  Missing on CPU (present on GPU)

  ┌────────────────────────────────────┬──────────────────────────────────────────┬─────────────────────────────┐
  │              Feature               │                   GPU                    │             CPU             │
  ├────────────────────────────────────┼──────────────────────────────────────────┼─────────────────────────────┤
  │ Fused Add+RMSNorm                  │ Yes (avoids FP16 truncation at residual) │ No (separate add + rmsnorm) │
  ├────────────────────────────────────┼──────────────────────────────────────────┼─────────────────────────────┤
  │ Sliding window in attention kernel │ Yes                                      │ No (handled externally)     │
  ├────────────────────────────────────┼──────────────────────────────────────────┼─────────────────────────────┤
  │ FP16 activations                   │ Yes                                      │ No (CPU is FP32 throughout) │
  └────────────────────────────────────┴──────────────────────────────────────────┴─────────────────────────────┘

  ---
  5. Summary: What Stands Out

  CPU backend is more optimized than GPU in several areas:
  - Fused decode dispatches (3→1 for QKV, 2→1 for GateUp)
  - Tiled online softmax for long sequences (GPU crashes at ~12K)
  - Pre-quantized input reuse across projections
  - Pre-computed RoPE tables vs per-thread powf
  - Per-tile KV dequant in attention (GPU materializes the full dequantized buffer)

  GPU backend has one key fusion CPU lacks:
  - Fused Add+RMSNorm (avoids residual precision loss)

  Quantized GEMV coverage gap on GPU: Q5_0 and Q5_K fall back to dequant→cuBLAS GEMV (expensive extra kernel + memory pass). CPU handles them natively.

  IKernelRunner and IAttentionMechanism are dead abstractions — neither backend uses them. The forward passes call kernels directly.

  Standalone softmax.cu appears unused in the GPU forward pass — softmax is embedded in attention.cu.