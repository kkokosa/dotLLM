# DESIGN — Nemotron-3 (`nemotron_h`) support in dotLLM

Branch: `feature/nemotron-and-mamba-2` (worktree: `C:/Development/dotLLM-mamba3`).
Target model: `NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf` (2.7 GB).
Goal: load and generate tokens with the dotLLM CPU backend.

> **Note on scope.** This work was started under a branch originally named
> `feature/mamba-3` with the intent of implementing the **Mamba-3** algorithm
> (Lahoti et al., [arXiv 2603.15569](https://hf.co/papers/2603.15569),
> Mar 2026 — complex-valued state updates, MIMO formulation). The plan below
> was then (wrongly) written against the **Mamba-2**-based `nemotron_h`
> hybrid architecture that NVIDIA ships in every Nemotron-3 family member
> (Nano 4B / Nano 30B-A3B MoE / Super / Ultra). The "3" in Nemotron-3 is
> the model family version, not the Mamba generation. Having already built
> out most of this path, we finish the Mamba-2 / `nemotron_h` implementation
> on the renamed branch and land it, *then* open a fresh `feature/mamba-3`
> for the actual Mamba-3 algorithm as a separate workstream.

## 1. Architecture summary

Nemotron-3 (NVIDIA) is a **hybrid SSM + Transformer** language model. The
`nemotron_h` GGUF architecture string identifies it. Each of the model's
`block_count` layers is one of three exclusive types:

```
embed -> N x [ attn_norm -> { SSM | Attention | FFN } -> residual ] -> output_norm -> lm_head
```

This is unlike a standard Llama/Qwen block where attention **and** an FFN
both run with two residuals; here each layer applies a *single* sub-layer
with one residual add (mirroring llama.cpp `src/models/nemotron-h.cpp`).

The three sub-layer flavours, in order from most novel to least:

| Type      | Sub-layer body (after RMSNorm)                                                                                                    |
|-----------|-----------------------------------------------------------------------------------------------------------------------------------|
| SSM       | linear `ssm_in` -> split (z, x, B, C, dt) -> conv1d+SiLU on (x|B|C) -> selective-scan -> +D*x -> SwiGLU(z, y) -> group-RMSNorm -> linear `ssm_out` |
| Attention | linear Q/K/V (no bias) -> RoPE on first 78 dims of head -> GQA attention -> linear `attn_output`                                    |
| FFN       | linear `ffn_up` -> ReLU^2 -> linear `ffn_down`                                                                                   |

There is **no SwiGLU MLP** (no `ffn_gate.weight`). The FFN is a parallel
non-gated MLP with squared-ReLU activation.

## 2. Target GGUF inventory (NVIDIA-Nemotron-3-Nano-4B-Q4_K_M)

Metadata extracted via `dotnet run --project src/DotLLM.Cli -c Debug -- debug gguf-metadata <file>`:

```
general.architecture            = "nemotron_h"
nemotron_h.block_count          = 42
nemotron_h.embedding_length     = 3136
nemotron_h.attention.head_count = 40
nemotron_h.attention.key_length = 128
nemotron_h.attention.value_length = 128
nemotron_h.attention.head_count_kv      = INT32[42]   <- per-layer, 0 means non-attn layer
nemotron_h.feed_forward_length          = INT32[42]   <- per-layer, 0 means non-FFN layer
nemotron_h.attention.layer_norm_rms_epsilon = 1e-5
nemotron_h.context_length       = 1048576   (1 M)
nemotron_h.rope.dimension_count = 78        (partial RoPE, only first 78 of head_dim=128)
nemotron_h.ssm.conv_kernel      = 4   (d_conv)
nemotron_h.ssm.group_count      = 8   (n_group)
nemotron_h.ssm.inner_size       = 7680 (d_inner)
nemotron_h.ssm.state_size       = 128 (d_state)
nemotron_h.ssm.time_step_rank   = 96  (n_head for Mamba2 — heads of size d_inner/n_head = 80)
nemotron_h.vocab_size           = 131072
tokenizer.ggml.model            = "gpt2"   (BPE)
```

Per-layer tensor inventory (sample from `debug gguf-tensors`):

* SSM layers (21x: blk 0,2,4,6,7,9,11,14,16,19,21,23,26,28,30,31,34,35,36,38,40):
    - `attn_norm.weight                 [3136]      F32`
    - `ssm_in.weight                    [3136, 17504] Q5_0`   (17504 = 2*7680 + 2*8*128 + 96)
    - `ssm_conv1d.weight                [4, 9728]    F32`     (9728 = 7680 + 2*8*128)
    - `ssm_conv1d.bias                  [9728]       F32`
    - `ssm_a                            [1, 96]      F32`     (Mamba2 scalar A per head)
    - `ssm_d                            [1, 96]      F32`     (per-head skip gain)
    - `ssm_dt.bias                      [96]         F32`
    - `ssm_norm.weight                  [960, 8]     F32`     (group RMSNorm, d_inner/n_group=960, n_group=8)
    - `ssm_out.weight                   [7680, 3136] Q4_K`
* Attention layers (4x: blk 12,17,24,32):
    - `attn_norm.weight                 [3136]      F32`
    - `attn_q.weight                    [3136, 5120] Q5_0`    (40 heads x 128)
    - `attn_k.weight                    [3136, 1024] Q5_0`    (8 KV heads x 128)
    - `attn_v.weight                    [3136, 1024] Q5_0/Q8_0`
    - `attn_output.weight               [5120, 3136] Q4_K`
* FFN layers (17x: blk 1,3,5,8,10,13,15,18,20,22,25,27,29,33,37,39,41):
    - `attn_norm.weight                 [3136]      F32`     (named attn_norm but is the FFN pre-norm)
    - `ffn_up.weight                    [3136, 12544] Q5_0`
    - `ffn_down.weight                  [12544, 3136] Q4_K/Q6_K`
    - **no `ffn_gate.weight`**
* Global:
    - `token_embd.weight                [3136, 131072] Q5_0`
    - `output_norm.weight               [3136]      F32`
    - `output.weight                    [3136, 131072] Q8_0`

Layer-type discriminator (matches llama.cpp `is_recurrent(il)` / `n_ff(il) == 0`):

```
if   tensors contain blk.{i}.ssm_in.weight       -> SSM layer
elif tensors contain blk.{i}.attn_q.weight       -> Attention layer
elif tensors contain blk.{i}.ffn_up.weight       -> FFN layer
else error
```

## 3. dotLLM gap analysis

What dotLLM has that we can reuse as-is:
* `GgufFile.Open` / `GgufMetadata` / `GgufTensorDescriptor`
* `RmsNorm`, `Attention`, `RoPE`, `SiLU`, `Add`, `Multiply` kernels (CPU)
* `MatMul` GEMM/GEMV for Q4_K, Q5_0, Q5_K, Q6_K, Q8_0, F16, F32
* `WeightRepacking` R4 interleave
* `BpeTokenizer` (gpt2 pre-tokenizer regex is needed; verify already supported)
* `TextGenerator`, KV cache (works for the attention layers we keep)
* `IModel` / `ITensor` interfaces

What is missing or needs change:
1. **Architecture enum:** `Architecture.NemotronH` (extend `DotLLM.Core.Configuration.Architecture`).
2. **Config extraction:** `GgufModelConfigExtractor.ParseArchitecture` recognises `nemotron_h`. We also need to parse the **array** metadata (`head_count_kv`, `feed_forward_length`) and the SSM hyperparameters into a new `MambaSsmConfig` record bag. `GgufMetadata` does not yet expose `int[]` arrays of length-N per-layer counts — `GetInt32Array` covers it because the underlying `GgufValueType.Array` already stores `int[]`. Good. Need to add an `int[]?` field to `ModelConfig` (or a side record `HybridLayerLayout`) for per-layer head_count_kv and ffn dim, plus the SSM dims.
3. **Layer-type tag:** A `LayerKind` enum (`Ssm | Attention | Ffn`) computed at load time from tensor presence (preferred over array-zero check, more robust).
4. **New CPU kernels:**
    a. `Conv1dCausal` — depthwise causal 1D conv with kernel size 4, applied per channel along the time dim (input shape `[d_conv-1 + n_seq_tokens, channels]`). Bias add and SiLU follow.
    b. `Mamba2SelectiveScan` — the recurrent state-space scan (see math below). Pure scalar reference first, vectorise later.
    c. `ReluSquared` activation (one-line: `y = ReLU(x); y *= y`). Trivial.
    d. **Group RMSNorm** — RMSNorm applied independently to each of n_group sub-vectors of length `d_inner/n_group = 960`. Reuse existing `RmsNorm.Execute` in a loop; weight tensor `ssm_norm.weight` has shape `[960, 8]` and is broadcast per group.
5. **Recurrent state buffer (SSM cache):** Two per-layer tensors per sequence:
    * `conv_state`  shape `[d_conv-1, d_inner + 2*n_group*d_state]` = `[3, 9728]`
    * `ssm_state`   shape `[d_state, head_dim, n_head]` = `[128, 80, 96]` = ~3.9M floats = ~15 MB per SSM layer per sequence (large!). For 21 SSM layers that is ~315 MB; fine for a 4B model with 2.7 GB weights.
    The current `IKvCache` in dotLLM is attention-only. We add a parallel `ISsmStateCache` (or a unified `IRecurrentStateCache`) that lives next to the KV cache. For the first cut: a non-paged `SsmStateCache` allocated in unmanaged memory, one per generation request, sized to the model's SSM-layer count.
6. **Hybrid block dispatch:** A new `NemotronHTransformerModel : IModel` with its own `Forward()` that walks layers consulting the per-layer `LayerKind` and dispatches to one of three sub-layer routines. Cannot reuse `TransformerModel` directly because:
    * its forward assumes attn+FFN per layer with two residuals,
    * it pre-allocates Q/K/V/Gate/Up scratch globally,
    * it has no place for SSM tensors or recurrent state.
    A clean separate type is the right call. We should still factor shared helpers (`EmbeddingLookup`, `GemmInterleaved`, `QuantizeInput`) so we do not duplicate the dispatch table.
7. **TransformerArchitecture factory:** add `Architecture.NemotronH` -> `NemotronHTransformerModel.LoadFromGguf` switch.
8. **BPE tokenizer pre-tokenizer:** metadata says `tokenizer.ggml.pre = "pixtral"`. Need to check whether `GgufBpeTokenizerFactory` knows this regex; fall back to gpt2 default if not. (Out-of-scope concern — log a TODO.)
9. **RoPE partial dimensions:** `rope.dimension_count = 78` means only the first 78 of each 128-dim head get RoPE; the rest are unrotated. The existing `RoPE.Execute` already takes a `ropeDim` parameter (`_ropeDim`). Confirm it implements the partial case correctly — it appears to.

## 4. Mamba2 forward maths (CPU reference)

Naming below matches the GGUF tensors. All shapes assume one sequence
(`n_seqs = 1`), `T = n_seq_tokens` in the current step.

Input `x_in`  shape `[T, d_model]` where `d_model = 3136`.

```
# 1. Big input projection: hidden -> z, x, B, C, dt
zxBCdt = x_in @ ssm_in            # [T, d_in_proj] where d_in_proj = 17504
                                  # = 2*d_inner + 2*n_group*d_state + n_head
                                  #   d_inner = 7680, n_group = 8, d_state = 128, n_head = 96
# Layout along d_in_proj:
#   [0                              .. d_inner)               -> z   (gate)
#   [d_inner                        .. 2*d_inner)             -> x_part of xBC
#   [2*d_inner                      .. 2*d_inner + n_group*d_state) -> B
#   [...                            .. 2*d_inner + 2*n_group*d_state) -> C
#   [...                            .. + n_head)              -> dt
xBC = concat(x_part, B, C)        # [T, d_inner + 2*n_group*d_state] = [T, 9728]

# 2. Causal depthwise 1D conv over last d_conv-1 cached steps + new T steps
conv_input = concat(conv_state, xBC^T)   # along time, [d_conv-1+T, 9728]
for ch in 0..9728:
    for t in 0..T:
        y[t, ch] = sum_{k in 0..4} conv_input[t+k, ch] * conv1d_w[k, ch]
xBC = SiLU(y + conv1d_bias)               # [T, 9728]
conv_state = last (d_conv-1) rows of conv_input  # [3, 9728]

# 3. Re-split xBC into x, B, C (in-place views)
x = xBC[:, 0..d_inner]                    # [T, 7680]
B = xBC[:, d_inner..d_inner+n_group*d_state]   # [T, 8*128]
C = xBC[:, d_inner+n_group*d_state..end]       # [T, 8*128]

# 4. dt with bias
dt = dt + ssm_dt_b                        # [T, n_head]

# 5. Selective scan (THE recurrent loop). Groups: each head h is in group g = h / (n_head/n_group).
#    State shape: [n_head, head_dim, d_state] = [96, 80, 128].
A = ssm_a                                 # [n_head]    (scalar per head)
D = ssm_d                                 # [n_head]    (scalar per head)
for t in 0..T:
    for h in 0..n_head:
        dt_sp = softplus(dt[t, h])
        dA    = exp(dt_sp * A[h])
        g     = h / (n_head / n_group)
        for i in 0..head_dim:               # head_dim = 80
            x_dt = x[t, h*head_dim + i] * dt_sp
            sumf = 0
            for k in 0..d_state:            # d_state = 128
                state[h, i, k] = state[h, i, k] * dA + B[t, g*d_state + k] * x_dt
                sumf += state[h, i, k] * C[t, g*d_state + k]
            y[t, h*head_dim + i] = sumf

# 6. Skip connection then SwiGLU gating with z
y = y + x * D_per_head_broadcast          # broadcast D across head_dim
y = SiLU(z) * y                           # SwiGLU (note: gating param is z, not y)

# 7. Group RMSNorm: split y [T, d_inner] -> [T, n_group, d_inner/n_group], per-group RMSNorm with ssm_norm
y = group_rms_norm(y, ssm_norm)           # [T, 7680]

# 8. Output projection back to model width
out = y @ ssm_out                         # [T, 3136]
```

Per Mamba2 paper: A is parameterised in log-space and `ggml_ssm_scan`
applies `exp(dt_sp * A[h])` directly (no negation). The GGUF converter
already stores `A` such that `A[h] < 0`, so `exp(dt_sp * A[h])` decays.
We mirror llama.cpp byte-for-byte to avoid sign confusion.

The expensive inner loop is `n_head * head_dim * d_state = 96 * 80 * 128 = 983,040`
fused multiply-adds per token per layer, plus a second pass for the dot with C
(or both fused, as llama.cpp does). For 21 SSM layers and a 1-token decode:
~21 M FMAs per token. Q4_K weight matmuls dominate compute regardless.

## 5. Staged implementation plan

Each stage is a separate commit on `feature/mamba-3`. After every stage:
`dotnet build src/dotLLM.sln -c Release` MUST pass before commit.

1. **DESIGN.md** (this file). Commit first.
2. **Architecture enum + ParseArchitecture + ModelConfig fields**
   * Add `Architecture.NemotronH`
   * Extend `ModelConfig` with optional `HybridLayoutConfig?` (per-layer `LayerKind[]`, per-layer `int[] HeadCountKv`, per-layer `int[] FeedForwardLength`) and `MambaSsmConfig?` (d_conv, d_inner, d_state, n_group, n_head/dt_rank).
   * Update `GgufModelConfigExtractor.Extract` to populate them when `nemotron_h`.
   * Defensive: existing transformer paths must still be unchanged for llama/qwen/...
   * Verify against `dotnet run -- debug gguf-config <nemotron-3>` (need to extend the CLI's debug command to print the new fields, optional).
3. **NemotronHTransformerModel skeleton + loader**
   * Walk all 42 blocks, classify by tensor presence, build per-layer weight bundles (`SsmLayerWeights`, `AttnLayerWeights`, `FfnLayerWeights`).
   * Allocate scratch and recurrent state.
   * `Forward()` returns logits when **all** layers are SSM-only or trivial — for now throws `NotImplementedException("Mamba2 selective scan not yet implemented")` with a clear pointer to step 5.
   * Wire `TransformerArchitecture.SupportedArchitectures` and `CreateModel` to dispatch to it.
   * Update `ModelLoader.LoadFromGguf` to dispatch to the new model class for `Architecture.NemotronH`.
   * Result: `dotllm model-info <nemotron-3>` no longer throws on architecture, but `dotllm run` throws a clean NotImplementedException.
4. **Squared ReLU activation kernel** (small, easy)
   * `DotLLM.Cpu/Kernels/ReluSquared.cs`. SIMD via `TensorPrimitives.Max` then `TensorPrimitives.Multiply`. Smoke test against scalar reference.
5. **FFN sub-layer forward (no SSM, no attention yet)**
   * Wire the 17 FFN layers in `NemotronHTransformerModel.Forward`.
   * Path: RmsNorm -> ffn_up GEMM -> ReluSquared -> ffn_down GEMM -> residual.
   * Still throws on first SSM layer encountered; useful for catching layout bugs in isolation.
6. **Attention sub-layer forward**
   * Reuse existing `Attention.Execute`, `RoPE.Execute`, `MatMul` GEMMs.
   * Biases: none for nemotron-3.
   * Per-layer KV cache slot — but only for the 4 attention layers; allocate a sparse KV cache that has entries only for those layer indices.
7. **Mamba2 SSM forward**
   a. `Conv1dCausal` kernel (depthwise, kernel size 4, channels = 9728). Scalar fallback first; SIMD later.
   b. `Mamba2SelectiveScan` kernel — direct port of `ggml_compute_forward_ssm_scan_f32` (Mamba-2 branch where `src3->ne[0] == 1`).
   c. Group RMSNorm helper (reuse RmsNorm in a loop initially).
   d. SsmStateCache (unmanaged, aligned).
   e. Wire SSM dispatch in `NemotronHTransformerModel.Forward`.
   f. Numerical validation against llama.cpp logits (capture a few prompt tokens in llama.cpp with `--logit-bias` debug).
8. **Smoke test** `tests/DotLLM.Tests.Integration/Engine/NemotronHTextGeneratorTests.cs`:
   * Loads model, generates 5 greedy tokens, asserts non-empty and non-NaN logits.
   * Skipped (or marked `Trait("RequiresModel")`) when GGUF is absent.

## 6. Risks and notes

* **SSM math correctness is fragile.** Tiny errors in head ordering, group repeat-interleave, or A sign convention will produce gibberish that *looks* like normal generation. Mitigate by capturing intermediate tensors (z, x_post_conv, y_pre_norm, y_post_norm) for token 0 of "Hello" in llama.cpp and comparing to ours element-wise within 1e-3.
* **Performance (later).** First cut is scalar; fine. AVX2 vectorisation of the inner `d_state=128` loop is straightforward (512-byte rows = 4 YMM registers). The conv1d and selective-scan are memory-bound with small per-channel work, so a thread-per-head-block parallelisation across `n_head=96` should scale well on 16-32 cores.
* **Memory:** 21 SSM layers x (3 * 9728 * 4B conv-state + 96*80*128 * 4B ssm-state) ~= 21 * (117 KB + 3.9 MB) ~= 84 MB recurrent state per sequence. Reasonable.
* **Tokenizer pre-tokenizer:** `tokenizer.ggml.pre = "pixtral"`. Must confirm the BPE tokenizer factory accepts that pre-key (otherwise tokens off by one will completely corrupt generation). We should log it loudly during loading and fall back to gpt2 split if unknown — the actual byte-level merges still apply.
* **Partial RoPE (rope_dim=78 of head_dim=128):** verify dotLLM's RoPE applies to first `ropeDim` dims of each head and leaves the rest untouched. From reading `TransformerModel.Forward` and `RoPE.Execute(... ropeDim ...)`, this looks correct. Add an assertion at load time that the four attention layers have `head_dim == 128` and `ropeDim == 78`.
* **Tied embeddings:** `output.weight` is present (Q8_0), separate from `token_embd.weight` (Q5_0). No tying.
* **MoE variant** `nemotron_h_moe` is **not** in scope; raise a clear error if encountered.
* **Build:** all Mamba2 work is CPU-only. No CUDA, no native lib changes. The CUDA backend already ignores unsupported architectures via `TransformerArchitecture` filter.

## 7. Done criteria

* `dotllm run NVIDIA-Nemotron-3-Nano-4B-GGUF` (or equivalent path) produces text without throwing.
* Greedy "The capital of France is" produces a sensible continuation containing "Paris" within the first 10 tokens (or, failing that, produces deterministic non-NaN logits whose argmax is in the vocab range).
* All existing tests still green.
* If we ship before step 7 (SSM forward) is done, the loader works end-to-end and the forward throws a clean `NotImplementedException("Mamba2 selective scan not yet implemented (stage 7 of feature/mamba-3)")` from a single named call site so a future agent can pick up exactly where we left off.
