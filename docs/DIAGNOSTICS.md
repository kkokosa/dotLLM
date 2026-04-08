# Diagnostics & Interpretability — dotLLM

## Hook System

### Hook Points

Fired at well-defined pipeline locations with the activation tensor:

| Hook Point | Location | Typical Use |
|-----------|----------|-------------|
| `PostEmbedding` | After token embedding, before layer 0 | Input analysis |
| `PreAttention(layer)` | After pre-attention norm | Attention input study |
| `PostAttention(layer)` | After attention, before residual | Attention output analysis |
| `PreFfn(layer)` | After post-attention norm | FFN input study |
| `PostFfn(layer)` | After FFN, before residual | FFN output analysis |
| `PostLayer(layer)` | After residual add (residual stream) | Layer-wise analysis, SAE |
| `PreLmHead` | Final hidden state | Embedding analysis |
| `PostLmHead` | Raw logits | Logit analysis |

### IInferenceHook Interface

```
IInferenceHook:
  HookPoint → HookPoint
  OnActivation(ReadOnlySpan<float> activation, HookContext ctx) → HookResult

HookResult:
  Continue     — Read-only inspection, no modification
  Replace(Span<float>) — Replace activation (interventions, steering, ablation)

HookContext:
  LayerIndex: int
  TokenPosition: int
  SequenceId: int
  CurrentStep: int
```

### Zero-Cost When Disabled

Hooks disabled by default. Implementation:
```csharp
// Hot path — simple null/flag check, no allocation
if (_hooks is not null && _hooks.HasHookAt(HookPoint.PostLayer, layer))
    _hooks.Fire(HookPoint.PostLayer, layer, activation, ctx);
```

No event invocation, no delegate allocation, no virtual dispatch when hooks are unregistered.

### Performance Impact When Enabled

Enabling hooks requires materializing activations at hook points (data may live on GPU). Cost:
- GPU → CPU copy for inspection hooks
- Memory allocation for captured tensors
- Synchronous execution on inference thread

Intended for research/debugging, not production serving.

## Logprobs

### API Support

OpenAI-compatible `logprobs` and `top_logprobs` fields on `/v1/chat/completions`:

```
Request:
  logprobs: true
  top_logprobs: 5  (1-20, default 5)

Response (per token in choices[].logprobs.content):
  token: "Hello"
  logprob: -0.0012
  top_logprobs: [
    { token: "Hello", logprob: -0.0012 },
    { token: "Hi", logprob: -2.34 },
    { token: "Hey", logprob: -3.12 },
    ...
  ]
```

Implementation: after LM head produces logits, apply log-softmax and capture top-k before sampling. No hook system dependency — operates directly on the sampling pipeline output. Works in both non-streaming and SSE streaming modes.

### Chat UI Visualization

Opt-in in the built-in web chat (`dotllm serve`):

- **Color-coded token confidence**: green (>90%), lime (>70%), yellow (>50%), orange (>30%), red (<30%)
- **Hover**: shows top-k alternative tokens with probabilities
- **Three diagnostic cues**:
  - Low confidence: token probability < 10%
  - Ambiguity: gap between top-2 tokens < 0.15
  - Sampling effect: chosen token ≠ argmax (sampling diverged from greedy)

Inspired by the logprobs visualization approach from [kokosa.dev/blog/2026/logprobs](https://kokosa.dev/blog/2026/logprobs/).

### Sample Project

`DotLLM.Sample.Logprobs` — console application demonstrating logprobs API usage with colored terminal output, confidence analysis, and ambiguity detection.

## Built-in Diagnostic Tools

### Activation Capture (`CaptureHook`)

Collects activations at specified layers/positions into a buffer:
```
var capture = new CaptureHook(HookPoint.PostLayer, layers: [0, 15, 31]);
engine.RegisterHook(capture);
// ... run inference ...
var activations = capture.GetCaptured();  // Dictionary<(layer, position), float[]>
```

Configurable: capture all tokens or specific positions, max buffer size.

### Logit Lens

Projects intermediate hidden states through the LM head at each layer:
```
For layer L:
  hidden = capture at PostLayer(L)
  logits = hidden @ lm_head_weight
  probs = softmax(logits)
  top_tokens = argmax(probs, k=10)
```

Reveals how the model's "belief" about the next token evolves through layers. Useful for understanding which layers are most important for specific predictions.

**Chat UI visualization** (opt-in in `dotllm serve`):
- Layer-by-layer prediction heatmap showing top predicted tokens at each layer
- Click a token to see how the prediction evolved through layers
- Per-layer confidence visualization (how certain the model was at each depth)

Sample project: `DotLLM.Sample.LogitLens`.

### Attention Pattern Capture

When enabled, attention mechanism exports full weight matrix (softmax output):
```
attention_weights[layer][head][query_pos][key_pos]
```

Expensive: O(n²) per head per layer. Use selectively (specific layers, specific tokens).

### Sparse Autoencoder (SAE) Integration

SAEs decompose residual stream activations into interpretable sparse features.

**Workflow**:
1. Register `Replace` hook at `PostLayer(layer)`.
2. Hook encodes activation: `features = relu(activation @ W_enc + b_enc)` → sparse vector.
3. Analyze features: log which features are active, their magnitudes.
4. Optionally modify: zero out features (ablation), amplify features (steering).
5. Decode back: `modified_activation = features @ W_dec + b_dec`.
6. Return modified activation to inference pipeline.

**SAE Loading**: Pre-trained SAEs loaded from SafeTensors. Each SAE targets a specific layer and has: encoder weight, encoder bias, decoder weight, decoder bias, feature labels (optional).

**SAE Format**: SafeTensors with `cfg.json` (SAELens convention). The `cfg.json` specifies architecture parameters (`d_in`, `d_sae`, activation function, hook point, layer index).

**SAE Discovery**: Manual path specification is the primary method. Optional curated registry: a shipped JSON file mapping base models to known SAE repositories (EleutherAI, Goodfire, Llama Scope). No unified programmatic API exists for SAE search across providers.

**Neuronpedia Integration**: [Neuronpedia](https://neuronpedia.org/) (MIT-licensed) provides auto-generated feature explanations for 50M+ features across 30+ models. `GET /api/feature/{modelId}/{source}/{index}` returns explanation text, top/bottom logits, and activation examples (no auth required). `POST /api/search-topk-by-token` finds top-k active features per token. Bulk S3 export available for offline label cache. Covers Llama 3.1-8B, Gemma 2/3, Qwen 2.5/3, DeepSeek R1. Integration is optional — dotLLM works offline with local SAEs; Neuronpedia enriches with human-readable labels when available.

```
ISparseAutoencoder:
  Encode(activation) → SparseFeatures
  Decode(features) → activation
  FeatureCount → int
```

Sample project: `DotLLM.Sample.Interpretability` demonstrates full workflow.

**Chat UI visualization** (opt-in in `dotllm serve`):
- SAE configuration panel: load SAE for a specific layer, browse active features and their labels
- Feature visualization in generated text: highlight tokens by active feature magnitudes
- Feature steering controls: amplify or suppress specific features directly from the UI
