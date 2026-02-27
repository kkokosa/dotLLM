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

```
ISparseAutoencoder:
  Encode(activation) → SparseFeatures
  Decode(features) → activation
  FeatureCount → int
```

Sample project: `DotLLM.Sample.Interpretability` demonstrates full workflow.
