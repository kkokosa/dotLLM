# Telemetry & Observability — dotLLM

## Metrics — IInferenceMetrics

All metrics via `System.Diagnostics.Metrics` (native .NET OpenTelemetry). Zero-cost when no listener.

### Throughput

| Metric | Type | Description |
|--------|------|-------------|
| `dotllm.tokens_per_second.prefill` | Gauge | Tokens/sec during prompt processing |
| `dotllm.tokens_per_second.decode` | Gauge | Tokens/sec during generation |
| `dotllm.requests.completed` | Counter | Total completed requests |
| `dotllm.tokens.generated` | Counter | Total tokens generated |
| `dotllm.tokens.prompt` | Counter | Total prompt tokens processed |

### Latency (Histograms)

| Metric | Description |
|--------|-------------|
| `dotllm.latency.time_to_first_token` | Request receipt → first generated token (TTFT) |
| `dotllm.latency.inter_token` | Time between consecutive tokens (ITL) |
| `dotllm.latency.request_duration` | Total request time including queue wait |
| `dotllm.latency.prefill_duration` | Time spent in prefill phase |

### Resource Utilization

| Metric | Description |
|--------|-------------|
| `dotllm.kv_cache.utilization` | Fraction of KV blocks in use |
| `dotllm.kv_cache.blocks.allocated` | Currently allocated blocks |
| `dotllm.kv_cache.blocks.free` | Available blocks |
| `dotllm.gpu.memory.used_bytes` | GPU memory in use |
| `dotllm.gpu.memory.total_bytes` | Total GPU memory |
| `dotllm.batch.size` | Active sequences in batch |
| `dotllm.queue.depth` | Requests waiting for admission |

### Scheduler

| Metric | Description |
|--------|-------------|
| `dotllm.scheduler.preemptions` | Sequence preemption count |
| `dotllm.prefix_cache.hit_ratio` | Prefix cache hit rate |
| `dotllm.prefix_cache.entries` | Cached prefix count |

## Implementation

```csharp
// Define meter once
private static readonly Meter s_meter = new("DotLLM.Engine");

// Create instruments
private static readonly Counter<long> s_tokensGenerated =
    s_meter.CreateCounter<long>("dotllm.tokens.generated");

private static readonly Histogram<double> s_ttft =
    s_meter.CreateHistogram<double>("dotllm.latency.time_to_first_token",
        unit: "s", description: "Time to first token");

// Record (zero-cost if no listener)
s_tokensGenerated.Add(1);
s_ttft.Record(elapsed.TotalSeconds);
```

## Request Tracing — IRequestTracer

Per-request distributed tracing via `System.Diagnostics.Activity` (OpenTelemetry-compatible).

### Trace Spans

```
dotllm.request                    (root)
├── dotllm.queue_wait             Time in scheduler queue
├── dotllm.tokenize               Prompt tokenization
├── dotllm.template               Chat template application
├── dotllm.prefix_lookup          Prefix cache lookup
├── dotllm.prefill                KV-cache computation
│   └── dotllm.layer.{n}         Per-layer (optional, verbose)
├── dotllm.decode                 Token generation loop
│   └── dotllm.sample            Sampling + constraint eval
└── dotllm.detokenize             Token-to-text
```

### Span Attributes

Each span carries: token counts, model name, adapter ID, constraint type, GPU device ID, batch position.

### Implementation

```csharp
private static readonly ActivitySource s_source = new("DotLLM.Engine");

using var activity = s_source.StartActivity("dotllm.prefill");
activity?.SetTag("dotllm.prompt_tokens", tokenCount);
activity?.SetTag("dotllm.model", modelName);
// ... do prefill ...
```

Zero-cost when no `ActivityListener` registered.

## Integration

- **Prometheus**: `OpenTelemetry.Exporter.Prometheus` package → `/metrics` endpoint.
- **Grafana**: Standard dashboards for LLM serving metrics.
- **Jaeger/Zipkin**: Trace visualization via OpenTelemetry trace exporters.
- **ASP.NET**: Automatically correlates HTTP request traces with inference spans.
