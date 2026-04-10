// ============================================================================
// dotLLM Chat UI — Vanilla JS + TailwindCSS
// ============================================================================

// ── STATE ──

const state = {
    messages: [],       // {role, content, stats?, rawPrompt?, rawResponse?}
    config: null,       // from /props
    isGenerating: false,
    verbose: true,
    showLogprobs: false,
    systemPrompt: '',
    abortController: null,
    toolsEnabled: false,
    toolsJson: '',
    awaitingToolResults: false,
    // Modal state
    modalSelectedRepo: null,
    modalSelectedFile: null,
    modalSelectedFilename: null,
};

const SAMPLE_TOOLS_JSON = `[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a location.",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "City name, e.g. 'Warsaw' or 'New York'"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "Temperature unit"
          }
        },
        "required": ["location"]
      },
      "response_example": {
        "temperature": 22,
        "unit": "celsius",
        "condition": "sunny",
        "humidity": 45
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_time",
      "description": "Get the current date and time in a given timezone.",
      "parameters": {
        "type": "object",
        "properties": {
          "timezone": {
            "type": "string",
            "description": "IANA timezone, e.g. 'Europe/Warsaw' or 'America/New_York'"
          }
        },
        "required": ["timezone"]
      },
      "response_example": {
        "datetime": "2026-04-03T14:30:00+02:00",
        "timezone": "Europe/Warsaw"
      }
    }
  }
]`;

// ── DOM REFS ──

const $ = (sel) => document.querySelector(sel);
const messagesEl = $('#messages');
const welcomeEl = $('#welcome');
const userInput = $('#user-input');
const sendBtn = $('#send-btn');
const stopBtn = $('#stop-btn');
const clearBtn = $('#clear-btn');
const exportBtn = $('#export-btn');
const exportDropdown = $('#export-dropdown');
const exportDownloadBtn = $('#export-download-btn');
const themeToggleBtn = $('#theme-toggle-btn');
const settingsBtn = $('#settings-btn');
const settingsPanel = $('#settings-panel');
const settingsOverlay = $('#settings-overlay');
const settingsClose = $('#settings-close');
const toolsBtn = $('#tools-btn');
const toolsPanel = $('#tools-panel');
const toolsOverlay = $('#tools-overlay');
const toolsClose = $('#tools-close');
const toolsEnabled = $('#tools-enabled');
const toolsJsonInput = $('#tools-json');
const toolsJsonError = $('#tools-json-error');
const toolsResetBtn = $('#tools-reset-btn');
const toolsValidateBtn = $('#tools-validate-btn');
const modelBadge = $('#model-badge');
const statusIndicator = $('#status-indicator');
const systemPromptBar = $('#system-prompt-bar');
const systemPromptInput = $('#system-prompt');
const systemPromptToggle = $('#system-prompt-toggle');
const systemPromptClear = $('#system-prompt-clear');
const applyConfigBtn = $('#apply-config-btn');
const reloadModelBtn = $('#reload-model-btn');

// Modal refs
const modelModalOverlay = $('#model-modal-overlay');
const modelModalClose = $('#model-modal-close');
const modalModelSelect = $('#modal-model-select');
const modalModelInfo = $('#modal-model-info');
const modalOptions = $('#modal-options');
const modalGpuSection = $('#modal-gpu-section');
const modalGpuLayers = $('#modal-gpu-layers');
const modalGpuLayersMax = $('#modal-gpu-layers-max');
const modalGpuLayersVal = $('#modal-gpu-layers-val');
const modalSizeEstimate = $('#modal-size-estimate');
const modalCacheK = $('#modal-cache-k');
const modalCacheV = $('#modal-cache-v');
const modalThreads = $('#modal-threads');
const modalDecodeThreads = $('#modal-decode-threads');
const modalStatus = $('#modal-status');
const modalCancelBtn = $('#modal-cancel-btn');
const modalLoadBtn = $('#modal-load-btn');
const modalSpeculativeSection = $('#modal-speculative-section');
const modalSpeculativeSelect = $('#modal-speculative-select');
const modalSpeculativeInfo = $('#modal-speculative-info');
const modalSpeculativeKSection = $('#modal-speculative-k-section');
const modalSpeculativeK = $('#modal-speculative-k');
const modalSpeculativeKVal = $('#modal-speculative-k-val');

// ── API LAYER ──

async function fetchProps() {
    const res = await fetch('/props');
    return res.json();
}

async function fetchAvailableModels() {
    const res = await fetch('/v1/models/available');
    return res.json();
}

async function inspectModel(fullPath) {
    const res = await fetch(`/v1/models/inspect?path=${encodeURIComponent(fullPath)}`);
    return res.ok ? res.json() : null;
}

async function loadModel(model, quant, opts) {
    const body = { model };
    if (quant) body.quant = quant;
    if (opts?.device) body.device = opts.device;
    if (opts?.device === 'gpu' && opts?.gpuLayers != null) body.gpu_layers = opts.gpuLayers;
    if (opts?.cacheTypeK && opts.cacheTypeK !== 'f32') body.cache_type_k = opts.cacheTypeK;
    if (opts?.cacheTypeV && opts.cacheTypeV !== 'f32') body.cache_type_v = opts.cacheTypeV;
    if (opts?.threads) body.threads = opts.threads;
    if (opts?.decodeThreads) body.decode_threads = opts.decodeThreads;
    if (opts?.speculativeModel) body.speculative_model = opts.speculativeModel;
    if (opts?.speculativeK) body.speculative_k = opts.speculativeK;
    const res = await fetch('/v1/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    return res;
}

async function updateConfig(params) {
    const res = await fetch('/v1/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
    });
    return res.json();
}

async function* streamChat(messages, params) {
    const controller = new AbortController();
    state.abortController = controller;

    const body = {
        messages,
        stream: true,
        temperature: params.temperature,
        top_p: params.top_p,
        top_k: params.top_k,
        min_p: params.min_p,
        max_tokens: params.max_tokens,
    };
    if (params.repetition_penalty && params.repetition_penalty !== 1.0) {
        body.repetition_penalty = params.repetition_penalty;
    }
    if (params.seed != null) {
        body.seed = params.seed;
    }
    if (params.tools) {
        body.tools = params.tools;
    }
    if (params.logprobs) {
        body.logprobs = true;
        body.top_logprobs = params.top_logprobs || 5;
    }

    const response = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed.startsWith('data: ')) continue;
                const data = trimmed.slice(6);
                if (data === '[DONE]') return;

                try {
                    const chunk = JSON.parse(data);
                    const choice = chunk.choices?.[0];

                    if (choice?.delta?.content) {
                        yield { type: 'delta', content: choice.delta.content, logprobs: choice?.logprobs?.content };
                    }
                    if (choice?.delta?.tool_calls) {
                        yield { type: 'tool_calls', toolCalls: choice.delta.tool_calls };
                    }
                    if (choice?.finish_reason) {
                        yield { type: 'finish', reason: choice.finish_reason };
                    }
                    if (chunk.usage || chunk.timings || chunk.prompt) {
                        yield { type: 'usage', usage: chunk.usage, timings: chunk.timings, prompt: chunk.prompt };
                    }
                } catch { /* skip malformed chunks */ }
            }
        }
    } finally {
        state.abortController = null;
    }
}

// ── MARKDOWN RENDERING ──

function renderMarkdown(text) {
    // Escape HTML
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');

    // Code blocks (```lang\n...\n```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Italic
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Links
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Unordered lists (simple single-level)
    html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);

    // Ordered lists
    html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

    // Paragraphs (double newline)
    html = html.replace(/\n\n+/g, '</p><p>');

    // Single newlines (not inside pre)
    html = html.replace(/(?<!<\/pre>)\n(?!<pre)/g, '<br>');

    return `<p>${html}</p>`;
}

// ── LOGPROBS RENDERING ──

function logprobToColor(logprob) {
    const p = Math.exp(logprob);
    if (p > 0.9) return 'rgba(34,197,94,0.2)';   // green
    if (p > 0.7) return 'rgba(132,204,22,0.2)';   // lime
    if (p > 0.5) return 'rgba(234,179,8,0.2)';    // yellow
    if (p > 0.3) return 'rgba(249,115,22,0.2)';   // orange
    return 'rgba(239,68,68,0.2)';                   // red
}

function logprobDiagnosticClass(entry) {
    const classes = [];
    const p = Math.exp(entry.logprob);
    if (p < 0.1) classes.push('lp-low-confidence');
    if (entry.top_logprobs && entry.top_logprobs.length >= 2) {
        const p1 = Math.exp(entry.top_logprobs[0].logprob);
        const p2 = Math.exp(entry.top_logprobs[1].logprob);
        if (p1 - p2 < 0.15) classes.push('lp-ambiguous');
    }
    if (entry.top_logprobs && entry.top_logprobs.length > 0 && entry.token !== entry.top_logprobs[0].token) {
        classes.push('lp-sampling-effect');
    }
    return classes.join(' ');
}

function renderTokensWithLogprobs(tokenEntries) {
    let html = '';
    for (const entry of tokenEntries) {
        const color = logprobToColor(entry.logprob);
        const p = (Math.exp(entry.logprob) * 100).toFixed(1);
        const diagClass = logprobDiagnosticClass(entry);
        const escapedToken = entry.text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        const displayToken = escapedToken.replace(/\n/g, '<br>');
        html += `<span class="lp-token ${diagClass}" style="background-color:${color}" data-logprob="${entry.logprob}" data-prob="${p}">${displayToken}</span>`;
    }
    return html;
}

function createLogprobTooltip(entry, spanEl) {
    const existing = document.querySelector('.lp-tooltip');
    if (existing) existing.remove();

    const p = (Math.exp(entry.logprob) * 100).toFixed(1);
    let html = `<div class="lp-tooltip-header">"${escapeHtml(entry.text)}" \u2014 logprob: ${entry.logprob.toFixed(3)} (p=${p}%)</div>`;
    if (entry.top_logprobs && entry.top_logprobs.length > 0) {
        html += '<div class="lp-tooltip-divider"></div>';
        html += '<table class="lp-tooltip-table">';
        for (let i = 0; i < entry.top_logprobs.length; i++) {
            const alt = entry.top_logprobs[i];
            const altP = (Math.exp(alt.logprob) * 100).toFixed(1);
            const isCurrent = alt.token === entry.token;
            const cls = isCurrent ? ' class="lp-tooltip-current"' : '';
            const marker = isCurrent ? '\u25b6' : '';
            html += `<tr${cls}>`
                + `<td class="lp-tt-marker">${marker}</td>`
                + `<td class="lp-tt-rank">${i+1}.</td>`
                + `<td class="lp-tt-token">${escapeHtml(alt.token)}</td>`
                + `<td class="lp-tt-logprob">${alt.logprob.toFixed(3)}</td>`
                + `<td class="lp-tt-prob">${altP}%</td>`
                + `</tr>`;
        }
        html += '</table>';
    }

    const tooltip = document.createElement('div');
    tooltip.className = 'lp-tooltip';
    tooltip.innerHTML = html;
    document.body.appendChild(tooltip);

    const rect = spanEl.getBoundingClientRect();
    tooltip.style.left = Math.min(rect.left, window.innerWidth - tooltip.offsetWidth - 10) + 'px';
    tooltip.style.top = (rect.bottom + 4) + 'px';
    return tooltip;
}

function escapeHtml(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function createLogprobsLegend() {
    const el = document.createElement('div');
    el.className = 'lp-legend';
    el.innerHTML = `<span class="lp-legend-label">logprobs</span>`;

    const tooltip = document.createElement('div');
    tooltip.className = 'lp-legend-tooltip';
    tooltip.innerHTML =
        `<div class="lp-legend-title">Token confidence</div>` +
        `<div class="lp-legend-row"><span class="lp-legend-swatch" style="background:rgba(34,197,94,0.35)"></span> &gt;90% — very confident</div>` +
        `<div class="lp-legend-row"><span class="lp-legend-swatch" style="background:rgba(132,204,22,0.35)"></span> 70–90% — confident</div>` +
        `<div class="lp-legend-row"><span class="lp-legend-swatch" style="background:rgba(234,179,8,0.35)"></span> 50–70% — moderate</div>` +
        `<div class="lp-legend-row"><span class="lp-legend-swatch" style="background:rgba(249,115,22,0.35)"></span> 30–50% — uncertain</div>` +
        `<div class="lp-legend-row"><span class="lp-legend-swatch" style="background:rgba(239,68,68,0.35)"></span> &lt;30% — low confidence</div>` +
        `<div class="lp-legend-divider"></div>` +
        `<div class="lp-legend-title">Diagnostic cues</div>` +
        `<div class="lp-legend-row"><span class="lp-legend-indicator lp-low-confidence">abc</span> red underline — p&lt;10%</div>` +
        `<div class="lp-legend-row"><span class="lp-legend-indicator lp-ambiguous">abc</span> dashed underline — top-2 gap&lt;15%</div>` +
        `<div class="lp-legend-row"><span class="lp-legend-indicator lp-sampling-effect">abc</span> wavy underline — chosen \u2260 argmax</div>` +
        `<div class="lp-legend-divider"></div>` +
        `<div class="lp-legend-hint">Hover any token to see top-K alternatives</div>`;
    el.appendChild(tooltip);
    return el;
}

// ── MODEL BADGE ──

function extractQuantFromPath(modelPath) {
    if (!modelPath) return null;
    // Match patterns like .Q4_K_M.gguf, .Q8_0.gguf, .IQ3_XXS.gguf, etc.
    const match = modelPath.match(/[.\-]((?:Q|IQ|F|BF)\w+)\.gguf$/i);
    return match ? match[1] : null;
}

function buildModelBadgeText(config) {
    if (!config || !config.is_ready) return 'no model loaded';

    const parts = [];

    // Model name
    parts.push(config.model_id || 'unknown');

    // Quant from filename
    const quant = extractQuantFromPath(config.model_path);
    if (quant) parts.push(quant);

    // Device label
    if (config.device === 'cpu' || (!config.device && !config.gpu_layers)) {
        const threads = config.threads || '?';
        parts.push(`CPU ${threads}t`);
    } else if (config.gpu_layers && config.num_layers && config.gpu_layers < config.num_layers) {
        parts.push(`Hybrid ${config.gpu_layers}/${config.num_layers} GPU layers`);
    } else {
        parts.push('GPU');
    }

    // Draft model indicator
    if (config.draft_model_path) {
        const draftFile = config.draft_model_path.split(/[/\\]/).pop() || '';
        const draftQuant = extractQuantFromPath(draftFile);
        const draftName = draftFile.replace(/\.gguf$/i, '').replace(/[.\-](Q|IQ|F|BF)\w+$/i, '');
        parts.push(`draft: ${draftName}${draftQuant ? ' ' + draftQuant : ''}`);
    }

    return parts.join(' | ');
}

function updateModelBadge() {
    modelBadge.textContent = buildModelBadgeText(state.config);
    updateSendButtonState();
}

function updateSendButtonState() {
    const ready = state.config?.is_ready;
    if (state.isGenerating || state.awaitingToolResults) {
        sendBtn.disabled = true;
    } else {
        sendBtn.disabled = !ready;
    }
    sendBtn.title = ready ? 'Send' : 'Load a model first';
}

// ── UI RENDERING ──

function setStatus(text, color = 'text-zinc-500') {
    statusIndicator.textContent = text;
    statusIndicator.className = `text-xs ${color}`;
}

function hideWelcome() {
    welcomeEl.classList.add('hidden');
}

function addMessageToDOM(role, content, stats, rawPrompt, rawResponse) {
    hideWelcome();

    const wrapper = document.createElement('div');
    wrapper.className = role === 'user'
        ? 'flex justify-end'
        : '';

    const bubble = document.createElement('div');
    bubble.className = role === 'user'
        ? 'bg-zinc-800 rounded-lg px-4 py-2.5 max-w-[85%] text-sm'
        : 'bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-2.5 max-w-full text-sm';

    // Role label
    const label = document.createElement('div');
    label.className = 'text-[10px] uppercase tracking-wider mb-1 ' +
        (role === 'user' ? 'text-zinc-500' : 'text-accent/60');
    label.textContent = role;
    bubble.appendChild(label);

    // Content
    const contentEl = document.createElement('div');
    contentEl.className = 'msg-content text-zinc-200 leading-relaxed';
    if (role === 'user') {
        contentEl.textContent = content;
    } else {
        contentEl.innerHTML = renderMarkdown(content);
    }
    bubble.appendChild(contentEl);

    // Stats
    if (stats) {
        bubble.appendChild(createStatsBar(stats));
    }

    // Verbose diagnostics: show full prompt as expandable details
    if (role === 'assistant' && state.verbose && (rawPrompt || rawResponse)) {
        bubble.appendChild(createVerboseDiagnostics(rawPrompt, rawResponse));
    }

    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    scrollToBottom();
    return { wrapper, bubble, contentEl };
}

function createAssistantPlaceholder() {
    hideWelcome();

    const wrapper = document.createElement('div');
    const bubble = document.createElement('div');
    bubble.className = 'bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-2.5 max-w-full text-sm';

    const label = document.createElement('div');
    label.className = 'text-[10px] uppercase tracking-wider mb-1 text-accent/60';
    label.textContent = 'assistant';
    bubble.appendChild(label);

    const contentEl = document.createElement('div');
    contentEl.className = 'msg-content text-zinc-200 leading-relaxed cursor-blink';
    bubble.appendChild(contentEl);

    // Live stats (during generation)
    const liveStats = document.createElement('div');
    liveStats.className = 'gen-live mt-1 hidden';
    bubble.appendChild(liveStats);

    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    scrollToBottom();
    return { wrapper, bubble, contentEl, liveStats };
}

function createStatsBar(stats) {
    const bar = document.createElement('div');
    bar.className = 'stats-bar';

    // ── Inline text (same as before) ──
    const parts = [];
    if (stats.usage) {
        parts.push(`${stats.usage.prompt_tokens} prompt`);
        parts.push(`${stats.usage.completion_tokens} gen`);
    }
    if (stats.timings?.cached_tokens > 0)
        parts.push(`${stats.timings.cached_tokens} cached`);
    if (stats.ttftMs != null)
        parts.push(`${stats.ttftMs.toFixed(0)}ms TTFT`);
    if (stats.timings) {
        if (stats.timings.prefill_tokens_per_sec > 0)
            parts.push(`${formatNum(stats.timings.prefill_tokens_per_sec)} pre t/s`);
        if (stats.timings.decode_tokens_per_sec > 0)
            parts.push(`${formatNum(stats.timings.decode_tokens_per_sec)} dec t/s`);
        if (stats.timings.prefill_time_ms != null)
            parts.push(`prefill: ${stats.timings.prefill_time_ms.toFixed(1)}ms`);
        if (stats.timings.decode_time_ms != null)
            parts.push(`decode: ${stats.timings.decode_time_ms.toFixed(0)}ms`);
        if (stats.timings.sampling_time_ms != null)
            parts.push(`sampling: ${stats.timings.sampling_time_ms.toFixed(1)}ms`);
        if (stats.timings.speculative_acceptance_rate > 0)
            parts.push(`spec: ${(stats.timings.speculative_acceptance_rate * 100).toFixed(0)}%`);
    }
    if (!parts.length) return bar;

    const textSpan = document.createElement('span');
    textSpan.textContent = `[${parts.join(' | ')}]`;
    bar.appendChild(textSpan);

    // ── Hover card ──
    const hover = document.createElement('div');
    hover.className = 'stats-hover';

    const t = stats.timings || {};
    const u = stats.usage || {};
    const promptTok = u.prompt_tokens || 0;
    const genTok = u.completion_tokens || 0;
    const cached = t.cached_tokens || 0;
    const newPrompt = promptTok - cached;
    const totalTok = promptTok + genTok;

    if (totalTok > 0) {
        // Build segments data
        const segs = [];
        let num = 1;
        if (cached > 0)    segs.push({ n: num++, cls: 'sh-seg-cached',  pct: cached / totalTok * 100,    tok: cached,    label: 'Cached tokens',   detail: 'reused from KV-cache',      rate: null });
        if (newPrompt > 0) segs.push({ n: num++, cls: 'sh-seg-prompt',  pct: newPrompt / totalTok * 100, tok: newPrompt, label: 'Prefill (new prompt)', detail: `${t.prefill_time_ms?.toFixed(0) ?? '?'}ms`, rate: t.prefill_tokens_per_sec > 0 ? `${formatNum(t.prefill_tokens_per_sec)} tok/s` : null });
        if (genTok > 0)    segs.push({ n: num++, cls: 'sh-seg-decode',  pct: genTok / totalTok * 100,    tok: genTok,    label: 'Generated tokens', detail: `${t.decode_time_ms?.toFixed(0) ?? '?'}ms`, rate: t.decode_tokens_per_sec > 0 ? `${formatNum(t.decode_tokens_per_sec)} tok/s` : null });

        // Segment bar with numbered markers
        let segHtml = '<div class="sh-seg-bar" style="position:relative">';
        let cumPct = 0;
        for (const s of segs) {
            segHtml += `<div class="sh-seg ${s.cls}" style="width:${s.pct.toFixed(1)}%"></div>`;
        }
        // Markers at segment midpoints
        cumPct = 0;
        for (const s of segs) {
            const mid = cumPct + s.pct / 2;
            segHtml += `<div class="sh-marker" style="left:calc(${mid.toFixed(1)}% - 8px)">${s.n}</div>`;
            cumPct += s.pct;
        }
        segHtml += '</div>';

        // Legend rows below
        segHtml += '<div class="sh-legend">';
        for (const s of segs) {
            const rateHtml = s.rate ? ` <span style="color:#71717a">@ ${s.rate}</span>` : '';
            segHtml += `<div class="sh-legend-row">` +
                `<div class="sh-legend-num">${s.n}</div>` +
                `<div class="sh-legend-color ${s.cls}"></div>` +
                `<span class="sh-legend-text">${s.label} <span style="color:#52525b">${s.detail}</span>${rateHtml}</span>` +
                `<span class="sh-legend-val">${s.tok.toLocaleString()}</span>` +
                `</div>`;
        }
        segHtml += '</div>';

        hover.innerHTML = segHtml;

        // TTFT + time row
        const metrics = [];
        if (stats.ttftMs != null)
            metrics.push({ val: `${stats.ttftMs.toFixed(0)}ms`, desc: 'Time to first token' });
        if (t.prefill_time_ms != null)
            metrics.push({ val: `${t.prefill_time_ms.toFixed(0)}ms`, desc: 'Prefill time' });
        if (t.decode_time_ms != null)
            metrics.push({ val: `${t.decode_time_ms.toFixed(0)}ms`, desc: 'Decode time' });
        if (t.sampling_time_ms != null && t.sampling_time_ms > 0.05)
            metrics.push({ val: `${t.sampling_time_ms.toFixed(1)}ms`, desc: 'Sampling time' });

        if (metrics.length) {
            const divider = document.createElement('div');
            divider.className = 'sh-divider';
            hover.appendChild(divider);

            const row = document.createElement('div');
            row.className = 'sh-row';
            for (const m of metrics) {
                row.innerHTML += `<div class="sh-metric"><span class="sh-val">${m.val}</span><span class="sh-desc">${m.desc}</span></div>`;
            }
            hover.appendChild(row);
        }

        // Speculative section
        if (t.speculative_draft_tokens > 0) {
            const accepted = t.speculative_accepted_tokens || 0;
            const drafted = t.speculative_draft_tokens || 0;
            const pct = ((t.speculative_acceptance_rate || 0) * 100).toFixed(0);
            const accPct = (accepted / drafted * 100).toFixed(1);

            const divider2 = document.createElement('div');
            divider2.className = 'sh-divider';
            hover.appendChild(divider2);

            const specSection = document.createElement('div');
            specSection.innerHTML =
                `<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">` +
                `<span>Speculative decoding</span>` +
                `<span class="sh-val">${pct}% accepted</span></div>` +
                `<div class="sh-spec-bar">` +
                `<div class="sh-spec-yes" style="width:${accPct}%"></div>` +
                `<div class="sh-spec-no" style="width:${(100 - parseFloat(accPct)).toFixed(1)}%"></div></div>` +
                `<div style="display:flex;justify-content:space-between;margin-top:2px">` +
                `<span class="sh-desc">${accepted} accepted</span>` +
                `<span class="sh-desc">${drafted - accepted} rejected</span></div>`;
            hover.appendChild(specSection);
        }
    }

    bar.appendChild(hover);
    return bar;
}

function createVerboseDiagnostics(rawPrompt, rawResponse) {
    const wrapper = document.createElement('div');
    wrapper.className = 'mt-2 border-t border-zinc-800 pt-2 space-y-1';

    // Raw prompt (after chat template)
    if (rawPrompt) {
        const promptDetails = document.createElement('details');
        promptDetails.className = '';
        const promptSummary = document.createElement('summary');
        promptSummary.className = 'text-[10px] uppercase tracking-wider text-zinc-600 cursor-pointer hover:text-zinc-400 select-none';
        promptSummary.textContent = 'Raw prompt (after template)';
        promptDetails.appendChild(promptSummary);
        const promptPre = document.createElement('pre');
        promptPre.className = 'mt-1 text-[10px] text-zinc-600 bg-zinc-950 border border-zinc-800 rounded p-2 overflow-x-auto max-h-60 overflow-y-auto whitespace-pre-wrap';
        promptPre.textContent = rawPrompt;
        promptDetails.appendChild(promptPre);
        wrapper.appendChild(promptDetails);
    }

    // Raw response
    if (rawResponse) {
        const respDetails = document.createElement('details');
        respDetails.className = '';
        const respSummary = document.createElement('summary');
        respSummary.className = 'text-[10px] uppercase tracking-wider text-zinc-600 cursor-pointer hover:text-zinc-400 select-none';
        respSummary.textContent = 'Raw response';
        respDetails.appendChild(respSummary);
        const respPre = document.createElement('pre');
        respPre.className = 'mt-1 text-[10px] text-zinc-600 bg-zinc-950 border border-zinc-800 rounded p-2 overflow-x-auto max-h-60 overflow-y-auto whitespace-pre-wrap';
        respPre.textContent = rawResponse;
        respDetails.appendChild(respPre);
        wrapper.appendChild(respDetails);
    }

    return wrapper;
}

function formatNum(n) {
    return n >= 1000 ? n.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ',') : n.toFixed(1);
}

function scrollToBottom() {
    const container = $('#chat-container');
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}

// ── SETTINGS PANEL ──

function openSettings() {
    settingsPanel.classList.add('open');
    settingsOverlay.classList.remove('hidden');
}

function closeSettings() {
    settingsPanel.classList.remove('open');
    settingsOverlay.classList.add('hidden');
}

function syncSettingsFromState() {
    const d = state.config?.sampling_defaults;
    if (!d) return;

    const setVal = (id, val, displayId, fmt) => {
        const el = $(`#${id}`);
        if (el) el.value = val;
        if (displayId) {
            const dEl = $(`#${displayId}`);
            if (dEl) dEl.textContent = fmt ? fmt(val) : val;
        }
    };

    setVal('opt-temperature', d.temperature, 'temp-val', v => Number(v).toFixed(2));
    setVal('opt-top-p', d.top_p, 'topp-val', v => Number(v).toFixed(2));
    setVal('opt-top-k', d.top_k, 'topk-val');
    setVal('opt-min-p', d.min_p, 'minp-val', v => Number(v).toFixed(2));
    setVal('opt-rep-penalty', d.repetition_penalty, 'rep-val', v => Number(v).toFixed(2));
    setVal('opt-max-tokens', d.max_tokens);
    setVal('opt-seed', d.seed ?? '');
    $('#opt-verbose').checked = state.verbose;
}

function getSettingsFromUI() {
    const logprobs = $('#opt-logprobs')?.checked ?? false;
    return {
        temperature: parseFloat($('#opt-temperature').value),
        top_p: parseFloat($('#opt-top-p').value),
        top_k: parseInt($('#opt-top-k').value) || 0,
        min_p: parseFloat($('#opt-min-p').value),
        repetition_penalty: parseFloat($('#opt-rep-penalty').value),
        max_tokens: parseInt($('#opt-max-tokens').value) || 2048,
        seed: $('#opt-seed').value ? parseInt($('#opt-seed').value) : null,
        logprobs: logprobs,
        top_logprobs: logprobs ? (parseInt($('#opt-top-logprobs')?.value) || 5) : 0,
    };
}

function updateRangeDisplays() {
    $('#temp-val').textContent = parseFloat($('#opt-temperature').value).toFixed(2);
    $('#topp-val').textContent = parseFloat($('#opt-top-p').value).toFixed(2);
    $('#topk-val').textContent = $('#opt-top-k').value;
    $('#minp-val').textContent = parseFloat($('#opt-min-p').value).toFixed(2);
    $('#rep-val').textContent = parseFloat($('#opt-rep-penalty').value).toFixed(2);
}

// ── TOOLS PANEL ──

function openTools() {
    toolsPanel.classList.add('open');
    toolsOverlay.classList.remove('hidden');
}

function closeTools() {
    toolsPanel.classList.remove('open');
    toolsOverlay.classList.add('hidden');
    // Save state
    state.toolsEnabled = toolsEnabled.checked;
    state.toolsJson = toolsJsonInput.value;
    saveConversation();
}

function validateToolsJson() {
    const text = toolsJsonInput.value.trim();
    toolsJsonError.classList.add('hidden');
    if (!text) return true;
    try {
        const parsed = JSON.parse(text);
        if (!Array.isArray(parsed)) {
            toolsJsonError.textContent = 'Must be a JSON array of tool definitions';
            toolsJsonError.classList.remove('hidden');
            return false;
        }
        return true;
    } catch (e) {
        toolsJsonError.textContent = `Invalid JSON: ${e.message}`;
        toolsJsonError.classList.remove('hidden');
        return false;
    }
}

function getToolsForRequest() {
    if (!state.toolsEnabled) return null;
    const text = toolsJsonInput.value.trim();
    if (!text) return null;
    try {
        const parsed = JSON.parse(text);
        if (!Array.isArray(parsed) || parsed.length === 0) return null;
        // Strip response_example before sending (not part of OpenAI spec)
        return parsed.map(t => ({
            ...t,
            function: {
                ...t.function,
                response_example: undefined,
            },
        }));
    } catch { /* invalid, skip */ }
    return null;
}

function getResponseExample(functionName) {
    try {
        const parsed = JSON.parse(toolsJsonInput.value.trim());
        if (!Array.isArray(parsed)) return null;
        const tool = parsed.find(t => t.function?.name === functionName);
        if (tool?.function?.response_example) {
            return JSON.stringify(tool.function.response_example);
        }
    } catch { /* ignore */ }
    return null;
}

function resetToolsJson() {
    toolsJsonInput.value = SAMPLE_TOOLS_JSON;
    state.toolsJson = SAMPLE_TOOLS_JSON;
    toolsJsonError.classList.add('hidden');
    saveConversation();
}

// ── MODEL LOAD MODAL ──

// Modal-local state
let modalModels = [];      // flat list from /v1/models/available
let modalInspect = null;   // inspect result for selected model
let modalSelectedFullPath = null;

function openModelModal() {
    modalSelectedFullPath = null;
    modalInspect = null;
    modalOptions.classList.add('hidden');
    modalGpuSection.classList.add('hidden');
    modalModelInfo.classList.add('hidden');
    modalSpeculativeSection.classList.add('hidden');
    modalSpeculativeSelect.innerHTML = '<option value="">None (disabled)</option>';
    modalSpeculativeInfo.classList.add('hidden');
    modalSpeculativeInfo.textContent = '';
    modalSpeculativeK.value = '5';
    modalSpeculativeKVal.textContent = '5';
    modalLoadBtn.disabled = true;
    modalStatus.innerHTML = '';
    modalCacheK.value = 'f32';
    modalCacheV.value = 'f32';
    document.querySelector('input[name="modal-device"][value="cpu"]').checked = true;
    modelModalOverlay.classList.remove('hidden');
    modelModalOverlay.style.display = 'flex';
    populateModalDropdown();
}

function closeModelModal() {
    modelModalOverlay.classList.add('hidden');
    modelModalOverlay.style.display = 'none';
}

function formatFileSize(bytes) {
    if (bytes == null) return '?';
    if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(2) + ' GB';
    return (bytes / 1048576).toFixed(0) + ' MB';
}

async function populateModalDropdown() {
    modalModelSelect.innerHTML = '<option value="">Loading...</option>';
    try {
        const data = await fetchAvailableModels();
        modalModels = data.models || [];

        if (modalModels.length === 0) {
            modalModelSelect.innerHTML = '<option value="">No models found — use dotllm model pull</option>';
            return;
        }

        // Group by repo
        const groups = {};
        for (const m of modalModels) {
            const repo = m.repo_id || 'local';
            if (!groups[repo]) groups[repo] = [];
            groups[repo].push(m);
        }

        modalModelSelect.innerHTML = '<option value="">Select a model...</option>';
        for (const [repo, files] of Object.entries(groups)) {
            const optgroup = document.createElement('optgroup');
            optgroup.label = repo;
            for (const f of files) {
                const opt = document.createElement('option');
                opt.value = f.full_path;
                const size = formatFileSize(f.size_bytes);
                const isCurrent = state.config?.model_path === f.full_path;
                opt.textContent = `${f.filename} (${size})${isCurrent ? ' ✓ loaded' : ''}`;
                opt.dataset.repo = repo;
                opt.dataset.filename = f.filename;
                opt.dataset.sizeBytes = f.size_bytes;
                optgroup.appendChild(opt);
            }
            modalModelSelect.appendChild(optgroup);
        }
    } catch {
        modalModelSelect.innerHTML = '<option value="">Failed to load models</option>';
    }
}

function populateSpeculativeDropdown(excludePath) {
    modalSpeculativeSelect.innerHTML = '<option value="">None (disabled)</option>';
    if (!modalModels || modalModels.length === 0) return;

    const groups = {};
    for (const m of modalModels) {
        if (m.full_path === excludePath) continue;
        const repo = m.repo_id || 'local';
        if (!groups[repo]) groups[repo] = [];
        groups[repo].push(m);
    }

    for (const [repo, files] of Object.entries(groups)) {
        const optgroup = document.createElement('optgroup');
        optgroup.label = repo;
        for (const f of files) {
            const opt = document.createElement('option');
            opt.value = f.full_path;
            const size = formatFileSize(f.size_bytes);
            opt.textContent = `${f.filename} (${size})`;
            opt.dataset.repo = repo;
            opt.dataset.filename = f.filename;
            optgroup.appendChild(opt);
        }
        modalSpeculativeSelect.appendChild(optgroup);
    }
}

async function onModalModelChange() {
    const fullPath = modalModelSelect.value;
    if (!fullPath) {
        modalOptions.classList.add('hidden');
        modalModelInfo.classList.add('hidden');
        modalSpeculativeSection.classList.add('hidden');
        modalLoadBtn.disabled = true;
        modalInspect = null;
        modalSelectedFullPath = null;
        return;
    }

    modalSelectedFullPath = fullPath;
    modalLoadBtn.disabled = false;
    modalModelInfo.classList.remove('hidden');
    modalModelInfo.textContent = 'Inspecting model...';

    // Inspect the selected model to get layer count
    modalInspect = await inspectModel(fullPath);
    if (modalInspect) {
        const size = formatFileSize(modalInspect.file_size_bytes);
        modalModelInfo.textContent = `${modalInspect.architecture} | ${modalInspect.num_layers} layers | ${modalInspect.hidden_size}H | ctx ${modalInspect.max_sequence_length?.toLocaleString() ?? '?'} | ${size}`;

        // Update GPU slider range
        modalGpuLayers.max = modalInspect.num_layers;
        modalGpuLayers.value = modalInspect.num_layers;
        modalGpuLayersMax.textContent = modalInspect.num_layers;
        updateGpuLayersDisplay();
    } else {
        modalModelInfo.textContent = 'Could not read model metadata';
    }

    modalOptions.classList.remove('hidden');
    updateGpuVisibility();

    // Show speculative section and populate draft model dropdown
    modalSpeculativeSection.classList.remove('hidden');
    populateSpeculativeDropdown(fullPath);
}

function getModalDevice() {
    return document.querySelector('input[name="modal-device"]:checked')?.value || 'cpu';
}

function updateGpuVisibility() {
    if (getModalDevice() === 'gpu') {
        modalGpuSection.classList.remove('hidden');
        updateGpuLayersDisplay();
    } else {
        modalGpuSection.classList.add('hidden');
    }
}

function updateGpuLayersDisplay() {
    const layers = parseInt(modalGpuLayers.value) || 0;
    const maxLayers = parseInt(modalGpuLayers.max) || 32;

    if (layers === 0) {
        modalGpuLayersVal.textContent = 'CPU only (no offloading)';
    } else if (layers >= maxLayers) {
        modalGpuLayersVal.textContent = `All ${maxLayers} layers on GPU`;
    } else {
        modalGpuLayersVal.textContent = `${layers}/${maxLayers} layers on GPU`;
    }

    // Size estimate
    if (modalInspect?.file_size_bytes) {
        const total = modalInspect.file_size_bytes;
        const frac = maxLayers > 0 ? layers / maxLayers : 0;
        const gpuBytes = total * frac;
        const cpuBytes = total - gpuBytes;
        modalSizeEstimate.innerHTML =
            `<span class="text-zinc-400">Estimated:</span> ` +
            `GPU ≈ ${formatFileSize(gpuBytes)} | RAM ≈ ${formatFileSize(cpuBytes)}` +
            `<span class="text-zinc-600"> (weights only, excludes KV-cache)</span>`;
    }
}

async function handleModalLoad() {
    if (!modalSelectedFullPath) return;

    const opt = modalModelSelect.selectedOptions[0];
    const repo = opt?.dataset.repo;
    const filename = opt?.dataset.filename;
    const quant = extractQuantFromPath(filename);
    const device = getModalDevice();
    const gpuLayers = device === 'gpu' ? parseInt(modalGpuLayers.value) : undefined;
    const threads = parseInt(modalThreads.value) || 0;
    const decodeThreads = parseInt(modalDecodeThreads.value) || 0;

    modalLoadBtn.disabled = true;
    modalCancelBtn.disabled = true;
    modalStatus.innerHTML = '<span class="spinner"></span> <span class="text-yellow-500">Loading and warming up model...</span>';
    setStatus('Loading and warming up model...', 'text-yellow-500');

    // Get speculative model selection
    const specPath = modalSpeculativeSelect.value || undefined;
    const specK = parseInt(modalSpeculativeK.value) || 5;

    try {
        const res = await loadModel(repo, quant, {
            device,
            gpuLayers,
            cacheTypeK: modalCacheK.value,
            cacheTypeV: modalCacheV.value,
            threads: threads || undefined,
            decodeThreads: decodeThreads || undefined,
            speculativeModel: specPath,
            speculativeK: specPath ? specK : undefined,
        });
        if (res.ok) {
            state.config = await fetchProps();
            updateModelBadge();
            syncSettingsFromState();
            setStatus('Ready', 'text-emerald-500');
            closeModelModal();
        } else {
            const err = await res.json().catch(() => ({}));
            const errMsg = err.error || `HTTP ${res.status}`;
            modalStatus.innerHTML = `<span class="text-red-400">Failed: ${esc(errMsg)}</span>`;
            setStatus(`Load failed: ${errMsg}`, 'text-red-400');
        }
    } catch (e) {
        modalStatus.innerHTML = `<span class="text-red-400">Failed: ${esc(e.message)}</span>`;
        setStatus('Load failed', 'text-red-400');
    } finally {
        modalLoadBtn.disabled = false;
        modalCancelBtn.disabled = false;
    }
}

// ── STREAMING HANDLER ──

const MAX_TOOL_ROUNDS = 5;

function buildApiMessages() {
    const apiMessages = [];
    if (state.systemPrompt) {
        apiMessages.push({ role: 'system', content: state.systemPrompt });
    }
    for (const m of state.messages) {
        const msg = { role: m.role, content: m.content };
        if (m.tool_calls) msg.tool_calls = m.tool_calls;
        if (m.tool_call_id) msg.tool_call_id = m.tool_call_id;
        // Assistant tool-call messages have null content per OpenAI spec
        if (m.role === 'assistant' && m.tool_calls) msg.content = null;
        apiMessages.push(msg);
    }
    return apiMessages;
}

async function handleSend() {
    const text = userInput.value.trim();
    if (!text || state.isGenerating || state.awaitingToolResults) return;
    if (!state.config?.is_ready) return;

    userInput.value = '';
    userInput.style.height = 'auto';

    // Add user message to state and DOM
    state.messages.push({ role: 'user', content: text });
    addMessageToDOM('user', text);

    await runGeneration(0);
}

async function runGeneration(round) {
    if (round >= MAX_TOOL_ROUNDS) {
        addSystemNote('Max tool call rounds reached');
        return;
    }

    const apiMessages = buildApiMessages();

    // Prepare for generation
    state.isGenerating = true;
    sendBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
    updateSendButtonState();
    setStatus(round > 0 ? `Generating (tool round ${round + 1})...` : 'Generating...', 'text-accent');

    const placeholder = createAssistantPlaceholder();
    let fullText = '';
    let tokenCount = 0;
    let usageData = null;
    let timingsData = null;
    let rawPrompt = null;
    let detectedToolCalls = null;
    let finishReason = null;
    const startTime = performance.now();
    let firstTokenTime = null;

    // Show live timer immediately (before first token)
    placeholder.liveStats.classList.remove('hidden');
    placeholder.liveStats.textContent = '0.0s';
    const liveTimer = setInterval(() => {
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        if (tokenCount === 0) {
            placeholder.liveStats.textContent = `${elapsed}s`;
        } else {
            placeholder.liveStats.textContent = `${tokenCount} tokens | ${elapsed}s`;
        }
    }, 100);

    // Get current sampling params + tools
    const params = getSettingsFromUI();
    const tools = getToolsForRequest();
    if (tools) params.tools = tools;

    const logprobEntries = [];
    const useLogprobs = state.showLogprobs && params.logprobs;

    try {
        for await (const event of streamChat(apiMessages, params)) {
            if (event.type === 'delta') {
                if (firstTokenTime === null) {
                    firstTokenTime = performance.now();
                }
                tokenCount++;
                fullText += event.content;
                if (useLogprobs && event.logprobs) {
                    for (const lp of event.logprobs) {
                        logprobEntries.push({ text: event.content, token: lp.token, logprob: lp.logprob, top_logprobs: lp.top_logprobs });
                    }
                    placeholder.contentEl.innerHTML = renderTokensWithLogprobs(logprobEntries);
                } else {
                    placeholder.contentEl.innerHTML = renderMarkdown(fullText);
                }
                scrollToBottom();
            } else if (event.type === 'tool_calls') {
                detectedToolCalls = event.toolCalls;
            } else if (event.type === 'finish') {
                finishReason = event.reason;
            } else if (event.type === 'usage') {
                usageData = event.usage;
                timingsData = event.timings;
                rawPrompt = event.prompt;
            }
        }
    } catch (e) {
        if (e.name === 'AbortError') {
            fullText += ' [stopped]';
        } else {
            fullText += ` [error: ${e.message}]`;
        }
    }

    // Finalize
    clearInterval(liveTimer);
    placeholder.contentEl.classList.remove('cursor-blink');
    placeholder.liveStats.remove();

    const ttftMs = firstTokenTime ? firstTokenTime - startTime : null;
    const statsInfo = { usage: usageData, timings: timingsData, ttftMs };

    // Tool call detected?
    if (detectedToolCalls && detectedToolCalls.length > 0 && finishReason === 'tool_calls') {
        // Replace streamed text with clean tool call display
        placeholder.contentEl.innerHTML = '';
        const toolCallEl = createToolCallDisplay(detectedToolCalls);
        placeholder.contentEl.appendChild(toolCallEl);

        placeholder.bubble.appendChild(createStatsBar(statsInfo));
        if (state.verbose) {
            placeholder.bubble.appendChild(createVerboseDiagnostics(rawPrompt, fullText));
        }

        // Save assistant message with tool_calls
        state.messages.push({
            role: 'assistant',
            content: fullText,
            tool_calls: detectedToolCalls,
            stats: statsInfo,
            rawPrompt,
            rawResponse: fullText,
        });
        saveConversation();

        // Show result inputs and wait for user
        const resultArea = createToolResultInputs(detectedToolCalls);
        placeholder.bubble.appendChild(resultArea.container);
        scrollToBottom();

        // Pause generation — user fills in tool results
        state.isGenerating = false;
        state.awaitingToolResults = true;
        sendBtn.classList.remove('hidden');
        sendBtn.disabled = true;
        stopBtn.classList.add('hidden');
        setStatus('Awaiting tool results...', 'text-yellow-500');

        // Wait for "Send Results" click
        const results = await resultArea.promise;
        state.awaitingToolResults = false;
        if (!results) {
            // User cancelled
            updateSendButtonState();
            setStatus('Ready', 'text-emerald-500');
            userInput.focus();
            saveConversation();
            return;
        }

        // Add tool result messages to state and show them
        resultArea.container.remove();
        for (const r of results) {
            state.messages.push({
                role: 'tool',
                content: r.content,
                tool_call_id: r.tool_call_id,
            });
            addToolResultToDOM(r.functionName, r.content);
        }
        saveConversation();

        // Re-generate with tool results
        await runGeneration(round + 1);
        return;
    }

    // Normal (non-tool-call) completion
    if (useLogprobs && logprobEntries.length > 0) {
        placeholder.contentEl.innerHTML = renderTokensWithLogprobs(logprobEntries);
        // Insert legend between label and content
        const label = placeholder.bubble.querySelector('.text-accent\\/60');
        if (label) label.appendChild(createLogprobsLegend());
    } else {
        placeholder.contentEl.innerHTML = renderMarkdown(fullText);
    }
    placeholder.bubble.appendChild(createStatsBar(statsInfo));
    if (state.verbose) {
        placeholder.bubble.appendChild(createVerboseDiagnostics(rawPrompt, fullText));
    }

    const msgIdx = state.messages.length;
    placeholder.bubble.setAttribute('data-msg-idx', msgIdx);
    state.messages.push({
        role: 'assistant', content: fullText, stats: statsInfo, rawPrompt, rawResponse: fullText,
        logprobs: useLogprobs ? logprobEntries : null,
    });

    // Reset UI
    state.isGenerating = false;
    sendBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    updateSendButtonState();
    setStatus('Ready', 'text-emerald-500');
    userInput.focus();
    saveConversation();
}

// ── TOOL CALL UI ──

function createToolCallDisplay(toolCalls) {
    const container = document.createElement('div');
    container.className = 'space-y-2';

    for (const tc of toolCalls) {
        const card = document.createElement('div');
        card.className = 'bg-zinc-800/50 border border-zinc-700 rounded px-3 py-2';

        const header = document.createElement('div');
        header.className = 'text-[10px] uppercase tracking-wider text-yellow-500/80 mb-1';
        header.textContent = 'tool call';
        card.appendChild(header);

        const name = document.createElement('div');
        name.className = 'text-xs text-accent font-medium';
        name.textContent = tc.function.name;
        card.appendChild(name);

        if (tc.function.arguments) {
            const args = document.createElement('pre');
            args.className = 'mt-1 text-[10px] text-zinc-400 bg-zinc-950 rounded p-2 overflow-x-auto';
            try {
                args.textContent = JSON.stringify(JSON.parse(tc.function.arguments), null, 2);
            } catch {
                args.textContent = tc.function.arguments;
            }
            card.appendChild(args);
        }

        container.appendChild(card);
    }

    return container;
}

function createToolResultInputs(toolCalls) {
    const container = document.createElement('div');
    container.className = 'mt-3 border-t border-zinc-700 pt-3 space-y-2';

    const heading = document.createElement('div');
    heading.className = 'text-[10px] uppercase tracking-wider text-zinc-500 mb-2';
    heading.textContent = 'Tool results (pre-filled from response_example, edit as needed)';
    container.appendChild(heading);

    const inputs = [];
    for (const tc of toolCalls) {
        const row = document.createElement('div');
        row.className = 'flex items-start gap-2';

        const label = document.createElement('span');
        label.className = 'text-xs text-accent shrink-0 pt-1';
        label.textContent = tc.function.name;
        row.appendChild(label);

        const input = document.createElement('input');
        input.type = 'text';
        input.value = getResponseExample(tc.function.name) || '{}';
        input.className = 'flex-1 bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-300 font-mono focus:outline-none focus:border-zinc-500';
        input.dataset.callId = tc.id;
        input.dataset.functionName = tc.function.name;
        row.appendChild(input);
        inputs.push(input);

        container.appendChild(row);
    }

    const btnRow = document.createElement('div');
    btnRow.className = 'flex gap-2 mt-2';

    const sendResultsBtn = document.createElement('button');
    sendResultsBtn.className = 'flex-1 py-1.5 rounded text-xs bg-accent/20 text-accent border border-accent/30 hover:bg-accent/30 transition';
    sendResultsBtn.textContent = 'Send Results';
    btnRow.appendChild(sendResultsBtn);

    const skipBtn = document.createElement('button');
    skipBtn.className = 'px-3 py-1.5 rounded text-xs border border-zinc-700 text-zinc-500 hover:text-zinc-300 hover:border-zinc-500 transition';
    skipBtn.textContent = 'Skip';
    btnRow.appendChild(skipBtn);

    container.appendChild(btnRow);

    // Focus the first input
    setTimeout(() => inputs[0]?.focus(), 50);

    const promise = new Promise((resolve) => {
        sendResultsBtn.addEventListener('click', () => {
            const results = inputs.map(inp => ({
                tool_call_id: inp.dataset.callId,
                functionName: inp.dataset.functionName,
                content: inp.value.trim() || '{}',
            }));
            resolve(results);
        });
        skipBtn.addEventListener('click', () => resolve(null));
    });

    return { container, promise };
}

function addToolResultToDOM(functionName, content) {
    hideWelcome();
    const wrapper = document.createElement('div');
    const bubble = document.createElement('div');
    bubble.className = 'bg-zinc-900 border border-yellow-500/20 rounded-lg px-4 py-2 max-w-full text-sm';

    const label = document.createElement('div');
    label.className = 'text-[10px] uppercase tracking-wider mb-1 text-yellow-500/60';
    label.textContent = `tool result: ${functionName}`;
    bubble.appendChild(label);

    const contentEl = document.createElement('div');
    contentEl.className = 'text-xs text-zinc-400 font-mono';
    contentEl.textContent = content;
    bubble.appendChild(contentEl);

    wrapper.appendChild(bubble);
    messagesEl.appendChild(wrapper);
    scrollToBottom();
}

function addSystemNote(text) {
    const note = document.createElement('div');
    note.className = 'text-center text-[10px] text-zinc-600 py-1';
    note.textContent = text;
    messagesEl.appendChild(note);
    scrollToBottom();
}

function handleStop() {
    if (state.abortController) {
        state.abortController.abort();
    }
}

function handleClear() {
    state.messages = [];
    messagesEl.innerHTML = '';
    welcomeEl.classList.remove('hidden');
    saveConversation();
    // Clear server-side prompt cache so stale KV-cache state isn't reused
    fetch('/v1/cache/clear', { method: 'POST' }).catch(() => {});
}

// ── EXPORT ──

function buildExportMarkdown() {
    const includeStats = $('#exp-stats').checked;
    const includeRaw = $('#exp-raw').checked;
    const includeModel = $('#exp-model').checked;

    const lines = [];
    lines.push('# dotLLM Chat Export');
    lines.push(`*Exported: ${new Date().toISOString()}*`);
    lines.push('');

    // Model & configuration section
    if (includeModel && state.config) {
        lines.push('## Model Configuration');
        lines.push('');
        lines.push('| Parameter | Value |');
        lines.push('|-----------|-------|');
        if (state.config.model_id) lines.push(`| Model | ${state.config.model_id} |`);
        if (state.config.model_path) {
            const quant = extractQuantFromPath(state.config.model_path);
            if (quant) lines.push(`| Quantization | ${quant} |`);
        }
        if (state.config.architecture) lines.push(`| Architecture | ${state.config.architecture} |`);
        if (state.config.num_layers) lines.push(`| Layers | ${state.config.num_layers} |`);
        if (state.config.hidden_size) lines.push(`| Hidden size | ${state.config.hidden_size} |`);
        if (state.config.max_sequence_length) lines.push(`| Max context | ${state.config.max_sequence_length.toLocaleString()} |`);
        lines.push(`| Device | ${state.config.device || 'cpu'} |`);
        if (state.config.gpu_layers) lines.push(`| GPU layers | ${state.config.gpu_layers}/${state.config.num_layers} |`);
        if (state.config.threads) lines.push(`| Threads | ${state.config.threads} |`);
        if (state.config.draft_model_path) {
            const draftFile = state.config.draft_model_path.split(/[/\\]/).pop() || '';
            const draftQuant = extractQuantFromPath(draftFile);
            const draftName = draftFile.replace(/\.gguf$/i, '');
            lines.push(`| Draft model | ${draftName} (speculative decoding) |`);
        }
        lines.push('');

        // Sampling defaults
        const d = state.config.sampling_defaults;
        if (d) {
            lines.push('**Sampling:** ' + [
                `temp=${d.temperature}`,
                `top_p=${d.top_p}`,
                d.top_k ? `top_k=${d.top_k}` : null,
                d.min_p ? `min_p=${d.min_p}` : null,
                d.repetition_penalty !== 1.0 ? `rep=${d.repetition_penalty}` : null,
                `max_tokens=${d.max_tokens}`,
                d.seed != null ? `seed=${d.seed}` : null,
            ].filter(Boolean).join(', '));
            lines.push('');
        }
    }

    if (state.systemPrompt) {
        lines.push('## System Prompt');
        lines.push('');
        lines.push('```');
        lines.push(state.systemPrompt);
        lines.push('```');
        lines.push('');
    }

    // Conversation
    lines.push('## Conversation');
    lines.push('');

    for (const m of state.messages) {
        const label = m.role === 'user' ? '**User**' : '**Assistant**';
        lines.push(`### ${label}`);
        lines.push('');
        lines.push(m.content);
        lines.push('');

        // Stats
        if (includeStats && m.role === 'assistant' && m.stats) {
            const s = m.stats;
            const parts = [];
            if (s.usage) {
                parts.push(`${s.usage.prompt_tokens} prompt tokens`);
                parts.push(`${s.usage.completion_tokens} generated tokens`);
            }
            if (s.timings?.cached_tokens > 0) parts.push(`${s.timings.cached_tokens} cached tokens`);
            if (s.ttftMs != null) parts.push(`TTFT: ${s.ttftMs.toFixed(0)}ms`);
            if (s.timings) {
                const t = s.timings;
                if (t.prefill_tokens_per_sec > 0) parts.push(`prefill: ${formatNum(t.prefill_tokens_per_sec)} tok/s`);
                if (t.decode_tokens_per_sec > 0) parts.push(`decode: ${formatNum(t.decode_tokens_per_sec)} tok/s`);
                if (t.prefill_time_ms != null) parts.push(`prefill time: ${t.prefill_time_ms.toFixed(1)}ms`);
                if (t.decode_time_ms != null) parts.push(`decode time: ${t.decode_time_ms.toFixed(0)}ms`);
                if (t.sampling_time_ms != null) parts.push(`sampling time: ${t.sampling_time_ms.toFixed(1)}ms`);
                if (t.speculative_draft_tokens > 0) {
                    const pct = Math.min(100, (t.speculative_acceptance_rate || 0) * 100).toFixed(0);
                    parts.push(`spec: ${pct}% accepted (${t.speculative_draft_tokens} drafted, ${t.speculative_accepted_tokens} produced)`);
                }
            }
            if (parts.length) {
                lines.push(`> *Stats: ${parts.join(' | ')}*`);
                lines.push('');
            }
        }

        // Raw prompt / response
        if (includeRaw && m.role === 'assistant') {
            if (m.rawPrompt) {
                lines.push('<details><summary>Raw prompt (after template)</summary>');
                lines.push('');
                lines.push('```');
                lines.push(m.rawPrompt);
                lines.push('```');
                lines.push('</details>');
                lines.push('');
            }
            if (m.rawResponse) {
                lines.push('<details><summary>Raw response</summary>');
                lines.push('');
                lines.push('```');
                lines.push(m.rawResponse);
                lines.push('```');
                lines.push('</details>');
                lines.push('');
            }
        }
    }

    lines.push('---');
    lines.push('*Generated by [dotLLM](https://github.com/kkokosa/dotLLM)*');
    return lines.join('\n');
}

function downloadExport() {
    const md = buildExportMarkdown();
    const blob = new Blob([md], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    a.href = url;
    a.download = `dotllm-chat-${timestamp}.md`;
    a.click();
    URL.revokeObjectURL(url);
    exportDropdown.classList.add('hidden');
}

// ── PERSISTENCE (localStorage) ──

function saveConversation() {
    try {
        localStorage.setItem('dotllm-messages', JSON.stringify(
            state.messages.map(m => {
                const msg = { role: m.role, content: m.content };
                if (m.tool_calls) msg.tool_calls = m.tool_calls;
                if (m.tool_call_id) msg.tool_call_id = m.tool_call_id;
                return msg;
            })
        ));
        localStorage.setItem('dotllm-system-prompt', state.systemPrompt);
        localStorage.setItem('dotllm-verbose', state.verbose ? '1' : '0');
        localStorage.setItem('dotllm-tools-enabled', state.toolsEnabled ? '1' : '0');
        localStorage.setItem('dotllm-tools-json', state.toolsJson);
    } catch { /* localStorage full or unavailable */ }
}

function loadConversation() {
    try {
        const msgs = JSON.parse(localStorage.getItem('dotllm-messages') || '[]');
        state.systemPrompt = localStorage.getItem('dotllm-system-prompt') || '';
        state.verbose = localStorage.getItem('dotllm-verbose') !== '0';
        state.toolsEnabled = localStorage.getItem('dotllm-tools-enabled') === '1';
        const savedToolsJson = localStorage.getItem('dotllm-tools-json');
        // Auto-upgrade: if saved JSON is the old sample (no response_example), replace with new sample
        state.toolsJson = (!savedToolsJson || !savedToolsJson.includes('response_example'))
            ? SAMPLE_TOOLS_JSON : savedToolsJson;

        if (state.systemPrompt) {
            systemPromptInput.value = state.systemPrompt;
            systemPromptBar.classList.remove('hidden');
        }

        // Restore tools UI
        toolsEnabled.checked = state.toolsEnabled;
        toolsJsonInput.value = state.toolsJson;

        for (const m of msgs) {
            const stateMsg = { role: m.role, content: m.content };
            if (m.tool_calls) stateMsg.tool_calls = m.tool_calls;
            if (m.tool_call_id) stateMsg.tool_call_id = m.tool_call_id;
            state.messages.push(stateMsg);

            if (m.role === 'tool') {
                // Find function name from tool_call_id
                const fnName = m.tool_call_id || 'tool';
                addToolResultToDOM(fnName, m.content);
            } else if (m.role === 'assistant' && m.tool_calls) {
                // Show tool call display instead of raw text
                const wrapper = document.createElement('div');
                const bubble = document.createElement('div');
                bubble.className = 'bg-zinc-900 border border-zinc-800 rounded-lg px-4 py-2.5 max-w-full text-sm';
                const label = document.createElement('div');
                label.className = 'text-[10px] uppercase tracking-wider mb-1 text-accent/60';
                label.textContent = 'assistant';
                bubble.appendChild(label);
                bubble.appendChild(createToolCallDisplay(m.tool_calls));
                wrapper.appendChild(bubble);
                messagesEl.appendChild(wrapper);
                hideWelcome();
            } else {
                addMessageToDOM(m.role, m.content);
            }
        }
    } catch { /* corrupted data, ignore */ }
}

// ── EVENT HANDLERS ──

sendBtn.addEventListener('click', handleSend);
stopBtn.addEventListener('click', handleStop);
clearBtn.addEventListener('click', handleClear);

// Theme toggle — the initial class is applied by an inline script in <head>
// (before paint, to avoid a flash). This handler only flips the class and
// persists the choice; the CSS in app.css renders the other theme on update.
themeToggleBtn.addEventListener('click', () => {
    const root = document.documentElement;
    const goingLight = !root.classList.contains('light');
    root.classList.toggle('light', goingLight);
    root.classList.toggle('dark', !goingLight);
    try {
        localStorage.setItem('dotllm-theme', goingLight ? 'light' : 'dark');
    } catch { /* localStorage unavailable */ }
});

// Export dropdown toggle + download
exportBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    exportDropdown.classList.toggle('hidden');
});
exportDownloadBtn.addEventListener('click', downloadExport);
document.addEventListener('click', (e) => {
    if (!exportDropdown.contains(e.target) && e.target !== exportBtn) {
        exportDropdown.classList.add('hidden');
    }
});

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

settingsBtn.addEventListener('click', openSettings);
settingsClose.addEventListener('click', closeSettings);
settingsOverlay.addEventListener('click', closeSettings);

toolsBtn.addEventListener('click', openTools);
toolsClose.addEventListener('click', closeTools);
toolsOverlay.addEventListener('click', closeTools);
toolsResetBtn.addEventListener('click', resetToolsJson);
toolsValidateBtn.addEventListener('click', validateToolsJson);
toolsEnabled.addEventListener('change', () => {
    state.toolsEnabled = toolsEnabled.checked;
    saveConversation();
});
toolsJsonInput.addEventListener('input', () => {
    state.toolsJson = toolsJsonInput.value;
    toolsJsonError.classList.add('hidden');
});

// Range slider live display
for (const id of ['opt-temperature', 'opt-top-p', 'opt-min-p', 'opt-rep-penalty']) {
    $(`#${id}`)?.addEventListener('input', updateRangeDisplays);
}

// Apply config button
applyConfigBtn.addEventListener('click', async () => {
    const params = getSettingsFromUI();
    await updateConfig(params);
    // Refresh props to confirm
    state.config = await fetchProps();
    syncSettingsFromState();
    setStatus('Config updated', 'text-emerald-500');
});

// Verbose toggle
$('#opt-verbose').addEventListener('change', (e) => {
    state.verbose = e.target.checked;
    saveConversation();
});

$('#opt-logprobs').addEventListener('change', (e) => {
    state.showLogprobs = e.target.checked;
    const topkRow = $('#logprobs-topk-row');
    if (topkRow) topkRow.classList.toggle('hidden', !e.target.checked);
    saveConversation();
});

// Logprobs tooltip delegation
document.addEventListener('mouseover', (e) => {
    const tokenEl = e.target.closest('.lp-token');
    if (!tokenEl) return;
    const msgEl = tokenEl.closest('.msg-content');
    if (!msgEl) return;
    // Find the logprob entry index
    const allTokens = msgEl.querySelectorAll('.lp-token');
    let idx = -1;
    allTokens.forEach((el, i) => { if (el === tokenEl) idx = i; });
    // Find the message in state
    const bubble = tokenEl.closest('[data-msg-idx]');
    if (!bubble || idx < 0) return;
    const msgIdx = parseInt(bubble.dataset.msgIdx);
    const msg = state.messages[msgIdx];
    if (!msg || !msg.logprobs || !msg.logprobs[idx]) return;
    createLogprobTooltip(msg.logprobs[idx], tokenEl);
});
document.addEventListener('mouseout', (e) => {
    if (e.target.closest('.lp-token')) {
        const tooltip = document.querySelector('.lp-tooltip');
        if (tooltip) tooltip.remove();
    }
});

// System prompt
systemPromptToggle.addEventListener('click', () => {
    systemPromptBar.classList.toggle('hidden');
});
systemPromptInput.addEventListener('input', () => {
    state.systemPrompt = systemPromptInput.value;
    saveConversation();
});
systemPromptClear.addEventListener('click', () => {
    systemPromptInput.value = '';
    state.systemPrompt = '';
    systemPromptBar.classList.add('hidden');
    saveConversation();
});

// Model modal events
reloadModelBtn.addEventListener('click', openModelModal);
modelModalClose.addEventListener('click', closeModelModal);
modalCancelBtn.addEventListener('click', closeModelModal);
modelModalOverlay.addEventListener('click', (e) => {
    if (e.target === modelModalOverlay) closeModelModal();
});
modalLoadBtn.addEventListener('click', handleModalLoad);
modalModelSelect.addEventListener('change', onModalModelChange);

// Device radio toggle → show/hide GPU section
document.querySelectorAll('input[name="modal-device"]').forEach(radio => {
    radio.addEventListener('change', updateGpuVisibility);
});

// GPU layers slider live update
modalGpuLayers.addEventListener('input', updateGpuLayersDisplay);

// Speculative draft model selection — inspect & vocab check
modalSpeculativeSelect.addEventListener('change', async () => {
    const draftPath = modalSpeculativeSelect.value;
    if (!draftPath) {
        modalSpeculativeInfo.classList.add('hidden');
        modalSpeculativeInfo.textContent = '';
        modalSpeculativeKSection.classList.add('hidden');
        return;
    }

    modalSpeculativeInfo.classList.remove('hidden');
    modalSpeculativeInfo.textContent = 'Inspecting draft model...';

    const draftInfo = await inspectModel(draftPath);
    if (!draftInfo) {
        modalSpeculativeInfo.textContent = 'Could not read draft model metadata';
        modalSpeculativeInfo.className = 'mt-1.5 text-[10px] text-red-400';
        modalSpeculativeKSection.classList.add('hidden');
        return;
    }

    const size = formatFileSize(draftInfo.file_size_bytes);
    const mainVocab = modalInspect?.vocab_size;
    const draftVocab = draftInfo.vocab_size;
    const maxDiff = 128; // matches SpeculativeConstants.MaxVocabSizeDifference
    const diff = Math.abs((mainVocab || 0) - (draftVocab || 0));
    const modelInfo = `${draftInfo.architecture} | ${draftInfo.num_layers} layers | ${size}`;

    if (!mainVocab || !draftVocab) {
        // Can't compare — show info only
        modalSpeculativeInfo.innerHTML = modelInfo;
        modalSpeculativeInfo.className = 'mt-1.5 text-[10px] text-zinc-600';
        modalSpeculativeKSection.classList.remove('hidden');
    } else if (mainVocab === draftVocab) {
        // Exact match — green
        modalSpeculativeInfo.innerHTML = modelInfo +
            `<br><span class="text-emerald-400">Exact vocab match (${draftVocab.toLocaleString()} tokens)</span>`;
        modalSpeculativeInfo.className = 'mt-1.5 text-[10px] text-zinc-600';
        modalSpeculativeKSection.classList.remove('hidden');
    } else if (diff <= maxDiff) {
        // Close match — yellow, will work
        modalSpeculativeInfo.innerHTML = modelInfo +
            `<br><span class="text-yellow-400">Compatible — vocab differs by ${diff} tokens ` +
            `(${draftVocab.toLocaleString()} vs ${mainVocab.toLocaleString()}). ` +
            `Shared range used for comparison.</span>`;
        modalSpeculativeInfo.className = 'mt-1.5 text-[10px] text-zinc-600';
        modalSpeculativeKSection.classList.remove('hidden');
    } else {
        // Incompatible — red
        modalSpeculativeInfo.innerHTML = modelInfo +
            `<br><span class="text-red-400">Incompatible — vocab differs by ${diff.toLocaleString()} tokens ` +
            `(${draftVocab.toLocaleString()} vs ${mainVocab.toLocaleString()}). ` +
            `Max allowed: ${maxDiff}.</span>`;
        modalSpeculativeInfo.className = 'mt-1.5 text-[10px] text-zinc-600';
        modalSpeculativeKSection.classList.add('hidden');
    }
});

// Speculative K slider live update
modalSpeculativeK.addEventListener('input', () => {
    modalSpeculativeKVal.textContent = modalSpeculativeK.value;
});

// Keyboard shortcut: Escape to close settings or modal
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (!modelModalOverlay.classList.contains('hidden')) {
            closeModelModal();
        } else if (toolsPanel.classList.contains('open')) {
            closeTools();
        } else {
            closeSettings();
        }
    }
});

// ── HELPERS ──

function esc(s) {
    const div = document.createElement('div');
    div.textContent = s ?? '';
    return div.innerHTML;
}

// ── INIT ──

async function init() {
    setStatus('Connecting...', 'text-yellow-500');

    try {
        state.config = await fetchProps();
        updateModelBadge();
        syncSettingsFromState();
        loadConversation();
        setStatus('Ready', 'text-emerald-500');
        updateSendButtonState();
        userInput.focus();
    } catch (e) {
        setStatus('Connection failed', 'text-red-400');
        modelBadge.textContent = 'offline';
        updateSendButtonState();
    }
}

init();
