// ============================================================================
// dotLLM Chat UI — Vanilla JS + TailwindCSS
// ============================================================================

// ── STATE ──

const state = {
    messages: [],       // {role, content, stats?, rawPrompt?, rawResponse?}
    config: null,       // from /props
    isGenerating: false,
    verbose: false,
    systemPrompt: '',
    abortController: null,
    // Modal state
    modalSelectedRepo: null,
    modalSelectedFile: null,
    modalSelectedFilename: null,
};

// ── DOM REFS ──

const $ = (sel) => document.querySelector(sel);
const messagesEl = $('#messages');
const welcomeEl = $('#welcome');
const userInput = $('#user-input');
const sendBtn = $('#send-btn');
const stopBtn = $('#stop-btn');
const clearBtn = $('#clear-btn');
const settingsBtn = $('#settings-btn');
const settingsPanel = $('#settings-panel');
const settingsOverlay = $('#settings-overlay');
const settingsClose = $('#settings-close');
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
                        yield { type: 'delta', content: choice.delta.content };
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
        parts.push(`Hybrid ${config.gpu_layers}/${config.num_layers} GPU`);
    } else {
        parts.push('GPU');
    }

    return parts.join(' | ');
}

function updateModelBadge() {
    modelBadge.textContent = buildModelBadgeText(state.config);
    updateSendButtonState();
}

function updateSendButtonState() {
    const ready = state.config?.is_ready;
    if (state.isGenerating) {
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

    const parts = [];
    if (stats.usage) {
        parts.push(`${stats.usage.prompt_tokens} prompt`);
        parts.push(`${stats.usage.completion_tokens} gen`);
    }
    if (stats.ttftMs != null) {
        parts.push(`${stats.ttftMs.toFixed(0)}ms TTFT`);
    }
    if (stats.timings) {
        if (stats.timings.prefill_tokens_per_sec > 0) {
            parts.push(`${formatNum(stats.timings.prefill_tokens_per_sec)} pre t/s`);
        }
        if (stats.timings.decode_tokens_per_sec > 0) {
            parts.push(`${formatNum(stats.timings.decode_tokens_per_sec)} dec t/s`);
        }
        if (stats.timings.prefill_time_ms != null) {
            parts.push(`prefill: ${stats.timings.prefill_time_ms.toFixed(1)}ms`);
        }
        if (stats.timings.decode_time_ms != null) {
            parts.push(`decode: ${stats.timings.decode_time_ms.toFixed(0)}ms`);
        }
        if (stats.timings.sampling_time_ms != null) {
            parts.push(`sampling: ${stats.timings.sampling_time_ms.toFixed(1)}ms`);
        }
    }

    bar.textContent = parts.length ? `[${parts.join(' | ')}]` : '';
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
    return {
        temperature: parseFloat($('#opt-temperature').value),
        top_p: parseFloat($('#opt-top-p').value),
        top_k: parseInt($('#opt-top-k').value) || 0,
        min_p: parseFloat($('#opt-min-p').value),
        repetition_penalty: parseFloat($('#opt-rep-penalty').value),
        max_tokens: parseInt($('#opt-max-tokens').value) || 2048,
        seed: $('#opt-seed').value ? parseInt($('#opt-seed').value) : null,
    };
}

function updateRangeDisplays() {
    $('#temp-val').textContent = parseFloat($('#opt-temperature').value).toFixed(2);
    $('#topp-val').textContent = parseFloat($('#opt-top-p').value).toFixed(2);
    $('#topk-val').textContent = $('#opt-top-k').value;
    $('#minp-val').textContent = parseFloat($('#opt-min-p').value).toFixed(2);
    $('#rep-val').textContent = parseFloat($('#opt-rep-penalty').value).toFixed(2);
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

async function onModalModelChange() {
    const fullPath = modalModelSelect.value;
    if (!fullPath) {
        modalOptions.classList.add('hidden');
        modalModelInfo.classList.add('hidden');
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
    modalStatus.innerHTML = '<span class="spinner"></span> <span class="text-yellow-500">Loading model...</span>';
    setStatus('Loading model...', 'text-yellow-500');

    try {
        const res = await loadModel(repo, quant, {
            device,
            gpuLayers,
            cacheTypeK: modalCacheK.value,
            cacheTypeV: modalCacheV.value,
            threads: threads || undefined,
            decodeThreads: decodeThreads || undefined,
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

async function handleSend() {
    const text = userInput.value.trim();
    if (!text || state.isGenerating) return;
    if (!state.config?.is_ready) return;

    userInput.value = '';
    userInput.style.height = 'auto';

    // Build messages array
    const apiMessages = [];
    if (state.systemPrompt) {
        apiMessages.push({ role: 'system', content: state.systemPrompt });
    }
    for (const m of state.messages) {
        apiMessages.push({ role: m.role, content: m.content });
    }
    apiMessages.push({ role: 'user', content: text });

    // Add user message to state and DOM
    state.messages.push({ role: 'user', content: text });
    addMessageToDOM('user', text);

    // Prepare for generation
    state.isGenerating = true;
    sendBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
    updateSendButtonState();
    setStatus('Generating...', 'text-accent');

    const placeholder = createAssistantPlaceholder();
    let fullText = '';
    let tokenCount = 0;
    let usageData = null;
    let timingsData = null;
    let rawPrompt = null;
    const startTime = performance.now();
    let firstTokenTime = null;

    // Get current sampling params
    const params = getSettingsFromUI();

    try {
        for await (const event of streamChat(apiMessages, params)) {
            if (event.type === 'delta') {
                if (firstTokenTime === null) {
                    firstTokenTime = performance.now();
                }
                tokenCount++;
                fullText += event.content;
                // Update content progressively
                placeholder.contentEl.innerHTML = renderMarkdown(fullText);
                scrollToBottom();

                // Update live stats
                const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
                placeholder.liveStats.classList.remove('hidden');
                placeholder.liveStats.textContent = `${tokenCount} tokens | ${elapsed}s`;
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
    placeholder.contentEl.classList.remove('cursor-blink');
    placeholder.contentEl.innerHTML = renderMarkdown(fullText);
    placeholder.liveStats.remove();

    // Compute TTFT
    const ttftMs = firstTokenTime ? firstTokenTime - startTime : null;

    // Add stats bar (always full breakdown)
    const statsInfo = { usage: usageData, timings: timingsData, ttftMs };
    const statsBar = createStatsBar(statsInfo);
    placeholder.bubble.appendChild(statsBar);

    // Add verbose diagnostics if enabled (raw prompt + raw response)
    if (state.verbose) {
        placeholder.bubble.appendChild(createVerboseDiagnostics(rawPrompt, fullText));
    }

    // Save to state
    state.messages.push({ role: 'assistant', content: fullText, stats: statsInfo, rawPrompt, rawResponse: fullText });

    // Reset UI
    state.isGenerating = false;
    sendBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    updateSendButtonState();
    setStatus('Ready', 'text-emerald-500');
    userInput.focus();
    saveConversation();
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
}

// ── PERSISTENCE (localStorage) ──

function saveConversation() {
    try {
        localStorage.setItem('dotllm-messages', JSON.stringify(
            state.messages.map(m => ({ role: m.role, content: m.content }))
        ));
        localStorage.setItem('dotllm-system-prompt', state.systemPrompt);
        localStorage.setItem('dotllm-verbose', state.verbose ? '1' : '0');
    } catch { /* localStorage full or unavailable */ }
}

function loadConversation() {
    try {
        const msgs = JSON.parse(localStorage.getItem('dotllm-messages') || '[]');
        state.systemPrompt = localStorage.getItem('dotllm-system-prompt') || '';
        state.verbose = localStorage.getItem('dotllm-verbose') === '1';

        if (state.systemPrompt) {
            systemPromptInput.value = state.systemPrompt;
            systemPromptBar.classList.remove('hidden');
        }

        for (const m of msgs) {
            state.messages.push({ role: m.role, content: m.content });
            addMessageToDOM(m.role, m.content);
        }
    } catch { /* corrupted data, ignore */ }
}

// ── EVENT HANDLERS ──

sendBtn.addEventListener('click', handleSend);
stopBtn.addEventListener('click', handleStop);
clearBtn.addEventListener('click', handleClear);

userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});

settingsBtn.addEventListener('click', openSettings);
settingsClose.addEventListener('click', closeSettings);
settingsOverlay.addEventListener('click', closeSettings);

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

// Keyboard shortcut: Escape to close settings or modal
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        if (!modelModalOverlay.classList.contains('hidden')) {
            closeModelModal();
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
