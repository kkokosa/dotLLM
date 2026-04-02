// ============================================================================
// dotLLM Chat UI — Vanilla JS + TailwindCSS
// ============================================================================

// ── STATE ──

const state = {
    messages: [],       // {role, content, stats?}
    config: null,       // from /props
    isGenerating: false,
    verbose: false,
    systemPrompt: '',
    abortController: null,
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
const modelInfo = $('#model-info');
const modelList = $('#model-list');
const modelLoading = $('#model-loading');
const refreshModelsBtn = $('#refresh-models-btn');
const applyConfigBtn = $('#apply-config-btn');

// ── API LAYER ──

async function fetchProps() {
    const res = await fetch('/props');
    return res.json();
}

async function fetchAvailableModels() {
    const res = await fetch('/v1/models/available');
    return res.json();
}

async function loadModel(model, quant) {
    const body = { model };
    if (quant) body.quant = quant;
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
                    if (chunk.usage || chunk.timings) {
                        yield { type: 'usage', usage: chunk.usage, timings: chunk.timings };
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

// ── UI RENDERING ──

function setStatus(text, color = 'text-zinc-500') {
    statusIndicator.textContent = text;
    statusIndicator.className = `text-xs ${color}`;
}

function hideWelcome() {
    welcomeEl.classList.add('hidden');
}

function addMessageToDOM(role, content, stats) {
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
    }

    bar.textContent = parts.length ? `[${parts.join(' | ')}]` : '';

    // Verbose: full breakdown
    if (state.verbose && stats.timings) {
        const detail = document.createElement('div');
        detail.className = 'mt-1 text-zinc-600';
        const t = stats.timings;
        detail.innerHTML = [
            `prefill: ${t.prefill_time_ms?.toFixed(1)}ms (${t.prompt_tokens} tok)`,
            `decode: ${t.decode_time_ms?.toFixed(1)}ms (${t.generated_tokens} tok)`,
            `sampling: ${t.sampling_time_ms?.toFixed(1)}ms`,
        ].join(' &middot; ');
        bar.appendChild(detail);
    }

    return bar;
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

function renderModelInfo() {
    if (!state.config) return;
    const c = state.config;
    modelInfo.innerHTML = [
        `<div><span class="stat-label">Model:</span> ${esc(c.model_id)}</div>`,
        `<div><span class="stat-label">Arch:</span> ${esc(c.architecture)} ${c.num_layers}L/${c.hidden_size}H</div>`,
        `<div><span class="stat-label">Vocab:</span> ${c.vocab_size?.toLocaleString() ?? '?'}</div>`,
        `<div><span class="stat-label">Context:</span> ${c.max_sequence_length?.toLocaleString() ?? '?'}</div>`,
        `<div><span class="stat-label">Device:</span> ${esc(c.device)}${c.gpu_layers ? ` (${c.gpu_layers} GPU layers)` : ''} | ${c.threads} threads</div>`,
    ].join('');
}

async function renderModelList() {
    modelList.innerHTML = '<div class="text-xs text-zinc-600">Loading...</div>';
    try {
        const data = await fetchAvailableModels();
        const models = data.models || [];
        if (models.length === 0) {
            modelList.innerHTML = '<div class="text-xs text-zinc-600">No local models found. Use <code>dotllm model pull</code>.</div>';
            return;
        }
        modelList.innerHTML = '';
        for (const m of models) {
            const item = document.createElement('div');
            item.className = 'model-item text-xs';
            const sizeMB = (m.size_bytes / 1048576).toFixed(0);
            const isCurrent = state.config?.model_path === m.full_path;
            if (isCurrent) item.classList.add('active');
            item.innerHTML = `<div class="text-zinc-300 truncate">${esc(m.repo_id)}/${esc(m.filename)}</div>` +
                `<div class="text-zinc-600">${sizeMB} MB${isCurrent ? ' <span class="text-accent">(loaded)</span>' : ''}</div>`;
            if (!isCurrent) {
                item.addEventListener('click', () => handleModelLoad(m.repo_id, m.filename));
            }
            modelList.appendChild(item);
        }
    } catch {
        modelList.innerHTML = '<div class="text-xs text-red-400">Failed to load models</div>';
    }
}

async function handleModelLoad(repoId, filename) {
    if (state.isGenerating) return;
    if (!confirm(`Load model ${repoId}/${filename}?`)) return;

    modelLoading.classList.remove('hidden');
    setStatus('Loading model...', 'text-yellow-500');

    try {
        const res = await loadModel(repoId);
        if (res.ok) {
            // Refresh state
            state.config = await fetchProps();
            modelBadge.textContent = state.config.model_id;
            renderModelInfo();
            syncSettingsFromState();
            await renderModelList();
            setStatus('Ready', 'text-emerald-500');
        } else {
            const err = await res.json().catch(() => ({}));
            setStatus(`Load failed: ${err.error || res.status}`, 'text-red-400');
        }
    } catch (e) {
        setStatus('Load failed', 'text-red-400');
    } finally {
        modelLoading.classList.add('hidden');
    }
}

// ── STREAMING HANDLER ──

async function handleSend() {
    const text = userInput.value.trim();
    if (!text || state.isGenerating) return;

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
    sendBtn.disabled = true;
    setStatus('Generating...', 'text-accent');

    const placeholder = createAssistantPlaceholder();
    let fullText = '';
    let tokenCount = 0;
    let usageData = null;
    let timingsData = null;
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

    // Add stats bar
    const statsInfo = { usage: usageData, timings: timingsData, ttftMs };
    const statsBar = createStatsBar(statsInfo);
    placeholder.bubble.appendChild(statsBar);

    // Save to state
    state.messages.push({ role: 'assistant', content: fullText, stats: statsInfo });

    // Reset UI
    state.isGenerating = false;
    sendBtn.classList.remove('hidden');
    stopBtn.classList.add('hidden');
    sendBtn.disabled = false;
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

// Model list refresh
refreshModelsBtn.addEventListener('click', () => renderModelList());

// Keyboard shortcut: Escape to close settings
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeSettings();
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
        modelBadge.textContent = state.config.model_id;
        syncSettingsFromState();
        renderModelInfo();
        loadConversation();
        setStatus('Ready', 'text-emerald-500');
        userInput.focus();
    } catch (e) {
        setStatus('Connection failed', 'text-red-400');
        modelBadge.textContent = 'offline';
    }
}

init();
