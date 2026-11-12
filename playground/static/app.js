// Tamil Tokenizer Playground Client Application

(function () {
  'use strict';

  // Sample test presets
  const PRESETS = {
    greeting: 'வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்? இன்று வானிலை நன்றாக உள்ளது.',
    kural: 'அகர முதல எழுத்தெல்லாம் ஆதி பகவன் முதற்றே உலகு.',
    tech: 'செயற்கை நுண்ணறிவு மற்றும் இயந்திர கற்றல் தொழில்நுட்பங்கள் தமிழ் மொழியில் வேகமாக வளர்ந்து வருகின்றன.',
    conjuncts: 'க்ஷ ஸ்ரீ க்ட ண்ட ஹலோ ஜன்னல் ஷங்கர் பக்ஷி',
    code: 'def greet(name: str) -> str:\n    # தமிழ் வாழ்த்துச் செய்தி\n    return f"வணக்கம் {name}! நலமா?"',
    tanglish: "வணக்கம் friends, today's video-ல we are going to learn modern LLM tokenization பற்றி."
  };

  // State
  let availableModels = [];
  let panels = ['tamil-120k', 'openai-o200k'];
  let panelViews = {}; // panelIndex -> 'tokens' | 'table'
  let latestResults = {};
  let debounceTimer = null;

  // DOM Elements
  const masterInput = document.getElementById('master-input');
  const panelsContainer = document.getElementById('panels-container');
  const addPanelBtn = document.getElementById('add-panel-btn');
  const clearInputBtn = document.getElementById('clear-input-btn');
  const themeToggle = document.getElementById('theme-toggle');
  const charCounter = document.getElementById('char-counter');
  const byteCounter = document.getElementById('byte-counter');
  const wordCounter = document.getElementById('word-counter');
  const popover = document.getElementById('token-popover');

  // Popover elements
  const popIndex = document.getElementById('pop-index');
  const popId = document.getElementById('pop-id');
  const popText = document.getElementById('pop-text');
  const popHex = document.getElementById('pop-hex');
  const popBytes = document.getElementById('pop-bytes');
  const popSpan = document.getElementById('pop-span');

  // Initialize
  async function init() {
    setupTheme();
    setupEventListeners();
    await fetchModels();
    renderPanels();

    // Load initial preset
    selectPreset('greeting');
  }

  function setupTheme() {
    const saved = localStorage.getItem('theme');
    if (saved) {
      document.documentElement.setAttribute('data-theme', saved);
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      document.documentElement.setAttribute('data-theme', 'light');
    }
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
  }

  async function fetchModels() {
    try {
      const resp = await fetch('/api/models');
      const data = await resp.json();
      availableModels = data.models || [];
    } catch (err) {
      console.error('Failed to load models list:', err);
      availableModels = [
        { id: 'tamil-120k', name: 'Tamil Tokenizer (120K)', category: 'This Project' },
        { id: 'tamil-30k', name: 'Tamil Tokenizer (30K)', category: 'This Project' },
        { id: 'openai-o200k', name: 'OpenAI GPT-4o (o200k_base)', category: 'OpenAI' },
        { id: 'openai-cl100k', name: 'OpenAI GPT-4 / 3.5 (cl100k_base)', category: 'OpenAI' },
        { id: 'gemma-4', name: 'Google Gemma 4 (262K)', category: 'Open LLMs' },
        { id: 'qwen-3.8', name: 'Alibaba Qwen 3.8 (248K)', category: 'Open LLMs' },
        { id: 'llama-4', name: 'Meta Llama 4 (201K)', category: 'Open LLMs' },
        { id: 'muse-glimmer', name: 'Meta Muse Glimmer (202K)', category: 'Open LLMs' },
        { id: 'indicbert', name: 'AI4Bharat IndicBERTv2 (256K)', category: 'Indic LLMs' },
        { id: 'sarvam-105b', name: 'Sarvam AI 105B (262K)', category: 'Indic LLMs' },
        { id: 'sarvam-1', name: 'Sarvam-1 (68K)', category: 'Indic LLMs' },
        { id: 'mbert', name: 'mBERT (119K)', category: 'Baselines' }
      ];
    }
  }

  function setupEventListeners() {
    masterInput.addEventListener('input', () => {
      updateInputCounters();
      scheduleTokenize();
    });

    clearInputBtn.addEventListener('click', () => {
      masterInput.value = '';
      updateInputCounters();
      scheduleTokenize();
      masterInput.focus();
    });

    addPanelBtn.addEventListener('click', () => {
      if (panels.length >= 4) {
        alert('Maximum of 4 comparison panels reached.');
        return;
      }
      // Pick a model not already displayed if possible
      const unused = availableModels.find(m => !panels.includes(m.id));
      panels.push(unused ? unused.id : 'openai-cl100k');
      renderPanels();
      tokenizeAll();
    });

    themeToggle.addEventListener('click', toggleTheme);

    // Preset chips
    document.querySelectorAll('.preset-chips .chip').forEach(chip => {
      chip.addEventListener('click', () => {
        document.querySelectorAll('.preset-chips .chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        const key = chip.dataset.preset;
        selectPreset(key);
      });
    });

    // Hide popover when mouse leaves document
    document.addEventListener('mousemove', (e) => {
      if (!e.target.closest('.token-span')) {
        hidePopover();
      }
    });
  }

  function selectPreset(key) {
    if (PRESETS[key]) {
      masterInput.value = PRESETS[key];
      updateInputCounters();
      tokenizeAll();
    }
  }

  function updateInputCounters() {
    const text = masterInput.value;
    const chars = text.length;
    const utf8Bytes = new TextEncoder().encode(text).length;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;

    charCounter.textContent = `${chars.toLocaleString()} chars`;
    byteCounter.textContent = `${utf8Bytes.toLocaleString()} bytes`;
    wordCounter.textContent = `${words.toLocaleString()} words`;
  }

  function scheduleTokenize() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(tokenizeAll, 40);
  }

  async function tokenizeAll() {
    const text = masterInput.value;
    const modelIds = [...new Set(panels)];

    if (!text) {
      latestResults = {};
      panels.forEach((_, idx) => renderPanelContent(idx, null));
      return;
    }

    // Set loading indicator
    panels.forEach((_, idx) => {
      const metricEl = document.getElementById(`metric-tokens-${idx}`);
      if (metricEl && !latestResults[panels[idx]]) {
        metricEl.innerHTML = '<span class="loading-indicator"></span>';
      }
    });

    try {
      const resp = await fetch('/api/tokenize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_ids: modelIds,
          text: text
        })
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}`);
      }

      const data = await resp.json();
      latestResults = data.results || {};

      panels.forEach((mid, idx) => {
        renderPanelContent(idx, latestResults[mid]);
      });
    } catch (err) {
      console.error('Tokenize failed:', err);
    }
  }

  function renderPanels() {
    panelsContainer.innerHTML = '';

    panels.forEach((modelId, idx) => {
      if (!panelViews[idx]) {
        panelViews[idx] = 'tokens';
      }

      const isProject = modelId.startsWith('tamil-');
      const panel = document.createElement('div');
      panel.className = `panel-card ${isProject ? 'is-project' : ''}`;
      panel.id = `panel-${idx}`;

      panel.innerHTML = `
        <div class="panel-header">
          <div class="model-select-wrapper">
            <select class="model-select" data-panel-index="${idx}">
              ${buildModelOptions(modelId)}
            </select>
            <span class="vocab-tag" id="vocab-tag-${idx}">--</span>
          </div>
          <div class="panel-actions">
            ${panels.length > 1 ? `<button class="btn-icon remove-panel-btn" data-panel-index="${idx}" title="Remove panel">✕</button>` : ''}
          </div>
        </div>

        <div class="metrics-strip">
          <div class="metric-item">
            <span class="metric-label">Tokens</span>
            <span class="metric-value" id="metric-tokens-${idx}">0</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Bytes / Tok</span>
            <span class="metric-value" id="metric-bytes-tok-${idx}">0.0</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Tok / Word</span>
            <span class="metric-value" id="metric-tok-word-${idx}">0.0</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Savings</span>
            <span class="metric-value highlight-diff" id="metric-savings-${idx}">--</span>
          </div>
        </div>

        <div class="view-nav">
          <div class="tab-buttons">
            <button class="tab-btn ${panelViews[idx] === 'tokens' ? 'active' : ''}" data-tab="tokens" data-panel-index="${idx}">Tokens</button>
            <button class="tab-btn ${panelViews[idx] === 'table' ? 'active' : ''}" data-tab="table" data-panel-index="${idx}">Table</button>
          </div>
          <button class="copy-tokens-btn" data-panel-index="${idx}" title="Copy token IDs as JSON">Copy IDs</button>
        </div>

        <div class="panel-body" id="panel-body-${idx}">
          <div class="empty-state">No text to tokenize</div>
        </div>
      `;

      panelsContainer.appendChild(panel);

      // Event listener for model selector
      panel.querySelector('.model-select').addEventListener('change', (e) => {
        panels[idx] = e.target.value;
        renderPanels();
        tokenizeAll();
      });

      // Event listener for remove panel
      const removeBtn = panel.querySelector('.remove-panel-btn');
      if (removeBtn) {
        removeBtn.addEventListener('click', () => {
          panels.splice(idx, 1);
          delete panelViews[idx];
          renderPanels();
          tokenizeAll();
        });
      }

      // Event listeners for tabs
      panel.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          panelViews[idx] = btn.dataset.tab;
          panel.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          renderPanelBodyOnly(idx, latestResults[panels[idx]]);
        });
      });

      // Event listener for copy IDs
      panel.querySelector('.copy-tokens-btn').addEventListener('click', () => {
        const res = latestResults[panels[idx]];
        if (res && res.tokens) {
          const ids = res.tokens.map(t => t.id);
          navigator.clipboard.writeText(JSON.stringify(ids));
          const copyBtn = panel.querySelector('.copy-tokens-btn');
          const originalText = copyBtn.textContent;
          copyBtn.textContent = 'Copied!';
          setTimeout(() => { copyBtn.textContent = originalText; }, 1200);
        }
      });
    });
  }

  function buildModelOptions(selectedId) {
    const categories = ['This Project', 'OpenAI', 'Open LLMs', 'Indic LLMs', 'Baselines'];
    let html = '';

    categories.forEach(cat => {
      const inCat = availableModels.filter(m => m.category === cat);
      if (inCat.length > 0) {
        html += `<optgroup label="${cat}">`;
        inCat.forEach(m => {
          const isSelected = m.id === selectedId ? 'selected' : '';
          html += `<option value="${m.id}" ${isSelected}>${m.name}</option>`;
        });
        html += `</optgroup>`;
      }
    });

    return html;
  }

  function renderPanelContent(idx, data) {
    const vocabTag = document.getElementById(`vocab-tag-${idx}`);
    const tokensEl = document.getElementById(`metric-tokens-${idx}`);
    const bytesTokEl = document.getElementById(`metric-bytes-tok-${idx}`);
    const tokWordEl = document.getElementById(`metric-tok-word-${idx}`);
    const savingsEl = document.getElementById(`metric-savings-${idx}`);

    if (!data || data.error) {
      if (vocabTag) vocabTag.textContent = '--';
      if (tokensEl) tokensEl.textContent = '0';
      if (bytesTokEl) bytesTokEl.textContent = '0.0';
      if (tokWordEl) tokWordEl.textContent = '0.0';
      if (savingsEl) savingsEl.textContent = '--';
      renderPanelBodyOnly(idx, null);
      return;
    }

    if (vocabTag) {
      vocabTag.textContent = `${(data.vocab_size || 0).toLocaleString()} vocab`;
    }

    if (tokensEl) {
      tokensEl.textContent = data.token_count.toLocaleString();
    }

    if (bytesTokEl) {
      bytesTokEl.textContent = data.bytes_per_token.toFixed(2);
      if (data.bytes_per_token >= 5.0) {
        bytesTokEl.classList.add('highlight-good');
      } else {
        bytesTokEl.classList.remove('highlight-good');
      }
    }

    if (tokWordEl) {
      tokWordEl.textContent = data.tokens_per_word.toFixed(2);
    }

    // Compute relative savings vs first baseline (OpenAI or another model)
    if (savingsEl) {
      calculateSavingsBadge(idx, data, savingsEl);
    }

    renderPanelBodyOnly(idx, data);
  }

  function calculateSavingsBadge(idx, currentData, savingsEl) {
    const currentCount = currentData.token_count;
    if (!currentCount) {
      savingsEl.textContent = '--';
      return;
    }

    // Compare against the first external baseline (e.g. o200k or cl100k)
    let baselineData = null;
    const baselineMid = panels.find(mid => !mid.startsWith('tamil-'));
    if (baselineMid && latestResults[baselineMid] && latestResults[baselineMid].token_count) {
      baselineData = latestResults[baselineMid];
    }

    if (baselineData && baselineData.token_count) {
      const baseCount = baselineData.token_count;
      const pct = Math.round(((baseCount - currentCount) / baseCount) * 100);

      if (pct > 0) {
        savingsEl.textContent = `-${pct}% tokens`;
        savingsEl.style.backgroundColor = 'var(--success-bg)';
        savingsEl.style.color = 'var(--success-color)';
      } else if (pct < 0) {
        savingsEl.textContent = `+${Math.abs(pct)}% tokens`;
        savingsEl.style.backgroundColor = 'rgba(248, 81, 73, 0.15)';
        savingsEl.style.color = 'var(--danger-color)';
      } else {
        savingsEl.textContent = 'Baseline';
        savingsEl.style.backgroundColor = 'var(--bg-tertiary)';
        savingsEl.style.color = 'var(--text-secondary)';
      }
    } else {
      savingsEl.textContent = `${currentCount} tok`;
      savingsEl.style.backgroundColor = 'var(--bg-tertiary)';
      savingsEl.style.color = 'var(--text-secondary)';
    }
  }

  function renderPanelBodyOnly(idx, data) {
    const bodyEl = document.getElementById(`panel-body-${idx}`);
    if (!bodyEl) return;

    if (!data || !data.tokens || data.tokens.length === 0) {
      bodyEl.innerHTML = '<div class="empty-state">No text to tokenize</div>';
      return;
    }

    const currentTab = panelViews[idx] || 'tokens';

    if (currentTab === 'tokens') {
      let html = '<div class="tokens-flow">';
      data.tokens.forEach((t, tIdx) => {
        const colorClass = `token-color-${tIdx % 7}`;
        const specialClass = t.is_special ? 'is-special' : '';
        const displayText = escapeHtml(t.text)
          .replace(/ /g, '<span class="whitespace-char">·</span>')
          .replace(/\n/g, '<span class="whitespace-char">↵</span>\n');

        html += `<span class="token-span ${colorClass} ${specialClass}" data-panel="${idx}" data-token-idx="${tIdx}">${displayText}</span>`;
      });
      html += '</div>';
      bodyEl.innerHTML = html;

      // Attach token hover listeners
      bodyEl.querySelectorAll('.token-span').forEach(span => {
        span.addEventListener('mouseenter', (e) => {
          const tIdx = parseInt(span.dataset.tokenIdx, 10);
          const token = data.tokens[tIdx];
          showPopover(token, e.target);
        });
      });

    } else if (currentTab === 'table') {
      let html = `
        <div class="token-table-wrapper">
          <table class="token-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Token ID</th>
                <th>Text</th>
                <th>Bytes</th>
                <th>Hex</th>
              </tr>
            </thead>
            <tbody>
      `;

      data.tokens.forEach(t => {
        const displaySub = escapeHtml(t.text).replace(/ /g, '·').replace(/\n/g, '↵');
        html += `
          <tr>
            <td>${t.index}</td>
            <td><strong>${t.id}</strong></td>
            <td>${displaySub}</td>
            <td>${t.bytes ? t.bytes.length : 0}</td>
            <td class="code-font">${t.hex || ''}</td>
          </tr>
        `;
      });

      html += `
            </tbody>
          </table>
        </div>
      `;
      bodyEl.innerHTML = html;
    }
  }

  function showPopover(token, targetElement) {
    if (!token || !targetElement) return;

    popIndex.textContent = `#${token.index}`;
    popId.textContent = `ID: ${token.id}`;
    popText.textContent = JSON.stringify(token.text);
    popHex.textContent = token.hex || '--';
    popBytes.textContent = `${token.bytes ? token.bytes.length : 0} bytes`;
    popSpan.textContent = `[${token.start}, ${token.end}]`;

    popover.style.display = 'flex';

    // Position popover relative to hovered token
    const rect = targetElement.getBoundingClientRect();
    const popRect = popover.getBoundingClientRect();

    let top = rect.bottom + window.scrollY + 6;
    let left = rect.left + window.scrollX;

    // Boundary detection
    if (left + popRect.width > window.innerWidth - 16) {
      left = window.innerWidth - popRect.width - 16;
    }
    if (top + popRect.height > window.innerHeight + window.scrollY - 16) {
      top = rect.top + window.scrollY - popRect.height - 6;
    }

    popover.style.top = `${Math.max(8, top)}px`;
    popover.style.left = `${Math.max(8, left)}px`;
  }

  function hidePopover() {
    popover.style.display = 'none';
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
