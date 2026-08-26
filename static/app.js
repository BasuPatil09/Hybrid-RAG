"use strict";
function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value == null ? "" : String(value);
  return div.innerHTML;
}

function formatPercent(fraction) {
  return `${Math.round(fraction * 100)}%`;
}

function confidenceClass(fraction) {
  if (fraction >= 0.7) return "is-high";
  if (fraction >= 0.4) return "is-mid";
  return "is-low";
}

const API_KEY_STORAGE_KEY = "rag-api-key";

function getStoredApiKey() {
  return sessionStorage.getItem(API_KEY_STORAGE_KEY) || "";
}

function promptForApiKey() {
  const key =
    window.prompt("This server requires an API key. Enter it now:") || "";
  sessionStorage.setItem(API_KEY_STORAGE_KEY, key);
  return key;
}

async function apiFetch(path, options) {
  const requestOptions = options || {};

  const attempt = async (apiKey) => {
    const headers = new Headers(requestOptions.headers || {});
    if (apiKey) headers.set("X-API-Key", apiKey);
    try {
      return await fetch(path, { ...requestOptions, headers });
    } catch (networkError) {
      throw new Error("Could not reach the server. Is the container running?");
    }
  };

  let response = await attempt(getStoredApiKey());

  if (response.status === 401) {
    response = await attempt(promptForApiKey());
  }

  let body = null;
  try {
    body = await response.json();
  } catch (parseError) {}

  if (!response.ok) {
    const detail =
      body && body.detail ? body.detail : `Request failed (${response.status})`;
    throw new Error(
      typeof detail === "string" ? detail : JSON.stringify(detail),
    );
  }

  return body;
}

function confidenceItemHtml(label, fraction, isComposite) {
  const cls = confidenceClass(fraction);
  const compositeClass = isComposite ? "is-composite" : "";
  return `
    <div class="confidence-item ${compositeClass}">
      <span class="confidence-label">${escapeHtml(label)}</span>
      <span class="confidence-value">${formatPercent(fraction)}</span>
      <div class="confidence-track">
        <div class="confidence-fill ${cls}" style="width:${Math.max(0, Math.min(100, fraction * 100))}%"></div>
      </div>
    </div>
  `;
}

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const targetTab = button.dataset.tab;

      buttons.forEach((b) => {
        b.classList.toggle("is-active", b === button);
        b.setAttribute("aria-selected", b === button ? "true" : "false");
      });

      document.querySelectorAll(".panel").forEach((panel) => {
        const isTarget = panel.id === `panel-${targetTab}`;
        panel.hidden = !isTarget;
        panel.classList.toggle("is-active", isTarget);
      });

      if (targetTab === "documents") {
        loadDocuments();
      }
      if (targetTab === "observability") {
        loadObservability();
      }
    });
  });
}

async function checkHealth() {
  const dot = document.getElementById("health-dot");
  const text = document.getElementById("health-text");
  try {
    await apiFetch("/health");
    dot.classList.add("is-ok");
    dot.classList.remove("is-down");
    text.textContent = "online";
  } catch (error) {
    dot.classList.add("is-down");
    dot.classList.remove("is-ok");
    text.textContent = "unreachable";
  }
}

function renderClaimVerdicts(claimVerdicts) {
  if (!claimVerdicts || claimVerdicts.length === 0) {
    return "";
  }
  const rows = claimVerdicts
    .map(
      (verdict) => `
      <div class="claim-row">
        <span class="verdict-badge ${verdict.supported ? "is-supported" : "is-unsupported"}">
          ${verdict.supported ? "Supported" : "Not supported"}
        </span>
        <div>
          <div class="claim-text">${escapeHtml(verdict.claim)}</div>
          <div class="claim-explanation">${escapeHtml(verdict.explanation)} &mdash; <span class="mono-cell">${escapeHtml(verdict.chunk_ids.join(", "))}</span></div>
        </div>
      </div>
    `,
    )
    .join("");
  return `<div class="claims-heading">Per-claim verification</div>${rows}`;
}

function renderSourceChips(reranked, citedChunkIds) {
  if (!reranked || reranked.length === 0) {
    return "";
  }
  const citedSet = new Set(citedChunkIds || []);
  const chips = reranked
    .map((scored) => {
      const chunk = scored.chunk;
      const isCited = citedSet.has(chunk.id);
      const pageLabel =
        chunk.metadata.page != null ? `p${chunk.metadata.page}` : "";
      return `
        <span class="source-chip ${isCited ? "is-cited" : ""}" title="${escapeHtml(chunk.content.slice(0, 200))}">
          ${isCited ? "&#10003; " : ""}${escapeHtml(chunk.metadata.source)} ${escapeHtml(pageLabel)} &middot; ${scored.score.toFixed(2)}
        </span>
      `;
    })
    .join("");
  return `<div class="sources-heading">Retrieved sources (checked = cited)</div>${chips}`;
}

function renderAnswer(query, data) {
  const answer = data.answer;
  const confidence = data.confidence;
  const claimVerdicts = data.verified ? data.verified.claim_verdicts : [];

  const block = document.createElement("div");
  block.className = "qa-block";
  block.innerHTML = `
    <div class="qa-question">${escapeHtml(query)}</div>
    <div class="qa-answer ${answer.is_insufficient ? "is-insufficient" : ""}">${escapeHtml(answer.answer)}</div>
    <div class="confidence-grid">
      ${confidenceItemHtml("Retrieval", confidence.retrieval_confidence, false)}
      ${confidenceItemHtml("Citation coverage", confidence.citation_coverage, false)}
      ${confidenceItemHtml("Completeness", confidence.answer_completeness, false)}
      ${confidenceItemHtml("Composite", confidence.composite, true)}
    </div>
    ${renderClaimVerdicts(claimVerdicts)}
    ${renderSourceChips(data.reranked, answer.cited_chunk_ids)}
  `;
  return block;
}

function initChat() {
  const form = document.getElementById("ask-form");
  const input = document.getElementById("question-input");
  const button = document.getElementById("ask-button");
  const log = document.getElementById("chat-log");
  const emptyState = document.getElementById("chat-empty");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = input.value.trim();
    if (!question) return;

    button.disabled = true;
    button.textContent = "Asking\u2026";
    emptyState.hidden = true;

    try {
      const data = await apiFetch("/v1/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      log.prepend(renderAnswer(question, data));
      input.value = "";
    } catch (error) {
      const banner = document.createElement("div");
      banner.className = "error-banner";
      banner.textContent = error.message;
      log.prepend(banner);
    } finally {
      button.disabled = false;
      button.textContent = "Ask";
      input.focus();
    }
  });
}

function renderIngestReport(report) {
  const container = document.createElement("div");

  const cards = [
    ["Documents loaded", report.documents_loaded],
    ["Chunks created", report.chunks_created],
    ["Already indexed", report.chunks_skipped_already_indexed],
    ["Duplicates skipped", report.duplicates_skipped],
    ["Total indexed", report.chunks_indexed],
  ];

  container.innerHTML = `
    <div class="report-grid">
      ${cards
        .map(
          ([label, value]) => `
        <div class="report-card">
          <div class="report-card-label">${escapeHtml(label)}</div>
          <div class="report-card-value">${escapeHtml(value)}</div>
        </div>
      `,
        )
        .join("")}
    </div>
    ${
      report.stopped_early
        ? `<div class="warning-banner">Stopped early: ${escapeHtml(report.stopped_early)} &mdash; already-embedded chunks are safely persisted; re-run to pick up where this left off.</div>`
        : ""
    }
  `;
  return container;
}

function initIngest() {
  const button = document.getElementById("ingest-button");
  const result = document.getElementById("ingest-result");

  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "Ingesting\u2026";
    result.innerHTML =
      '<div class="loading-line">Loading, chunking, embedding, and indexing the corpus</div>';

    try {
      const report = await apiFetch("/v1/ingest", { method: "POST" });
      result.innerHTML = "";
      result.appendChild(renderIngestReport(report));
    } catch (error) {
      result.innerHTML = `<div class="error-banner">${escapeHtml(error.message)}</div>`;
    } finally {
      button.disabled = false;
      button.textContent = "Run Ingestion";
    }
  });
}

function renderDocumentsTable(data) {
  if (data.documents.length === 0) {
    return '<div class="empty-state">Nothing indexed yet. Run ingestion first.</div>';
  }
  const rows = data.documents
    .map(
      (doc) => `
      <tr>
        <td>${escapeHtml(doc.source)}</td>
        <td class="mono-cell">${escapeHtml(doc.chunk_count)}</td>
      </tr>
    `,
    )
    .join("");
  return `
    <div class="report-grid">
      <div class="report-card">
        <div class="report-card-label">Total chunks</div>
        <div class="report-card-value">${escapeHtml(data.total_chunks)}</div>
      </div>
      <div class="report-card">
        <div class="report-card-label">Sources</div>
        <div class="report-card-value">${escapeHtml(data.documents.length)}</div>
      </div>
    </div>
    <table class="data-table">
      <thead><tr><th>Source</th><th>Chunks</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function loadDocuments() {
  const result = document.getElementById("documents-result");
  result.innerHTML = '<div class="loading-line">Loading</div>';
  try {
    const data = await apiFetch("/v1/documents");
    result.innerHTML = renderDocumentsTable(data);
  } catch (error) {
    result.innerHTML = `<div class="error-banner">${escapeHtml(error.message)}</div>`;
  }
}

function initDocuments() {
  document
    .getElementById("documents-refresh")
    .addEventListener("click", loadDocuments);
}

function renderEvalSummary(summary) {
  const categoryPills = Object.entries(summary.behavior_match_rate_by_category)
    .map(
      ([category, rate]) =>
        `<span class="category-pill">${escapeHtml(category)}: ${formatPercent(rate)}</span>`,
    )
    .join("");

  return `
    <div class="report-grid">
      <div class="report-card">
        <div class="report-card-label">Questions</div>
        <div class="report-card-value">${escapeHtml(summary.total_questions)}</div>
      </div>
      <div class="report-card">
        <div class="report-card-label">Behavior match</div>
        <div class="report-card-value">${formatPercent(summary.behavior_match_rate)}</div>
      </div>
      <div class="report-card">
        <div class="report-card-label">Citation accuracy</div>
        <div class="report-card-value">${formatPercent(summary.mean_citation_accuracy)}</div>
      </div>
      <div class="report-card">
        <div class="report-card-label">Composite confidence</div>
        <div class="report-card-value">${formatPercent(summary.mean_composite_confidence)}</div>
      </div>
    </div>
    <div class="category-breakdown">${categoryPills}</div>
  `;
}

function renderEvalResultsTable(results) {
  const rows = results
    .map((result, index) => {
      const matchIcon = result.behavior_matched
        ? '<span class="match-icon is-match">&#10003;</span>'
        : '<span class="match-icon is-mismatch">&#10007;</span>';
      return `
        <tr class="is-expandable" data-expand-target="eval-row-${index}">
          <td>${matchIcon}</td>
          <td class="mono-cell">${escapeHtml(result.question_id)}</td>
          <td>${escapeHtml(result.category)}</td>
          <td>${escapeHtml(result.expected_behavior)}</td>
          <td class="mono-cell">${formatPercent(result.citation_accuracy)}</td>
          <td class="mono-cell">${formatPercent(result.composite_confidence)}</td>
        </tr>
        <tr class="expand-row" id="eval-row-${index}" hidden>
          <td colspan="6"><strong>${escapeHtml(result.question)}</strong>\n\n${escapeHtml(result.answer)}</td>
        </tr>
      `;
    })
    .join("");

  return `
    <table class="data-table">
      <thead>
        <tr><th></th><th>ID</th><th>Category</th><th>Expected</th><th>Citation acc.</th><th>Composite</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function attachEvalRowExpansion(container) {
  container.querySelectorAll("tr.is-expandable").forEach((row) => {
    row.addEventListener("click", () => {
      const target = document.getElementById(row.dataset.expandTarget);
      if (target) target.hidden = !target.hidden;
    });
  });
}

function initEval() {
  const button = document.getElementById("eval-button");
  const result = document.getElementById("eval-result");

  button.addEventListener("click", async () => {
    button.disabled = true;
    button.textContent = "Running\u2026";
    result.innerHTML =
      '<div class="loading-line">Running the golden set through the full pipeline &mdash; this can take a few minutes</div>';

    try {
      const data = await apiFetch("/v1/eval", { method: "POST" });
      result.innerHTML =
        renderEvalSummary(data.summary) + renderEvalResultsTable(data.results);
      attachEvalRowExpansion(result);
    } catch (error) {
      result.innerHTML = `<div class="error-banner">${escapeHtml(error.message)}</div>`;
    } finally {
      button.disabled = false;
      button.textContent = "Run Eval Suite";
    }
  });
}

function eventDetail(event) {
  if (event.event_type === "ask") return event.question || "";
  if (event.event_type === "ingest")
    return event.chunks_indexed != null
      ? `${event.chunks_indexed} chunks indexed`
      : "";
  if (event.event_type === "eval") {
    return event.eval_questions != null
      ? `${event.eval_questions} questions, ${formatPercent(event.eval_behavior_match_rate || 0)} match`
      : "";
  }
  return "";
}

function renderObservabilitySummary(summary) {
  return `
    <div class="report-grid">
      <div class="report-card">
        <div class="report-card-label">Total events</div>
        <div class="report-card-value">${escapeHtml(summary.total_events)}</div>
      </div>
      <div class="report-card">
        <div class="report-card-label">Errors</div>
        <div class="report-card-value">${escapeHtml(summary.error_count)}</div>
      </div>
      <div class="report-card">
        <div class="report-card-label">Mean latency</div>
        <div class="report-card-value">${Math.round(summary.mean_latency_ms)}ms</div>
      </div>
      <div class="report-card">
        <div class="report-card-label">Mean composite</div>
        <div class="report-card-value">${formatPercent(summary.mean_composite_confidence)}</div>
      </div>
    </div>
    <div class="category-breakdown">
      <span class="category-pill">ask: ${escapeHtml(summary.ask_count)}</span>
      <span class="category-pill">ingest: ${escapeHtml(summary.ingest_count)}</span>
      <span class="category-pill">eval: ${escapeHtml(summary.eval_count)}</span>
      <span class="category-pill">insufficient-context rate: ${formatPercent(summary.insufficient_context_rate)}</span>
    </div>
  `;
}

function renderObservabilityTable(events) {
  if (events.length === 0) {
    return '<div class="empty-state">No requests recorded yet this session. Ask a question, run ingestion, or run eval, then come back.</div>';
  }
  const rows = events
    .map((event) => {
      const statusIcon = event.success
        ? '<span class="match-icon is-match">&#10003;</span>'
        : '<span class="match-icon is-mismatch">&#10007;</span>';
      const composite =
        event.composite_confidence != null
          ? formatPercent(event.composite_confidence)
          : "\u2014";
      const time = event.timestamp
        ? event.timestamp.replace("T", " ").slice(0, 19)
        : "";
      return `
        <tr>
          <td>${statusIcon}</td>
          <td class="mono-cell">${escapeHtml(event.event_type)}</td>
          <td class="mono-cell">${escapeHtml(time)}</td>
          <td class="mono-cell">${Math.round(event.latency_ms)}ms</td>
          <td class="mono-cell">${composite}</td>
          <td>${escapeHtml(event.error || eventDetail(event))}</td>
        </tr>
      `;
    })
    .join("");

  return `
    <table class="data-table">
      <thead><tr><th></th><th>Type</th><th>Time (UTC)</th><th>Latency</th><th>Composite</th><th>Detail</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function loadObservability() {
  const result = document.getElementById("observability-result");
  result.innerHTML = '<div class="loading-line">Loading</div>';
  try {
    const data = await apiFetch("/v1/observability");
    result.innerHTML =
      renderObservabilitySummary(data.summary) +
      renderObservabilityTable(data.recent_events);
  } catch (error) {
    result.innerHTML = `<div class="error-banner">${escapeHtml(error.message)}</div>`;
  }
}

function initObservability() {
  document
    .getElementById("observability-refresh")
    .addEventListener("click", loadObservability);
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  checkHealth();
  initChat();
  initIngest();
  initDocuments();
  initEval();
  initObservability();
});
