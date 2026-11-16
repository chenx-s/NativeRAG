/* ============================================================
   RAG 知识库问答 — 前端逻辑
   对接 FastAPI 接口：/health、/query/stream、/documents/*
   ============================================================ */

(function () {
  "use strict";

  /* ---------------------------------------------------------- 状态 */
  const state = {
    token: localStorage.getItem("rag_token") || "",
    streaming: false,
    steps: [],        // 当前回答的执行路径（label 数组）
    answerText: "",   // 当前回答已累积的文本
  };

  /* ---------------------------------------------------------- DOM 引用 */
  const $ = (id) => document.getElementById(id);
  const statusPills = $("statusPills");
  const docStats = $("docStats");
  const docList = $("docList");
  const messages = $("messages");
  const emptyState = $("emptyState");
  const questionInput = $("questionInput");
  const sendBtn = $("sendBtn");
  const allowWeb = $("allowWeb");
  const sessionInput = $("sessionInput");
  const tokenInput = $("tokenInput");
  const themeToggle = $("themeToggle");
  const fileInput = $("fileInput");

  /* ---------------------------------------------------------- 主题 */
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    themeToggle.textContent = theme === "dark" ? "☀️" : "🌙";
    localStorage.setItem("rag_theme", theme);
  }
  applyTheme(localStorage.getItem("rag_theme") || "light");
  themeToggle.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });

  /* ---------------------------------------------------------- Token */
  tokenInput.value = state.token;
  tokenInput.addEventListener("change", () => {
    state.token = tokenInput.value.trim();
    localStorage.setItem("rag_token", state.token);
    // 换 token 后立即重试拉取数据（之前可能因 401 失败）
    loadHealth();
    loadDocuments();
  });

  function authHeaders(extra) {
    const headers = Object.assign({}, extra || {});
    if (state.token) headers["Authorization"] = "Bearer " + state.token;
    return headers;
  }

  /* ---------------------------------------------------------- 通用 API */
  async function api(path, options) {
    const resp = await fetch(path, options);
    if (!resp.ok) {
      let detail = resp.statusText;
      try {
        const data = await resp.json();
        detail = data.detail || detail;
      } catch (_) { /* ignore */ }
      throw new Error(detail || `请求失败（HTTP ${resp.status}）`);
    }
    return resp.json();
  }

  function fmtSize(bytes) {
    if (bytes == null) return "";
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1024 / 1024).toFixed(1) + " MB";
  }

  /* ---------------------------------------------------------- 健康状态 */
  async function loadHealth() {
    try {
      const h = await api("/health", { headers: authHeaders() });
      renderHealth(h);
    } catch (e) {
      statusPills.innerHTML = pill("服务连接失败", "err");
    }
  }

  function pill(text, kind) {
    return `<span class="pill ${kind || ""}"><span class="dot"></span>${escapeHtml(text)}</span>`;
  }

  function renderHealth(h) {
    const parts = [];
    parts.push(pill(h.status === "ok" ? "服务正常" : "服务异常", h.status === "ok" ? "ok" : "err"));
    if (!h.api_key_configured) parts.push(pill("未配置 LLM Key", "warn"));
    parts.push(pill("模型 " + h.llm_model));
    const n = h.vector_store_stats ? h.vector_store_stats.index_count : 0;
    parts.push(pill("文档 " + (n ?? 0)));
    if (h.web_search) parts.push(pill("联网", h.web_search.ready ? "ok" : ""));
    if (h.rerank) parts.push(pill("精排", h.rerank.ready ? "ok" : ""));
    if (h.memory) parts.push(pill("记忆 " + h.memory.backend));
    statusPills.innerHTML = parts.join("");
  }

  /* ---------------------------------------------------------- 知识库 */
  async function loadDocuments() {
    try {
      const data = await api("/documents?limit=200", { headers: authHeaders() });
      renderDocuments(data);
    } catch (e) {
      docList.innerHTML = `<div class="doc-empty">加载失败：${escapeHtml(e.message)}</div>`;
    }
  }

  function renderDocuments(data) {
    const docs = data.documents || [];
    const total = data.total ?? docs.length;
    const ready = docs.filter((d) => d.status === "ready").length;

    docStats.textContent = `共 ${total} 篇 · 就绪 ${ready} · 队列 ${data.queue_pending ?? 0}`;

    if (!docs.length) {
      docList.innerHTML = `<div class="doc-empty">还没有文档<br>上传或扫描 data/ 目录开始</div>`;
      return;
    }

    docList.innerHTML = docs.map((d) => {
      const icon = d.origin === "upload" ? "📄" : "📁";
      const origin = d.origin === "upload" ? "上传" : "目录";
      const meta = [];
      if (d.chunks) meta.push(d.chunks + " 块");
      if (d.size) meta.push(fmtSize(d.size));
      meta.push(origin);
      const err = d.status === "failed" && d.error ? `<div class="doc-meta" style="color:var(--err)">${escapeHtml(d.error.slice(0, 60))}</div>` : "";
      return `
        <div class="doc-item" title="${escapeHtml(d.filename)}">
          <span class="doc-icon">${icon}</span>
          <div class="doc-main">
            <div class="doc-name">${escapeHtml(d.filename)}</div>
            <div class="doc-meta"><span class="badge ${d.status}"><span class="dot"></span>${statusLabel(d.status)}</span>${meta.map((m) => "<span>" + escapeHtml(m) + "</span>").join("")}</div>
            ${err}
          </div>
          <button class="doc-del" data-id="${escapeHtml(d.doc_id)}" title="删除">✕</button>
        </div>`;
    }).join("");

    docList.querySelectorAll(".doc-del").forEach((btn) => {
      btn.addEventListener("click", () => deleteDocument(btn.dataset.id));
    });
  }

  function statusLabel(s) {
    return { ready: "就绪", pending: "等待", processing: "处理中", failed: "失败" }[s] || s;
  }

  async function deleteDocument(docId) {
    if (!confirm("确认删除这篇文档及其全部向量？")) return;
    try {
      await api("/documents/" + encodeURIComponent(docId), { method: "DELETE", headers: authHeaders() });
      loadDocuments();
      loadHealth();
    } catch (e) {
      alert("删除失败：" + e.message);
    }
  }

  async function uploadDocument(file) {
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await api("/documents", { method: "POST", headers: authHeaders(), body: fd });
      const action = r.action;
      if (action === "duplicate") {
        alert("内容相同的文档已存在：" + r.document.filename);
      }
    } catch (e) {
      alert("上传失败：" + e.message);
    }
    loadDocuments();
    loadHealth();
  }

  async function scanDocuments() {
    try {
      const r = await api("/documents/scan", { method: "POST", headers: authHeaders() });
      alert(`扫描完成：新排队 ${r.queued} 篇，重复跳过 ${r.duplicates} 篇`);
    } catch (e) {
      alert("扫描失败：" + e.message);
    }
    loadDocuments();
    loadHealth();
  }

  async function reindex() {
    if (!confirm("重建索引会清空向量库并用当前模型重新入库，确定吗？")) return;
    try {
      const r = await api("/documents/reindex", { method: "POST", headers: authHeaders() });
      alert(r.detail || "已触发重建");
    } catch (e) {
      alert("重建失败：" + e.message);
    }
    loadDocuments();
  }

  async function clearIndex() {
    if (!confirm("清空全部：删除所有向量、登记表和上传文件（data/ 源文件保留）。确定吗？")) return;
    try {
      await api("/index", { method: "DELETE", headers: authHeaders() });
      loadDocuments();
      loadHealth();
    } catch (e) {
      alert("清空失败：" + e.message);
    }
  }

  /* ---------------------------------------------------------- 聊天 */
  function hideEmptyState() {
    if (emptyState) emptyState.style.display = "none";
  }

  function addMessage(role) {
    hideEmptyState();
    const wrap = document.createElement("div");
    wrap.className = "message " + role;
    wrap.innerHTML = `
      <div class="avatar">${role === "user" ? "你" : "AI"}</div>
      <div class="bubble"></div>`;
    messages.appendChild(wrap);
    scrollToBottom();
    return wrap.querySelector(".bubble");
  }

  function scrollToBottom() {
    requestAnimationFrame(() => { messages.scrollTop = messages.scrollHeight; });
  }

  function sendMessage() {
    const question = questionInput.value.trim();
    if (!question || state.streaming) return;

    questionInput.value = "";
    autoResize(questionInput);
    state.steps = [];
    state.answerText = "";

    // 用户气泡
    const userBubble = addMessage("user");
    userBubble.textContent = question;

    // 助手气泡
    const assistantBubble = addMessage("assistant");
    const pipeline = document.createElement("div");
    pipeline.className = "pipeline";
    const answer = document.createElement("div");
    answer.className = "answer-text streaming";
    answer.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    assistantBubble.appendChild(pipeline);
    assistantBubble.appendChild(answer);

    const body = { question };
    if (!allowWeb.checked) body.allow_web = false;
    const sid = sessionInput.value.trim();
    if (sid) body.session_id = sid;

    state.streaming = true;
    sendBtn.disabled = true;
    sendBtn.textContent = "生成中…";

    fetch("/query/stream", {
      method: "POST",
      headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body),
    })
      .then((resp) => {
        if (!resp.ok) {
          return resp.json().then((d) => { throw new Error(d.detail || ("HTTP " + resp.status)); });
        }
        return consumeSSE(resp.body, (event, data) => handleEvent(event, data, assistantBubble, answer, pipeline));
      })
      .then(() => {
        // done 事件内部已收尾；这里兜底处理提前断流
      })
      .catch((e) => {
        answer.classList.remove("streaming");
        answer.textContent = "⚠️ 请求失败：" + e.message;
      })
      .finally(() => {
        state.streaming = false;
        sendBtn.disabled = false;
        sendBtn.textContent = "发送";
      });
  }

  async function consumeSSE(body, onEvent) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        const parsed = parseSSEChunk(chunk);
        if (parsed) onEvent(parsed.event, parsed.data);
      }
    }
  }

  function parseSSEChunk(chunk) {
    let event = "message";
    const dataLines = [];
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (!dataLines.length) return null;
    let data;
    try { data = JSON.parse(dataLines.join("\n")); } catch (_) { data = dataLines.join("\n"); }
    return { event, data };
  }

  function handleEvent(event, data, assistantBubble, answer, pipeline) {
    if (event === "step") {
      state.steps.push(data.label || data.node);
      renderPipeline(pipeline);
    } else if (event === "token") {
      if (answer.querySelector(".typing")) answer.innerHTML = "";
      state.answerText += data.text || "";
      answer.textContent = state.answerText;
      // 追加光标
      answer.innerHTML = escapeHtml(state.answerText) + '<span class="cursor"></span>';
      scrollToBottom();
    } else if (event === "reset") {
      // 草稿被撤回（严格重生成 / 替换为兜底），清空已显示内容
      state.answerText = "";
      answer.textContent = "";
    } else if (event === "done") {
      finalizeAnswer(data, assistantBubble, answer, pipeline);
    } else if (event === "error") {
      answer.classList.remove("streaming");
      answer.textContent = "⚠️ " + (data.detail || "未知错误");
    }
  }

  function renderPipeline(pipeline) {
    const items = state.steps.map((label, i) =>
      `<span class="pipe-step ${i === state.steps.length - 1 ? "current" : ""}">${escapeHtml(label)}</span>`
    );
    pipeline.innerHTML = items.join('<span class="pipe-sep">→</span>');
  }

  function finalizeAnswer(result, assistantBubble, answer, pipeline) {
    answer.classList.remove("streaming");

    // 最终答案以 done 结果为准（兜底/替换时已通过 reset+token 同步，这里兜底覆盖）
    const finalText = (result.answer || "").trim();
    answer.textContent = finalText || "（无回答）";

    // 执行路径以最终 steps 为准（含节点名 -> 中文标签）
    if (result.steps && result.steps.length) {
      state.steps = result.steps.map((s) => STEP_LABELS[s] || s);
      renderPipeline(pipeline);
    }

    // 来源
    if (result.sources && result.sources.length) {
      assistantBubble.appendChild(renderSources(result.sources));
    }

    // 元信息
    assistantBubble.appendChild(renderMeta(result));

    scrollToBottom();
  }

  const STEP_LABELS = {
    route_question: "理解问题",
    transform_query: "查询变换",
    retrieve: "检索知识库",
    rerank_documents: "精排",
    grade_document: "相关性评估",
    rewrite_query: "改写查询",
    web_search: "联网搜索",
    generate: "生成回答",
    check_groundedness: "核对出处",
  };

  function renderSources(sources) {
    const box = document.createElement("div");
    box.className = "sources";
    const title = document.createElement("div");
    title.className = "sources-title";
    title.textContent = "📚 参考来源";
    box.appendChild(title);

    sources.forEach((s) => {
      const a = document.createElement("a");
      a.className = "source-card";
      a.target = "_blank";
      a.rel = "noopener";

      if (s.source_type === "web") {
        a.href = s.source || "#";
        a.title = "打开网页";
      } else if (s.doc_id) {
        a.href = "/documents/" + encodeURIComponent(s.doc_id) + "/file" + (s.page ? "#page=" + s.page : "");
        a.title = "打开原文";
      } else {
        a.href = "javascript:void(0)";
      }

      const head = document.createElement("div");
      head.className = "source-head";
      const name = document.createElement("span");
      name.className = "src-name";
      const page = s.page ? ` · 第 ${s.page} 页` : "";
      name.textContent = (s.filename || s.source || "来源") + page;
      head.appendChild(name);
      if (s.relevance_score != null) {
        const score = document.createElement("span");
        score.className = "score";
        score.textContent = s.relevance_score.toFixed(3);
        head.appendChild(score);
      }
      a.appendChild(head);

      if (s.snippet) {
        const snip = document.createElement("div");
        snip.className = "source-snippet";
        snip.textContent = s.snippet;
        a.appendChild(snip);
      }
      box.appendChild(a);
    });
    return box;
  }

  function renderMeta(result) {
    const meta = document.createElement("div");
    meta.className = "msg-meta";

    const g = result.groundedness;
    const gText = { yes: "出处已核对", no: "出处未通过", skipped: "兜底回答", off: "核对已关闭", unknown: "核对未知" }[g] || g;
    const gKind = g === "yes" ? "ready" : (g === "no" ? "failed" : "neutral");
    meta.appendChild(badgeEl(gText, gKind));

    if (result.web_searched) {
      meta.appendChild(badgeEl("已联网搜索", "processing"));
    } else if (result.source_type === "web") {
      meta.appendChild(badgeEl("来源：互联网", "processing"));
    }

    if (result.web_error) {
      meta.appendChild(badgeEl("联网失败", "failed"));
    }

    if (result.memory_turns) {
      meta.appendChild(badgeEl(`记忆 ${result.memory_turns} 轮`, "neutral"));
    }

    return meta;
  }

  function badgeEl(text, kind) {
    const b = document.createElement("span");
    b.className = "badge " + kind;
    b.textContent = text;
    return b;
  }

  /* ---------------------------------------------------------- 工具 */
  function escapeHtml(str) {
    return String(str == null ? "" : str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function autoResize(ta) {
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 180) + "px";
  }

  /* ---------------------------------------------------------- 事件绑定 */
  sendBtn.addEventListener("click", sendMessage);
  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  questionInput.addEventListener("input", () => autoResize(questionInput));

  $("refreshDocs").addEventListener("click", loadDocuments);
  $("scanBtn").addEventListener("click", scanDocuments);
  $("reindexBtn").addEventListener("click", reindex);
  $("clearIndexBtn").addEventListener("click", clearIndex);

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length) {
      uploadDocument(fileInput.files[0]);
      fileInput.value = "";
    }
  });

  // 空状态示例问题
  document.querySelectorAll(".sample").forEach((btn) => {
    btn.addEventListener("click", () => {
      questionInput.value = btn.dataset.q;
      sendMessage();
    });
  });

  /* ---------------------------------------------------------- 初始化 */
  loadHealth();
  loadDocuments();
  setInterval(loadHealth, 30000);
  setInterval(loadDocuments, 4000);
})();
