/* Webview shell: trees, viewer, options, run log. */
(function () {
  "use strict";

  const $ = sel => document.querySelector(sel);
  const t = k => window.I18N.t(k);

  const state = {
    selectedInput: null,     // relative path of the chosen input file
    current: null,           // { side, path, kind }
    optionFields: {},        // name -> { input, kind, original }
    logOffset: 0,
    running: false,
    pdf: null,               // { pages, page, dpi, text }
    dataset: null            // { side, path, offset, limit }
  };

  // ── helpers ──────────────────────────────────────────────────────────
  function fmtBytes(n) {
    if (!n) return "0 B";
    const u = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
    return (n / Math.pow(1024, i)).toFixed(i ? 1 : 0) + " " + u[i];
  }

  function fmtNum(n) { return (n || 0).toLocaleString(); }

  async function api(url, opts) {
    const res = await fetch(url, opts);
    const text = await res.text();
    let data;
    try { data = text ? JSON.parse(text) : {}; } catch (e) { data = { raw: text }; }
    if (!res.ok) throw new Error(data.error || res.statusText);
    return data;
  }

  function postJSON(url, body) {
    return api(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    });
  }

  let toastTimer = null;
  function toast(msg, kind) {
    const box = $("#toast");
    box.textContent = msg;
    box.className = "toast " + (kind || "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => box.classList.add("hidden"), 3200);
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function iconFor(name, ext, isDir) {
    if (isDir) return "📁";
    const map = { ".pdf": "📄", ".ifc": "🏗", ".jsonl": "🗂", ".json": "🗂",
                  ".png": "🖼", ".jpg": "🖼", ".jpeg": "🖼", ".txt": "📝" };
    return map[ext] || "•";
  }

  function q(side, path, extra) {
    let url = "side=" + encodeURIComponent(side) + "&path=" + encodeURIComponent(path);
    if (extra) url += "&" + extra;
    return url;
  }

  // ── trees ────────────────────────────────────────────────────────────
  function renderTree(host, node, side, opts) {
    host.innerHTML = "";
    if (!node || (node.missing)) {
      host.appendChild(el("div", "tree-empty", t("msg.noOutput")));
      return;
    }
    host.appendChild(buildNode(node, side, opts || {}, 0, true));
  }

  function buildNode(node, side, opts, depth, isRoot) {
    const wrap = el("div");
    const row = el("div", "node" + (node.type === "file" && !node.supported && side === "input" ? " dim" : ""));
    row.style.paddingLeft = (6 + depth * 12) + "px";

    if (node.type === "dir") {
      const tw = el("span", "tw", "▾");
      row.appendChild(tw);
      row.appendChild(el("span", "ic", iconFor(node.name, "", true)));
      row.appendChild(el("span", "nm", node.name || "/"));
      wrap.appendChild(row);

      const kids = el("div", "children");
      (node.children || []).forEach(c => kids.appendChild(buildNode(c, side, opts, depth + 1)));
      wrap.appendChild(kids);

      if (!isRoot && depth > 1 && !opts.expandAll) {
        kids.classList.add("collapsed");
        tw.textContent = "▸";
      }
      row.addEventListener("click", () => {
        kids.classList.toggle("collapsed");
        tw.textContent = kids.classList.contains("collapsed") ? "▸" : "▾";
      });
    } else {
      row.appendChild(el("span", "tw", ""));
      row.appendChild(el("span", "ic", iconFor(node.name, node.ext, false)));
      row.appendChild(el("span", "nm", node.name));
      row.appendChild(el("span", "sz", fmtBytes(node.size)));
      row.dataset.path = node.path;
      row.dataset.side = side;
      row.addEventListener("click", () => selectFile(side, node.path, row));
      wrap.appendChild(row);
    }
    return wrap;
  }

  function markSelected(row) {
    document.querySelectorAll(".node.selected").forEach(n => n.classList.remove("selected"));
    if (row) row.classList.add("selected");
  }

  async function loadInputTree() {
    const query = $("#filter-input").value.trim();
    const tree = await api("/api/tree/input?q=" + encodeURIComponent(query));
    renderTree($("#tree-input"), tree, "input", { expandAll: !!query });
  }

  async function loadOutputTree(sourcePath) {
    const query = $("#filter-output").value.trim();
    const url = "/api/tree/output?q=" + encodeURIComponent(query) +
      "&path=" + encodeURIComponent(sourcePath || "");
    const data = await api(url);
    const host = $("#tree-output");
    host.innerHTML = "";
    if (!data.roots.length) {
      host.appendChild(el("div", "tree-empty", t("msg.noOutput")));
    } else {
      data.roots.forEach(root => {
        const box = el("div");
        box.appendChild(buildNode(root, "output", { expandAll: true }, 0, true));
        host.appendChild(box);
      });
    }
    const s = data.summary || { files: 0, records: 0, bytes: 0 };
    $("#dataset-meta").textContent =
      `${fmtNum(s.files)} ${t("stat.files")} · ${fmtNum(s.records)} ${t("stat.records")} · ${fmtBytes(s.bytes)}`;
    return data;
  }

  // ── file selection ───────────────────────────────────────────────────
  async function selectFile(side, path, row) {
    markSelected(row);
    if (side === "input") {
      state.selectedInput = path;
      loadOutputTree(path).then(autoOpenDataset).catch(e => toast(e.message, "err"));
    }
    try {
      const info = await api("/api/file?" + q(side, path));
      state.current = { side, path, kind: info.kind };
      await showInViewer(side, path, info);
      if (side === "output" && info.kind === "jsonl") openDataset(side, path);
    } catch (e) {
      toast(e.message, "err");
    }
  }

  async function autoOpenDataset(data) {
    if (!data || !data.roots.length) { $("#dataset").innerHTML = ""; $("#dataset-tools").innerHTML = ""; return; }
    const found = [];
    (function walk(node) {
      (node.children || []).forEach(c => {
        if (c.type === "dir") walk(c);
        else if (c.ext === ".jsonl") found.push(c.path);
      });
    })({ children: data.roots });
    if (found.length) openDataset("output", found[0]);
  }

  // ── centre viewer ────────────────────────────────────────────────────
  function clearViewer() {
    if (window.AECViewer3D) window.AECViewer3D.dispose();
    $("#viewer").innerHTML = "";
    $("#viewer-tools").innerHTML = "";
  }

  async function showInViewer(side, path, info) {
    clearViewer();
    $("#viewer-title").removeAttribute("data-i18n");
    $("#viewer-title").textContent = info.name;
    $("#viewer-meta").textContent = fmtBytes(info.size) +
      (info.pages ? ` · ${info.pages} p` : "") +
      (info.records ? ` · ${fmtNum(info.records)} ${t("view.records")}` : "");

    switch (info.kind) {
      case "pdf": return showPdf(side, path, info);
      case "ifc": return showIfc(side, path);
      case "image": return showImage(side, path);
      case "jsonl": return showJsonl(side, path);
      case "json":
      case "text": return showText(side, path);
      default:
        $("#viewer").appendChild(el("div", "viewer-empty", t("msg.binary")));
    }
  }

  function showImage(side, path) {
    const img = el("img", "raw");
    img.src = "/api/file/raw?" + q(side, path);
    $("#viewer").appendChild(img);
  }

  // PDF: server-rendered page images + a full-text index for search.
  async function showPdf(side, path, info) {
    if (!info.pages) {
      $("#viewer").appendChild(el("div", "viewer-empty", info.error || t("msg.binary")));
      return;
    }
    state.pdf = { side, path, pages: info.pages, page: 0, dpi: 110, text: null };
    const tools = $("#viewer-tools");

    const prev = el("button", "btn tiny", "‹");
    const next = el("button", "btn tiny", "›");
    const pageIn = el("input");
    pageIn.type = "number"; pageIn.min = 1; pageIn.max = state.pdf.pages; pageIn.value = 1;
    pageIn.className = "search"; pageIn.style.maxWidth = "70px"; pageIn.style.flex = "none";
    const total = el("span", "stats", "/ " + state.pdf.pages);
    total.style.marginLeft = "0";

    const zoomOut = el("button", "btn tiny", "−");
    const zoomIn = el("button", "btn tiny", "+");
    const find = el("input", "search");
    find.type = "search"; find.placeholder = t("view.find");
    const hits = el("span", "stats", "");
    hits.style.marginLeft = "0";

    tools.append(prev, pageIn, total, next, zoomOut, zoomIn, find, hits);

    const goto = n => {
      state.pdf.page = Math.max(0, Math.min(n, state.pdf.pages - 1));
      pageIn.value = state.pdf.page + 1;
      drawPdfPage();
    };
    prev.onclick = () => goto(state.pdf.page - 1);
    next.onclick = () => goto(state.pdf.page + 1);
    pageIn.onchange = () => goto(parseInt(pageIn.value, 10) - 1 || 0);
    zoomIn.onclick = () => { state.pdf.dpi = Math.min(260, state.pdf.dpi + 25); drawPdfPage(); };
    zoomOut.onclick = () => { state.pdf.dpi = Math.max(60, state.pdf.dpi - 25); drawPdfPage(); };

    let timer = null;
    find.oninput = () => {
      clearTimeout(timer);
      timer = setTimeout(() => runPdfSearch(find.value.trim(), hits, goto), 250);
    };

    drawPdfPage();
  }

  function drawPdfPage() {
    const p = state.pdf;
    const host = $("#viewer");
    const keep = host.querySelector(".hit-list");
    host.innerHTML = "";
    const box = el("div", "pdf-pages");
    const img = el("img", "pdf-page");
    img.src = "/api/pdf/page?" + q(p.side, p.path, `page=${p.page}&dpi=${p.dpi}`);
    box.append(img, el("div", "pdf-page-no", `${p.page + 1} / ${p.pages}`));
    host.appendChild(box);
    if (keep) host.appendChild(keep);
  }

  async function runPdfSearch(term, hitsLabel, goto) {
    const host = $("#viewer");
    const old = host.querySelector(".hit-list");
    if (old) old.remove();
    if (!term) { hitsLabel.textContent = ""; return; }

    if (!state.pdf.text) {
      hitsLabel.textContent = t("msg.loading");
      const data = await api("/api/pdf/text?" + q(state.pdf.side, state.pdf.path));
      state.pdf.text = data.pages || [];
    }
    const needle = term.toLowerCase();
    const list = el("div", "hit-list");
    let count = 0;
    state.pdf.text.forEach((text, page) => {
      const lower = (text || "").toLowerCase();
      let at = lower.indexOf(needle);
      while (at !== -1 && count < 300) {
        const from = Math.max(0, at - 40);
        const snippet = text.slice(from, at + needle.length + 60).replace(/\s+/g, " ");
        const item = el("div", "hit");
        item.appendChild(el("b", null, `p.${page + 1}  `));
        const mark = escapeHtml(snippet).replace(
          new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "ig"),
          m => `<mark>${m}</mark>`);
        const span = document.createElement("span");
        span.innerHTML = mark;
        item.appendChild(span);
        item.onclick = () => goto(page);
        list.appendChild(item);
        count++;
        at = lower.indexOf(needle, at + needle.length);
      }
    });
    hitsLabel.textContent = `${count} ${t("view.hits")}`;
    if (count) host.appendChild(list);
  }

  // IFC: tessellated server-side, drawn by the three.js module.
  async function showIfc(side, path) {
    const host = $("#viewer");
    host.appendChild(el("div", "viewer-empty", t("msg.loading")));
    let mesh;
    try {
      mesh = await api("/api/ifc/mesh?" + q(side, path));
    } catch (e) {
      host.innerHTML = "";
      host.appendChild(el("div", "viewer-empty", t("msg.noIfc") + " — " + e.message));
      return;
    }
    host.innerHTML = "";
    if (!window.AECViewer3D) {
      host.appendChild(el("div", "viewer-empty", t("msg.no3d")));
      return;
    }
    const stage = el("div");
    stage.id = "three-host";
    host.appendChild(stage);

    const reset = el("button", "btn tiny", t("view.reset"));
    reset.onclick = () => window.AECViewer3D.reset();
    $("#viewer-tools").appendChild(reset);
    $("#viewer-tools").appendChild(
      el("span", "stats", `${fmtNum(mesh.elements)} ${t("view.elements")}` +
        (mesh.truncated ? ` / ${fmtNum(mesh.total)}` : "")));

    window.AECViewer3D.show(stage, mesh);

    const legend = el("div", "legend");
    mesh.groups.forEach(g => {
      const row = el("div");
      const swatch = el("i");
      swatch.style.background = g.colour;
      row.append(swatch, el("span", null, g.type));
      legend.appendChild(row);
    });
    if (mesh.groups.length) host.appendChild(legend);
  }

  async function showText(side, path) {
    const data = await api("/api/text?" + q(side, path));
    const tools = $("#viewer-tools");
    const find = el("input", "search");
    find.type = "search"; find.placeholder = t("view.find");
    const hits = el("span", "stats", "");
    hits.style.marginLeft = "0";
    tools.append(find, hits);

    const pre = el("pre", "text");
    pre.textContent = data.text + (data.truncated ? "\n\n… truncated …" : "");
    $("#viewer").appendChild(pre);

    find.oninput = () => {
      const term = find.value.trim();
      if (!term) { pre.textContent = data.text; hits.textContent = ""; return; }
      const rx = new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "ig");
      const escaped = escapeHtml(data.text);
      let count = 0;
      pre.innerHTML = escaped.replace(rx, m => { count++; return `<mark>${m}</mark>`; });
      hits.textContent = `${count} ${t("view.hits")}`;
      const first = pre.querySelector("mark");
      if (first) first.scrollIntoView({ block: "center" });
    };
  }

  function showJsonl(side, path) {
    const host = $("#viewer");
    const box = el("div");
    box.style.padding = "6px 8px";
    host.appendChild(box);
    renderRecords(box, side, path, 0, 25, $("#viewer-tools"));
  }

  // ── dataset panel (bottom right) ─────────────────────────────────────
  function openDataset(side, path) {
    state.dataset = { side, path };
    renderRecords($("#dataset"), side, path, 0, 15, $("#dataset-tools"));
  }

  async function renderRecords(host, side, path, offset, limit, toolsHost) {
    host.innerHTML = "";
    const data = await api("/api/jsonl?" + q(side, path, `offset=${offset}&limit=${limit}`));
    const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";
    const kind = path.includes("dapt_") ? "dapt" : path.includes("vlm_") ? "vlm" : "sft";

    data.records.forEach(rec => host.appendChild(recordCard(rec, kind, side, dir)));
    if (!data.records.length) host.appendChild(el("div", "tree-empty", "—"));

    if (toolsHost) {
      toolsHost.innerHTML = "";
      const pager = el("div", "pager");
      const prev = el("button", "btn tiny", "‹ " + t("view.prev"));
      const next = el("button", "btn tiny", t("view.next") + " ›");
      prev.disabled = offset <= 0;
      next.disabled = offset + limit >= data.total;
      prev.onclick = () => renderRecords(host, side, path, Math.max(0, offset - limit), limit, toolsHost);
      next.onclick = () => renderRecords(host, side, path, offset + limit, limit, toolsHost);
      pager.append(prev,
        el("span", null, `${offset + 1}–${Math.min(offset + limit, data.total)} / ${fmtNum(data.total)}`),
        next,
        el("span", null, path.split("/").pop()));
      toolsHost.appendChild(pager);
    }
  }

  function recordCard(rec, kind, side, dir) {
    const card = el("details", "rec");
    const head = el("summary");
    const data = rec.data || {};
    head.appendChild(el("span", "tag " + kind, kind.toUpperCase()));
    head.appendChild(el("b", null, data.id || `#${rec.index}`));
    const title = data.instruction || data.text || (data.output && data.output.answer) || "";
    head.appendChild(el("span", null, String(title).slice(0, 90)));
    card.appendChild(head);

    const body = el("div", "rec-body");
    const kv = el("div", "kv");
    const add = (k, v) => {
      if (v === undefined || v === null || v === "") return;
      kv.appendChild(el("div", "k", k));
      kv.appendChild(el("div", "v", typeof v === "string" ? v : JSON.stringify(v, null, 1)));
    };

    if (rec.error) {
      add("error", rec.error);
      add("raw", rec.raw);
    } else if (kind === "vlm") {
      add("task_type", data.task_type);
      add("instruction", data.instruction);
      add("answer", data.output && data.output.answer);
      add("label", data.output && data.output.label);
      add("evidence", data.output && data.output.evidence);
      add("metadata", data.metadata);
    } else if (kind === "dapt") {
      add("source_name", data.source_name);
      add("page_range", data.page_range);
      add("domain_tags", data.domain_tags);
      add("text", data.text);
    } else {
      add("task_type", data.task_type);
      add("instruction", data.instruction);
      add("context", data.input && data.input.context);
      add("answer", data.output && data.output.answer);
      add("final_label", data.output && data.output.final_label);
      add("evidence", data.output && data.output.evidence);
      add("domain_tags", data.domain_tags);
    }
    body.appendChild(kv);

    if (Array.isArray(data.images) && data.images.length) {
      const strip = el("div", "thumbs");
      data.images.forEach(rel => {
        const img = el("img");
        const full = dir ? dir + "/" + rel : rel;
        img.src = "/api/file/raw?" + q(side, full);
        img.title = rel;
        img.onclick = () => selectFile(side, full, null);
        strip.appendChild(img);
      });
      body.appendChild(strip);
    }

    const raw = el("pre", "rec-json", JSON.stringify(data, null, 2));
    body.appendChild(raw);
    card.appendChild(body);
    return card;
  }

  // ── options panel ────────────────────────────────────────────────────
  async function loadOptions() {
    const data = await api("/api/config");
    const host = $("#options");
    host.innerHTML = "";
    state.optionFields = {};

    data.groups.forEach((group, index) => {
      const box = el("details", "optgroup");
      if (index < 2) box.open = true;
      const head = el("summary");
      head.dataset.i18n = "group." + group.key;
      head.textContent = t("group." + group.key);
      box.appendChild(head);
      group.fields.forEach(field => box.appendChild(optionRow(field)));
      host.appendChild(box);
    });
  }

  function optionRow(field) {
    const wide = field.multiline || field.kind === "json";
    const row = el("div", "optrow" + (wide ? " wide" : ""));
    const label = el("label", null, field.name);
    label.title = field.name;
    row.appendChild(label);

    let input;
    if (field.kind === "multi") {
      input = el("div", "optchecks");
      const boxes = [];
      (field.choices || []).forEach(choice => {
        const wrap = el("label");
        const check = el("input");
        check.type = "checkbox";
        check.value = choice;
        check.checked = (field.value || []).indexOf(choice) !== -1;
        wrap.append(check, el("span", null, choice));
        input.appendChild(wrap);
        boxes.push(check);
      });
      input.getValue = () => boxes.filter(b => b.checked).map(b => b.value);
    } else if (field.choices) {
      input = el("select");
      field.choices.forEach(choice => {
        const opt = el("option", null, choice);
        opt.value = choice;
        input.appendChild(opt);
      });
      input.value = field.value;
      input.getValue = () => input.value;
    } else if (field.kind === "bool") {
      input = el("input");
      input.type = "checkbox";
      input.checked = !!field.value;
      input.getValue = () => input.checked;
    } else if (wide) {
      input = el("textarea");
      input.value = field.value;
      input.getValue = () => input.value;
    } else {
      input = el("input");
      input.type = field.secret ? "password" : (field.kind === "int" || field.kind === "float" ? "number" : "text");
      if (field.kind === "float") input.step = "any";
      input.value = field.value;
      input.getValue = () => input.value;
    }
    row.appendChild(input);

    const original = JSON.stringify(input.getValue());
    const mark = () => row.classList.toggle("changed", JSON.stringify(input.getValue()) !== original);
    row.addEventListener("input", mark);
    row.addEventListener("change", mark);

    state.optionFields[field.name] = { input, kind: field.kind, original };
    return row;
  }

  function collectOptions() {
    const values = {};
    Object.keys(state.optionFields).forEach(name => {
      const field = state.optionFields[name];
      const value = field.input.getValue();
      if (JSON.stringify(value) !== field.original) values[name] = value;
    });
    return values;
  }

  async function applyOptions(save) {
    const values = collectOptions();
    if (!Object.keys(values).length && !save) { toast(t("msg.noChange")); return; }
    try {
      const res = await postJSON("/api/config", { values, save: !!save });
      toast(save ? t("msg.saved") : t("msg.applied"), "ok");
      if (res.changed.length) { await loadOptions(); await refreshState(); await loadInputTree(); }
    } catch (e) {
      toast(e.message, "err");
    }
  }

  // ── run control ──────────────────────────────────────────────────────
  async function startRun() {
    if (!confirm(t("msg.confirmRun"))) return;
    const values = collectOptions();
    if (Object.keys(values).length) {
      try {
        await postJSON("/api/config", { values, save: false });
      } catch (e) {
        toast(e.message, "err");
        return;
      }
    }
    const body = {
      only_new: $("#opt-only-new").checked,
      dry_run: $("#opt-dry-run").checked,
      targets: ($("#opt-selected-only").checked && state.selectedInput) ? [state.selectedInput] : []
    };
    try {
      await postJSON("/api/run", body);
      state.logOffset = 0;
      $("#log").textContent = "";
      switchTab("log");
      toast(t("msg.started"), "ok");
      pollRun();
    } catch (e) {
      toast(e.message, "err");
    }
  }

  function logClass(line) {
    if (/ - ERROR - |\[error\]/.test(line)) return "err";
    if (/ - WARNING - |\[cancel\]/.test(line)) return "warn";
    if (/\[done\] exit code 0/.test(line)) return "ok";
    return "";
  }

  async function pollRun() {
    let status;
    try {
      status = await api("/api/run/status?since=" + state.logOffset);
    } catch (e) {
      return;
    }
    const log = $("#log");
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 30;
    status.lines.forEach(line => {
      const span = el("span", logClass(line), line + "\n");
      log.appendChild(span);
    });
    state.logOffset = status.log_total;
    if (atBottom) log.scrollTop = log.scrollHeight;

    const wasRunning = state.running;
    state.running = status.running;
    $("#btn-generate").disabled = status.running;
    $("#btn-stop").disabled = !status.running;

    const label = $("#run-state");
    label.removeAttribute("data-i18n");
    if (status.running) label.textContent = t("log.running");
    else if (status.cancelled) label.textContent = t("log.cancelled");
    else if (status.returncode === 0) label.textContent = t("log.done");
    else if (status.returncode === null) label.textContent = t("log.idle");
    else label.textContent = t("log.failed") + " (" + status.returncode + ")";

    if (wasRunning && !status.running) {
      await refreshState();
      if (state.selectedInput) loadOutputTree(state.selectedInput).then(autoOpenDataset);
    }
    if (status.running) setTimeout(pollRun, 1000);
  }

  // ── header state ─────────────────────────────────────────────────────
  async function refreshState() {
    const data = await api("/api/state");
    $("#path-input").textContent = data.input_dir;
    $("#path-output").textContent = data.output_dir;

    const inStats = data.input_stats;
    const kinds = inStats.by_ext.filter(e => [".pdf", ".ifc"].indexOf(e.ext) !== -1)
      .map(e => `${e.ext.slice(1)} ${e.count}`).join(" · ");
    $("#stats-input").textContent =
      `${kinds ? kinds + " · " : ""}${fmtNum(inStats.files)} ${t("stat.files")} · ${fmtBytes(inStats.bytes)}`;

    const out = data.output_stats;
    const ds = out.datasets.filter(d => d.records).map(d => `${d.kind} ${fmtNum(d.records)}`).join(" · ");
    $("#stats-output").textContent =
      `${ds ? ds + " · " : ""}${fmtNum(out.images.count)} ${t("stat.images")} · ${fmtBytes(out.bytes)}`;
  }

  // ── tabs / modal / theme ─────────────────────────────────────────────
  function switchTab(name) {
    document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
    $("#tab-options").classList.toggle("active", name === "options");
    $("#tab-log").classList.toggle("active", name === "log");
  }

  async function showConfigModal() {
    try {
      const data = await api("/api/config/raw");
      $("#modal-title").textContent = data.path || "config.json";
      $("#modal-body").textContent = data.text;
      $("#modal").classList.remove("hidden");
    } catch (e) { toast(e.message, "err"); }
  }

  function initTheme() {
    const saved = localStorage.getItem("aec.theme") || "dark";
    document.documentElement.dataset.theme = saved;
  }

  // ── boot ─────────────────────────────────────────────────────────────
  function debounce(fn, ms) {
    let timer = null;
    return function () { clearTimeout(timer); timer = setTimeout(fn, ms); };
  }

  document.addEventListener("DOMContentLoaded", async () => {
    initTheme();
    window.I18N.apply();
    $("#btn-lang").textContent = window.I18N.lang.toUpperCase();

    $("#btn-lang").onclick = () => {
      window.I18N.toggle();
      $("#btn-lang").textContent = window.I18N.lang.toUpperCase();
      refreshState().catch(() => {});
    };
    $("#btn-theme").onclick = () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem("aec.theme", next);
      if (window.AECViewer3D) window.AECViewer3D.retheme();
    };
    $("#btn-config").onclick = showConfigModal;
    $("#modal-close").onclick = () => $("#modal").classList.add("hidden");
    $("#modal").onclick = e => { if (e.target.id === "modal") $("#modal").classList.add("hidden"); };

    $("#btn-generate").onclick = startRun;
    $("#btn-stop").onclick = async () => {
      try { await postJSON("/api/run/stop"); toast(t("msg.stopped")); } catch (e) { toast(e.message, "err"); }
    };
    $("#btn-apply").onclick = () => applyOptions(false);
    $("#btn-save").onclick = () => applyOptions(true);
    $("#btn-clear-log").onclick = () => { $("#log").textContent = ""; };

    $("#btn-refresh-input").onclick = () => loadInputTree().catch(e => toast(e.message, "err"));
    $("#btn-refresh-output").onclick = () =>
      loadOutputTree(state.selectedInput).then(autoOpenDataset).catch(e => toast(e.message, "err"));

    $("#filter-input").addEventListener("input", debounce(() => loadInputTree(), 250));
    $("#filter-output").addEventListener("input",
      debounce(() => loadOutputTree(state.selectedInput), 250));

    document.querySelectorAll(".tab").forEach(b => b.onclick = () => switchTab(b.dataset.tab));

    window.addEventListener("aec:lang", () => {
      $("#btn-lang").textContent = window.I18N.lang.toUpperCase();
    });

    try {
      await refreshState();
      await loadInputTree();
      await loadOptions();
      await loadOutputTree("");
      pollRun();
    } catch (e) {
      toast(e.message, "err");
    }
  });
})();
