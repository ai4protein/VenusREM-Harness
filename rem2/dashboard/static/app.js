(function () {
  "use strict";

  var PAGE_SIZE = 80;
  var POLL_MS = 4000;
  var HEALTH_MS = 15000;
  var MOL_SOURCES = [
    "vendor/3Dmol-min.js",
    "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js",
    "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js",
  ];
  var HYDRO_KD = {
    I: 4.5, V: 4.2, L: 3.8, F: 2.8, C: 2.5, M: 1.9, A: 1.8, G: -0.4,
    T: -0.7, S: -0.8, W: -0.9, Y: -1.3, P: -1.6, H: -3.2, E: -3.5,
    Q: -3.5, D: -3.5, N: -3.5, K: -3.9, R: -4.5,
  };
  var STORE_RUN = "rem2.activeRunId";

  var FALLBACK_RECIPES = [
    { id: "full", label: "Full rem2", argv: [] },
    {
      id: "raw",
      label: "Raw backbone",
      argv: ["--alpha", "0", "--background_weight", "0", "--no_rsa_decay", "--no_plddt_decay"],
    },
    {
      id: "msa",
      label: "+ MSA",
      argv: ["--background_weight", "0", "--no_rsa_decay", "--no_plddt_decay"],
    },
    {
      id: "ccd",
      label: "+ MSA + CCD",
      argv: ["--no_rsa_decay", "--no_plddt_decay"],
    },
  ];

  var FALLBACK_MODELS = [
    { name: "esm2", description: "ESM-2 650M", supports_mask: true, needs_pdb: false, size_hint: "first download ~2.5 GB" },
    { name: "esm2-8m", description: "ESM-2 8M", supports_mask: true, needs_pdb: false, size_hint: "first download ~30 MB" },
    { name: "saprot", description: "SaProt", supports_mask: true, needs_pdb: true },
    { name: "venusrem2", description: "Official ProSST ensemble", supports_mask: false, needs_pdb: true },
    { name: "proteinmpnn", description: "ProteinMPNN", supports_tf: true, needs_pdb: true },
  ];

  var PIPELINE = [
    { id: "fwd", label: "Forward", on: ["full", "raw", "msa", "ccd"] },
    { id: "msa", label: "MSA (α)", on: ["full", "msa", "ccd"] },
    { id: "ccd", label: "CCD (β)", on: ["full", "ccd"] },
    { id: "rsa", label: "RSA", on: ["full"] },
    { id: "plddt", label: "pLDDT", on: ["full"] },
    { id: "export", label: "Export", on: ["full", "raw", "msa", "ccd"] },
  ];

  var state = {
    route: { page: "runs" },
    health: null,
    connected: false,
    models: FALLBACK_MODELS.slice(),
    recipes: FALLBACK_RECIPES.slice(),
    runs: [],
    job: null,
    scores: null,
    histogram: null,
    top: null,
    selectedRow: null,
    activeRunId: sessionStorage.getItem(STORE_RUN) || "",
    lastPoll: null,
    sort: "-score",
    offset: 0,
    query: "",
    k: 20,
    molFailed: false,
    pdbText: null,
    structureSource: "",
    fastaSeq: null,
    viewer: null,
    picked: null,
    view: {
      rep: "cartoon",
      color: "spectrum",
      sidechains: "selection",
      surface: false,
      labels: true,
      hetero: true,
      water: false,
      bg: "dark",
    },
    logText: null,
    logMissing: "",
  };

  var els = {};
  var pollTimer = null;
  var healthTimer = null;
  var scoresAbort = null;
  var topAbort = null;
  var lastFocus = null;
  var workspaceBound = false;
  var selectTimer = null;

  function $(id) {
    return document.getElementById(id);
  }

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function pad2(n) {
    return (n < 10 ? "0" : "") + n;
  }

  function fmtClock(date) {
    return pad2(date.getHours()) + ":" + pad2(date.getMinutes()) + ":" + pad2(date.getSeconds());
  }

  function parseDate(value) {
    if (value == null || value === "") return null;
    if (typeof value === "number") {
      return new Date(value < 1e12 ? value * 1000 : value);
    }
    var d = new Date(value);
    return isNaN(d.getTime()) ? null : d;
  }

  function fmtTime(value) {
    var d = parseDate(value);
    return d ? d.toLocaleString() : value ? String(value) : "—";
  }

  function fmtScore(value) {
    if (value == null || value === "") return "—";
    var n = Number(value);
    if (!isFinite(n)) return esc(value);
    return n.toFixed(4);
  }

  function csvEscape(value) {
    var s = value == null ? "" : String(value);
    if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  }

  function setHidden(el, hidden) {
    if (!el) return;
    if (hidden) el.setAttribute("hidden", "");
    else el.removeAttribute("hidden");
  }

  function flash(msg) {
    if (!els.flash) return;
    if (!msg) {
      setHidden(els.flash, true);
      els.flash.textContent = "";
      return;
    }
    els.flash.textContent = msg;
    setHidden(els.flash, false);
  }

  function isLive(status) {
    return status === "queued" || status === "running";
  }

  function anyLive(runs) {
    var list = runs || state.runs;
    for (var i = 0; i < list.length; i++) {
      if (isLive(list[i].status)) return true;
    }
    return false;
  }

  function badge(status) {
    var s = String(status || "unknown");
    return '<span class="badge badge-' + esc(s) + '">' + esc(s) + "</span>";
  }

  function fileName(input) {
    return input && input.files && input.files[0] ? input.files[0].name : "";
  }

  function modelByName(name) {
    var key = String(name || "").toLowerCase();
    for (var i = 0; i < state.models.length; i++) {
      var m = state.models[i];
      if (String(m.name || "").toLowerCase() === key) return m;
      var aliases = m.aliases || [];
      for (var j = 0; j < aliases.length; j++) {
        if (String(aliases[j]).toLowerCase() === key) return m;
      }
    }
    return null;
  }

  function recipeById(id) {
    for (var i = 0; i < state.recipes.length; i++) {
      if (state.recipes[i].id === id) return state.recipes[i];
    }
    return state.recipes[0] || FALLBACK_RECIPES[0];
  }

  function jobSequence(job) {
    if (!job) return state.fastaSeq || "";
    if (job.sequence) return String(job.sequence);
    var proteins = job.proteins;
    if (Array.isArray(proteins) && proteins[0]) {
      return String(proteins[0].sequence || proteins[0].aa_seq || proteins[0].seq || "");
    }
    if (proteins && typeof proteins === "object") {
      var keys = Object.keys(proteins);
      if (keys.length) {
        var p = proteins[keys[0]];
        if (typeof p === "string") return p;
        if (p) return String(p.sequence || p.aa_seq || "");
      }
    }
    return state.fastaSeq || "";
  }

  function parseMutants(mutant) {
    if (!mutant) return [];
    var parts = String(mutant).split(/[:;,]/);
    var out = [];
    for (var i = 0; i < parts.length; i++) {
      var raw = parts[i].trim();
      var m = raw.match(/^([A-Z*])(\d+)([A-Z*])$/i);
      if (m) {
        out.push({
          wt: m[1].toUpperCase(),
          pos: parseInt(m[2], 10),
          mut: m[3].toUpperCase(),
          raw: raw,
        });
      }
    }
    return out;
  }

  function mutantField(row) {
    if (!row) return "";
    return row.mutant || row.Mutant || row.aa_mut || row.substitution || row.mutations || "";
  }

  function rowScore(row, primary) {
    if (!row) return null;
    if (primary && row[primary] != null && row[primary] !== "") return row[primary];
    if (row.score != null && row.score !== "") return row.score;
    var keys = Object.keys(row);
    for (var i = 0; i < keys.length; i++) {
      if (/rem2$/i.test(keys[i]) || keys[i] === "score") return row[keys[i]];
    }
    return null;
  }

  function rowDms(row) {
    if (!row) return null;
    if (row.DMS_score != null) return row.DMS_score;
    if (row.dms_score != null) return row.dms_score;
    if (row.DMS != null) return row.DMS;
    return null;
  }

  function showDms(job, rows) {
    if (job && job.has_dms) return true;
    var sample = rows && rows[0];
    return sample != null && rowDms(sample) != null;
  }

  function primaryScore(job, payload) {
    if (payload && payload.primary_score) return payload.primary_score;
    if (job && job.primary_score) return job.primary_score;
    if (job && job.score_columns && job.score_columns.length) return job.score_columns[0];
    return "score";
  }

  function shellArg(value) {
    var s = String(value);
    if (/^[A-Za-z0-9_./:+,-]+$/.test(s)) return s;
    return "'" + s.replace(/'/g, "'\\''") + "'";
  }

  function buildCli(opts) {
    if (opts.demo) return "rem2 demo --model esm2-8m";
    var parts = ["rem2", "--model", shellArg(opts.model || "esm2")];
    if (opts.fasta) parts.push("--fasta", shellArg(opts.fasta));
    if (opts.pdb) parts.push("--pdb", shellArg(opts.pdb));
    if (!opts.pdb && (opts.pdb_id || opts.uniprot_id) && opts.fetch_structure !== "none") {
      parts.push("--pdb", "<fetched.pdb>");
    }
    if (opts.msa) parts.push("--aa_seq_aln", shellArg(opts.msa));
    if (opts.mutants) {
      parts.push("--mutants", shellArg(opts.mutants));
    } else {
      parts.push("--mutant_sites", shellArg(opts.mutant_sites || "1"));
      if (opts.positions) parts.push("--positions", shellArg(opts.positions));
      if (opts.residue_range) parts.push("--residue_range", shellArg(opts.residue_range));
    }
    if (opts.scoring_strategy && opts.scoring_strategy !== "wt") {
      parts.push("--scoring_strategy", shellArg(opts.scoring_strategy));
    }
    if (opts.max_mutants) parts.push("--max_mutants", shellArg(opts.max_mutants));
    var recipe = recipeById(opts.recipe);
    var argv = (recipe && recipe.argv) || [];
    for (var i = 0; i < argv.length; i++) parts.push(shellArg(argv[i]));
    return parts.join(" ");
  }

  function formSnapshot() {
    var form = els.form;
    if (!form) {
      return { model: "esm2", recipe: "full", mutant_sites: "1", scoring_strategy: "wt" };
    }
    var data = new FormData(form);
    return {
      model: data.get("model") || "esm2",
      recipe: data.get("recipe") || "full",
      fasta: fileName(form.fasta),
      pdb: fileName(form.pdb),
      mutants: fileName(form.mutants),
      msa: fileName(form.msa),
      mutant_sites: (data.get("mutant_sites") || "1").toString().trim(),
      positions: (data.get("positions") || "").toString().trim(),
      residue_range: (data.get("residue_range") || "").toString().trim(),
      scoring_strategy: data.get("scoring_strategy") || "wt",
      max_mutants: (data.get("max_mutants") || "").toString().trim(),
      pdb_id: (data.get("pdb_id") || "").toString().trim(),
      uniprot_id: (data.get("uniprot_id") || "").toString().trim(),
      fetch_structure: (data.get("fetch_structure") || "auto").toString().trim(),
    };
  }

  function updateCli() {
    if (!els.cli) return;
    els.cli.textContent = buildCli(formSnapshot());
  }

  function updateFileLabels() {
    var nodes = document.querySelectorAll(".file-name[data-for]");
    for (var i = 0; i < nodes.length; i++) {
      var id = nodes[i].getAttribute("data-for");
      var input = $(id);
      var name = fileName(input);
      if (id === "file-mutants") {
        nodes[i].textContent = name || "empty → saturation";
      } else if (id === "file-msa" || id === "file-pdb") {
        nodes[i].textContent = name || "optional";
      } else {
        nodes[i].textContent = name || "choose a file";
      }
      var field = input && input.closest(".file-field");
      if (field) field.classList.toggle("has-file", !!name);
    }
    var sat = $("saturation-fields");
    if (sat) sat.classList.toggle("is-off", !!fileName($("file-mutants")));
  }

  function updateModelMeta() {
    var sel = els.modelSelect;
    var meta = $("model-meta");
    var hint = $("strategy-hint");
    if (!sel || !meta) return;
    var spec = modelByName(sel.value);
    if (!spec) {
      meta.textContent = "";
      if (hint) hint.textContent = "";
      return;
    }
    var bits = [];
    if (spec.description) bits.push(spec.description);
    if (spec.default_model_id) bits.push("id " + spec.default_model_id);
    if (spec.needs_pdb) bits.push("PDB required");
    if (spec.extras) bits.push("extras: " + spec.extras);
    if (spec.size_hint) bits.push(spec.size_hint);
    if (spec.notes) bits.push(spec.notes);
    meta.textContent = bits.join(" · ");
    if (hint) {
      var strat = ($("f-strategy") && $("f-strategy").value) || "wt";
      var warns = [];
      if (strat === "mask" && spec.supports_mask === false) {
        warns.push("This backbone does not advertise mask forwards.");
      }
      if (strat === "tf" && spec.supports_tf === false) {
        warns.push("tf is for ProteinMPNN-style models.");
      }
      hint.textContent = warns.join(" ");
    }
  }

  async function api(path, options) {
    var opts = options || {};
    var res = await fetch(path, opts);
    if (!res.ok) {
      var text = "";
      try {
        text = await res.text();
      } catch (err) {
        text = "";
      }
      var err = new Error(text || res.status + " " + res.statusText);
      err.status = res.status;
      err.body = text;
      throw err;
    }
    return res;
  }

  async function apiJson(path, options) {
    var res = await api(path, options);
    return res.json();
  }

  function setActiveRun(id) {
    state.activeRunId = id || "";
    if (id) sessionStorage.setItem(STORE_RUN, id);
    else sessionStorage.removeItem(STORE_RUN);
    updateSelectTab();
  }

  function enableRunTab(tab, hrefWhenOn, hrefWhenOff) {
    if (!tab) return;
    var id = state.activeRunId;
    if (id) {
      tab.href = hrefWhenOn.replace("{id}", encodeURIComponent(id));
      tab.classList.remove("is-disabled");
      tab.removeAttribute("aria-disabled");
      tab.removeAttribute("title");
    } else {
      tab.href = hrefWhenOff;
      tab.classList.add("is-disabled");
      tab.setAttribute("aria-disabled", "true");
      tab.title = "Choose a run first";
    }
  }

  function updateSelectTab() {
    enableRunTab(els.tabSelect, "#/select/{id}", "#/select");
    enableRunTab(els.tabStructure, "#/structure/{id}", "#/structure");
  }

  function setTabs(page) {
    var map = {
      runs: els.tabRuns,
      workspace: els.tabRuns,
      predict: els.tabPredict,
      select: els.tabSelect,
      structure: els.tabStructure,
    };
    var tabs = [els.tabRuns, els.tabPredict, els.tabSelect, els.tabStructure];
    for (var i = 0; i < tabs.length; i++) {
      if (tabs[i]) tabs[i].classList.remove("is-active");
    }
    if (map[page]) map[page].classList.add("is-active");
  }

  function showPage(id) {
    var pages = ["page-runs", "page-predict", "page-workspace", "page-select", "page-structure"];
    for (var i = 0; i < pages.length; i++) {
      setHidden($(pages[i]), pages[i] !== id);
    }
  }

  function parseHash() {
    var raw = (location.hash || "#/").replace(/^#/, "");
    if (!raw || raw === "/") return { page: "runs" };
    var parts = raw.split("/").filter(Boolean);
    if (parts[0] === "predict") return { page: "predict" };
    if (parts[0] === "select") {
      if (parts[1]) return { page: "select", id: decodeURIComponent(parts[1]) };
      return { page: "select" };
    }
    if (parts[0] === "structure") {
      if (parts[1]) return { page: "structure", id: decodeURIComponent(parts[1]) };
      return { page: "structure" };
    }
    if (parts[0] === "runs") {
      if (parts[1]) return { page: "workspace", id: decodeURIComponent(parts[1]) };
      return { page: "runs" };
    }
    return { page: "runs" };
  }

  function go(hash) {
    if (location.hash === hash) onRoute();
    else location.hash = hash;
  }

  function updateChrome() {
    var conn = els.conn;
    if (conn) {
      conn.textContent = state.connected ? "Connected" : "Offline";
      conn.classList.toggle("is-on", state.connected);
      conn.classList.toggle("is-off", !state.connected);
    }
    var jobPill = els.jobStatus;
    var running = anyLive();
    if (jobPill) {
      setHidden(jobPill, !running);
    }
    if (els.lastPoll) {
      els.lastPoll.textContent = state.lastPoll ? "poll " + fmtClock(state.lastPoll) : "poll --:--:--";
    }
    if (els.appVersion && state.health && state.health.version) {
      els.appVersion.textContent = state.health.version;
      setHidden(els.appVersion, false);
    }
    updateSelectTab();
  }

  async function refreshHealth() {
    try {
      var data = await apiJson("/api/health");
      state.health = data;
      state.connected = !!(data && data.ok !== false);
      state.lastPoll = new Date();
    } catch (err) {
      state.connected = false;
    }
    updateChrome();
  }

  async function refreshRuns() {
    var data = await apiJson("/api/runs");
    state.runs = (data && data.runs) || [];
    state.lastPoll = new Date();
    state.connected = true;
    updateChrome();
    if (state.route.page === "runs") renderRuns();
    return state.runs;
  }

  function fillModels() {
    var sel = els.modelSelect;
    if (!sel) return;
    var prev = sel.value || "esm2";
    sel.innerHTML = "";
    for (var i = 0; i < state.models.length; i++) {
      var m = state.models[i];
      var opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = m.name;
      sel.appendChild(opt);
    }
    if (modelByName(prev)) sel.value = prev;
    else if (modelByName("esm2")) sel.value = "esm2";
    updateModelMeta();
    updateCli();
  }

  function fillRecipes() {
    var host = $("recipe-radios");
    if (!host) return;
    var current = formSnapshot().recipe || "full";
    host.innerHTML = "";
    for (var i = 0; i < state.recipes.length; i++) {
      var r = state.recipes[i];
      var lab = document.createElement("label");
      lab.className = "seg-item";
      var input = document.createElement("input");
      input.type = "radio";
      input.name = "recipe";
      input.value = r.id;
      if (r.id === current) input.checked = true;
      lab.appendChild(input);
      lab.appendChild(document.createTextNode(" " + r.label));
      host.appendChild(lab);
    }
    if (!host.querySelector("input:checked")) {
      var first = host.querySelector("input");
      if (first) first.checked = true;
    }
    updateCli();
  }

  async function loadCatalog() {
    try {
      var models = await apiJson("/api/models");
      if (models && Array.isArray(models.models) && models.models.length) {
        state.models = models.models;
      }
    } catch (err) {
      /* keep fallback */
    }
    try {
      var recipes = await apiJson("/api/recipes");
      if (recipes && Array.isArray(recipes.recipes) && recipes.recipes.length) {
        state.recipes = recipes.recipes;
      }
    } catch (err) {
      /* keep fallback */
    }
    fillModels();
    fillRecipes();
  }

  function renderRuns() {
    var host = $("runs-body");
    if (!host) return;
    var runs = state.runs || [];
    if (!runs.length) {
      host.innerHTML =
        '<div class="empty-state"><h2>No runs yet</h2>' +
        "<p>Submit a FASTA, or fetch a structure from RCSB / AlphaFold DB.</p>" +
        '<a class="btn btn-primary" href="#/predict">New job</a></div>';
      return;
    }
    var rows = runs
      .map(function (run) {
        var id = encodeURIComponent(run.id);
        return (
          "<tr data-id=\"" +
          esc(run.id) +
          "\">" +
          '<td class="mono">' +
          esc(run.protein || "—") +
          "</td>" +
          "<td>" +
          esc(run.model || "—") +
          "</td>" +
          "<td>" +
          esc(run.recipe || "—") +
          "</td>" +
          '<td class="num">' +
          esc(run.n_mutants != null ? run.n_mutants : "—") +
          "</td>" +
          "<td>" +
          badge(run.status) +
          "</td>" +
          "<td>" +
          esc(fmtTime(run.created_at)) +
          "</td>" +
          "<td>" +
          '<a class="btn-link" href="#/runs/' +
          id +
          '">Open</a>' +
          '<a class="btn-link" href="#/select/' +
          id +
          '">Select</a>' +
          '<a class="btn-link" href="#/structure/' +
          id +
          '">Structure</a>' +
          "</td>" +
          "</tr>"
        );
      })
      .join("");
    host.innerHTML =
      '<div class="table-scroll"><table class="data" id="runs-table"><thead><tr>' +
      "<th>protein</th><th>model</th><th>recipe</th><th>mutants</th>" +
      "<th>status</th><th>created</th><th>actions</th>" +
      "</tr></thead><tbody>" +
      rows +
      "</tbody></table></div>";
  }

  function sortMark(key) {
    var cur = state.sort || "-score";
    var desc = cur.charAt(0) === "-";
    var col = desc ? cur.slice(1) : cur;
    if (col !== key) return "";
    return desc ? " ↓" : " ↑";
  }

  function thSort(key, label, extraClass) {
    var cur = state.sort || "-score";
    var desc = cur.charAt(0) === "-";
    var col = desc ? cur.slice(1) : cur;
    var cls = "sortable" + (col === key ? " is-sort" : "") + (extraClass ? " " + extraClass : "");
    return (
      '<th class="' +
      cls +
      '" data-sort="' +
      esc(key) +
      '">' +
      esc(label) +
      sortMark(key) +
      "</th>"
    );
  }

  function renderSequence(seq, positions) {
    if (!seq) return '<p class="muted">Sequence not available.</p>';
    var set = {};
    for (var i = 0; i < (positions || []).length; i++) set[positions[i]] = true;
    var picked = state.picked && state.picked.resi;
    var html = "";
    for (var idx = 0; idx < seq.length; idx++) {
      var pos = idx + 1;
      if (idx % 50 === 0) {
        if (idx) html += "\n";
        html += '<span class="seq-pos">' + String(pos).padStart(4, " ") + "</span> ";
      } else if (idx % 10 === 0) {
        html += " ";
      }
      var ch = esc(seq.charAt(idx));
      var cls = "seq-aa";
      if (set[pos]) cls += " seq-mut";
      if (picked === pos) cls += " seq-pick";
      html += '<span class="' + cls + '" data-pos="' + pos + '" title="' + pos + '">' + ch + "</span>";
    }
    return '<pre class="seq">' + html + "</pre>";
  }

  function drawHistogram(canvas, data) {
    if (!canvas || !data) return;
    var bins = data.bins || [];
    var dpr = window.devicePixelRatio || 1;
    var cssW = canvas.clientWidth || canvas.parentElement.clientWidth || 320;
    var cssH = 150;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    canvas._bins = bins;
    canvas._w = cssW;
    canvas._h = cssH;
    if (!bins.length) {
      ctx.fillStyle = "#6d675c";
      ctx.font = "13px ui-sans-serif, system-ui, sans-serif";
      ctx.fillText("No score histogram yet.", 8, 24);
      return;
    }
    var max = 1;
    for (var i = 0; i < bins.length; i++) max = Math.max(max, Number(bins[i].count) || 0);
    var pad = { l: 8, r: 8, t: 8, b: 18 };
    var innerW = cssW - pad.l - pad.r;
    var innerH = cssH - pad.t - pad.b;
    var gap = bins.length > 40 ? 0 : 1;
    var barW = innerW / bins.length;
    ctx.fillStyle = "#145246";
    for (var b = 0; b < bins.length; b++) {
      var h = ((Number(bins[b].count) || 0) / max) * innerH;
      var x = pad.l + b * barW;
      var y = pad.t + (innerH - h);
      ctx.fillRect(x, y, Math.max(barW - gap, 0.5), h);
    }
    ctx.fillStyle = "#6d675c";
    ctx.font = "11px ui-monospace, Menlo, Consolas, monospace";
    var lo = data.min != null ? data.min : bins[0].lo;
    var hi = data.max != null ? data.max : bins[bins.length - 1].hi;
    ctx.fillText(fmtScore(lo), pad.l, cssH - 4);
    var hiText = fmtScore(hi);
    ctx.fillText(hiText, cssW - pad.r - ctx.measureText(hiText).width, cssH - 4);
  }

  function histTip(canvas, ev) {
    var bins = canvas._bins;
    if (!bins || !bins.length) return "";
    var rect = canvas.getBoundingClientRect();
    var x = ev.clientX - rect.left;
    var padL = 8;
    var innerW = (canvas._w || rect.width) - 16;
    var idx = Math.floor(((x - padL) / innerW) * bins.length);
    if (idx < 0 || idx >= bins.length) return "";
    var bin = bins[idx];
    return fmtScore(bin.lo) + " … " + fmtScore(bin.hi) + "  n=" + (bin.count || 0);
  }

  function pipelineHtml(recipe) {
    var rec = String(recipe || "full");
    var html = [];
    for (var i = 0; i < PIPELINE.length; i++) {
      if (i) html.push('<span class="chip-arrow">→</span>');
      var step = PIPELINE[i];
      var on = step.on.indexOf(rec) !== -1;
      html.push('<span class="chip' + (on ? "" : " is-dim") + '">' + esc(step.label) + "</span>");
    }
    return '<div class="chips" aria-label="Pipeline">' + html.join("") + "</div>";
  }

  function scoreTableHtml(job, payload) {
    var rows = (payload && payload.rows) || [];
    var total = (payload && payload.total) || rows.length;
    var offset = (payload && payload.offset) != null ? payload.offset : state.offset;
    var primary = primaryScore(job, payload);
    var dms = showDms(job, rows);
    var selMut = mutantField(state.selectedRow);
    var head =
      "<tr>" +
      thSort("mutant", "mutant") +
      thSort("score", "rem2 score") +
      (dms ? thSort("DMS_score", "DMS_score") : "") +
      thSort("rank", "rank") +
      "</tr>";
    var body = rows
      .map(function (row, i) {
        var mut = mutantField(row);
        var rank = row.rank != null ? row.rank : offset + i + 1;
        var sel = mut && mut === selMut ? " is-sel" : "";
        return (
          '<tr class="' +
          sel +
          '" data-i="' +
          i +
          '">' +
          '<td class="mono">' +
          esc(mut || "—") +
          "</td>" +
          '<td class="num">' +
          fmtScore(rowScore(row, primary)) +
          "</td>" +
          (dms ? '<td class="num">' + fmtScore(rowDms(row)) + "</td>" : "") +
          '<td class="num">' +
          esc(rank) +
          "</td>" +
          "</tr>"
        );
      })
      .join("");
    if (!body) {
      body = '<tr><td colspan="' + (dms ? 4 : 3) + '"><div class="empty">No rows.</div></td></tr>';
    }
    var from = total ? offset + 1 : 0;
    var to = Math.min(offset + rows.length, total);
    return (
      '<div class="table-scroll"><table class="data" id="score-table"><thead>' +
      head +
      "</thead><tbody>" +
      body +
      "</tbody></table></div>" +
      '<div class="pager">' +
      "<span>" +
      from +
      "–" +
      to +
      " of " +
      total +
      (primary && primary !== "score" ? " · " + esc(primary) : "") +
      "</span>" +
      '<span class="btn-row">' +
      '<button type="button" class="btn" id="btn-prev"' +
      (offset <= 0 ? " disabled" : "") +
      ">Prev</button>" +
      '<button type="button" class="btn" id="btn-next"' +
      (offset + PAGE_SIZE >= total ? " disabled" : "") +
      ">Next</button>" +
      "</span></div>"
    );
  }

  function selectedDetailHtml(job, row) {
    if (!row) return '<p class="muted">Click a row to inspect substitutions.</p>';
    var muts = parseMutants(mutantField(row));
    var primary = primaryScore(job, state.scores);
    var bits = muts
      .map(function (m) {
        return m.wt + m.pos + m.mut + " (1-based " + m.pos + ")";
      })
      .join("  ·  ");
    return (
      '<div class="detail">' +
      esc(mutantField(row) || "—") +
      "  score=" +
      fmtScore(rowScore(row, primary)) +
      (rowDms(row) != null ? "  DMS=" + fmtScore(rowDms(row)) : "") +
      (bits ? "<br>" + esc(bits) : "") +
      "</div>"
    );
  }

  function renderWorkspace() {
    var host = $("workspace-body");
    if (!host) return;
    var job = state.job;
    if (!job) {
      host.innerHTML = '<div class="empty">Run not loaded.</div>';
      return;
    }
    var nMut = job.n_mutants != null ? job.n_mutants : job.n_rows;
    var primary = primaryScore(job, state.scores);
    var seq = jobSequence(job);
    var muts = parseMutants(mutantField(state.selectedRow));
    var positions = muts.map(function (m) {
      return m.pos;
    });
    var viewerBlock = "";
    if (job.has_pdb && !state.molFailed) {
      viewerBlock =
        '<div class="panel view-box">' +
        '<div class="toolbar"><h2 style="margin:0;flex:1">Structure</h2>' +
        '<a class="btn" href="#/structure/' +
        encodeURIComponent(job.id) +
        '">Open PyMOL view</a></div>' +
        originNoteHtml(job) +
        molPanelHtml("compact") +
        "</div>";
    } else {
      viewerBlock =
        '<div class="panel view-box"><p class="muted">No PDB in this run. ' +
        '<a href="#/structure/' +
        encodeURIComponent(job.id) +
        '">Fetch from RCSB or AlphaFold DB</a>.</p></div>';
    }
    var err = job.error
      ? '<div class="error-box">' + esc(job.error) + "</div>"
      : "";
    var cancel =
      isLive(job.status)
        ? '<button type="button" class="btn btn-danger" id="btn-cancel">Cancel</button>'
        : "";
    host.innerHTML =
      '<div class="page-head"><h1 class="mono">' +
      esc(job.protein || job.id) +
      "</h1></div>" +
      '<div class="panel overview">' +
      ov("protein", job.protein || "—") +
      ov("model", job.model || "—") +
      '<div class="overview-item"><div class="overview-k">status</div><div class="overview-v">' +
      badge(job.status) +
      "</div></div>" +
      ov("mutants", nMut != null ? nMut : "—") +
      ov("primary score", primary) +
      ov("recipe", job.recipe || "—") +
      "</div>" +
      pipelineHtml(job.recipe) +
      err +
      '<div class="inspector' +
      (viewerBlock ? "" : " has-no-viewer") +
      '">' +
      '<div class="panel seq-box"><h2>Sequence</h2>' +
      renderSequence(seq, positions) +
      "</div>" +
      '<div class="panel hist-box"><h2>Score distribution</h2>' +
      '<canvas id="hist-canvas" class="hist-canvas"></canvas>' +
      '<div class="hist-meta"><span id="hist-tip"> </span><span>' +
      esc(primary) +
      "</span></div></div>" +
      viewerBlock +
      "</div>" +
      '<div class="panel">' +
      '<div class="toolbar">' +
      '<input type="search" id="score-q" placeholder="Search mutants" title="Press Enter" value="' +
      esc(state.query) +
      '">' +
      '<button type="button" class="btn" id="btn-copy-mutant">Copy mutant</button>' +
      '<button type="button" class="btn" id="btn-copy-selected">Copy selected</button>' +
      '<a class="btn" href="#/select/' +
      encodeURIComponent(job.id) +
      '">Select top-K</a>' +
      cancel +
      (job.has_pdb
        ? '<a class="btn btn-ghost" href="/api/runs/' +
          encodeURIComponent(job.id) +
          '/artifact?kind=pdb' +
          (structureSources(job).query ? "&source=query" : "") +
          '">PDB</a>'
        : "") +
      '<a class="btn btn-ghost" href="/api/runs/' +
      encodeURIComponent(job.id) +
      '/artifact?kind=fasta">FASTA</a>' +
      "</div>" +
      scoreTableHtml(job, state.scores) +
      selectedDetailHtml(job, state.selectedRow) +
      "</div>" +
      (job.argv
        ? '<details class="advanced"><summary>Job argv</summary><pre class="cli">' +
          esc(Array.isArray(job.argv) ? job.argv.join(" ") : job.argv) +
          "</pre></details>"
        : "") +
      (state.logText
        ? '<details class="advanced"><summary>Log</summary><pre class="log-pre">' +
          esc(state.logText) +
          "</pre></details>"
        : "");

    var canvas = $("hist-canvas");
    if (canvas) drawHistogram(canvas, state.histogram || { bins: [] });
    maybeLoadViewer(job, positions);
  }

  function ov(k, v) {
    return (
      '<div class="overview-item"><div class="overview-k">' +
      esc(k) +
      '</div><div class="overview-v" title="' +
      esc(v) +
      '">' +
      esc(v) +
      "</div></div>"
    );
  }

  function bindWorkspaceOnce() {
    if (workspaceBound) return;
    workspaceBound = true;
    var host = $("workspace-body");
    if (!host) return;
    host.addEventListener("click", function (ev) {
      var th = ev.target.closest("th[data-sort]");
      if (th && host.contains(th)) {
        toggleSort(th.getAttribute("data-sort"));
        return;
      }
      var tr = ev.target.closest("#score-table tbody tr[data-i]");
      if (tr) {
        var i = parseInt(tr.getAttribute("data-i"), 10);
        var rows = (state.scores && state.scores.rows) || [];
        state.selectedRow = rows[i] || null;
        highlightSelection();
        return;
      }
      var id = ev.target.id;
      if (id === "btn-prev") {
        state.offset = Math.max(0, state.offset - PAGE_SIZE);
        loadScores().then(patchScores);
      } else if (id === "btn-next") {
        state.offset += PAGE_SIZE;
        loadScores().then(patchScores);
      } else if (id === "btn-copy-mutant") {
        copyMutant();
      } else if (id === "btn-copy-selected") {
        copySelected();
      } else if (id === "btn-cancel") {
        cancelJob(state.job && state.job.id);
      }
      var molEl = ev.target.closest("[data-mol]");
      if (molEl && host.contains(molEl)) {
        handleMolControl(molEl);
        return;
      }
      var aa = ev.target.closest(".seq-aa[data-pos]");
      if (aa && host.contains(aa)) {
        pickSequencePos(parseInt(aa.getAttribute("data-pos"), 10));
      }
    });
    host.addEventListener("change", function (ev) {
      var molEl = ev.target.closest("[data-mol]");
      if (molEl) handleMolControl(molEl);
    });
    host.addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter" || ev.target.id !== "score-q") return;
      state.query = ev.target.value.trim();
      state.offset = 0;
      loadScores().then(patchScores);
    });
    host.addEventListener("mousemove", function (ev) {
      if (ev.target.id !== "hist-canvas") return;
      var tip = $("hist-tip");
      if (tip) tip.textContent = histTip(ev.target, ev);
    });
    host.addEventListener("mouseleave", function (ev) {
      if (ev.target.id !== "hist-canvas") return;
      var tip = $("hist-tip");
      if (tip) tip.textContent = " ";
    }, true);
  }

  function patchScores() {
    var table = $("score-table");
    if (!table) {
      renderWorkspace();
      return;
    }
    var wrap = table.closest(".panel");
    if (!wrap) {
      renderWorkspace();
      return;
    }
    var toolbar = wrap.querySelector(".toolbar");
    var qVal = $("score-q") ? $("score-q").value : state.query;
    wrap.innerHTML = "";
    if (toolbar) wrap.appendChild(toolbar);
    var box = document.createElement("div");
    box.innerHTML = scoreTableHtml(state.job, state.scores) + selectedDetailHtml(state.job, state.selectedRow);
    while (box.firstChild) wrap.appendChild(box.firstChild);
    var q = $("score-q");
    if (q) q.value = qVal;
  }

  function highlightSelection() {
    var job = state.job;
    var seqHost = document.querySelector(".seq-box");
    if (seqHost) {
      var h2 = seqHost.querySelector("h2");
      seqHost.innerHTML = "";
      if (h2) seqHost.appendChild(h2);
      else {
        var title = document.createElement("h2");
        title.textContent = "Sequence";
        seqHost.appendChild(title);
      }
      var tmp = document.createElement("div");
      var muts = parseMutants(mutantField(state.selectedRow));
      tmp.innerHTML = renderSequence(
        jobSequence(job),
        muts.map(function (m) {
          return m.pos;
        })
      );
      while (tmp.firstChild) seqHost.appendChild(tmp.firstChild);
    }
    var panel = $("score-table") && $("score-table").closest(".panel");
    if (panel) {
      var next = selectedDetailHtml(job, state.selectedRow);
      var old = panel.querySelector(".detail") || panel.querySelector(":scope > .muted");
      var holder = document.createElement("div");
      holder.innerHTML = next;
      if (old) old.replaceWith(holder.firstChild);
      else panel.appendChild(holder.firstChild);
    }
    var rows = document.querySelectorAll("#score-table tbody tr");
    var sel = mutantField(state.selectedRow);
    for (var i = 0; i < rows.length; i++) {
      rows[i].classList.toggle("is-sel", mutantField(((state.scores && state.scores.rows) || [])[i]) === sel);
    }
    applyView({ zoom: "sel" });
  }

  function toggleSort(key) {
    var cur = state.sort || "-score";
    var desc = cur.charAt(0) === "-";
    var col = desc ? cur.slice(1) : cur;
    if (col === key) state.sort = desc ? key : "-" + key;
    else state.sort = key === "mutant" || key === "rank" ? key : "-" + key;
    state.offset = 0;
    loadScores().then(patchScores);
  }

  function copyText(text) {
    if (!text) {
      flash("Select a row first.");
      return;
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () {
          flash("");
        },
        function () {
          fallbackCopy(text);
        }
      );
    } else {
      fallbackCopy(text);
    }
  }

  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.left = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand("copy");
    } catch (err) {
      flash("Copy failed.");
    }
    document.body.removeChild(ta);
  }

  function copyMutant() {
    copyText(mutantField(state.selectedRow));
  }

  function copySelected() {
    var row = state.selectedRow;
    if (!row) {
      flash("Select a row first.");
      return;
    }
    var primary = primaryScore(state.job, state.scores);
    var parts = [mutantField(row), rowScore(row, primary), row.rank != null ? row.rank : ""];
    if (rowDms(row) != null) parts.push(rowDms(row));
    copyText(parts.join("\t"));
  }

  async function cancelJob(id) {
    if (!id) return;
    if (!window.confirm("Cancel this job?")) return;
    try {
      await api("/api/runs/" + encodeURIComponent(id) + "/cancel", { method: "POST" });
      await loadWorkspace(id, { keepSelection: true });
    } catch (err) {
      flash(err.message || "Cancel failed.");
    }
  }

  function teardownViewer() {
    document.body.classList.remove("mol-fs");
    if (state.viewer) {
      try {
        state.viewer.clear();
      } catch (err) {
        /* ignore */
      }
      state.viewer = null;
    }
  }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement("script");
      s.src = src;
      s.async = true;
      s.onload = function () {
        resolve();
      };
      s.onerror = function () {
        reject(new Error("script"));
      };
      document.head.appendChild(s);
    });
  }

  async function ensureMol() {
    if (state.molFailed) return null;
    if (window.$3Dmol) return window.$3Dmol;
    for (var i = 0; i < MOL_SOURCES.length; i++) {
      try {
        await loadScript(MOL_SOURCES[i]);
        if (window.$3Dmol) return window.$3Dmol;
      } catch (err) {
        /* try next */
      }
    }
    state.molFailed = true;
    return null;
  }

  function bgColor() {
    if (state.view.bg === "white") return "#f3efe6";
    if (state.view.bg === "black") return "#000000";
    return "#11140f";
  }

  function plddtColor(b) {
    if (b >= 90) return "#0053d6";
    if (b >= 70) return "#65cbf3";
    if (b >= 50) return "#ffdb13";
    return "#ff7d45";
  }

  function hydroColor(resn) {
    var aa = RESN_TO_AA[resn] || resn;
    var v = HYDRO_KD[aa];
    if (v == null) return "#888888";
    var t = (v + 4.5) / 9;
    var r = Math.round(40 + t * 200);
    var b = Math.round(200 - t * 160);
    return "rgb(" + r + ",90," + b + ")";
  }

  var RESN_TO_AA = {
    ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E",
    GLY: "G", HIS: "H", ILE: "I", LEU: "L", LYS: "K", MET: "M", PHE: "F",
    PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V",
  };

  function mutationPositions() {
    return parseMutants(mutantField(state.selectedRow)).map(function (m) {
      return m.pos;
    });
  }

  function selectionResis() {
    var pos = mutationPositions();
    if (state.picked && state.picked.resi && pos.indexOf(state.picked.resi) < 0) {
      pos = pos.concat([state.picked.resi]);
    }
    return pos;
  }

  function structureSources(job) {
    return (job && job.structure_sources) || {};
  }

  function currentSourceKey(job) {
    var sources = structureSources(job);
    if (state.structureSource && sources[state.structureSource]) return state.structureSource;
    if (job && job.preferred_source && sources[job.preferred_source]) return job.preferred_source;
    if (sources.afdb) return "afdb";
    if (sources.query) return "query";
    if (sources.rcsb) return "rcsb";
    return "";
  }

  function currentSourceMeta(job) {
    var key = currentSourceKey(job);
    var sources = structureSources(job);
    return (key && sources[key]) || null;
  }

  function sourceHasPlddt(job) {
    var meta = currentSourceMeta(job);
    if (meta) return !!meta.has_plddt;
    return !!(job && job.has_plddt);
  }

  function colorStyle() {
    var mode = state.view.color;
    if (mode === "spectrum") return { color: "spectrum" };
    if (mode === "chain") return { color: "chainHetatm" };
    if (mode === "ss") return { color: "ssPyMol" };
    if (mode === "white") return { color: "white" };
    if (mode === "plddt") {
      if (!sourceHasPlddt(state.job)) return { color: "spectrum" };
      return {
        colorfunc: function (atom) {
          return plddtColor(atom.b || 0);
        },
      };
    }
    if (mode === "hydrophobic") {
      return {
        colorfunc: function (atom) {
          return hydroColor(atom.resn);
        },
      };
    }
    return { color: "spectrum" };
  }

  function molPanelHtml(kind) {
    var v = state.view;
    var large = kind === "full";
    function opt(name, value, label) {
      return (
        '<option value="' +
        value +
        '"' +
        (v[name] === value ? " selected" : "") +
        ">" +
        label +
        "</option>"
      );
    }
    function chk(name, label) {
      return (
        '<label><input type="checkbox" data-mol="' +
        name +
        '"' +
        (v[name] ? " checked" : "") +
        "> " +
        label +
        "</label>"
      );
    }
    var controls =
      '<div class="mol-controls">' +
      "<h3>Display</h3>" +
      '<label>Representation<select data-mol="rep">' +
      opt("rep", "cartoon", "Cartoon") +
      opt("rep", "cartoon-stick", "Cartoon + sticks") +
      opt("rep", "stick", "Sticks") +
      opt("rep", "line", "Lines") +
      opt("rep", "sphere", "Spheres") +
      "</select></label>" +
      '<label>Color<select data-mol="color">' +
      opt("color", "spectrum", "Spectrum (N→C)") +
      opt("color", "chain", "Chain") +
      opt("color", "ss", "Secondary structure") +
      opt("color", "plddt", sourceHasPlddt(state.job) ? "pLDDT" : "pLDDT (unavailable on crystal)") +
      opt("color", "hydrophobic", "Hydrophobicity") +
      opt("color", "white", "Uniform") +
      "</select></label>" +
      sourceSelectHtml() +
      '<label>Side chains<select data-mol="sidechains">' +
      opt("sidechains", "off", "Hidden") +
      opt("sidechains", "selection", "Selection") +
      opt("sidechains", "polar", "Polar") +
      opt("sidechains", "all", "All") +
      "</select></label>" +
      '<div class="mol-toggles">' +
      chk("surface", "Surface") +
      chk("labels", "Residue labels") +
      chk("hetero", "Hetero / ligand") +
      chk("water", "Waters") +
      "</div>" +
      '<label>Background<select data-mol="bg">' +
      opt("bg", "dark", "Dark") +
      opt("bg", "black", "Black") +
      opt("bg", "white", "White") +
      "</select></label>" +
      '<div class="mol-actions">' +
      '<button type="button" class="btn" data-mol="reset">Reset view</button>' +
      '<button type="button" class="btn" data-mol="zoom">Zoom selection</button>' +
      '<button type="button" class="btn" data-mol="fs">' +
      (document.body.classList.contains("mol-fs") ? "Exit full" : "Fullscreen") +
      "</button>" +
      "</div>" +
      '<p class="mol-hint">Drag rotate · scroll zoom · click a residue or sequence letter to select. Shift-click adds.</p>' +
      "</div>";
    var canvas =
      '<div class="mol-fs-target" id="mol-fs-box">' +
      '<div id="mol-host" class="viewer-host' +
      (large ? " viewer-host-lg" : "") +
      '"></div>' +
      '<div class="mol-legend" id="mol-legend"></div>' +
      '<div class="mol-pick" id="mol-pick"></div></div>';
    if (large) return '<div class="mol-stage">' + controls + canvas + "</div>";
    return (
      '<div class="mol-toolbar">' +
      controls +
      "</div>" +
      canvas
    );
  }

  function sourceSelectHtml() {
    var sources = structureSources(state.job);
    var keys = Object.keys(sources);
    if (keys.length < 2) return "";
    var current = currentSourceKey(state.job);
    function opt(value, label) {
      return (
        '<option value="' +
        value +
        '"' +
        (current === value ? " selected" : "") +
        ">" +
        label +
        "</option>"
      );
    }
    var options = keys
      .map(function (key) {
        var meta = sources[key];
        var label = (meta.label || key) + (meta.has_plddt ? " · pLDDT" : " · no pLDDT");
        return opt(key, label);
      })
      .join("");
    return '<label>Structure<select data-mol="source">' + options + "</select></label>";
  }

  function legendHtml() {
    var mode = state.view.color;
    if (mode === "plddt") {
      if (!sourceHasPlddt(state.job)) {
        return "<span>Crystal / experimental PDB has no pLDDT (B-factor is a temperature factor). Showing spectrum. Fetch an AlphaFold model to color by confidence.</span>";
      }
      return (
        '<span><i class="mol-swatch" style="background:#0053d6"></i>pLDDT ≥90</span>' +
        '<span><i class="mol-swatch" style="background:#65cbf3"></i>70–90</span>' +
        '<span><i class="mol-swatch" style="background:#ffdb13"></i>50–70</span>' +
        '<span><i class="mol-swatch" style="background:#ff7d45"></i>&lt;50</span>'
      );
    }
    if (mode === "spectrum") {
      return "<span>N terminus → C terminus (spectrum)</span>";
    }
    if (mode === "ss") {
      return (
        '<span><i class="mol-swatch" style="background:#e6cd6a"></i>helix</span>' +
        '<span><i class="mol-swatch" style="background:#7ebc6f"></i>sheet</span>' +
        '<span><i class="mol-swatch" style="background:#d0d0d0"></i>loop</span>'
      );
    }
    if (mode === "hydrophobic") {
      return "<span>Kyte–Doolittle: blue hydrophilic → red hydrophobic</span>";
    }
    return "<span>Click a residue to inspect. Mutants from the table stay highlighted.</span>";
  }

  function pickHtml() {
    var p = state.picked;
    if (!p) return "No residue selected.";
    var aa = RESN_TO_AA[p.resn] || "";
    var mut = mutationPositions().indexOf(p.resi) >= 0 ? "  ·  in selected mutant" : "";
    return (
      (p.chain ? p.chain + "/" : "") +
      (aa || p.resn) +
      p.resi +
      (p.resn ? "  " + p.resn : "") +
      (p.b != null && isFinite(p.b)
        ? (sourceHasPlddt(state.job) ? "  pLDDT=" : "  B=") + Number(p.b).toFixed(1)
        : "") +
      mut
    );
  }

  function applyView(opts) {
    opts = opts || {};
    var viewer = state.viewer;
    if (!viewer || !window.$3Dmol) return;
    var mol = window.$3Dmol;
    var style = colorStyle();
    var selPos = selectionResis();
    try {
      viewer.setBackgroundColor(bgColor());
      viewer.removeAllSurfaces();
      viewer.removeAllLabels();
      viewer.setStyle({}, {});
      var protein = { hetflag: false };
      var water = { resn: ["HOH", "WAT", "H2O", "DOD"] };
      if (state.view.rep === "stick") {
        viewer.setStyle(protein, { stick: Object.assign({ radius: 0.18 }, style) });
      } else if (state.view.rep === "line") {
        viewer.setStyle(protein, { line: style });
      } else if (state.view.rep === "sphere") {
        viewer.setStyle(protein, { sphere: Object.assign({ scale: 0.35 }, style) });
      } else {
        viewer.setStyle(protein, { cartoon: Object.assign({ thickness: 0.3 }, style) });
        if (state.view.rep === "cartoon-stick") {
          viewer.setStyle(protein, {
            cartoon: Object.assign({ thickness: 0.3 }, style),
            stick: { radius: 0.08, color: "white" },
          });
        }
      }
      var side = state.view.sidechains;
      if (side === "all") {
        viewer.addStyle(protein, { stick: { radius: 0.12, colorscheme: "amino" } });
      } else if (side === "polar") {
        viewer.addStyle(
          { resn: ["SER", "THR", "ASN", "GLN", "TYR", "TRP", "HIS", "LYS", "ARG", "ASP", "GLU", "CYS"] },
          { stick: { radius: 0.14, colorscheme: "amino" } }
        );
      } else if (side === "selection" && selPos.length) {
        viewer.addStyle({ resi: selPos }, { stick: { radius: 0.16, color: "#d9a441" } });
      }
      if (selPos.length) {
        viewer.addStyle(
          { resi: selPos },
          { cartoon: { color: "#d9a441", thickness: 0.5 } }
        );
      }
      if (state.view.hetero) {
        viewer.setStyle({ hetflag: true, not: water }, { stick: { radius: 0.18, colorscheme: "Jmol" } });
      }
      if (state.view.water) {
        viewer.setStyle(water, { sphere: { scale: 0.25, color: "#7eb6ff" } });
      } else {
        viewer.setStyle(water, {});
      }
      if (state.view.surface) {
        viewer.addSurface(mol.SurfaceType.VDW, { opacity: 0.28, color: "white" }, protein);
      }
      if (state.view.labels && selPos.length) {
        viewer.addResLabels({ resi: selPos }, { fontSize: 12, backgroundColor: "black", showBackground: true });
      }
      if (opts.zoom === "sel" && selPos.length) {
        try {
          viewer.zoomTo({ resi: selPos });
        } catch (err) {
          viewer.zoomTo();
        }
      } else if (opts.zoom === "all") {
        viewer.zoomTo();
      }
      viewer.render();
    } catch (err) {
      /* keep last good frame */
    }
    var legend = $("mol-legend");
    if (legend) legend.innerHTML = legendHtml();
    var pick = $("mol-pick");
    if (pick) pick.textContent = pickHtml();
  }

  function colorViewer(positions) {
    applyView({ zoom: positions && positions.length ? "sel" : undefined });
  }

  function bindViewerClicks(viewer) {
    if (!viewer || !viewer.setClickable) return;
    try {
      viewer.setClickable({}, true, function (atom) {
        if (!atom) return;
        state.picked = {
          chain: atom.chain,
          resi: atom.resi,
          resn: atom.resn,
          b: atom.b,
        };
        applyView();
        refreshSequenceHighlight();
      });
      if (viewer.setHoverable) {
        viewer.setHoverable(
          {},
          true,
          function (atom, _v, event) {
            if (!atom || !viewer) return;
            viewer.removeAllLabels();
            viewer.addLabel((RESN_TO_AA[atom.resn] || atom.resn) + atom.resi, {
              position: atom,
              backgroundColor: "black",
              fontColor: "white",
              fontSize: 12,
            });
            viewer.render();
          },
          function () {
            applyView();
          }
        );
      }
    } catch (err) {
      /* older 3Dmol */
    }
  }

  async function maybeLoadViewer(job, positions) {
    if (!job || (!job.has_pdb && !currentSourceKey(job)) || state.molFailed) return;
    var host = $("mol-host");
    if (!host) return;
    if (state.viewer && !host.querySelector("canvas")) state.viewer = null;
    if (state.viewer && host.querySelector("canvas")) {
      applyView({ zoom: positions && positions.length ? "sel" : undefined });
      return;
    }
    var mol = await ensureMol();
    host = $("mol-host");
    if (!mol || !host) return;
    try {
      if (state.pdbText == null) {
        var src = currentSourceKey(job);
        var q = src ? "&source=" + encodeURIComponent(src) : "";
        var res = await api("/api/runs/" + encodeURIComponent(job.id) + "/artifact?kind=pdb" + q);
        state.pdbText = await res.text();
      }
      host.innerHTML = "";
      var viewer = mol.createViewer(host, {
        backgroundColor: bgColor(),
        antialias: true,
        cartoonQuality: 10,
      });
      var fmt = "pdb";
      if (/\n\s*data_/m.test(state.pdbText) || /\n_atom_site\./.test(state.pdbText)) fmt = "cif";
      viewer.addModel(state.pdbText, fmt);
      state.viewer = viewer;
      bindViewerClicks(viewer);
      applyView({ zoom: positions && positions.length ? "sel" : "all" });
      setTimeout(function () {
        try {
          viewer.resize();
          viewer.render();
        } catch (err) {
          /* ignore */
        }
      }, 40);
    } catch (err) {
      var box = $("mol-fs-box");
      if (box) box.innerHTML = '<p class="muted">Could not load the structure viewer.</p>';
    }
  }

  function handleMolControl(el) {
    var key = el.getAttribute("data-mol");
    if (!key) return;
    if (el.tagName === "SELECT") {
      if (key === "source") {
        state.structureSource = el.value;
        state.pdbText = null;
        teardownViewer();
        maybeLoadViewer(state.job, selectionResis());
        return;
      }
      state.view[key] = el.value;
      applyView();
      return;
    }
    if (el.type === "checkbox") {
      state.view[key] = el.checked;
      applyView();
      return;
    }
    if (key === "reset") {
      applyView({ zoom: "all" });
      return;
    }
    if (key === "zoom") {
      applyView({ zoom: "sel" });
      return;
    }
    if (key === "fs") {
      document.body.classList.toggle("mol-fs");
      var btn = document.querySelector('[data-mol="fs"]');
      if (btn) btn.textContent = document.body.classList.contains("mol-fs") ? "Exit full" : "Fullscreen";
      if (state.viewer) {
        setTimeout(function () {
          try {
            state.viewer.resize();
            state.viewer.render();
          } catch (err) {
            /* ignore */
          }
        }, 30);
      }
    }
  }

  function pickSequencePos(pos) {
    if (!pos) return;
    state.picked = { resi: pos, resn: "", chain: "", b: null };
    var seq = jobSequence(state.job);
    if (seq && pos >= 1 && pos <= seq.length) {
      var aa = seq.charAt(pos - 1);
      state.picked.resn = aa;
    }
    applyView({ zoom: "sel" });
    refreshSequenceHighlight();
  }

  function refreshSequenceHighlight() {
    var host = document.querySelector(".seq-box");
    if (!host) return;
    var h2 = host.querySelector("h2");
    var muts = parseMutants(mutantField(state.selectedRow));
    host.innerHTML = "";
    if (h2) host.appendChild(h2);
    var tmp = document.createElement("div");
    tmp.innerHTML = renderSequence(
      jobSequence(state.job),
      muts.map(function (m) {
        return m.pos;
      })
    );
    while (tmp.firstChild) host.appendChild(tmp.firstChild);
  }

  async function loadScores() {
    var id = state.route.id;
    if (!id) return;
    if (scoresAbort) scoresAbort.abort();
    scoresAbort = new AbortController();
    var params = new URLSearchParams();
    params.set("offset", String(state.offset));
    params.set("limit", String(PAGE_SIZE));
    params.set("sort", state.sort || "-score");
    if (state.query) params.set("q", state.query);
    try {
      state.scores = await apiJson(
        "/api/runs/" + encodeURIComponent(id) + "/scores?" + params.toString(),
        { signal: scoresAbort.signal }
      );
    } catch (err) {
      if (err.name === "AbortError") return;
      state.scores = { rows: [], total: 0, offset: state.offset, limit: PAGE_SIZE };
    }
  }

  async function loadHistogram(id) {
    try {
      state.histogram = await apiJson("/api/runs/" + encodeURIComponent(id) + "/histogram");
    } catch (err) {
      state.histogram = { bins: [] };
    }
  }

  async function loadLog(id) {
    if (state.logMissing === id) return;
    try {
      var res = await api("/api/runs/" + encodeURIComponent(id) + "/log");
      state.logText = await res.text();
      state.logMissing = "";
    } catch (err) {
      if (err.status === 404) state.logMissing = id;
      state.logText = null;
    }
  }

  async function loadFastaIfNeeded(job) {
    if (jobSequence(job)) return;
    try {
      var res = await api("/api/runs/" + encodeURIComponent(job.id) + "/artifact?kind=fasta");
      var text = await res.text();
      state.fastaSeq = parseFasta(text);
    } catch (err) {
      state.fastaSeq = "";
    }
  }

  function parseFasta(text) {
    var lines = String(text || "").split(/\r?\n/);
    var seq = [];
    for (var i = 0; i < lines.length; i++) {
      if (!lines[i] || lines[i].charAt(0) === ">") continue;
      seq.push(lines[i].trim());
    }
    return seq.join("").replace(/[^A-Za-z*]/g, "").toUpperCase();
  }

  async function loadWorkspace(id, opts) {
    opts = opts || {};
    if (!opts.keepSelection) state.selectedRow = null;
    if (!opts.keepPage) {
      state.offset = 0;
      state.query = "";
      state.sort = "-score";
    }
    if (state.job && state.job.id !== id) {
      teardownViewer();
      state.pdbText = null;
      state.structureSource = "";
      state.fastaSeq = null;
      state.logText = null;
      state.logMissing = "";
    }
    try {
      state.job = await apiJson("/api/runs/" + encodeURIComponent(id));
      setActiveRun(id);
      state.lastPoll = new Date();
      state.connected = true;
      updateChrome();
    } catch (err) {
      state.job = null;
      var host = $("workspace-body");
      if (host) {
        host.innerHTML =
          '<div class="error-box">Run ' + esc(id) + " could not be loaded.\n" + esc(err.message) + "</div>";
      }
      return;
    }
    await loadFastaIfNeeded(state.job);
    await Promise.all([loadScores(), loadHistogram(id), loadLog(id)]);
    renderWorkspace();
  }

  function topColumns(rows, primary) {
    var cols = ["mutant", primary || "score", "rank"];
    if (rows && rows[0] && rowDms(rows[0]) != null) cols.push("DMS_score");
    return cols;
  }

  function renderSelect() {
    var host = $("select-body");
    if (!host) return;
    var job = state.job;
    var payload = state.top || { rows: [], k: state.k };
    var rows = payload.rows || [];
    var primary = primaryScore(job, payload);
    var dms = showDms(job, rows);
    var body = rows
      .map(function (row, i) {
        var rank = row.rank != null ? row.rank : i + 1;
        return (
          "<tr>" +
          '<td class="mono">' +
          esc(mutantField(row) || "—") +
          "</td>" +
          '<td class="num">' +
          fmtScore(rowScore(row, primary)) +
          "</td>" +
          (dms ? '<td class="num">' + fmtScore(rowDms(row)) + "</td>" : "") +
          '<td class="num">' +
          esc(rank) +
          "</td></tr>"
        );
      })
      .join("");
    if (!body) body = '<tr><td colspan="' + (dms ? 4 : 3) + '"><div class="empty">No variants yet.</div></td></tr>';
    host.innerHTML =
      '<div class="page-head"><h1>Select</h1>' +
      '<p class="muted">Top variants for <span class="mono">' +
      esc((job && (job.protein || job.id)) || state.route.id || "") +
      "</span></p></div>" +
      '<div class="panel">' +
      '<div class="k-row"><label for="k-slider">k</label>' +
      '<input type="range" id="k-slider" min="5" max="100" step="1" value="' +
      esc(state.k) +
      '">' +
      '<span class="k-val" id="k-val">' +
      esc(state.k) +
      "</span>" +
      '<button type="button" class="btn" id="btn-dl-csv">Download CSV</button>' +
      '<button type="button" class="btn" id="btn-copy-list">Copy mutant list</button>' +
      (job
        ? '<a class="btn btn-ghost" href="#/runs/' + encodeURIComponent(job.id) + '">Back to run</a>'
        : "") +
      "</div>" +
      '<div class="table-scroll"><table class="data"><thead><tr>' +
      "<th>mutant</th><th>rem2 score</th>" +
      (dms ? "<th>DMS_score</th>" : "") +
      "<th>rank</th></tr></thead><tbody>" +
      body +
      "</tbody></table></div></div>";
    var slider = $("k-slider");
    var kval = $("k-val");
    if (slider) {
      slider.addEventListener("input", function () {
        state.k = clampK(slider.value);
        if (kval) kval.textContent = String(state.k);
        clearTimeout(selectTimer);
        selectTimer = setTimeout(function () {
          loadTop(state.route.id).then(renderSelect);
        }, 280);
      });
    }
    var dl = $("btn-dl-csv");
    if (dl) dl.addEventListener("click", downloadTopCsv);
    var cp = $("btn-copy-list");
    if (cp) {
      cp.addEventListener("click", function () {
        var list = ((state.top && state.top.rows) || [])
          .map(function (r) {
            return mutantField(r);
          })
          .filter(Boolean)
          .join("\n");
        copyText(list);
      });
    }
  }

  function clampK(v) {
    var n = parseInt(v, 10);
    if (!isFinite(n)) n = 20;
    return Math.min(100, Math.max(5, n));
  }

  async function loadTop(id) {
    if (!id) return;
    if (topAbort) topAbort.abort();
    topAbort = new AbortController();
    try {
      state.top = await apiJson(
        "/api/runs/" + encodeURIComponent(id) + "/top?k=" + encodeURIComponent(state.k),
        { signal: topAbort.signal }
      );
    } catch (err) {
      if (err.name === "AbortError") return;
      state.top = { rows: [], k: state.k };
    }
  }

  function downloadTopCsv() {
    var rows = (state.top && state.top.rows) || [];
    var primary = primaryScore(state.job, state.top);
    var cols = topColumns(rows, primary);
    var lines = [cols.join(",")];
    for (var i = 0; i < rows.length; i++) {
      var row = rows[i];
      var rec = {
        mutant: mutantField(row),
        rank: row.rank != null ? row.rank : i + 1,
        DMS_score: rowDms(row),
      };
      rec[primary || "score"] = rowScore(row, primary);
      lines.push(
        cols
          .map(function (c) {
            return csvEscape(rec[c] != null ? rec[c] : row[c]);
          })
          .join(",")
      );
    }
    var blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "rem2-" + (state.route.id || "run") + "-top" + state.k + ".csv";
    document.body.appendChild(a);
    a.click();
    setTimeout(function () {
      URL.revokeObjectURL(a.href);
      a.remove();
    }, 0);
  }

  function originNoteHtml(job) {
    var meta = currentSourceMeta(job);
    var origin = (meta && meta.origin) || job.pdb_origin || "";
    var sources = structureSources(job);
    var hasAf = !!sources.afdb;
    var bits = [];
    if (origin === "experimental" || (meta && meta.has_plddt === false)) {
      bits.push(
        "This is a crystal / experimental structure. The B column is a temperature factor, not pLDDT. rem2 already skips pLDDT decay on it."
      );
      if (!hasAf) bits.push("Fetch an AlphaFold DB model to color by confidence.");
    } else if (origin === "predicted" || (meta && meta.has_plddt)) {
      bits.push("Predicted model — B-factor is pLDDT.");
    }
    if (job.pdb_id) bits.push("PDB " + job.pdb_id);
    if (job.uniprot_id) bits.push("UniProt " + job.uniprot_id);
    if ((job.fetch_errors || []).length) bits.push(job.fetch_errors.join(" · "));
    if (!bits.length) return "";
    return '<p class="origin-note">' + esc(bits.join(" ")) + "</p>";
  }

  function fetchBarHtml(job) {
    return (
      '<div class="panel fetch-bar">' +
      "<h2>RCSB / AlphaFold DB</h2>" +
      originNoteHtml(job) +
      '<div class="field-row">' +
      '<label>PDB id<input type="text" id="fetch-pdb-id" value="' +
      esc(job.pdb_id || "") +
      '" placeholder="2L6Q"></label>' +
      '<label>UniProt<input type="text" id="fetch-uniprot" value="' +
      esc(job.uniprot_id || "") +
      '" placeholder="P0A6Y8"></label>' +
      "</div>" +
      '<div class="mol-actions" style="margin-top:8px">' +
      '<button type="button" class="btn btn-primary" data-fetch="auto">Fetch RCSB + AFDB</button>' +
      '<button type="button" class="btn" data-fetch="rcsb">RCSB only</button>' +
      '<button type="button" class="btn" data-fetch="afdb">AlphaFold only</button>' +
      "</div>" +
      '<p class="mol-hint">Downloads go into this run. Scoring is not re-run.</p></div>'
    );
  }

  function renderStructure() {
    var host = $("structure-body");
    if (!host) return;
    var job = state.job;
    if (!job) {
      host.innerHTML = '<div class="empty">Run not loaded.</div>';
      return;
    }
    var seq = jobSequence(job);
    var muts = parseMutants(mutantField(state.selectedRow));
    var positions = muts.map(function (m) {
      return m.pos;
    });
    var viewer =
      job.has_pdb || currentSourceKey(job)
        ? '<div class="panel">' + molPanelHtml("full") + "</div>"
        : '<div class="empty">No coordinate file yet. Fetch from RCSB or AlphaFold DB, or re-submit with a PDB.</div>';
    host.innerHTML =
      '<div class="page-head"><h1>Structure</h1>' +
      '<p class="muted">' +
      esc(job.protein || job.id) +
      " · " +
      esc(job.model || "") +
      ' · <a href="#/runs/' +
      encodeURIComponent(job.id) +
      '">back to scores</a></p></div>' +
      fetchBarHtml(job) +
      viewer +
      '<div class="inspector has-no-viewer" style="margin-top:10px">' +
      '<div class="panel seq-box"><h2>Sequence</h2>' +
      renderSequence(seq, positions) +
      "</div>" +
      '<div class="panel"><h2>Selection</h2>' +
      selectedDetailHtml(job, state.selectedRow) +
      '<p class="mol-hint">Table selection and residue clicks stay in sync with the 3D view.</p></div>' +
      "</div>";
  }

  async function fetchRunStructure(source) {
    var job = state.job;
    if (!job || !job.id) return;
    var fd = new FormData();
    fd.append("source", source || "auto");
    var pdbEl = $("fetch-pdb-id");
    var uniEl = $("fetch-uniprot");
    if (pdbEl && pdbEl.value.trim()) fd.append("pdb_id", pdbEl.value.trim());
    if (uniEl && uniEl.value.trim()) fd.append("uniprot_id", uniEl.value.trim());
    flash("");
    try {
      var res = await api("/api/runs/" + encodeURIComponent(job.id) + "/fetch_structure", {
        method: "POST",
        body: fd,
      });
      state.job = await res.json();
      state.pdbText = null;
      if (source === "afdb" || (state.job.structure_sources && state.job.structure_sources.afdb)) {
        state.structureSource = "afdb";
      }
      teardownViewer();
      renderStructure();
      await maybeLoadViewer(state.job, selectionResis());
      flash("Structure downloaded.");
    } catch (err) {
      flash(err.message || "Fetch failed.");
    }
  }

  async function loadStructure(id) {
    try {
      state.job = await apiJson("/api/runs/" + encodeURIComponent(id));
      setActiveRun(id);
      state.connected = true;
      state.lastPoll = new Date();
      updateChrome();
    } catch (err) {
      state.job = { id: id, has_pdb: false };
      flash(err.message || "Could not load run.");
    }
    await loadFastaIfNeeded(state.job);
    if (!state.scores) {
      try {
        await loadScores();
      } catch (err) {
        /* optional */
      }
    }
    renderStructure();
    var muts = parseMutants(mutantField(state.selectedRow));
    await maybeLoadViewer(
      state.job,
      muts.map(function (m) {
        return m.pos;
      })
    );
  }

  async function loadSelect(id) {
    try {
      state.job = await apiJson("/api/runs/" + encodeURIComponent(id));
      setActiveRun(id);
      state.connected = true;
      state.lastPoll = new Date();
      updateChrome();
    } catch (err) {
      state.job = { id: id };
      flash(err.message || "Could not load run.");
    }
    await loadTop(id);
    renderSelect();
  }

  function setBusy(busy) {
    if (els.btnDemo) els.btnDemo.disabled = busy;
    if (els.btnSubmit) els.btnSubmit.disabled = busy;
  }

  function showFormError(msg) {
    var box = $("form-error");
    if (!box) return;
    box.textContent = msg || "";
    setHidden(box, !msg);
  }

  function buildFormData(demo) {
    var fd = new FormData();
    if (demo) {
      fd.append("model", "esm2-8m");
      fd.append("recipe", "full");
      fd.append("mutant_mode", "demo");
      fd.append("demo", "true");
      return fd;
    }
    var snap = formSnapshot();
    var form = els.form;
    fd.append("model", snap.model);
    fd.append("recipe", snap.recipe);
    fd.append("mutant_mode", snap.mutants ? "upload" : "saturation");
    if (form.fasta.files[0]) fd.append("fasta", form.fasta.files[0]);
    if (form.pdb.files[0]) fd.append("pdb", form.pdb.files[0]);
    if (form.mutants.files[0]) fd.append("mutants", form.mutants.files[0]);
    if (form.msa.files[0]) fd.append("msa", form.msa.files[0]);
    if (!snap.mutants) {
      fd.append("mutant_sites", snap.mutant_sites || "1");
      if (snap.positions) fd.append("positions", snap.positions);
      if (snap.residue_range) fd.append("residue_range", snap.residue_range);
    }
    if (snap.scoring_strategy) fd.append("scoring_strategy", snap.scoring_strategy);
    if (snap.max_mutants) fd.append("max_mutants", snap.max_mutants);
    if (snap.pdb_id) fd.append("pdb_id", snap.pdb_id);
    if (snap.uniprot_id) fd.append("uniprot_id", snap.uniprot_id);
    if (snap.fetch_structure) fd.append("fetch_structure", snap.fetch_structure);
    return fd;
  }

  async function submitJob(demo) {
    flash("");
    showFormError("");
    if (!demo) {
      var snap = formSnapshot();
      if (!snap.fasta && !snap.pdb && !snap.pdb_id && !snap.uniprot_id) {
        showFormError("Provide a FASTA and/or PDB, a PDB/UniProt id, or use Score demo assay.");
        return;
      }
    }
    setBusy(true);
    try {
      var res = await api("/api/runs", { method: "POST", body: buildFormData(demo) });
      var job = await res.json();
      var id = job && job.id;
      if (!id) throw new Error("Server did not return a run id.");
      setActiveRun(id);
      go("#/runs/" + encodeURIComponent(id));
    } catch (err) {
      showFormError(err.message || "Submit failed.");
    } finally {
      setBusy(false);
    }
  }

  function renderDoctor(data) {
    var host = els.doctorBody;
    if (!host) return;
    if (!data) {
      host.innerHTML = '<p class="muted">Doctor is unavailable.</p>';
      return;
    }
    var extras = data.extras || [];
    var extraRows = extras
      .map(function (ex) {
        var ok = ex.ok ? '<span class="ok">ok</span>' : '<span class="miss">missing</span>';
        return (
          "<tr><td>" +
          esc(ex.name) +
          "</td><td>" +
          ok +
          "</td><td class=\"muted\">" +
          esc(ex.note || "") +
          "</td></tr>"
        );
      })
      .join("");
    var problems = data.problems || [];
    var cache = data.cache;
    var cacheText = typeof cache === "string" ? cache : JSON.stringify(cache || {}, null, 0);
    host.innerHTML =
      '<dl class="kv">' +
      "<dt>version</dt><dd>" +
      esc(data.version || "—") +
      "</dd>" +
      "<dt>python</dt><dd>" +
      esc(data.python || "—") +
      "</dd>" +
      "<dt>torch</dt><dd>" +
      esc(data.torch || "—") +
      "</dd>" +
      "<dt>cuda</dt><dd>" +
      esc(data.cuda == null ? "—" : String(data.cuda)) +
      "</dd>" +
      "<dt>device</dt><dd>" +
      esc(data.device || "—") +
      "</dd>" +
      "<dt>cache</dt><dd>" +
      esc(cacheText) +
      "</dd>" +
      "<dt>demo</dt><dd>" +
      esc(typeof data.demo === "string" ? data.demo : JSON.stringify(data.demo || "")) +
      "</dd></dl>" +
      (extraRows
        ? '<div class="table-scroll"><table class="data"><thead><tr><th>extra</th><th>status</th><th>note</th></tr></thead><tbody>' +
          extraRows +
          "</tbody></table></div>"
        : "") +
      (problems.length
        ? '<div class="error-box">' +
          problems
            .map(function (p) {
              return esc(p);
            })
            .join("\n") +
          "</div>"
        : '<p class="muted">No problems reported.</p>');
  }

  async function openDoctor() {
    lastFocus = document.activeElement;
    setHidden(els.drawerRoot, false);
    els.doctorBody.innerHTML = "Checking install…";
    if (els.btnDoctorClose) els.btnDoctorClose.focus();
    try {
      var data = await apiJson("/api/doctor");
      renderDoctor(data);
    } catch (err) {
      els.doctorBody.innerHTML =
        '<div class="error-box">Doctor request failed.\n' + esc(err.message) + "</div>";
    }
  }

  function closeDoctor() {
    setHidden(els.drawerRoot, true);
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  async function onRoute() {
    flash("");
    var route = parseHash();
    if ((route.page === "select" || route.page === "structure") && !route.id) {
      if (state.activeRunId) {
        go("#/" + route.page + "/" + encodeURIComponent(state.activeRunId));
        return;
      }
      go("#/runs");
      return;
    }
    state.route = route;
    setTabs(route.page);
    if (route.page !== "workspace" && route.page !== "structure") teardownViewer();
    if (route.page === "runs") {
      showPage("page-runs");
      try {
        await refreshRuns();
      } catch (err) {
        state.connected = false;
        updateChrome();
        $("runs-body").innerHTML =
          '<div class="error-box">Could not list runs.\n' + esc(err.message) + "</div>";
      }
    } else if (route.page === "predict") {
      showPage("page-predict");
      updateCli();
    } else if (route.page === "workspace") {
      showPage("page-workspace");
      setActiveRun(route.id);
      await loadWorkspace(route.id);
    } else if (route.page === "structure") {
      showPage("page-structure");
      setActiveRun(route.id);
      teardownViewer();
      await loadStructure(route.id);
    } else if (route.page === "select") {
      showPage("page-select");
      setActiveRun(route.id);
      await loadSelect(route.id);
    }
    schedulePoll();
  }

  function patchOverview(job) {
    var nMut = job.n_mutants != null ? job.n_mutants : job.n_rows;
    var items = document.querySelectorAll("#workspace-body .overview-item");
    for (var i = 0; i < items.length; i++) {
      var k = items[i].querySelector(".overview-k");
      var v = items[i].querySelector(".overview-v");
      if (!k || !v) continue;
      var key = k.textContent;
      if (key === "status") v.innerHTML = badge(job.status);
      if (key === "mutants" && nMut != null) v.textContent = String(nMut);
    }
    var cancel = $("btn-cancel");
    if (cancel && !isLive(job.status)) cancel.remove();
    if (!cancel && isLive(job.status)) {
      var bar = document.querySelector("#workspace-body .toolbar");
      if (bar) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.id = "btn-cancel";
        btn.className = "btn btn-danger";
        btn.textContent = "Cancel";
        bar.appendChild(btn);
      }
    }
  }

  async function pollTick() {
    try {
      try {
        await refreshRuns();
      } catch (err) {
        state.connected = false;
      }
      if (state.route.page === "workspace" && state.route.id) {
        var prev = state.job && state.job.status;
        var hadPdb = state.job && state.job.has_pdb;
        state.job = await apiJson("/api/runs/" + encodeURIComponent(state.route.id));
        state.lastPoll = new Date();
        state.connected = true;
        await Promise.all([
          loadScores(),
          loadHistogram(state.route.id),
          loadLog(state.route.id),
          loadFastaIfNeeded(state.job),
        ]);
        var pdbArrived = state.job.has_pdb && !hadPdb;
        var statusChanged = prev !== state.job.status;
        if (!$("score-table") || pdbArrived || (statusChanged && !isLive(state.job.status))) {
          renderWorkspace();
        } else {
          patchOverview(state.job);
          patchScores();
          var canvas = $("hist-canvas");
          if (canvas) drawHistogram(canvas, state.histogram || { bins: [] });
        }
        updateChrome();
      } else if (state.route.page === "select" && state.route.id && state.job && isLive(state.job.status)) {
        state.job = await apiJson("/api/runs/" + encodeURIComponent(state.route.id));
        await loadTop(state.route.id);
        renderSelect();
        updateChrome();
      }
    } catch (err) {
      state.connected = false;
      updateChrome();
    }
    schedulePoll();
  }

  function schedulePoll() {
    clearTimeout(pollTimer);
    var need =
      anyLive() ||
      (state.job && isLive(state.job.status) && (state.route.page === "workspace" || state.route.page === "select"));
    if (need) pollTimer = setTimeout(pollTick, POLL_MS);
  }

  function bindPredict() {
    if (!els.form) return;
    els.form.addEventListener("input", function () {
      updateFileLabels();
      updateModelMeta();
      updateCli();
    });
    els.form.addEventListener("change", function () {
      updateFileLabels();
      updateModelMeta();
      updateCli();
    });
    els.form.addEventListener("submit", function (ev) {
      ev.preventDefault();
      submitJob(false);
    });
    if (els.btnDemo) {
      els.btnDemo.addEventListener("click", function () {
        submitJob(true);
      });
    }
    updateFileLabels();
    updateCli();
  }

  function bindChrome() {
    if (els.btnDoctor) els.btnDoctor.addEventListener("click", openDoctor);
    if (els.btnDoctorClose) els.btnDoctorClose.addEventListener("click", closeDoctor);
    if (els.drawerBackdrop) els.drawerBackdrop.addEventListener("click", closeDoctor);
    document.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape" && els.drawerRoot && !els.drawerRoot.hasAttribute("hidden")) {
        closeDoctor();
      }
    });
    window.addEventListener("hashchange", onRoute);
    window.addEventListener("resize", function () {
      var canvas = $("hist-canvas");
      if (canvas && state.histogram) drawHistogram(canvas, state.histogram);
      if (state.viewer) {
        try {
          state.viewer.resize();
          state.viewer.render();
        } catch (err) {
          /* ignore */
        }
      }
    });
  }

  async function init() {
    els.flash = $("flash");
    els.conn = $("conn-status");
    els.jobStatus = $("job-status");
    els.lastPoll = $("last-poll");
    els.appVersion = $("app-version");
    els.tabRuns = $("tab-runs");
    els.tabPredict = $("tab-predict");
    els.tabSelect = $("tab-select");
    els.tabStructure = $("tab-structure");
    els.form = $("predict-form");
    els.modelSelect = $("model-select");
    els.cli = $("cli-preview");
    els.btnDemo = $("btn-demo");
    els.btnSubmit = $("btn-submit");
    els.drawerRoot = $("drawer-root");
    els.drawerBackdrop = $("drawer-backdrop");
    els.btnDoctor = $("btn-doctor");
    els.btnDoctorClose = $("btn-doctor-close");
    els.doctorBody = $("doctor-body");

    fillModels();
    fillRecipes();
    bindChrome();
    bindPredict();
    bindWorkspaceOnce();
    var structHost = $("structure-body");
    if (structHost) {
      structHost.addEventListener("click", function (ev) {
        var fetchEl = ev.target.closest("[data-fetch]");
        if (fetchEl) {
          fetchRunStructure(fetchEl.getAttribute("data-fetch"));
          return;
        }
        var molEl = ev.target.closest("[data-mol]");
        if (molEl) {
          handleMolControl(molEl);
          return;
        }
        var aa = ev.target.closest(".seq-aa[data-pos]");
        if (aa) pickSequencePos(parseInt(aa.getAttribute("data-pos"), 10));
      });
      structHost.addEventListener("change", function (ev) {
        var molEl = ev.target.closest("[data-mol]");
        if (molEl) handleMolControl(molEl);
      });
    }
    var runsHost = $("runs-body");
    if (runsHost) {
      runsHost.addEventListener("click", function (ev) {
        if (ev.target.closest("a")) return;
        var tr = ev.target.closest("tr[data-id]");
        if (tr) go("#/runs/" + encodeURIComponent(tr.getAttribute("data-id")));
      });
    }
    updateSelectTab();
    updateChrome();

    await refreshHealth();
    await loadCatalog();
    try {
      await refreshRuns();
    } catch (err) {
      state.connected = false;
      updateChrome();
    }
    await onRoute();
    healthTimer = setInterval(refreshHealth, HEALTH_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
