(function () {
  "use strict";

  var PAGE_SIZE = 80;
  var POLL_MS = 1000;
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
  var STORE_SPLIT = {
    review: "rem2.split.review",
    structure: "rem2.split.structure",
  };

  var FALLBACK_RECIPES = [
    { id: "full", label: "Full rem2", argv: [], msa: "optional" },
    {
      id: "raw",
      label: "Raw backbone",
      argv: ["--alpha", "0", "--background_weight", "0", "--no_rsa_decay", "--no_plddt_decay"],
      msa: "off",
    },
    {
      id: "msa",
      label: "+ MSA",
      argv: ["--background_weight", "0", "--no_rsa_decay", "--no_plddt_decay"],
      msa: "optional",
    },
    {
      id: "ccd",
      label: "+ MSA + CCD",
      argv: ["--no_rsa_decay", "--no_plddt_decay"],
      msa: "optional",
    },
  ];

  var SERIES_ORDER = ["venusrem2", "esm", "prosst", "saprot", "progen", "proteinmpnn", "rita", "other"];
  var SERIES_LABELS = {
    venusrem2: "VenusREM2",
    esm: "ESM",
    prosst: "ProSST",
    saprot: "SaProt",
    progen: "ProGen",
    proteinmpnn: "ProteinMPNN",
    rita: "RITA",
    other: "Other",
  };
  var VARIANT_LABELS = {
    venusrem2: "VenusREM2",
    "prosst": "Default (K=2048)",
    "prosst-20": "K=20",
    "prosst-128": "K=128",
    "prosst-512": "K=512",
    "prosst-1024": "K=1024",
    "prosst-2048": "K=2048",
    "prosst-4096": "K=4096",
    esm2: "ESM-2 650M",
    "esm2-8m": "ESM-2 8M",
    "esm2-35m": "ESM-2 35M",
    "esm2-150m": "ESM-2 150M",
    "esm2-3b": "ESM-2 3B",
    esm1b: "ESM-1b",
    esm1v: "ESM-1v",
    esm3: "ESM3",
    esmc: "ESM-C 300M",
    "esmc-600m": "ESM-C 600M",
    esm_if: "ESM-IF",
    saprot: "650M AF2",
    "saprot-35m-af2": "35M AF2",
    "saprot-650m-pdb": "650M PDB",
    progen2: "ProGen2 L",
    "progen2-s": "ProGen2 S",
    "progen2-m": "ProGen2 M",
    "progen2-b": "ProGen2 B",
    "progen2-xl": "ProGen2 XL",
    progen3: "ProGen3 1B",
    "progen3-112m": "ProGen3 112M",
    "progen3-219m": "ProGen3 219M",
    "progen3-339m": "ProGen3 339M",
    "progen3-762m": "ProGen3 762M",
    "progen3-3b": "ProGen3 3B",
    protein_mpnn: "v_48_020",
    "protein_mpnn-v_48_002": "v_48_002",
    "protein_mpnn-v_48_010": "v_48_010",
    "protein_mpnn-v_48_030": "v_48_030",
    "protein_mpnn-soluble-v_48_002": "soluble v_48_002",
    "protein_mpnn-soluble-v_48_010": "soluble v_48_010",
    "protein_mpnn-soluble-v_48_020": "soluble v_48_020",
    "protein_mpnn-soluble-v_48_030": "soluble v_48_030",
    rita: "RITA XL",
    "rita-s": "RITA S",
    "rita-m": "RITA M",
    "rita-l": "RITA L",
    protssn: "ProtSSN",
    carp: "CARP-640M",
    mifst: "MIF-ST",
    s2f: "S2F",
    s3f: "S3F",
    protgpt2: "ProtGPT2",
    auto: "HF AutoMLM",
  };
  var FALLBACK_MODELS = [
    { name: "venusrem2", description: "Official ProSST ensemble", supports_mask: false, needs_pdb: true, needs_msa: false, input_kind: "structure" },
    { name: "esm2-8m", description: "ESM-2 8M", supports_mask: true, input_kind: "sequence", size_hint: "first download ~30 MB" },
    { name: "esm2-35m", description: "ESM-2 35M", supports_mask: true, input_kind: "sequence" },
    { name: "esm2-150m", description: "ESM-2 150M", supports_mask: true, input_kind: "sequence" },
    { name: "esm2", description: "ESM-2 650M", supports_mask: true, input_kind: "sequence", size_hint: "first download ~2.5 GB" },
    { name: "esm2-3b", description: "ESM-2 3B", supports_mask: true, input_kind: "sequence" },
    { name: "esm1b", description: "ESM-1b masked language model", supports_mask: true, input_kind: "sequence" },
    { name: "esm1v", description: "ESM-1v 5-seed ensemble", supports_mask: true, input_kind: "sequence" },
    { name: "esmc", description: "ESM-C 300M", supports_mask: true, input_kind: "sequence" },
    { name: "esmc-600m", description: "ESM-C 600M", supports_mask: true, input_kind: "sequence" },
    { name: "esm3", description: "ESM3 small open", supports_mask: true, input_kind: "sequence" },
    { name: "esm_if", description: "ESM-IF1 inverse folding", needs_pdb: true, input_kind: "structure" },
    { name: "prosst", description: "ProSST structure-aware MLM", needs_pdb: true, input_kind: "structure" },
    { name: "prosst-20", description: "ProSST-K20", needs_pdb: true, input_kind: "structure" },
    { name: "prosst-128", description: "ProSST-K128", needs_pdb: true, input_kind: "structure" },
    { name: "prosst-512", description: "ProSST-K512", needs_pdb: true, input_kind: "structure" },
    { name: "prosst-1024", description: "ProSST-K1024", needs_pdb: true, input_kind: "structure" },
    { name: "prosst-2048", description: "ProSST-K2048", needs_pdb: true, input_kind: "structure" },
    { name: "prosst-4096", description: "ProSST-K4096", needs_pdb: true, input_kind: "structure" },
    { name: "saprot", description: "SaProt 650M AF2", supports_mask: true, needs_pdb: true, input_kind: "structure" },
    { name: "saprot-35m-af2", description: "SaProt 35M AF2", supports_mask: true, needs_pdb: true, input_kind: "structure" },
    { name: "saprot-650m-pdb", description: "SaProt 650M PDB", supports_mask: true, needs_pdb: true, input_kind: "structure" },
    { name: "progen2-s", description: "ProGen2-small", input_kind: "sequence" },
    { name: "progen2-m", description: "ProGen2-medium", input_kind: "sequence" },
    { name: "progen2-b", description: "ProGen2-base", input_kind: "sequence" },
    { name: "progen2", description: "ProGen2-L", input_kind: "sequence" },
    { name: "progen2-xl", description: "ProGen2-xlarge", input_kind: "sequence" },
    { name: "progen3-112m", description: "progen3-112m", input_kind: "sequence" },
    { name: "progen3-219m", description: "progen3-219m", input_kind: "sequence" },
    { name: "progen3-339m", description: "progen3-339m", input_kind: "sequence" },
    { name: "progen3-762m", description: "progen3-762m", input_kind: "sequence" },
    { name: "progen3", description: "ProGen3-1B", input_kind: "sequence" },
    { name: "progen3-3b", description: "progen3-3b", input_kind: "sequence" },
    { name: "protein_mpnn", description: "ProteinMPNN v_48_020", supports_tf: true, needs_pdb: true, input_kind: "structure", aliases: ["proteinmpnn"] },
    { name: "protein_mpnn-v_48_002", description: "ProteinMPNN v_48_002", supports_tf: true, needs_pdb: true, input_kind: "structure" },
    { name: "protein_mpnn-v_48_010", description: "ProteinMPNN v_48_010", supports_tf: true, needs_pdb: true, input_kind: "structure" },
    { name: "protein_mpnn-v_48_030", description: "ProteinMPNN v_48_030", supports_tf: true, needs_pdb: true, input_kind: "structure" },
    { name: "protein_mpnn-soluble-v_48_002", description: "ProteinMPNN soluble_v_48_002", supports_tf: true, needs_pdb: true, input_kind: "structure" },
    { name: "protein_mpnn-soluble-v_48_010", description: "ProteinMPNN soluble_v_48_010", supports_tf: true, needs_pdb: true, input_kind: "structure" },
    { name: "protein_mpnn-soluble-v_48_020", description: "ProteinMPNN soluble_v_48_020", supports_tf: true, needs_pdb: true, input_kind: "structure" },
    { name: "protein_mpnn-soluble-v_48_030", description: "ProteinMPNN soluble_v_48_030", supports_tf: true, needs_pdb: true, input_kind: "structure" },
    { name: "rita-s", description: "RITA_s", input_kind: "sequence" },
    { name: "rita-m", description: "RITA_m", input_kind: "sequence" },
    { name: "rita-l", description: "RITA_l", input_kind: "sequence" },
    { name: "rita", description: "RITA-XL", input_kind: "sequence" },
    { name: "protssn", description: "ProtSSN structure GNN ensemble", supports_mask: true, needs_pdb: true, input_kind: "structure" },
    { name: "carp", description: "CARP-640M", supports_mask: true, input_kind: "sequence" },
    { name: "mifst", description: "MIF-ST masked inverse folding", needs_pdb: true, input_kind: "structure" },
    { name: "s2f", description: "S2F", input_kind: "sequence" },
    { name: "s3f", description: "S3F structure+sequence", supports_mask: true, needs_pdb: true, input_kind: "structure" },
    { name: "protgpt2", description: "ProtGPT2 causal LM", input_kind: "sequence" },
    { name: "auto", description: "Any HuggingFace AutoModelForMaskedLM", supports_mask: true, input_kind: "sequence" },
  ];

  var FALLBACK_PAPER = [
    { name: "VenusREM2", model: "venusrem2", highlight: true, inputs: ["seq", "str", "evo"], note: "ProSST ensemble + rem2", average: 0.556, activity: 0.541, binding: 0.495, expression: 0.557, organismal: 0.494, stability: 0.691 },
    { name: "AIDO Protein-RAG (16B)", inputs: ["str", "evo"], average: 0.518, activity: 0.517, binding: 0.426, expression: 0.522, organismal: 0.491, stability: 0.635 },
    { name: "VenusREM", inputs: ["seq", "str", "evo"], average: 0.518, activity: 0.495, binding: 0.454, expression: 0.533, organismal: 0.459, stability: 0.650 },
    { name: "ProSST (K=2048)", inputs: ["seq", "str"], average: 0.507, activity: 0.476, binding: 0.445, expression: 0.530, organismal: 0.431, stability: 0.653 },
    { name: "S3F-MSA", inputs: ["str", "evo"], average: 0.496, activity: 0.502, binding: 0.440, expression: 0.479, organismal: 0.477, stability: 0.581 },
    { name: "Protriever", inputs: ["evo"], average: 0.479, activity: 0.487, binding: 0.396, expression: 0.496, organismal: 0.479, stability: 0.537 },
    { name: "ESCOTT", inputs: ["str", "evo"], average: 0.476, activity: 0.499, binding: 0.389, expression: 0.468, organismal: 0.466, stability: 0.557 },
    { name: "PoET (200M)", inputs: ["evo"], average: 0.470, activity: 0.494, binding: 0.396, expression: 0.466, organismal: 0.475, stability: 0.519 },
    { name: "ESM3 open (1.4B)", inputs: ["seq", "str"], average: 0.466, activity: 0.430, binding: 0.400, expression: 0.470, organismal: 0.389, stability: 0.641 },
    { name: "RSALOR", inputs: ["str", "evo"], average: 0.465, activity: 0.479, binding: 0.416, expression: 0.427, organismal: 0.426, stability: 0.575 },
    { name: "VespaG", inputs: ["seq"], average: 0.458, activity: 0.493, binding: 0.370, expression: 0.456, organismal: 0.437, stability: 0.533 },
    { name: "SaProt (650M)", model: "saprot", inputs: ["seq", "str"], average: 0.457, activity: 0.458, binding: 0.378, expression: 0.488, organismal: 0.366, stability: 0.592 },
    { name: "TranceptEVE-L", inputs: ["seq", "evo"], average: 0.456, activity: 0.487, binding: 0.376, expression: 0.457, organismal: 0.459, stability: 0.500 },
    { name: "GEMME", inputs: ["evo"], average: 0.455, activity: 0.482, binding: 0.383, expression: 0.438, organismal: 0.452, stability: 0.519 },
    { name: "ProtSSN ensemble", inputs: ["seq", "str"], average: 0.449, activity: 0.466, binding: 0.366, expression: 0.449, organismal: 0.396, stability: 0.568 },
  ];
  function fallbackBoardRows(metric) {
    return FALLBACK_PAPER.slice()
      .sort(function (a, b) { return b[metric] - a[metric]; })
      .map(function (row, i) {
        return {
          rank: i + 1,
          name: row.name,
          model: row.model,
          inputs: row.inputs,
          note: row.note,
          highlight: !!row.highlight,
          score: row[metric],
        };
      });
  }
  var FALLBACK_CATALOG = {
    default: "substitutions",
    url: "https://proteingym.org/benchmarks",
    metric: "Mean Spearman",
    boards: [
      { id: "substitutions", label: "Substitutions", title: "ProteinGym substitutions", n: 217, note: "", rows: fallbackBoardRows("average") },
      { id: "stability", label: "Stability", title: "ProteinGym stability", n: 217, note: "", rows: fallbackBoardRows("stability") },
      { id: "activity", label: "Activity", title: "ProteinGym activity", n: 217, note: "", rows: fallbackBoardRows("activity") },
      { id: "binding", label: "Binding", title: "ProteinGym binding", n: 217, note: "", rows: fallbackBoardRows("binding") },
      { id: "expression", label: "Expression", title: "ProteinGym expression", n: 217, note: "", rows: fallbackBoardRows("expression") },
      { id: "organismal", label: "Organismal", title: "ProteinGym organismal fitness", n: 217, note: "", rows: fallbackBoardRows("organismal") },
      {
        id: "ablations",
        label: "Full rem2",
        title: "VenusREM2 recipe ladder",
        n: 217,
        note: "",
        rows: [
          { rank: 1, name: "VenusREM2 · rem2", model: "venusrem2", score: 0.556, inputs: ["seq", "str", "evo"], highlight: true, note: "+ pLDDT (full)" },
          { rank: 2, name: "VenusREM2 · + RSA", model: "venusrem2", score: 0.554, inputs: ["seq", "str", "evo"], note: "gated CCD + RSA" },
          { rank: 3, name: "VenusREM2 · + gated CCD", model: "venusrem2", score: 0.550, inputs: ["seq", "str", "evo"], note: "adaptive mix + coherence gate" },
          { rank: 4, name: "VenusREM2 · adaptive mix", model: "venusrem2", score: 0.542, inputs: ["seq", "str", "evo"], note: "entropy-α MSA" },
          { rank: 5, name: "VenusREM2 · + ungated CCD", model: "venusrem2", score: 0.538, inputs: ["seq", "str", "evo"], note: "mix + κ=1" },
          { rank: 6, name: "VenusREM2 · raw", model: "venusrem2", score: 0.524, inputs: ["seq", "str", "evo"], note: "uncalibrated ProSST ensemble" },
        ],
      },
    ],
  };
  var FALLBACK_BOARD = FALLBACK_CATALOG.boards[0];

  var PIPELINE = [
    { id: "fwd", label: "Forward", on: ["full", "raw", "msa", "ccd"] },
    { id: "msa", label: "MSA (α)", on: ["msa", "ccd"], opt: ["full"] },
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
    catalog: FALLBACK_CATALOG,
    board: FALLBACK_BOARD,
    boardId: "substitutions",
    features: null,
    runs: [],
    job: null,
    scores: null,
    histogram: null,
    top: null,
    selectedRow: null,
    experimentRows: {},
    experimentReady: false,
    evidenceMode: "evidence",
    activeRunId: sessionStorage.getItem(STORE_RUN) || "",
    lastPoll: null,
    sort: "-score",
    offset: 0,
    query: "",
    k: 30,
    seqRange: null,
    molFailed: false,
    pdbText: null,
    structureSource: "",
    fastaSeq: null,
    viewer: null,
    picked: null,
    view: {
      rep: "cartoon",
      color: "slate",
      sidechains: "selection",
      surface: false,
      labels: true,
      hetero: true,
      water: false,
      bg: "white",
    },
    logText: null,
    logMissing: "",
    seqDrag: null,
    split: {
      review: 56,
      structure: 64,
    },
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
      return String(proteins[0].sequence || proteins[0].aa_seq || proteins[0].seq || state.fastaSeq || "");
    }
    if (proteins && typeof proteins === "object") {
      var keys = Object.keys(proteins);
      if (keys.length) {
        var p = proteins[keys[0]];
        if (typeof p === "string") return p;
        if (p) return String(p.sequence || p.aa_seq || state.fastaSeq || "");
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
    var parts = ["rem2", "--model", shellArg(opts.model || "venusrem2")];
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

  function classifyAccession(raw) {
    var v = String(raw || "").trim();
    if (!v) return { pdb_id: "", uniprot_id: "" };
    var uni = v.replace(/^AF-/i, "").split("-")[0];
    if (/^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$/i.test(uni)) {
      return { pdb_id: "", uniprot_id: uni.toUpperCase() };
    }
    if (/^[0-9][A-Za-z0-9]{3}$/.test(v) && !/^20[0-2][0-9]$/.test(v)) {
      return { pdb_id: v.toUpperCase(), uniprot_id: "" };
    }
    if (v.length >= 6) return { pdb_id: "", uniprot_id: v.toUpperCase() };
    return { pdb_id: v.toUpperCase(), uniprot_id: "" };
  }

  function syncStructAccession() {
    var el = $("f-struct-id");
    if (!el) return;
    var acc = classifyAccession(el.value);
    var pdb = $("f-pdb-id");
    var uni = $("f-uniprot");
    if (pdb) pdb.value = acc.pdb_id;
    if (uni) uni.value = acc.uniprot_id;
  }

  function formSnapshot() {
    var form = els.form;
    if (!form) {
      return { model: "venusrem2", recipe: "full", mutant_sites: "1", scoring_strategy: "wt" };
    }
    syncStructAccession();
    var data = new FormData(form);
    var structEl = $("f-struct-id");
    var classified = classifyAccession(structEl ? structEl.value : "");
    return {
      model: data.get("model") || "venusrem2",
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
      seq_id: (data.get("seq_id") || "").toString().trim(),
      pdb_id: classified.pdb_id || (data.get("pdb_id") || "").toString().trim(),
      uniprot_id: classified.uniprot_id || (data.get("uniprot_id") || "").toString().trim(),
      fetch_structure: (data.get("fetch_structure") || "auto").toString().trim(),
    };
  }

  function hasSequenceInput(snap) {
    snap = snap || formSnapshot();
    return !!(snap.fasta || snap.pdb || snap.seq_id);
  }

  function hasStructureInput(snap) {
    snap = snap || formSnapshot();
    return !!(snap.pdb || snap.pdb_id || snap.uniprot_id);
  }

  function canFetchStructure(snap) {
    snap = snap || formSnapshot();
    return snap.fetch_structure !== "none" && !!(snap.pdb_id || snap.uniprot_id);
  }

  function structureReady(snap) {
    snap = snap || formSnapshot();
    return hasStructureInput(snap) || canFetchStructure(snap);
  }

  function hasMsaInput(snap) {
    snap = snap || formSnapshot();
    return !!snap.msa;
  }

  function modelLockReason(spec, snap) {
    snap = snap || formSnapshot();
    if (!spec) return "Unknown model";
    if (spec.needs_pdb && !structureReady(snap)) {
      return "needs a PDB file or a PDB / UniProt id";
    }
    return "";
  }

  function modelUnlocked(spec, snap) {
    return !modelLockReason(spec, snap);
  }

  function preferredModel(snap) {
    snap = snap || formSnapshot();
    var names = ["venusrem2", "esm2", "esm2-8m"];
    var i;
    for (i = 0; i < names.length; i++) {
      var spec = modelByName(names[i]);
      if (spec && modelUnlocked(spec, snap)) return names[i];
    }
    for (i = 0; i < state.models.length; i++) {
      if (modelUnlocked(state.models[i], snap)) return state.models[i].name;
    }
    return "";
  }

  function inputChip(on, label) {
    return '<span class="gate-chip' + (on ? " is-on" : " is-off") + '">' + esc(label) + "</span>";
  }

  function inputGateMessage(snap) {
    snap = snap || formSnapshot();
    var bits = [];
    if (!hasSequenceInput(snap) && !hasStructureInput(snap)) {
      bits.push("Add a sequence or structure.");
    } else if (!structureReady(snap)) {
      bits.push("FASTA only — sequence models unlocked.");
    } else {
      bits.push("Structure ready.");
    }
    if (!hasMsaInput(snap)) {
      bits.push("No MSA — α=0.");
    } else {
      bits.push("MSA attached.");
    }
    return bits.join(" ");
  }

  function updateInputGate() {
    var snap = formSnapshot();
    var html =
      '<div class="gate-chips">' +
      inputChip(!!(snap.fasta || snap.seq_id), "FASTA") +
      inputChip(structureReady(snap), "PDB") +
      inputChip(hasMsaInput(snap), "MSA") +
      "</div><p class=\"gate-msg\">" +
      esc(inputGateMessage(snap)) +
      "</p>";
    var a = $("input-gate");
    var b = $("model-gate");
    if (a) a.innerHTML = html;
    if (b) b.innerHTML = html;
    updateRecipeHint();
  }

  function updateRecipeHint() {
    var hint = $("recipe-hint");
    if (!hint) return;
    var snap = formSnapshot();
    var rec = snap.recipe || "full";
    var parts = [];
    if ((rec === "full" || rec === "msa" || rec === "ccd") && !hasMsaInput(snap)) {
      parts.push("This recipe can use MSA, but you did not attach one — α=0.");
    }
    if (rec === "full" && !structureReady(snap)) {
      parts.push("No structure — RSA / pLDDT decay will be skipped.");
    }
    if (rec === "raw") parts.push("Raw backbone only. No MSA mix, CCD, RSA, or pLDDT.");
    hint.textContent = parts.join(" ");
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
      } else if (id === "file-msa") {
        nodes[i].textContent = name || "optional — skip → α=0";
      } else if (id === "file-pdb") {
        nodes[i].textContent = name || "optional — unlocks structure models";
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
    if (spec.needs_msa) bits.push("MSA required");
    else bits.push("MSA optional (none → α=0)");
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
      try {
        var parsed = JSON.parse(text);
        if (parsed && parsed.detail) {
          text = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail);
        }
      } catch (err2) {
        /* keep text */
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

  function activeJob() {
    if (state.job && state.job.id && state.job.id === state.activeRunId) return state.job;
    var id = state.activeRunId;
    if (!id) return null;
    var runs = state.runs || [];
    for (var i = 0; i < runs.length; i++) {
      if (runs[i].id === id) return runs[i];
    }
    return null;
  }

  function setTabEnabled(tab, href, titleWhenOff) {
    if (!tab) return;
    if (href) {
      tab.href = href;
      tab.classList.remove("is-disabled");
      tab.removeAttribute("aria-disabled");
      tab.removeAttribute("title");
    } else {
      tab.href = "#/runs";
      tab.classList.add("is-disabled");
      tab.setAttribute("aria-disabled", "true");
      tab.title = titleWhenOff;
    }
  }

  function updateSelectTab() {
    var job = activeJob();
    var id = job && job.id;
    setTabEnabled(
      els.tabReview,
      job && job.status === "done" ? "#/review/" + encodeURIComponent(id) : "",
      id ? "Wait until scoring finishes" : "Open a finished run first"
    );
  }

  function setTabs(page) {
    var map = {
      runs: els.tabRuns,
      workspace: els.tabRuns,
      predict: els.tabPredict,
      benchmarks: els.tabBenchmarks,
      review: els.tabReview,
      select: els.tabReview,
      structure: els.tabReview,
    };
    var tabs = [els.tabRuns, els.tabPredict, els.tabReview, els.tabBenchmarks];
    for (var i = 0; i < tabs.length; i++) {
      if (tabs[i]) tabs[i].classList.remove("is-active");
    }
    if (map[page]) map[page].classList.add("is-active");
  }

  function showPage(id) {
    var pages = ["page-runs", "page-predict", "page-benchmarks", "page-workspace", "page-select", "page-structure"];
    for (var i = 0; i < pages.length; i++) {
      setHidden($(pages[i]), pages[i] !== id);
    }
  }

  function parseHash() {
    var raw = (location.hash || "#/").replace(/^#/, "");
    if (!raw || raw === "/") return { page: "runs" };
    var parts = raw.split("/").filter(Boolean);
    if (parts[0] === "predict") return { page: "predict" };
    if (parts[0] === "benchmarks") return { page: "benchmarks" };
    if (parts[0] === "review" || parts[0] === "select" || parts[0] === "structure") {
      if (parts[1]) return { page: "review", id: decodeURIComponent(parts[1]) };
      return { page: "review" };
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

  var openModelSeries = "";
  var modelSeriesBound = false;

  function inferModelSeries(name) {
    var raw = String(name || "").toLowerCase();
    if (raw === "venusrem2" || raw === "venusrem") return "venusrem2";
    if (raw.indexOf("prosst") === 0) return "prosst";
    if (raw.indexOf("saprot") === 0) return "saprot";
    if (raw.indexOf("progen") === 0) return "progen";
    if (raw.indexOf("protein_mpnn") === 0 || raw.indexOf("proteinmpnn") === 0) return "proteinmpnn";
    if (raw.indexOf("rita") === 0) return "rita";
    if (raw.indexOf("esm") === 0) return "esm";
    return "other";
  }

  function modelSeriesOf(spec) {
    if (!spec) return "other";
    if (spec.series) return spec.series;
    return inferModelSeries(spec.name);
  }

  function modelVariantLabel(spec) {
    if (!spec) return "";
    if (spec.label) return spec.label;
    if (VARIANT_LABELS[spec.name]) return VARIANT_LABELS[spec.name];
    return spec.description || spec.name;
  }

  function seriesLabelOf(seriesId, variants) {
    if (variants && variants[0] && variants[0].series_label) return variants[0].series_label;
    return SERIES_LABELS[seriesId] || seriesId;
  }

  function groupModelSeries() {
    var groups = {};
    var i;
    for (i = 0; i < state.models.length; i++) {
      var spec = state.models[i];
      var sid = modelSeriesOf(spec);
      if (!groups[sid]) groups[sid] = [];
      groups[sid].push(spec);
    }
    Object.keys(groups).forEach(function (sid) {
      groups[sid].sort(function (a, b) {
        var ao = a.variant_order;
        var bo = b.variant_order;
        if (ao == null) ao = 1000;
        if (bo == null) bo = 1000;
        if (ao !== bo) return ao - bo;
        return String(a.name) < String(b.name) ? -1 : 1;
      });
    });
    var out = [];
    var seen = {};
    for (i = 0; i < SERIES_ORDER.length; i++) {
      var id = SERIES_ORDER[i];
      if (groups[id] && groups[id].length) {
        out.push({ id: id, label: seriesLabelOf(id, groups[id]), variants: groups[id] });
        seen[id] = true;
      }
    }
    Object.keys(groups).forEach(function (sid) {
      if (!seen[sid] && groups[sid].length) {
        out.push({ id: sid, label: seriesLabelOf(sid, groups[sid]), variants: groups[sid] });
      }
    });
    return out;
  }

  function bindModelSeriesHost(host) {
    if (modelSeriesBound || !host) return;
    modelSeriesBound = true;
    host.addEventListener("click", function (ev) {
      ev.stopPropagation();
      var variant = ev.target.closest("[data-model]");
      if (variant) {
        setModel(variant.getAttribute("data-model"));
        return;
      }
      var seriesBtn = ev.target.closest("[data-series]");
      if (seriesBtn) toggleModelSeries(seriesBtn.getAttribute("data-series"));
    });
    document.addEventListener("click", function (ev) {
      if (!openModelSeries) return;
      if (host.contains(ev.target)) return;
      openModelSeries = "";
      renderModelSeries();
    });
  }

  function toggleModelSeries(seriesId) {
    var groups = groupModelSeries();
    var group = null;
    for (var i = 0; i < groups.length; i++) {
      if (groups[i].id === seriesId) {
        group = groups[i];
        break;
      }
    }
    if (!group) return;
    if (group.variants.length === 1) {
      openModelSeries = "";
      setModel(group.variants[0].name);
      return;
    }
    openModelSeries = openModelSeries === seriesId ? "" : seriesId;
    renderModelSeries();
  }

  function fillModels() {
    var sel = els.modelSelect;
    if (!sel) return;
    var snap = formSnapshot();
    var prev = sel.value || snap.model || "venusrem2";
    sel.innerHTML = "";
    var models = state.models.slice().sort(function (a, b) {
      if (a.name === "venusrem2") return -1;
      if (b.name === "venusrem2") return 1;
      var ao = a.variant_order;
      var bo = b.variant_order;
      if (ao == null) ao = 1000;
      if (bo == null) bo = 1000;
      if (ao !== bo) return ao - bo;
      return a.name < b.name ? -1 : 1;
    });
    for (var i = 0; i < models.length; i++) {
      var m = models[i];
      var opt = document.createElement("option");
      opt.value = m.name;
      opt.textContent = (m.label || VARIANT_LABELS[m.name] || m.name) + (m.needs_pdb ? "  (structure)" : "  (sequence)");
      sel.appendChild(opt);
    }
    if (prev && modelByName(prev)) sel.value = prev;
    else sel.value = "venusrem2";
    renderModelSeries();
    updateModelMeta();
    updateInputGate();
    updateCli();
  }

  function renderModelSeries() {
    var host = $("model-series");
    if (!host) return;
    bindModelSeriesHost(host);
    var current = (els.modelSelect && els.modelSelect.value) || "venusrem2";
    var currentSpec = modelByName(current);
    var currentSeries = modelSeriesOf(currentSpec) || inferModelSeries(current);
    var groups = groupModelSeries();
    var html = "";
    for (var i = 0; i < groups.length; i++) {
      var group = groups[i];
      var selected = currentSeries === group.id;
      var open = openModelSeries === group.id && group.variants.length > 1;
      var picked = null;
      var v;
      for (v = 0; v < group.variants.length; v++) {
        if (group.variants[v].name === current) {
          picked = group.variants[v];
          break;
        }
      }
      var sub = selected && picked ? modelVariantLabel(picked) : group.variants.length === 1 ? "" : group.variants.length + " variants";
      html += '<div class="model-series-item">';
      html +=
        '<button type="button" class="model-series-btn' +
        (selected ? " is-on" : "") +
        (open ? " is-open" : "") +
        '" data-series="' +
        esc(group.id) +
        '"' +
        (group.variants.length === 1 ? ' data-model="' + esc(group.variants[0].name) + '"' : "") +
        ' aria-expanded="' +
        (open ? "true" : "false") +
        '">' +
        esc(group.label) +
        (sub ? "<small>" + esc(sub) + "</small>" : "") +
        "</button>";
      if (open) {
        html += '<div class="model-variant-panel" role="listbox">';
        for (v = 0; v < group.variants.length; v++) {
          var spec = group.variants[v];
          html +=
            '<button type="button" class="model-variant' +
            (spec.name === current ? " is-on" : "") +
            '" data-model="' +
            esc(spec.name) +
            '" role="option" aria-selected="' +
            (spec.name === current ? "true" : "false") +
            '">' +
            esc(modelVariantLabel(spec)) +
            "</button>";
        }
        html += "</div>";
      }
      html += "</div>";
    }
    host.innerHTML = html;
  }

  function renderModelCards() {
    renderModelSeries();
  }

  function setModel(name) {
    if (!name || !els.modelSelect) return;
    if (modelByName(name)) els.modelSelect.value = name;
    openModelSeries = "";
    renderModelSeries();
    updateModelMeta();
    updateInputGate();
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
    updateRecipeHint();
    updateCli();
  }

  function currentBoard() {
    var catalog = state.catalog || FALLBACK_CATALOG;
    var boards = catalog.boards || [];
    var id = state.boardId || catalog.default || "substitutions";
    for (var i = 0; i < boards.length; i++) {
      if (boards[i].id === id) return boards[i];
    }
    return boards[0] || FALLBACK_BOARD;
  }

  function renderLeaderboard() {
    var catalog = state.catalog || FALLBACK_CATALOG;
    var board = currentBoard();
    state.board = board;
    var note = $("leaderboard-note");
    var title = $("board-title");
    var sub = $("board-sub");
    var filters = $("board-filters");
    var host = $("leaderboard-body");
    if (note) {
      note.textContent = board.note || "";
      note.hidden = !board.note;
    }
    if (title) title.textContent = board.title || "ProteinGym substitutions";
    if (sub) {
      sub.innerHTML =
        esc(String(board.n || 217)) +
        " proteins · mean Spearman · <a href=\"" +
        esc(catalog.url || "https://proteingym.org/benchmarks") +
        '" target="_blank" rel="noreferrer">proteingym.org</a>';
    }
    if (filters) {
      filters.innerHTML = (catalog.boards || [])
        .map(function (item) {
          return (
            '<button type="button" class="board-filter' +
            (item.id === board.id ? " is-on" : "") +
            '" data-board="' +
            esc(item.id) +
            '">' +
            esc(item.label || item.title) +
            "</button>"
          );
        })
        .join("");
    }
    if (!host) return;
    var rows = board.rows || [];
    var body = rows
      .map(function (row) {
        var tags = (row.inputs || [])
          .map(function (tag) {
            return '<span class="board-tag">' + esc(String(tag).toUpperCase()) + "</span>";
          })
          .join("");
        var first = row.highlight || row.rank === 1;
        var score = Number(row.score);
        var barPct = (Math.max(0, Math.min(1, isFinite(score) ? score : 0)) * 100).toFixed(1);
        return (
          '<tr class="' +
          (first ? "is-first" : "") +
          '"><td class="board-rank">' +
          esc(String(row.rank)) +
          '</td><td><span class="board-name">' +
          esc(row.name) +
          (row.note ? "<small>" + esc(row.note) + "</small>" : "") +
          '</span></td><td class="board-tags">' +
          tags +
          '</td><td class="board-score">' +
          '<span class="board-score-num">' +
          (isFinite(score) ? score.toFixed(3) : "") +
          '</span><span class="board-score-track" aria-hidden="true"><span class="board-score-bar" style="width:' +
          barPct +
          '%"></span></span></td></tr>'
        );
      })
      .join("");
    host.innerHTML =
      "<thead><tr><th>Rank</th><th>Model</th><th>Inputs</th><th class=\"num\">Score</th></tr></thead><tbody>" +
      body +
      "</tbody>";
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
    try {
      var board = await apiJson("/api/leaderboard");
      if (board && Array.isArray(board.boards) && board.boards.length) {
        state.catalog = board;
        if (!state.boardId) state.boardId = board.default || "substitutions";
        state.board = currentBoard();
      } else if (board && Array.isArray(board.rows) && board.rows.length) {
        state.board = board;
      }
    } catch (err) {
      /* keep fallback */
    }
    fillModels();
    fillRecipes();
    renderLeaderboard();
  }

  function classifyIntakeFile(file) {
    var n = ((file && file.name) || "").toLowerCase();
    if (/\.(pdb|ent|cif)$/.test(n)) return "pdb";
    if (/\.(a2m|a3m|sto)$/.test(n)) return "msa";
    if (/\.(csv|tsv)$/.test(n)) return "mutants";
    if (/\.(fa|fasta|faa|fna|txt)$/.test(n)) return "fasta";
    return "";
  }

  function setInputFile(input, file) {
    if (!input || !file) return;
    try {
      var dt = new DataTransfer();
      dt.items.add(file);
      input.files = dt.files;
    } catch (err) {
      /* browsers without DataTransfer keep the original picker */
    }
  }

  function assignDroppedFiles(fileList, slot) {
    var files = fileList || [];
    for (var i = 0; i < files.length; i++) {
      var kind = classifyIntakeFile(files[i]);
      if (slot === "fasta") {
        if (kind === "fasta") setInputFile($("file-fasta"), files[i]);
      } else if (slot === "pdb") {
        if (kind === "pdb") setInputFile($("file-pdb"), files[i]);
      } else if (slot === "msa") {
        if (kind === "msa" || kind === "fasta") setInputFile($("file-msa"), files[i]);
      } else if (kind === "fasta") setInputFile($("file-fasta"), files[i]);
      else if (kind === "pdb") setInputFile($("file-pdb"), files[i]);
      else if (kind === "msa") setInputFile($("file-msa"), files[i]);
      else if (kind === "mutants") setInputFile($("file-mutants"), files[i]);
    }
    updateFileLabels();
    renderIntakeFiles();
    updateInputGate();
    updateCli();
  }

  function renderSlotChips(id, items) {
    var host = $(id);
    if (!host) return;
    host.innerHTML = items
      .filter(Boolean)
      .map(function (text) {
        return '<span class="slot-chip">' + esc(text) + "</span>";
      })
      .join("");
  }

  function renderIntakeFiles() {
    var snap = formSnapshot();
    renderSlotChips("slot-sequence-chips", [
      snap.fasta ? "FASTA · " + snap.fasta : "",
      snap.seq_id || "",
    ]);
    renderSlotChips("slot-structure-chips", [
      snap.pdb ? "PDB · " + snap.pdb : "",
      snap.pdb_id || "",
      snap.uniprot_id || "",
    ]);
    renderSlotChips("slot-msa-chips", [snap.msa ? "MSA · " + snap.msa : ""]);
  }

  function runPhase(job) {
    if (!job) return "queue";
    var stage = ((job.progress || {}).stage || job.status || "").toLowerCase();
    if (job.status === "failed" || job.status === "cancelled") {
      return stage === "fetch" ? "fetch" : "score";
    }
    if (job.status === "done") return "review";
    if (stage === "fetch") return "fetch";
    if (stage === "score" || job.status === "running") return "score";
    return "queue";
  }

  function runFlowHtml(job) {
    var phase = runPhase(job);
    var order = ["queue", "fetch", "score", "review"];
    var labels = { queue: "1. Queued", fetch: "2. Structure", score: "3. Score", review: "4. Review" };
    var current = order.indexOf(phase);
    var html = '<ol class="flow-rail" aria-label="Run progress">';
    for (var i = 0; i < order.length; i++) {
      var cls = "flow-dot";
      if (job && (job.status === "failed" || job.status === "cancelled") && order[i] === phase) cls += " is-fail";
      else if (i === current) cls += " is-on";
      else if (i < current) cls += " is-done";
      html += "<li><span class=\"" + cls + "\">" + labels[order[i]] + "</span></li>";
    }
    return html + "</ol>";
  }

  function nextActionsHtml(job, opts) {
    if (!job || job.status !== "done") return "";
    opts = opts || {};
    var cards = "";
    if (!opts.hideReview) {
      cards +=
        '<a class="next-card" href="#/review/' +
        encodeURIComponent(job.id) +
        '"><strong>Review</strong><span>Sequence, structure, top 30, and score / pLDDT / RSA.</span></a>';
    }
    cards +=
      '<a class="next-card" href="#/predict"><strong>New job</strong><span>Score another sequence.</span></a>';
    return '<div class="next-row">' + cards + "</div>";
  }

  function readSplit(kind, fallback) {
    try {
      var n = parseFloat(localStorage.getItem(STORE_SPLIT[kind] || ""));
      if (isFinite(n)) return n;
    } catch (err) {
      /* ignore */
    }
    return fallback;
  }

  function writeSplit(kind, pct) {
    state.split[kind] = pct;
    try {
      localStorage.setItem(STORE_SPLIT[kind], String(pct));
    } catch (err) {
      /* ignore */
    }
  }

  function clampSplit(pct) {
    return Math.max(50, Math.min(82, pct));
  }

  function applySplit(bench, pct) {
    if (!bench) return;
    var viewer = bench.querySelector(".protein-viewer-pane");
    if (!viewer) return;
    viewer.style.flex = "0 0 " + pct + "%";
  }

  function resizeViewerSoon() {
    if (!state.viewer) return;
    setTimeout(function () {
      try {
        state.viewer.resize();
        state.viewer.render();
      } catch (err) {
        /* ignore */
      }
    }, 30);
  }

  function setBenchMode(on) {
    document.body.classList.toggle("is-bench", !!on);
    if (!on) document.body.classList.remove("is-splitting");
  }

  function mountBench(kind) {
    var bench = document.querySelector(".protein-bench");
    if (!bench) {
      setBenchMode(false);
      return;
    }
    var fallback = kind === "structure" ? 64 : 56;
    var pct = clampSplit(readSplit(kind, state.split[kind] || fallback));
    state.split[kind] = pct;
    applySplit(bench, pct);
    setBenchMode(true);
    resizeViewerSoon();
  }

  function proteinBenchHtml(kind, sideHtml, viewerHtml) {
    var seq = jobSequence(state.job);
    return (
      '<div class="review-wrap">' +
      '<div class="review-seq">' +
      aaNavHtml(seq) +
      aaStripHtml(seq, highlightPositions()) +
      "</div>" +
      '<div class="protein-bench" data-split="' +
      esc(kind) +
      '" data-viewer="end">' +
      '<aside class="protein-side-pane">' +
      sideHtml +
      "</aside>" +
      '<div class="protein-gutter" data-gutter role="separator" aria-orientation="vertical" title="Drag to resize"></div>' +
      '<section class="protein-viewer-pane">' +
      viewerHtml +
      "</section></div></div>"
    );
  }

  function lastLogLines(text, n) {
    var lines = String(text || "").split(/\r?\n/).filter(Boolean);
    return lines.slice(Math.max(0, lines.length - (n || 24))).join("\n");
  }

  function renderRuns() {
    var host = $("runs-body");
    if (!host) return;
    var runs = state.runs || [];
    if (!runs.length) {
      host.innerHTML =
        '<div class="empty-state"><h2>No runs yet</h2>' +
        "<p>Start a job, wait for scoring, then review and select.</p>" +
        '<a class="btn btn-primary" href="#/predict">New job</a></div>';
      return;
    }
    var rows = runs
      .map(function (run) {
        var id = encodeURIComponent(run.id);
        var dest = run.status === "done" ? "#/review/" + id : "#/runs/" + id;
        var actions =
          '<a class="btn-link" href="' +
          dest +
          '">' +
          (isLive(run.status) ? "Watch" : "Review") +
          "</a>";
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
          actions +
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

  function mutationPosSet(positions) {
    var set = {};
    for (var i = 0; i < (positions || []).length; i++) set[positions[i]] = true;
    return set;
  }

  function topMutPositions() {
    var top = (state.top && state.top.rows) || [];
    var set = {};
    for (var i = 0; i < top.length; i++) {
      var muts = parseMutants(mutantField(top[i]));
      for (var j = 0; j < muts.length; j++) set[muts[j].pos] = true;
    }
    return Object.keys(set).map(function (k) {
      return parseInt(k, 10);
    });
  }

  function rangePositions() {
    var range = state.seqRange;
    if (!range || !range.lo || !range.hi) return [];
    var out = [];
    var lo = Math.min(range.lo, range.hi);
    var hi = Math.max(range.lo, range.hi);
    for (var p = lo; p <= hi; p++) out.push(p);
    return out;
  }

  function highlightPositions() {
    return topMutPositions();
  }

  function currentMutPositions() {
    var fromRow = selectionResis();
    if (fromRow.length) return fromRow;
    var ranged = rangePositions();
    if (ranged.length) return ranged;
    return topMutPositions();
  }

  function renderSequence(seq, positions) {
    if (!seq) return '<p class="muted">Sequence not available.</p>';
    var set = mutationPosSet(positions);
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

  function aaStripHtml(seq, positions) {
    if (!seq) {
      return '<div class="aa-strip" id="aa-strip"><p class="aa-empty">No sequence yet.</p></div>';
    }
    var set = mutationPosSet(positions);
    var picked = state.picked && state.picked.resi;
    var html = "";
    for (var i = 0; i < seq.length; i++) {
      var pos = i + 1;
      var cls = "aa-cell seq-aa";
      if (set[pos]) cls += " is-mut";
      if (state.seqRange && pos >= Math.min(state.seqRange.lo, state.seqRange.hi) && pos <= Math.max(state.seqRange.lo, state.seqRange.hi)) {
        cls += " is-range";
      }
      if (picked === pos) cls += " is-pick";
      var showNum = pos === 1 || pos % 10 === 0 || picked === pos;
      html +=
        '<button type="button" class="' +
        cls +
        '" data-pos="' +
        pos +
        '" title="' +
        esc(seq.charAt(i)) +
        pos +
        '"><span class="aa-num' +
        (showNum ? " is-on" : "") +
        '">' +
        pos +
        '</span><span class="aa-letter">' +
        esc(seq.charAt(i)) +
        "</span></button>";
    }
    return (
      '<div class="aa-strip" id="aa-strip" tabindex="0" role="listbox" aria-label="Select a residue">' +
      html +
      "</div>"
    );
  }

  function aaNavHtml(seq) {
    var n = (seq && seq.length) || 0;
    var p = state.picked;
    var val = p && p.resi ? String(p.resi) : "";
    var status = p ? pickHtml() : n ? n + " residues — click a letter or type a position" : "Sequence not loaded";
    return (
      '<div class="aa-nav" id="aa-nav">' +
      '<button type="button" class="btn" data-aa="prev" title="Previous residue">‹</button>' +
      '<label class="aa-jump-lab">Residue <input id="aa-jump" type="number" min="1"' +
      (n ? ' max="' + n + '"' : "") +
      ' value="' +
      esc(val) +
      '" placeholder="1"></label>' +
      '<button type="button" class="btn" data-aa="go">Go</button>' +
      '<button type="button" class="btn" data-aa="next" title="Next residue">›</button>' +
      '<span class="aa-status" id="aa-status">' +
      esc(status) +
      "</span></div>"
    );
  }

  function scrollAaStripTo(pos) {
    if (!pos) return;
    var el = document.querySelector('.aa-cell[data-pos="' + pos + '"]');
    if (el && el.scrollIntoView) {
      el.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
    }
  }

  function refreshAaStripHighlight() {
    var strip = $("aa-strip");
    var picked = state.picked && state.picked.resi;
    if (strip) {
      var cells = strip.querySelectorAll(".aa-cell");
      for (var i = 0; i < cells.length; i++) {
        var pos = parseInt(cells[i].getAttribute("data-pos"), 10);
        var lo = state.seqRange ? Math.min(state.seqRange.lo, state.seqRange.hi) : 0;
        var hi = state.seqRange ? Math.max(state.seqRange.lo, state.seqRange.hi) : 0;
        cells[i].classList.toggle("is-pick", pos === picked);
        cells[i].classList.toggle("is-range", !!(lo && hi && pos >= lo && pos <= hi));
        var num = cells[i].querySelector(".aa-num");
        if (num) num.classList.toggle("is-on", pos === 1 || pos % 10 === 0 || pos === picked);
      }
      scrollAaStripTo(picked);
    }
    var status = $("aa-status");
    if (status) status.textContent = pickHtml();
    var jump = $("aa-jump");
    if (jump && picked) jump.value = String(picked);
    var host = document.querySelector(".seq-box .seq");
    if (host) {
      var letters = host.querySelectorAll(".seq-aa[data-pos]");
      for (var j = 0; j < letters.length; j++) {
        letters[j].classList.toggle(
          "seq-pick",
          parseInt(letters[j].getAttribute("data-pos"), 10) === picked
        );
      }
    }
  }

  function stepResidue(delta) {
    var seq = jobSequence(state.job) || "";
    var n = seq.length;
    if (!n) return;
    var cur = (state.picked && state.picked.resi) || 0;
    var next = cur ? cur + delta : delta > 0 ? 1 : n;
    if (next < 1) next = n;
    if (next > n) next = 1;
    pickSequencePos(next);
  }

  function goResidueInput() {
    var input = $("aa-jump");
    var n = parseInt(input && input.value, 10);
    if (n) pickSequencePos(n);
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
      ctx.fillText((data && data.empty) || "No histogram yet.", 8, 24);
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

  function jobProgress(job) {
    var p = (job && job.progress) || {};
    var pct = Number(p.pct);
    if (!isFinite(pct)) {
      if (job && job.status === "queued") pct = 4;
      else if (job && job.status === "running") pct = 18;
      else if (job && job.status === "done") pct = 100;
      else pct = 0;
    }
    return {
      pct: Math.max(0, Math.min(100, pct)),
      message: p.message || (job && job.status) || "",
      stage: p.stage || (job && job.status) || "",
    };
  }

  function progressHtml(job) {
    if (!job || (!isLive(job.status) && job.status !== "failed")) {
      if (job && job.status === "done") return "";
    }
    var p = jobProgress(job);
    if (!isLive(job.status) && job.status !== "failed") return "";
    return (
      '<div class="progress" id="run-progress">' +
      '<div class="progress-track"><div class="progress-bar' +
      (job.status === "failed" ? " is-fail" : "") +
      '" style="width:' +
      p.pct +
      '%"></div></div>' +
      '<div class="progress-meta"><span>' +
      esc(p.message || job.status) +
      "</span><span class=\"mono\">" +
      p.pct +
      "%</span></div></div>"
    );
  }

  function pipelineHtml(recipe) {
    var rec = String(recipe || "full");
    var html = [];
    for (var i = 0; i < PIPELINE.length; i++) {
      if (i) html.push('<span class="chip-arrow">→</span>');
      var step = PIPELINE[i];
      var on = step.on.indexOf(rec) !== -1;
      var opt = !on && step.opt && step.opt.indexOf(rec) !== -1;
      var cls = "chip";
      if (opt) cls += " is-opt";
      else if (!on) cls += " is-dim";
      var title = opt ? ' title="optional; skip → α=0"' : "";
      html.push('<span class="' + cls + '"' + title + ">" + esc(step.label) + "</span>");
    }
    return '<div class="chips" aria-label="Pipeline">' + html.join("") + "</div>";
  }

  function scoreTableHtml(job, payload) {
    var rows = (payload && payload.rows) || [];
    var total = (payload && payload.total) || rows.length;
    var rankingTotal = (job && (job.n_mutants != null ? job.n_mutants : job.n_rows)) || total;
    var offset = (payload && payload.offset) != null ? payload.offset : state.offset;
    var primary = primaryScore(job, payload);
    var selMut = mutantField(state.selectedRow);
    var head =
      "<tr>" +
      '<th class="experiment-check"><span class="sr-only">Experiment</span></th>' +
      thSort("rank", "Rank") +
      thSort("mutant", "Mutation") +
      thSort("score", "Ranking score") +
      '<th>Rank percentile</th>' +
      '<th>Status</th>' +
      "</tr>";
    var body = rows
      .map(function (row, i) {
        var mut = mutantField(row);
        var rank = row.rank != null ? row.rank : offset + i + 1;
        var sel = mut && mut === selMut ? " is-sel" : "";
        var checked = !!state.experimentRows[mut];
        var percentile = rankingTotal ? Math.max(0, Math.min(100, 100 - ((rank - 1) / rankingTotal) * 100)) : 0;
        var score = rowScore(row, primary);
        return (
          '<tr class="' +
          sel +
          '" data-i="' +
          i +
          '">' +
          '<td class="experiment-check"><input type="checkbox" data-experiment-mutant="' +
          esc(mut) +
          '" aria-label="Add ' +
          esc(mut) +
          ' to experiment"' +
          (checked ? " checked" : "") +
          '></td><td class="num rank-cell">' +
          esc(rank) +
          '</td><td class="mono mutation-cell">' +
          esc(mut || "—") +
          "</td>" +
          '<td class="num score-cell"><strong>' +
          fmtScore(score) +
          '</strong><span class="score-mark' +
          (Number(score) < 0 ? " is-negative" : "") +
          '"><i style="width:' +
          Math.max(4, Math.min(100, percentile)).toFixed(1) +
          '%"></i></span></td>' +
          '<td class="percentile-cell"><span class="confidence-dot"></span>' +
          percentile.toFixed(1) +
          '%</td><td class="status-cell">' +
          (checked ? '<span class="selected-status">Selected</span>' : '<span class="muted">Candidate</span>') +
          "</td>" +
          "</tr>"
        );
      })
      .join("");
    if (!body) {
      body = '<tr><td colspan="6"><div class="empty">No candidates match these filters.</div></td></tr>';
    }
    var from = total ? offset + 1 : 0;
    var to = Math.min(offset + rows.length, total);
    return (
      '<div class="table-scroll candidate-table-scroll"><table class="data candidate-table" id="score-table"><thead>' +
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
      " · ranking score, not ΔΔG" +
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

  function experimentRows() {
    return Object.keys(state.experimentRows || {}).map(function (key) {
      return state.experimentRows[key];
    });
  }

  function toggleExperimentRow(row, checked) {
    var mut = mutantField(row);
    if (!mut) return;
    if (checked) state.experimentRows[mut] = row;
    else delete state.experimentRows[mut];
    state.experimentReady = false;
  }

  function featureAt(name, pos) {
    var values = (state.features && state.features[name]) || [];
    var value = pos ? Number(values[pos - 1]) : NaN;
    return isFinite(value) ? value : null;
  }

  function percentileFor(row) {
    var total = (state.job && (state.job.n_mutants != null ? state.job.n_mutants : state.job.n_rows)) ||
      (state.scores && state.scores.total) || 0;
    var rank = row && row.rank != null ? Number(row.rank) : 0;
    if (!total || !rank) return 0;
    return Math.max(0, Math.min(100, 100 - ((rank - 1) / total) * 100));
  }

  function normalizedScore(row) {
    var value = Number(rowScore(row, primaryScore(state.job, state.scores)));
    var lo = Number(state.histogram && state.histogram.min);
    var hi = Number(state.histogram && state.histogram.max);
    if (!isFinite(value)) return 0;
    if (!isFinite(lo) || !isFinite(hi) || hi === lo) return Math.max(0, Math.min(100, 50 + value * 10));
    return Math.max(0, Math.min(100, ((value - lo) / (hi - lo)) * 100));
  }

  function comparisonMetricBand(metric, value) {
    if (value == null || value === "") return "na";
    value = Number(value);
    if (!isFinite(value)) return "na";
    if (metric === "rank") {
      if (value >= 90) return "rank-high";
      if (value >= 70) return "rank-mid";
      return "rank-low";
    }
    if (metric === "plddt") {
      if (value >= 90) return "plddt-very-high";
      if (value >= 70) return "plddt-confident";
      if (value >= 50) return "plddt-low";
      return "plddt-very-low";
    }
    if (metric === "rsa") {
      if (value <= 0.2) return "rsa-buried";
      if (value <= 0.5) return "rsa-partial";
      return "rsa-exposed";
    }
    return "na";
  }

  function comparisonMetricHtml(metric, width, value, label, title) {
    var band = comparisonMetricBand(metric, value);
    return '<span class="metric metric--' + band + '" title="' + esc(title) + '">' +
      '<b style="width:' + Math.max(0, Math.min(100, width)).toFixed(1) + '%"></b>' +
      '<em>' + esc(label) + '</em></span>';
  }

  function reviewModeTabsHtml() {
    return (
      '<div class="evidence-tabs" role="tablist" aria-label="Evidence view">' +
      '<button type="button" role="tab" data-evidence-mode="evidence" class="' +
      (state.evidenceMode === "evidence" ? "is-on" : "") +
      '" aria-selected="' +
      (state.evidenceMode === "evidence") +
      '">Single candidate</button>' +
      '<button type="button" role="tab" data-evidence-mode="compare" class="' +
      (state.evidenceMode === "compare" ? "is-on" : "") +
      '" aria-selected="' +
      (state.evidenceMode === "compare") +
      '">Compare selected</button></div>'
    );
  }

  function singleEvidenceHtml(row) {
    if (!row) return '<div class="review-empty">Select a candidate to inspect its evidence.</div>';
    var mut = mutantField(row);
    var parsed = parseMutants(mut);
    var m = parsed[0] || {};
    var plddt = featureAt("plddt", m.pos);
    var rsa = featureAt("rsa", m.pos);
    return (
      '<div class="single-evidence">' +
      '<div class="evidence-chart"><canvas id="hist-canvas" class="hist-canvas"></canvas>' +
      '<div class="hist-meta"><span id="hist-canvas-tip">Score distribution</span><span>' +
      esc(mut) +
      " " +
      fmtScore(rowScore(row, primaryScore(state.job, state.scores))) +
      '</span></div></div>' +
      '<dl class="evidence-facts">' +
      '<div><dt>Position</dt><dd>' + esc(m.pos || "—") + '</dd></div>' +
      '<div><dt>Substitution</dt><dd>' + esc(m.wt && m.mut ? m.wt + " → " + m.mut : mut) + '</dd></div>' +
      '<div><dt>Rank percentile</dt><dd class="positive">' + percentileFor(row).toFixed(1) + '%</dd></div>' +
      '<div><dt>pLDDT</dt><dd>' + (plddt == null ? "n/a" : plddt.toFixed(1)) + '</dd></div>' +
      '<div><dt>RSA</dt><dd>' + (rsa == null ? "n/a" : rsa.toFixed(3)) + '</dd></div>' +
      '</dl></div>'
    );
  }

  function compareEvidenceHtml() {
    var rows = experimentRows();
    if (!rows.length && state.selectedRow) rows = [state.selectedRow];
    rows = rows.slice(0, 6);
    if (!rows.length) return '<div class="review-empty">Select candidates in the table to compare them here.</div>';
    var body = rows.map(function (row, index) {
      var mut = mutantField(row);
      var m = parseMutants(mut)[0] || {};
      var plddt = featureAt("plddt", m.pos);
      var rsa = featureAt("rsa", m.pos);
      var percentile = percentileFor(row);
      var plddtPct = plddt == null ? 0 : Math.max(0, Math.min(100, plddt));
      var rsaPct = rsa == null ? 0 : Math.max(0, Math.min(100, rsa * 100));
      return (
        '<div class="comparison-row"><strong><i style="--series:' + index + '"></i>' + esc(mut) + '</strong>' +
        comparisonMetricHtml("rank", normalizedScore(row), percentile, fmtScore(rowScore(row, primaryScore(state.job, state.scores))), "Ranking band follows rank percentile") +
        comparisonMetricHtml("rank", percentile, percentile, percentile.toFixed(0) + "%", "High ≥90 · medium 70–89 · low <70") +
        comparisonMetricHtml("plddt", plddtPct, plddt, plddt == null ? "n/a" : plddt.toFixed(0), "pLDDT: very high ≥90 · confident 70–89 · low 50–69 · very low <50") +
        comparisonMetricHtml("rsa", rsaPct, rsa, rsa == null ? "n/a" : rsa.toFixed(2), "RSA: buried ≤0.20 · partial 0.21–0.50 · exposed >0.50") + '</div>'
      );
    }).join("");
    var uniquePositions = experimentMutationPositions();
    var structureNote = rows.length > 1
      ? rows.length + " candidates map to " + uniquePositions.length + " unique structure position" + (uniquePositions.length === 1 ? "" : "s") + "."
      : "";
    var plddtNote = state.features && state.features.has_pdb && !state.features.has_plddt
      ? " Experimental structure: pLDDT is unavailable; RSA is calculated from the PDB."
      : "";
    return (
      '<div class="comparison-grid"><div class="comparison-head"><span>Mutation</span><span>Score</span><span>Percentile</span><span>pLDDT</span><span>RSA</span></div>' +
      body +
      '<div class="comparison-legend" aria-label="Metric color thresholds">' +
      '<span><i class="rank-high"></i>Rank ≥90</span><span><i class="rank-mid"></i>70–89</span><span><i class="rank-low"></i>&lt;70</span>' +
      '<span><i class="plddt-very-high"></i>pLDDT ≥90</span><span><i class="plddt-confident"></i>70–89</span><span><i class="plddt-low"></i>50–69</span><span><i class="plddt-very-low"></i>&lt;50</span>' +
      '<span><i class="rsa-buried"></i>RSA buried</span><span><i class="rsa-partial"></i>partial</span><span><i class="rsa-exposed"></i>exposed</span></div>' +
      '<p class="comparison-note">' + esc(structureNote + plddtNote) +
      (experimentRows().length > 6 ? " Showing the first 6 selected candidates." : "") + '</p>' +
      '</div>'
    );
  }

  function reviewEvidenceHtml() {
    var row = state.selectedRow;
    var mut = mutantField(row) || "Candidate";
    return (
      '<section class="review-evidence" aria-live="polite"><div class="review-section-head"><div><span class="section-kicker">Decision support</span><h2>' +
      (state.evidenceMode === "compare" ? "Candidate comparison" : "Evidence for " + esc(mut)) +
      '</h2></div><div class="evidence-head-actions"><span class="score-disclaimer">Ranking score · not ΔΔG</span>' +
      reviewModeTabsHtml() +
      '</div></div>' +
      (state.evidenceMode === "compare" ? compareEvidenceHtml() : singleEvidenceHtml(row)) +
      '</section>'
    );
  }

  function experimentTrayHtml() {
    var rows = experimentRows();
    var chips = rows.slice(0, 6).map(function (row) {
      var mut = mutantField(row);
      return '<button type="button" class="experiment-chip" data-remove-experiment="' + esc(mut) + '" title="Remove ' + esc(mut) + '">' + esc(mut) + '<span>Remove</span></button>';
    }).join("");
    if (rows.length > 6) chips += '<span class="more-selected">+' + (rows.length - 6) + ' more</span>';
    return (
      '<footer class="experiment-tray"><div class="experiment-count"><strong>' + rows.length + ' / 24 selected</strong><span>' +
      (state.experimentReady ? "Experiment batch ready" : "Choose variants for experimental validation") +
      '</span></div><div class="experiment-chips">' + (chips || '<span class="tray-empty">No candidates selected yet</span>') +
      '</div><div class="experiment-actions"><button type="button" class="btn" id="btn-export-experiment"' +
      (!rows.length ? " disabled" : "") +
      '>Export CSV</button><button type="button" class="btn btn-primary" id="btn-add-experiment"' +
      (!rows.length ? " disabled" : "") +
      '>' + (state.experimentReady ? "Experiment ready" : "Add to experiment") + '</button></div></footer>'
    );
  }

  function downloadExperimentCsv() {
    var rows = experimentRows();
    if (!rows.length) return;
    var primary = primaryScore(state.job, state.scores);
    var lines = ["mutant," + csvEscape(primary || "score") + ",rank"];
    rows.forEach(function (row, i) {
      lines.push([csvEscape(mutantField(row)), csvEscape(rowScore(row, primary)), csvEscape(row.rank != null ? row.rank : i + 1)].join(","));
    });
    var blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "rem2-" + (state.route.id || "run") + "-experiment.csv";
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 0);
  }

  function renderWorkspace() {
    var host = $("workspace-body");
    if (!host) return;
    var job = state.job;
    if (!job) {
      setBenchMode(false);
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
    var err = job.error
      ? '<div class="error-box">' + esc(job.error) + "</div>"
      : "";
    var cancel =
      isLive(job.status)
        ? '<button type="button" class="btn btn-danger" id="btn-cancel">Cancel</button>'
        : "";
    var head =
      '<div class="page-head"><h1 class="mono">' +
      esc(job.protein || job.id) +
      "</h1>" +
      '<p class="muted">Scoring this run. Review opens when it finishes.</p></div>' +
      runFlowHtml(job) +
      '<div class="panel overview">' +
      ov("protein", job.protein || "—") +
      ov("model", job.model || "—") +
      '<div class="overview-item"><div class="overview-k">status</div><div class="overview-v">' +
      badge(job.status) +
      "</div></div>" +
      ov("mutants", nMut != null ? nMut : "—") +
      ov("recipe", job.recipe || "—") +
      "</div>" +
      progressHtml(job) +
      err;
    if (isLive(job.status) || (job.status === "failed" && !(state.scores && state.scores.total))) {
      setBenchMode(false);
      host.innerHTML =
        head +
        '<div class="panel"><div class="toolbar"><h2 style="margin:0;flex:1">Live log</h2>' +
        cancel +
        "</div><pre class=\"run-log\" id=\"run-log\">" +
        esc(lastLogLines(state.logText, 30) || "Waiting for rem2…") +
        "</pre></div>";
      return;
    }
    if (!state.selectedRow && state.scores && state.scores.rows && state.scores.rows.length) {
      state.selectedRow = state.scores.rows[0];
    }
    var hasCoord = !!(job.has_pdb || currentSourceKey(job));
    var seq = jobSequence(job);
    var total = nMut || (state.scores && state.scores.total) || 0;
    var selectedMut = mutantField(state.selectedRow) || "—";
    var selectedRows = experimentRows();
    var selectedPositions = experimentMutationPositions();
    var structureSelection = selectedRows.length
      ? selectedRows.length + " candidates · " + selectedPositions.length + " positions"
      : selectedMut;
    var header =
      '<header class="review-context"><div><div class="review-title-row"><h1>' +
      esc(job.protein || job.id) +
      '</h1><span class="badge badge-done">Complete</span></div><p>' +
      esc(job.description || "Candidate review for experimental validation") +
      '</p></div><dl><div><dt>Variants</dt><dd>' +
      esc(total) +
      '</dd></div><div><dt>Model</dt><dd>' +
      esc(job.model || "—") +
      '</dd></div><div><dt>Recipe</dt><dd>' +
      esc(job.recipe || "—") +
      '</dd></div></dl></header>';
    var candidates =
      '<section class="candidate-pane"><div class="candidate-head"><div><span class="section-kicker">Ranked output</span><h2>Candidates <small>(' +
      esc(total) +
      ')</small></h2></div><button type="button" class="btn btn-ghost" id="btn-copy-selected">Copy row</button></div>' +
      '<div class="candidate-filters"><input type="search" id="score-q" placeholder="Search mutations, e.g. M55K" title="Press Enter" value="' +
      esc(state.query) +
      '"><span class="filter-chip is-on">All positions</span><span class="filter-chip">Single substitutions</span></div>' +
      scoreTableHtml(job, state.scores) +
      '</section>';
    var structure =
      '<section class="structure-pane"><div class="review-section-head structure-head"><div><span class="section-kicker">Molecular context</span><h2>Structure <small>' +
      esc(structureSelection) +
      '</small></h2></div></div>' +
      (hasCoord
        ? molPanelHtml("bench")
        : '<div class="viewer-empty"><h2>No coordinates</h2><p>Add or fetch a PDB structure to link candidates to their molecular context.</p></div>') +
      '<div class="sequence-panel"><div class="sequence-label"><strong>Amino acid sequence</strong><span>' +
      esc((seq && seq.length) || 0) +
      ' residues</span></div>' +
      aaStripHtml(seq, currentMutPositions()) +
      '</div></section>';
    host.innerHTML =
      '<div class="review-workbench">' +
      header +
      err +
      '<div class="review-grid">' +
      candidates +
      '<div class="review-right">' +
      structure +
      reviewEvidenceHtml() +
      '</div></div>' +
      experimentTrayHtml() +
      '</div>';
    setBenchMode(true);
    drawReviewHists();
  }

  function rerenderReview() {
    teardownViewer();
    renderWorkspace();
    if ($("mol-host")) maybeLoadViewer(state.job, []);
  }

  function histPanelHtml(id, title, extra) {
    return (
      '<div class="panel hist-box"><h2>' +
      esc(title) +
      "</h2>" +
      '<canvas id="' +
      id +
      '" class="hist-canvas"></canvas>' +
      '<div class="hist-meta"><span id="' +
      id +
      '-tip"> </span><span>' +
      esc(extra || "") +
      "</span></div></div>"
    );
  }

  function drawReviewHists() {
    var score = $("hist-canvas");
    if (score) drawHistogram(score, state.histogram || { bins: [] });
    var feats = state.features || {};
    var plddt = $("plddt-canvas");
    if (plddt) {
      var pHist = feats.plddt_histogram || { bins: [] };
      if (!(feats.plddt && feats.plddt.length)) {
        pHist = {
          bins: [],
          empty: feats.has_pdb && !feats.has_plddt
            ? "Crystal / experimental PDB — no pLDDT."
            : "No pLDDT yet.",
        };
      }
      drawHistogram(plddt, pHist);
    }
    var rsa = $("rsa-canvas");
    if (rsa) {
      var rHist = feats.rsa_histogram || { bins: [] };
      if (!(feats.rsa && feats.rsa.length)) rHist = { bins: rHist.bins || [], empty: rHist.bins && rHist.bins.length ? "" : "No RSA yet." };
      drawHistogram(rsa, rHist.bins && rHist.bins.length ? rHist : { bins: [], empty: "No RSA yet." });
    }
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
      var experimentInput = ev.target.closest("[data-experiment-mutant]");
      if (experimentInput && host.contains(experimentInput)) {
        var experimentIndex = parseInt(experimentInput.closest("tr").getAttribute("data-i"), 10);
        var experimentRow = ((state.scores && state.scores.rows) || [])[experimentIndex];
        state.selectedRow = experimentRow || state.selectedRow;
        var experimentMutation = parseMutants(mutantField(experimentRow))[0];
        state.picked = experimentMutation
          ? { chain: "", resi: experimentMutation.pos, resn: experimentMutation.wt, b: null }
          : null;
        toggleExperimentRow(experimentRow, experimentInput.checked);
        rerenderReview();
        return;
      }
      var mode = ev.target.closest("[data-evidence-mode]");
      if (mode && host.contains(mode)) {
        state.evidenceMode = mode.getAttribute("data-evidence-mode") === "compare" ? "compare" : "evidence";
        rerenderReview();
        return;
      }
      var remove = ev.target.closest("[data-remove-experiment]");
      if (remove && host.contains(remove)) {
        delete state.experimentRows[remove.getAttribute("data-remove-experiment")];
        state.experimentReady = false;
        rerenderReview();
        return;
      }
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
        var rowMutation = parseMutants(mutantField(state.selectedRow))[0];
        state.picked = rowMutation
          ? { chain: "", resi: rowMutation.pos, resn: rowMutation.wt, b: null }
          : null;
        rerenderReview();
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
      } else if (id === "btn-export-experiment") {
        downloadExperimentCsv();
      } else if (id === "btn-add-experiment") {
        if (experimentRows().length) {
          state.experimentReady = true;
          flash(experimentRows().length + " candidates prepared for experimental validation.");
          rerenderReview();
        }
      } else if (id === "btn-cancel") {
        cancelJob(state.job && state.job.id);
      }
      var molEl = ev.target.closest("[data-mol]");
      if (molEl && host.contains(molEl)) {
        handleMolControl(molEl);
        return;
      }
      var aaNav = ev.target.closest("[data-aa]");
      if (aaNav && host.contains(aaNav)) {
        if (aaNav.getAttribute("data-aa") === "go") goResidueInput();
        else if (aaNav.getAttribute("data-aa") === "prev") stepResidue(-1);
        else if (aaNav.getAttribute("data-aa") === "next") stepResidue(1);
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
      if (ev.target.id === "aa-jump" && ev.key === "Enter") {
        ev.preventDefault();
        goResidueInput();
        return;
      }
      if (ev.target.closest && ev.target.closest("#aa-strip")) {
        if (ev.key === "ArrowLeft") {
          ev.preventDefault();
          stepResidue(-1);
        } else if (ev.key === "ArrowRight") {
          ev.preventDefault();
          stepResidue(1);
        }
        return;
      }
      if (ev.key !== "Enter" || ev.target.id !== "score-q") return;
      state.query = ev.target.value.trim();
      state.offset = 0;
      loadScores().then(patchScores);
    });
    host.addEventListener("mousemove", function (ev) {
      if (!ev.target.classList || !ev.target.classList.contains("hist-canvas")) return;
      var tip = $(ev.target.id + "-tip") || $("hist-tip");
      if (tip) tip.textContent = histTip(ev.target, ev);
    });
    host.addEventListener("mouseleave", function (ev) {
      if (!ev.target.classList || !ev.target.classList.contains("hist-canvas")) return;
      var tip = $(ev.target.id + "-tip") || $("hist-tip");
      if (tip) tip.textContent = " ";
    }, true);
    host.addEventListener("pointerdown", function (ev) {
      var cell = ev.target.closest && ev.target.closest(".aa-cell[data-pos]");
      if (!cell || !host.contains(cell)) return;
      var pos = parseInt(cell.getAttribute("data-pos"), 10);
      if (!pos) return;
      state.seqDrag = { start: pos, end: pos };
      state.seqRange = { lo: pos, hi: pos };
      state.selectedRow = null;
      pickSequencePos(pos);
      refreshAaStripHighlight();
      try {
        host.setPointerCapture(ev.pointerId);
      } catch (err) {
        /* ignore */
      }
    });
    host.addEventListener("pointermove", function (ev) {
      if (!state.seqDrag) return;
      var el = document.elementFromPoint(ev.clientX, ev.clientY);
      var cell = el && el.closest && el.closest(".aa-cell[data-pos]");
      if (!cell) return;
      var pos = parseInt(cell.getAttribute("data-pos"), 10);
      if (!pos) return;
      state.seqDrag.end = pos;
      state.seqRange = {
        lo: Math.min(state.seqDrag.start, pos),
        hi: Math.max(state.seqDrag.start, pos),
      };
      refreshAaStripHighlight();
    });
    host.addEventListener("pointerup", function () {
      if (state.seqDrag) {
        pickSequencePos(state.seqDrag.end);
        applyView({ zoom: "sel" });
      }
      state.seqDrag = null;
    });
  }

  function patchScores() {
    if (state.route.page === "review") {
      rerenderReview();
      return;
    }
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
    refreshSequenceHighlight();
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
    if (state.view.bg === "white") return "#f7f8f6";
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

  function experimentMutationPositions() {
    var set = {};
    experimentRows().forEach(function (row) {
      parseMutants(mutantField(row)).forEach(function (m) {
        if (m.pos) set[m.pos] = true;
      });
    });
    return Object.keys(set).map(function (pos) { return parseInt(pos, 10); }).sort(function (a, b) { return a - b; });
  }

  function selectionResis() {
    var seen = {};
    var pos = experimentMutationPositions().concat(mutationPositions()).filter(function (resi) {
      if (!resi || seen[resi]) return false;
      seen[resi] = true;
      return true;
    });
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
    if (mode === "slate") return { color: "#7f9fc4" };
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
    var bench = kind === "bench" || kind === "full";
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
        '<label class="mol-check"><input type="checkbox" data-mol="' +
        name +
        '"' +
        (v[name] ? " checked" : "") +
        "> " +
        label +
        "</label>"
      );
    }
    var seq = jobSequence(state.job);
    var overlay =
      '<div class="mol-overlay">' +
      '<select data-mol="rep" title="Representation">' +
      opt("rep", "cartoon", "Cartoon") +
      opt("rep", "cartoon-stick", "Cartoon + sticks") +
      opt("rep", "stick", "Sticks") +
      opt("rep", "line", "Lines") +
      opt("rep", "sphere", "Spheres") +
      "</select>" +
      '<select data-mol="color" title="Color">' +
      opt("color", "spectrum", "Spectrum") +
      opt("color", "slate", "Slate") +
      opt("color", "chain", "Chain") +
      opt("color", "ss", "Secondary structure") +
      opt("color", "plddt", sourceHasPlddt(state.job) ? "pLDDT" : "pLDDT (n/a)") +
      opt("color", "hydrophobic", "Hydrophobicity") +
      opt("color", "white", "Uniform") +
      "</select>" +
      sourceSelectHtml() +
      '<select data-mol="sidechains" title="Side chains">' +
      opt("sidechains", "off", "SC off") +
      opt("sidechains", "selection", "SC selection") +
      opt("sidechains", "polar", "SC polar") +
      opt("sidechains", "all", "SC all") +
      "</select>" +
      '<select data-mol="bg" title="Background">' +
      opt("bg", "dark", "Dark") +
      opt("bg", "black", "Black") +
      opt("bg", "white", "White") +
      "</select>" +
      chk("surface", "Surface") +
      chk("labels", "Labels") +
      chk("hetero", "Het") +
      chk("water", "HOH") +
      '<button type="button" class="btn" data-mol="reset">Reset</button>' +
      '<button type="button" class="btn" data-mol="zoom">Zoom sel</button>' +
      '<button type="button" class="btn" data-mol="fs">' +
      (document.body.classList.contains("mol-fs") ? "Exit full" : "Full") +
      "</button></div>";
    var canvas =
      '<div class="mol-fs-target" id="mol-fs-box">' +
      '<div class="mol-top">' +
      overlay +
      (bench ? "" : aaNavHtml(seq) + aaStripHtml(seq, currentMutPositions())) +
      "</div>" +
      '<div id="mol-host" class="viewer-host viewer-host-fill"></div>' +
      '<div class="mol-chrome"><div class="mol-legend" id="mol-legend"></div>' +
      '<div class="mol-pick" id="mol-pick"></div></div></div>';
    if (bench) return '<div class="mol-stage mol-stage-fill">' + canvas + "</div>";
    return '<div class="mol-toolbar">' + overlay + "</div>" + canvas;
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
    return '<select data-mol="source" title="Structure">' + options + "</select>";
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
    var candidates = experimentRows();
    var positions = experimentMutationPositions();
    if (candidates.length) {
      return "<span>Experiment selection: " + candidates.length + " candidates across " + positions.length + " highlighted positions.</span>";
    }
    return "<span>Click a residue or select candidates in the table to inspect the structure.</span>";
  }

  function pickHtml() {
    var p = state.picked;
    if (!p) {
      var selected = experimentRows();
      var positions = experimentMutationPositions();
      if (selected.length) return positions.length + " selected positions: " + positions.join(", ");
      var current = parseMutants(mutantField(state.selectedRow))[0];
      return current ? mutantField(state.selectedRow) + " · residue " + current.pos : "No residue selected.";
    }
    var aa = RESN_TO_AA[p.resn] || "";
    var mut = selectionResis().indexOf(p.resi) >= 0 ? "  ·  linked to selection" : "";
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
    if (!viewer || !window.$3Dmol) {
      refreshAaStripHighlight();
      return;
    }
    var mol = window.$3Dmol;
    var style = colorStyle();
    var selPos = selectionResis();
    var experimentPos = experimentMutationPositions();
    var inspectedPos = mutationPositions().filter(function (pos) {
      return experimentPos.indexOf(pos) < 0;
    });
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
        viewer.addStyle({ resi: selPos }, { stick: { radius: 0.16, color: "#8769e8" } });
      }
      if (experimentPos.length) {
        viewer.addStyle(
          { resi: experimentPos },
          { cartoon: { color: "#8769e8", thickness: 0.5 } }
        );
      }
      if (inspectedPos.length) {
        viewer.addStyle(
          { resi: inspectedPos },
          { cartoon: { color: "#168676", thickness: 0.5 } }
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
    refreshAaStripHighlight();
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
      if (!positions || !positions.length) {
        viewer.zoom(1.3);
        viewer.render();
      }
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
  }

  function refreshSequenceHighlight() {
    var host = document.querySelector(".seq-box");
    var muts = highlightPositions();
    if (host) {
      var h2 = host.querySelector("h2");
      host.innerHTML = "";
      if (h2) host.appendChild(h2);
      var tmp = document.createElement("div");
      tmp.innerHTML = renderSequence(jobSequence(state.job), muts);
      while (tmp.firstChild) host.appendChild(tmp.firstChild);
    }
    var strip = $("aa-strip");
    var nav = $("aa-nav");
    var seq = jobSequence(state.job);
    if (strip) {
      var wrap = document.createElement("div");
      wrap.innerHTML = aaStripHtml(seq, muts);
      strip.replaceWith(wrap.firstChild);
    }
    if (nav) {
      var wrapNav = document.createElement("div");
      wrapNav.innerHTML = aaNavHtml(seq);
      nav.replaceWith(wrapNav.firstChild);
    }
    refreshAaStripHighlight();
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

  async function loadFeatures(id) {
    try {
      state.features = await apiJson("/api/runs/" + encodeURIComponent(id) + "/features");
    } catch (err) {
      state.features = { plddt_histogram: { bins: [] }, rsa_histogram: { bins: [] }, has_plddt: false };
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
      state.experimentRows = {};
      state.experimentReady = false;
      state.evidenceMode = "evidence";
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
    await Promise.all([loadScores(), loadHistogram(id), loadLog(id), loadTop(id), loadFeatures(id)]);
    renderWorkspace();
    if ($("mol-host")) {
      await maybeLoadViewer(state.job, []);
    }
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
    if (!job) {
      host.innerHTML =
        '<div class="empty-state"><h2>No run selected</h2>' +
        "<p>Finish a scoring job first, then export top-K from here.</p>" +
        '<a class="btn btn-primary" href="#/runs">Back to runs</a></div>';
      return;
    }
    if (isLive(job.status)) {
      host.innerHTML =
        '<div class="page-head"><h1>Select</h1>' +
        '<p class="muted">This step waits for scores.</p></div>' +
        runFlowHtml(job) +
        '<div class="empty-state"><h2>Scoring is still running</h2>' +
        "<p>Top-K is available after the score table is written.</p>" +
        '<a class="btn btn-primary" href="#/runs/' +
        encodeURIComponent(job.id) +
        '">Watch this run</a></div>';
      return;
    }
    if (job.status !== "done") {
      host.innerHTML =
        '<div class="page-head"><h1>Select</h1></div>' +
        runFlowHtml(job) +
        '<div class="empty-state"><h2>No scores to export</h2>' +
        "<p>This run " +
        esc(job.status || "stopped") +
        ". Open it to read the log, or start a new job.</p>" +
        '<a class="btn btn-primary" href="#/runs/' +
        encodeURIComponent(job.id) +
        '">Open run</a></div>';
      return;
    }
    if (!body) body = '<tr><td colspan="' + (dms ? 4 : 3) + '"><div class="empty">No variants yet.</div></td></tr>';
    host.innerHTML =
      '<div class="page-head"><h1>Select</h1>' +
      '<p class="muted">Step 5 — export top-ranked mutants for <span class="mono">' +
      esc((job && (job.protein || job.id)) || state.route.id || "") +
      "</span></p></div>" +
      (job ? runFlowHtml(job) : "") +
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
    if (!isFinite(n)) n = 30;
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
    if (!job || !job.id) {
      setBenchMode(false);
      host.innerHTML =
        '<div class="empty-state"><h2>No run selected</h2>' +
        "<p>Open a run first. If it has no PDB, fetch RCSB / AlphaFold here.</p>" +
        '<a class="btn btn-primary" href="#/runs">Back to runs</a></div>';
      return;
    }
    var seq = jobSequence(job);
    var muts = parseMutants(mutantField(state.selectedRow));
    var positions = muts.map(function (m) {
      return m.pos;
    });
    var hasCoord = !!(job.has_pdb || currentSourceKey(job));
    var side =
      '<div class="bench-head">' +
      "<div><h1>Structure</h1><p class=\"muted\">" +
      esc(job.protein || job.id) +
      ' · <a href="#/runs/' +
      encodeURIComponent(job.id) +
      '">back to scores</a>. Crystal B-factors are not pLDDT.</p></div>' +
      runFlowHtml(job) +
      "</div>" +
      '<div class="panel seq-box"><h2>Sequence</h2>' +
      renderSequence(seq, positions) +
      "</div>" +
      '<div class="panel"><h2>Selection</h2>' +
      selectedDetailHtml(job, state.selectedRow) +
      '<p class="mol-hint">Use the residue bar above the 3D view, or click the cartoon. Drag the gutter to grow the protein.</p></div>' +
      fetchBarHtml(job);
    var viewer = hasCoord
      ? molPanelHtml("bench")
      : '<div class="viewer-empty"><h2>' +
        (isLive(job.status) ? "Waiting for coordinates" : "No coordinates") +
        "</h2><p>Paste a PDB id or UniProt on the left. This does not re-score.</p></div>";
    host.innerHTML = proteinBenchHtml("structure", side, viewer);
    mountBench("structure");
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
    if (els.btnSubmit) {
      els.btnSubmit.disabled = busy;
      els.btnSubmit.textContent = busy ? "Starting…" : "Start scoring";
    }
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
    if (snap.seq_id) fd.append("seq_id", snap.seq_id);
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
      if (!snap.fasta && !snap.pdb && !snap.pdb_id && !snap.uniprot_id && !snap.seq_id) {
        showFormError("Provide a FASTA, PDB, or an accession.");
        return;
      }
      var spec = modelByName(snap.model);
      var reason = modelLockReason(spec, snap);
      if (reason) {
        showFormError((spec && spec.name ? spec.name + " " : "") + reason + ".");
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
      var dest = "#/runs/" + encodeURIComponent(id);
      go(dest);
      if (location.hash !== dest) location.hash = dest;
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
    if (route.page !== "workspace" && route.page !== "review") setBenchMode(false);
    if (route.page === "review" && !route.id) {
      if (state.activeRunId) {
        go("#/review/" + encodeURIComponent(state.activeRunId));
        return;
      }
      go("#/runs");
      return;
    }
    state.route = route;
    setTabs(route.page);
    if (route.page !== "workspace" && route.page !== "review") teardownViewer();
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
      fillModels();
      updateCli();
    } else if (route.page === "benchmarks") {
      showPage("page-benchmarks");
      renderLeaderboard();
    } else if (route.page === "workspace" || route.page === "review") {
      showPage("page-workspace");
      setActiveRun(route.id);
      await loadWorkspace(route.id);
      if (route.page === "workspace" && state.job && state.job.status === "done") {
        go("#/review/" + encodeURIComponent(state.job.id));
        return;
      }
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
    var rail = document.querySelector("#workspace-body .flow-rail");
    if (rail) {
      var tmp = document.createElement("div");
      tmp.innerHTML = runFlowHtml(job);
      if (tmp.firstChild) rail.replaceWith(tmp.firstChild);
    }
    var host = $("run-progress");
    if (isLive(job.status) || job.status === "failed") {
      var html = progressHtml(job);
      if (host) host.outerHTML = html;
      else {
        var pipe = document.querySelector("#workspace-body .chips");
        if (pipe) pipe.insertAdjacentHTML("afterend", html);
      }
    } else if (host) {
      host.remove();
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
      if ((state.route.page === "workspace" || state.route.page === "review") && state.route.id) {
        var prev = state.job && state.job.status;
        var prevPhase = runPhase(state.job);
        var hadPdb = state.job && state.job.has_pdb;
        state.job = await apiJson("/api/runs/" + encodeURIComponent(state.route.id));
        state.lastPoll = new Date();
        state.connected = true;
        await Promise.all([
          loadScores(),
          loadHistogram(state.route.id),
          loadLog(state.route.id),
          loadFastaIfNeeded(state.job),
          loadTop(state.route.id),
          loadFeatures(state.route.id),
        ]);
        if (prev !== "done" && state.job.status === "done" && state.route.page === "workspace") {
          go("#/review/" + encodeURIComponent(state.job.id));
          return;
        }
        var pdbArrived = state.job.has_pdb && !hadPdb;
        var statusChanged = prev !== state.job.status;
        var phaseChanged = prevPhase !== runPhase(state.job);
        if (statusChanged || pdbArrived || phaseChanged || (!$("run-progress") && !$("run-log") && !$("score-table") && !$("mol-host"))) {
          renderWorkspace();
          if ($("mol-host")) await maybeLoadViewer(state.job, selectionResis());
        } else {
          patchOverview(state.job);
          var logEl = $("run-log");
          if (logEl) logEl.textContent = lastLogLines(state.logText, 30) || "Waiting for rem2…";
          if ($("score-table")) patchScores();
          drawReviewHists();
        }
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
      (state.job &&
        isLive(state.job.status) &&
        (state.route.page === "workspace" || state.route.page === "review"));
    if (need) pollTimer = setTimeout(pollTick, POLL_MS);
  }

  function bindPredict() {
    if (!els.form) return;
    els.form.addEventListener("input", function () {
      updateFileLabels();
      renderIntakeFiles();
      fillModels();
      updateModelMeta();
      updateCli();
    });
    els.form.addEventListener("change", function () {
      updateFileLabels();
      renderIntakeFiles();
      fillModels();
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
    var seriesHost = $("model-series");
    if (seriesHost) bindModelSeriesHost(seriesHost);
    var filters = $("board-filters");
    if (filters) {
      filters.addEventListener("click", function (ev) {
        var btn = ev.target.closest("[data-board]");
        if (!btn) return;
        state.boardId = btn.getAttribute("data-board");
        renderLeaderboard();
      });
    }
    function bindSlotDrop(id, slot) {
      var el = $(id);
      if (!el) return;
      el.addEventListener("dragover", function (ev) {
        ev.preventDefault();
        el.classList.add("is-drag");
      });
      el.addEventListener("dragleave", function () {
        el.classList.remove("is-drag");
      });
      el.addEventListener("drop", function (ev) {
        ev.preventDefault();
        el.classList.remove("is-drag");
        assignDroppedFiles(ev.dataTransfer && ev.dataTransfer.files, slot);
      });
    }
    bindSlotDrop("slot-sequence", "fasta");
    bindSlotDrop("slot-structure", "pdb");
    bindSlotDrop("slot-msa", "msa");
    updateFileLabels();
    renderIntakeFiles();
    updateCli();
    renderLeaderboard();
    updateInputGate();
  }

  function bindSplit() {
    document.addEventListener("pointerdown", function (ev) {
      var gutter = ev.target.closest("[data-gutter]");
      if (!gutter) return;
      var bench = gutter.closest(".protein-bench");
      if (!bench) return;
      ev.preventDefault();
      var kind = bench.getAttribute("data-split") || "review";
      var rect = bench.getBoundingClientRect();
      var viewerAtEnd = bench.getAttribute("data-viewer") === "end";
      document.body.classList.add("is-splitting");
      gutter.classList.add("is-drag");
      function pctFromX(x) {
        var raw = viewerAtEnd ? ((rect.right - x) / rect.width) * 100 : ((x - rect.left) / rect.width) * 100;
        return clampSplit(raw);
      }
      function move(e) {
        var pct = pctFromX(e.clientX);
        applySplit(bench, pct);
        writeSplit(kind, pct);
        resizeViewerSoon();
      }
      function end(e) {
        gutter.classList.remove("is-drag");
        document.body.classList.remove("is-splitting");
        try {
          gutter.releasePointerCapture(e.pointerId);
        } catch (err) {
          /* ignore */
        }
        gutter.removeEventListener("pointermove", move);
        gutter.removeEventListener("pointerup", end);
        gutter.removeEventListener("pointercancel", end);
        resizeViewerSoon();
      }
      try {
        gutter.setPointerCapture(ev.pointerId);
      } catch (err) {
        /* ignore */
      }
      gutter.addEventListener("pointermove", move);
      gutter.addEventListener("pointerup", end);
      gutter.addEventListener("pointercancel", end);
    });
    document.addEventListener("dblclick", function (ev) {
      var gutter = ev.target.closest("[data-gutter]");
      if (!gutter) return;
      var bench = gutter.closest(".protein-bench");
      if (!bench) return;
      var kind = bench.getAttribute("data-split") || "review";
      var fallback = kind === "structure" ? 64 : 56;
      writeSplit(kind, fallback);
      applySplit(bench, fallback);
      resizeViewerSoon();
    });
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
      drawReviewHists();
      resizeViewerSoon();
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
    els.tabReview = $("tab-review");
    els.tabBenchmarks = $("tab-benchmarks");
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
    bindSplit();
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
        var aaNav = ev.target.closest("[data-aa]");
        if (aaNav) {
          if (aaNav.getAttribute("data-aa") === "go") goResidueInput();
          else if (aaNav.getAttribute("data-aa") === "prev") stepResidue(-1);
          else if (aaNav.getAttribute("data-aa") === "next") stepResidue(1);
          return;
        }
        var aa = ev.target.closest(".seq-aa[data-pos]");
        if (aa) pickSequencePos(parseInt(aa.getAttribute("data-pos"), 10));
      });
      structHost.addEventListener("change", function (ev) {
        var molEl = ev.target.closest("[data-mol]");
        if (molEl) handleMolControl(molEl);
      });
      structHost.addEventListener("keydown", function (ev) {
        if (ev.target.id === "aa-jump" && ev.key === "Enter") {
          ev.preventDefault();
          goResidueInput();
        } else if (ev.target.closest && ev.target.closest("#aa-strip")) {
          if (ev.key === "ArrowLeft") {
            ev.preventDefault();
            stepResidue(-1);
          } else if (ev.key === "ArrowRight") {
            ev.preventDefault();
            stepResidue(1);
          }
        }
      });
    }
    var runsHost = $("runs-body");
    if (runsHost) {
      runsHost.addEventListener("click", function (ev) {
        if (ev.target.closest("a")) return;
        var tr = ev.target.closest("tr[data-id]");
        if (tr) {
          var rid = tr.getAttribute("data-id");
          var run = (state.runs || []).filter(function (item) {
            return item.id === rid;
          })[0];
          go((run && run.status === "done" ? "#/review/" : "#/runs/") + encodeURIComponent(rid));
        }
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
