"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const appPath = path.join(__dirname, "..", "rem2", "dashboard", "static", "app.js");
const htmlPath = path.join(__dirname, "..", "rem2", "dashboard", "static", "index.html");
const cssPath = path.join(__dirname, "..", "rem2", "dashboard", "static", "app.css");

const storage = Object.create(null);
const localStorage = {
  getItem(key) {
    return Object.prototype.hasOwnProperty.call(storage, key) ? storage[key] : null;
  },
  setItem(key, value) {
    storage[key] = String(value);
  },
  removeItem(key) {
    delete storage[key];
  },
};

let src = fs.readFileSync(appPath, "utf8");
src = src.replace(/^\(function \(\) \{/, "");
src = src.replace(/\n  if \(document\.readyState === "loading"\) \{[\s\S]*$/, "");
src += `
this.runPhase = runPhase;
this.clampSplit = clampSplit;
this.clampK = clampK;
this.isLive = isLive;
this.parseMutants = parseMutants;
this.mutantField = mutantField;
this.csvEscape = csvEscape;
this.fmtScore = fmtScore;
this.parseHash = parseHash;
this.proteinBenchHtml = proteinBenchHtml;
this.pipelineHtml = pipelineHtml;
this.aaStripHtml = aaStripHtml;
this.aaNavHtml = aaNavHtml;
this.molPanelHtml = molPanelHtml;
this.nextActionsHtml = nextActionsHtml;
this.jobSequence = jobSequence;
this.readSplit = readSplit;
this.writeSplit = writeSplit;
this.activeJob = activeJob;
this.modelUnlocked = modelUnlocked;
this.modelLockReason = modelLockReason;
this.preferredModel = preferredModel;
this.inferModelSeries = inferModelSeries;
this.groupModelSeries = groupModelSeries;
this.inputGateMessage = inputGateMessage;
this.highlightPositions = highlightPositions;
this.experimentMutationPositions = experimentMutationPositions;
this.selectionResis = selectionResis;
this.comparisonMetricBand = comparisonMetricBand;
this.classifyIntakeFile = classifyIntakeFile;
this.currentBoard = currentBoard;
`;

const context = vm.createContext({
  console,
  sessionStorage: { getItem: () => null, setItem() {}, removeItem() {} },
  localStorage,
  location: { hash: "#/runs" },
  document: {
    readyState: "loading",
    addEventListener() {},
    getElementById() {
      return null;
    },
    querySelector() {
      return null;
    },
    querySelectorAll() {
      return [];
    },
    body: {
      classList: {
        contains() {
          return false;
        },
        toggle() {},
        add() {},
        remove() {},
      },
    },
  },
  window: { addEventListener() {}, devicePixelRatio: 1 },
  navigator: {},
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
});

vm.runInContext(src, context);

function check(name, fn) {
  try {
    fn();
  } catch (err) {
    err.message = name + ": " + err.message;
    throw err;
  }
}

check("runPhase stages", () => {
  assert.strictEqual(context.runPhase(null), "queue");
  assert.strictEqual(context.runPhase({ status: "queued" }), "queue");
  assert.strictEqual(context.runPhase({ status: "running", progress: { stage: "fetch" } }), "fetch");
  assert.strictEqual(context.runPhase({ status: "running", progress: { stage: "score" } }), "score");
  assert.strictEqual(context.runPhase({ status: "running" }), "score");
  assert.strictEqual(context.runPhase({ status: "done" }), "review");
  assert.strictEqual(context.runPhase({ status: "failed", progress: { stage: "fetch" } }), "fetch");
  assert.strictEqual(context.runPhase({ status: "failed", progress: { stage: "score" } }), "score");
  assert.strictEqual(context.runPhase({ status: "cancelled" }), "score");
});

check("clampSplit keeps protein >= 50%", () => {
  assert.strictEqual(context.clampSplit(10), 50);
  assert.strictEqual(context.clampSplit(50), 50);
  assert.strictEqual(context.clampSplit(64), 64);
  assert.strictEqual(context.clampSplit(99), 82);
});

check("clampK", () => {
  assert.strictEqual(context.clampK("2"), 5);
  assert.strictEqual(context.clampK(20), 20);
  assert.strictEqual(context.clampK(999), 100);
  assert.strictEqual(context.clampK("nope"), 30);
});

check("isLive", () => {
  assert.strictEqual(context.isLive("queued"), true);
  assert.strictEqual(context.isLive("running"), true);
  assert.strictEqual(context.isLive("done"), false);
  assert.strictEqual(context.isLive("failed"), false);
});

check("parseMutants", () => {
  const one = context.parseMutants("A1C");
  assert.strictEqual(one.length, 1);
  assert.strictEqual(one[0].wt, "A");
  assert.strictEqual(one[0].pos, 1);
  assert.strictEqual(one[0].mut, "C");
  const multi = context.parseMutants("A10Y:B12D");
  assert.strictEqual(multi.length, 2);
  assert.strictEqual(multi[1].pos, 12);
  assert.strictEqual(context.parseMutants("").length, 0);
  assert.strictEqual(context.parseMutants("not-a-mut").length, 0);
});

check("mutantField / jobSequence / fmtScore / csvEscape", () => {
  assert.strictEqual(context.mutantField({ mutant: "A1C" }), "A1C");
  assert.strictEqual(context.mutantField({ aa_mut: "V20A" }), "V20A");
  assert.strictEqual(context.jobSequence({ sequence: "ACDE" }), "ACDE");
  assert.strictEqual(context.jobSequence({ proteins: [{ sequence: "WW" }] }), "WW");
  assert.strictEqual(context.fmtScore(1.23456), "1.2346");
  assert.strictEqual(context.fmtScore(null), "—");
  assert.strictEqual(context.csvEscape("a,b"), '"a,b"');
});

check("parseHash routes", () => {
  context.location.hash = "#/";
  assert.strictEqual(context.parseHash().page, "runs");
  context.location.hash = "#/predict";
  assert.strictEqual(context.parseHash().page, "predict");
  context.location.hash = "#/benchmarks";
  assert.strictEqual(context.parseHash().page, "benchmarks");
  context.location.hash = "#/runs/abc12";
  const run = context.parseHash();
  assert.strictEqual(run.page, "workspace");
  assert.strictEqual(run.id, "abc12");
  context.location.hash = "#/structure/abc12";
  assert.strictEqual(context.parseHash().page, "review");
  assert.strictEqual(context.parseHash().id, "abc12");
  context.location.hash = "#/select/abc12";
  assert.strictEqual(context.parseHash().page, "review");
  context.location.hash = "#/review/abc12";
  assert.strictEqual(context.parseHash().page, "review");
  context.location.hash = "#/select";
  assert.strictEqual(context.parseHash().page, "review");
  context.location.hash = "#/nope";
  assert.strictEqual(context.parseHash().page, "runs");
});

check("residue bar sits on the review page", () => {
  context.state.job = { sequence: "ACDEY" };
  context.state.picked = { resi: 3, resn: "D", chain: "A" };
  const strip = context.aaStripHtml("ACDEY", [2]);
  assert.ok(strip.includes('id="aa-strip"'));
  assert.ok(strip.includes('data-pos="3"'));
  assert.ok(strip.includes("is-pick"));
  assert.ok(strip.includes("is-mut"));
  context.state.picked = { resi: 239, resn: "S", chain: "A" };
  const longStrip = context.aaStripHtml("A".repeat(250), [239]);
  assert.ok(longStrip.includes('<span class="aa-num is-on">239</span>'));
  assert.ok(longStrip.includes('<span class="aa-num">240</span>'));
  const nav = context.aaNavHtml("ACDEY");
  assert.ok(nav.includes('id="aa-jump"'));
  assert.ok(nav.includes("data-aa=\"prev\""));
  const bench = context.proteinBenchHtml("review", "<p>side</p>", "<div id='mol-host'></div>");
  assert.ok(bench.includes('id="aa-strip"'));
  assert.ok(bench.includes('id="aa-nav"'));
  assert.ok(bench.includes("review-seq"));
  assert.ok(bench.indexOf("aa-strip") < bench.indexOf("mol-host"));
});

check("protein bench markup", () => {
  const html = context.proteinBenchHtml("structure", "<p>side</p>", "<div id='mol-host'></div>");
  assert.ok(html.includes('data-split="structure"'));
  assert.ok(html.includes("data-gutter"));
  assert.ok(html.includes("protein-viewer-pane"));
  assert.ok(html.includes("protein-side-pane"));
});

check("nextActions point at the merged review page", () => {
  const job = { id: "r1", status: "done", has_pdb: true };
  const full = context.nextActionsHtml(job);
  assert.ok(full.includes("#/review/r1"));
  const slim = context.nextActionsHtml(job, { hideReview: true });
  assert.ok(!slim.includes("#/review/r1"));
  assert.ok(slim.includes("#/predict"));
  assert.strictEqual(context.nextActionsHtml({ id: "r1", status: "running" }), "");
});

check("split persistence clamps on read path via write/read", () => {
  context.writeSplit("review", 70);
  assert.strictEqual(context.readSplit("review", 56), 70);
  assert.strictEqual(context.readSplit("missing", 64), 64);
});

check("activeJob prefers state.job then runs list", () => {
  context.state.activeRunId = "a";
  context.state.job = { id: "a", status: "running" };
  assert.strictEqual(context.activeJob().status, "running");
  context.state.job = { id: "other", status: "done" };
  context.state.runs = [{ id: "a", status: "queued" }];
  assert.strictEqual(context.activeJob().status, "queued");
  context.state.activeRunId = "";
  context.state.job = null;
  assert.strictEqual(context.activeJob(), null);
});

check("experiment selections drive molecular positions", () => {
  context.state.experimentRows = {
    V1M: { mutant: "V1M" },
    K20A: { mutant: "K20A" },
    K20C: { mutant: "K20C" },
  };
  context.state.selectedRow = { mutant: "S40A" };
  context.state.picked = null;
  assert.deepStrictEqual(Array.from(context.experimentMutationPositions()), [1, 20]);
  assert.deepStrictEqual(Array.from(context.selectionResis()), [1, 20, 40]);
});

const html = fs.readFileSync(htmlPath, "utf8");
const css = fs.readFileSync(cssPath, "utf8");

check("full rem2 treats MSA as optional", () => {
  const full = context.pipelineHtml("full");
  assert.ok(full.includes("is-opt"));
  assert.ok(full.includes("optional; skip → α=0"));
  const raw = context.pipelineHtml("raw");
  assert.ok(raw.includes("is-dim"));
  assert.ok(!raw.includes("is-opt"));
});

check("html flow chrome", () => {
  assert.ok(html.includes('id="tab-predict"'));
  assert.ok(html.includes('id="tab-benchmarks"'));
  assert.ok(html.includes('id="tab-review"') && html.includes("is-disabled"));
  assert.ok(!html.includes("1. Inputs"));
  assert.ok(html.includes('id="btn-submit"') && html.includes("Start scoring"));
  assert.ok(html.includes('id="intake-box"'));
  assert.ok(html.includes('id="slot-sequence"'));
  assert.ok(html.includes('id="slot-structure"'));
  assert.ok(html.includes('id="slot-msa"'));
  assert.ok(html.includes('name="seq_id"'));
  assert.ok(!html.includes('id="f-struct-id"'));
  assert.ok(html.includes("default is all positions × 19 substitutions"));
  assert.ok(html.includes("They are not physical ΔΔG measurements"));
  assert.ok(html.includes("<b>5</b> Export"));
  assert.ok(src.includes('data-delete-run="'));
  assert.ok(src.includes("refreshReviewSelection();"));
  assert.ok(!html.includes('id="f-struct-id"'));
  assert.ok(html.includes("default is all positions × 19 substitutions"));
  assert.ok(html.includes("They are not physical ΔΔG measurements"));
  assert.ok(!html.includes('id="file-bundle"'));
  assert.ok(html.includes('id="model-series"'));
  assert.ok(html.includes('id="model-select"') && html.includes('name="model"'));
  assert.ok(html.includes('id="board-filters"'));
  assert.ok(html.includes('value="auto" selected'));
  assert.ok(html.includes('id="btn-demo"'));
  assert.ok(!html.includes("Skip the form"));
  assert.ok(!html.includes("Full ProteinGym-level scoring needs at least a PDB and an MSA"));
  assert.ok(html.includes('id="leaderboard-body"'));
  assert.ok(html.includes("skip → α=0"));
  assert.ok(src.includes('data-delete-run="'));
  assert.ok(src.includes("refreshReviewSelection"));
});

check("fasta-only unlocks sequence models only", () => {
  const fasta = { fasta: "a.fa", pdb: "", pdb_id: "", uniprot_id: "", fetch_structure: "auto", msa: "" };
  assert.strictEqual(context.modelUnlocked({ name: "esm2", needs_pdb: false }, fasta), true);
  assert.strictEqual(context.modelUnlocked({ name: "venusrem2", needs_pdb: true }, fasta), false);
  assert.ok(context.modelLockReason({ name: "venusrem2", needs_pdb: true }, fasta).includes("PDB"));
  assert.strictEqual(context.preferredModel(fasta), "esm2");
  const msg = context.inputGateMessage(fasta);
  assert.ok(msg.includes("sequence models"));
  assert.ok(msg.includes("α=0"));
});

check("intake classifies dropped files", () => {
  assert.strictEqual(context.classifyIntakeFile({ name: "query.fasta" }), "fasta");
  assert.strictEqual(context.classifyIntakeFile({ name: "2l6q.pdb" }), "pdb");
  assert.strictEqual(context.classifyIntakeFile({ name: "aln.a2m" }), "msa");
  assert.strictEqual(context.classifyIntakeFile({ name: "mut.csv" }), "mutants");
});

check("default highlight uses top mutants", () => {
  context.state.selectedRow = null;
  context.state.seqRange = null;
  context.state.top = { rows: [{ mutant: "A1C" }, { mutant: "V10A" }], k: 30 };
  const pos = context.highlightPositions().map(Number).sort((a, b) => a - b);
  assert.strictEqual(pos.join(","), "1,10");
});

check("comparison metric thresholds preserve scientific meaning", () => {
  assert.strictEqual(context.comparisonMetricBand("rank", 90), "rank-high");
  assert.strictEqual(context.comparisonMetricBand("rank", 89.9), "rank-mid");
  assert.strictEqual(context.comparisonMetricBand("rank", 69.9), "rank-low");
  assert.strictEqual(context.comparisonMetricBand("plddt", 90), "plddt-very-high");
  assert.strictEqual(context.comparisonMetricBand("plddt", 70), "plddt-confident");
  assert.strictEqual(context.comparisonMetricBand("plddt", 50), "plddt-low");
  assert.strictEqual(context.comparisonMetricBand("plddt", 49.9), "plddt-very-low");
  assert.strictEqual(context.comparisonMetricBand("rsa", 0.2), "rsa-buried");
  assert.strictEqual(context.comparisonMetricBand("rsa", 0.5), "rsa-partial");
  assert.strictEqual(context.comparisonMetricBand("rsa", 0.51), "rsa-exposed");
  assert.strictEqual(context.comparisonMetricBand("rsa", null), "na");
  assert.strictEqual(context.comparisonMetricBand("rsa", undefined), "na");
});

check("leaderboard catalog has filter boards", () => {
  context.state.boardId = "stability";
  assert.strictEqual(context.currentBoard().id, "stability");
  assert.strictEqual(context.currentBoard().rows[0].name, "VenusREM2");
  assert.strictEqual(context.currentBoard().rows[0].score, 0.691);
  context.state.boardId = "ablations";
  assert.strictEqual(context.currentBoard().id, "ablations");
  assert.strictEqual(context.currentBoard().rows[0].score, 0.556);
});

check("model picker groups by series", () => {
  assert.strictEqual(context.inferModelSeries("esm2-8m"), "esm");
  assert.strictEqual(context.inferModelSeries("prosst-2048"), "prosst");
  assert.strictEqual(context.inferModelSeries("progen3-1b") || context.inferModelSeries("progen3"), "progen");
  assert.strictEqual(context.inferModelSeries("protein_mpnn-soluble-v_48_020"), "proteinmpnn");
  assert.strictEqual(context.inferModelSeries("rita-s"), "rita");
  assert.strictEqual(context.inferModelSeries("venusrem2"), "venusrem2");
  const groups = context.groupModelSeries();
  const ids = groups.map((g) => g.id);
  assert.ok(ids.includes("venusrem2"));
  assert.ok(ids.includes("esm"));
  assert.ok(ids.includes("prosst"));
  const esm = groups.find((g) => g.id === "esm");
  assert.ok(esm.variants.some((v) => v.name === "esm2-8m"));
  assert.ok(esm.variants.some((v) => v.name === "esm_if"));
});

check("pdb or accession unlocks VenusREM2", () => {
  const pdb = { fasta: "a.fa", pdb: "x.pdb", pdb_id: "", uniprot_id: "", fetch_structure: "none", msa: "q.a2m" };
  assert.strictEqual(context.modelUnlocked({ name: "venusrem2", needs_pdb: true }, pdb), true);
  assert.strictEqual(context.preferredModel(pdb), "venusrem2");
  const acc = { fasta: "a.fa", pdb: "", pdb_id: "2L6Q", uniprot_id: "", fetch_structure: "auto", msa: "" };
  assert.strictEqual(context.modelUnlocked({ name: "venusrem2", needs_pdb: true }, acc), true);
});

check("css protein pane is at least half", () => {
  assert.ok(css.includes("min-width: 50%"));
  assert.ok(css.includes(".protein-gutter"));
  assert.ok(css.includes("cursor: col-resize"));
  assert.ok(css.includes("flex: 0 0 56%"));
  assert.ok(css.includes(".aa-strip"));
  assert.ok(css.includes(".aa-cell"));
  assert.ok(css.includes(".board-table"));
  assert.ok(css.includes(".board-filter"));
  assert.ok(css.includes(".model-series"));
  assert.ok(css.includes(".model-variant"));
  assert.ok(css.includes(".intake"));
  assert.ok(css.includes(".input-slot"));
  assert.ok(css.includes(".review-seq"));
  assert.ok(css.includes(".board-score-bar"));
});

check("experiment tray exposes only the real CSV action", () => {
  assert.ok(src.includes(">Add to CSV</button>"));
  assert.ok(!src.includes("Add to experiment"));
  assert.ok(!src.includes("btn-add-experiment"));
});

console.log("dashboard ui logic: ok");
