"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const appPath = path.join(__dirname, "..", "vrh", "dashboard", "static", "app.js");
const htmlPath = path.join(__dirname, "..", "vrh", "dashboard", "static", "index.html");
const cssPath = path.join(__dirname, "..", "vrh", "dashboard", "static", "app.css");

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
this.currentBenchmark = currentBenchmark;
this.benchmarkRows = benchmarkRows;
this.benchmarkPairRow = benchmarkPairRow;
this.benchmarkVisibleInputs = benchmarkVisibleInputs;
this.EXAMPLE_PRESETS = EXAMPLE_PRESETS;
this.catalogPairs = catalogPairs;
this.featuredPair = featuredPair;
this.benchmarkRows = benchmarkRows;
this.benchmarkPairRow = benchmarkPairRow;
this.benchmarkFamily = benchmarkFamily;
this.familyColor = familyColor;
this.renderProductBenchmark = renderProductBenchmark;
this.renderCatalogAwaiting = renderCatalogAwaiting;
this.benchmarkTabMeta = benchmarkTabMeta;
this.featuredRank = featuredRank;
this.benchmarkFeaturedHtml = benchmarkFeaturedHtml;
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
  assert.strictEqual(context.parseHash().benchmark, "proteingym");
  context.location.hash = "#/benchmarks/venusvirohub";
  assert.strictEqual(context.parseHash().page, "benchmarks");
  assert.strictEqual(context.parseHash().benchmark, "venusvirohub");
  context.location.hash = "#/benchmarks/venusmuthub";
  assert.strictEqual(context.parseHash().benchmark, "venusmuthub");
  context.location.hash = "#/benchmarks/nope";
  assert.strictEqual(context.parseHash().benchmark, "proteingym");
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

check("review geometry is locked before route loading", () => {
  const routeStart = src.indexOf("async function onRoute()");
  const routeEnd = src.indexOf("function patchOverview", routeStart);
  const routeSource = src.slice(routeStart, routeEnd);
  assert.ok(routeSource.includes("setBenchMode(isWorkbenchRoute)"));
  assert.ok(routeSource.indexOf("setBenchMode(isWorkbenchRoute)") < routeSource.indexOf("await loadWorkspace(route.id)"));
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

check("full vrh treats MSA as optional", () => {
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
  assert.ok(html.includes("all sites × 19"));
  assert.ok(html.includes("Not ΔΔG"));
  assert.ok(html.includes("<b>5</b> Export"));
  assert.ok(src.includes('data-delete-run="'));
  assert.ok(src.includes("refreshReviewSelection();"));
  assert.ok(!html.includes('id="f-struct-id"'));
  assert.ok(!html.includes('id="file-bundle"'));
  assert.ok(html.includes('id="model-series"'));
  assert.ok(html.includes('id="model-select"') && html.includes('name="model"'));
  assert.ok(html.includes('id="pg-board"'));
  assert.ok(html.includes('value="auto" selected'));
  assert.ok(html.includes('id="example-row"'));
  assert.ok(html.includes('id="example-status"'));
  assert.ok(html.includes('id="btn-demo"'));
  assert.ok(html.includes('data-example="full"'));
  assert.ok(!html.includes('data-example="sequence"'));
  assert.ok(!html.includes('data-example="structure"'));
  assert.ok(src.includes("toggleExamplePreset"));
  assert.ok(src.includes("clearExamplePreset"));
  assert.ok(!src.includes("maybePrefillDemo"));
  assert.strictEqual(context.EXAMPLE_PRESETS.full.join(","), "fasta,pdb,msa");
  assert.strictEqual(context.EXAMPLE_PRESETS.sequence.join(","), "fasta,pdb,msa");
  assert.strictEqual(context.EXAMPLE_PRESETS.structure.join(","), "fasta,pdb,msa");
  assert.ok(!html.includes("Skip the form"));
  assert.ok(!html.includes("Full ProteinGym-level scoring needs at least a PDB and an MSA"));
  assert.ok(html.includes("Benchmarks"));
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
  assert.ok(msg.includes("Sequence only"));
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

check("product benchmark compares the same base with VRH", () => {
  const benchmark = context.currentBenchmark();
  assert.strictEqual(benchmark.id, "proteingym");
  assert.strictEqual(benchmark.pairs.length, 4);
  const prosst = benchmark.pairs.find((row) => row.key === "prosst_ensemble");
  assert.strictEqual(prosst.base, 0.529);
  assert.strictEqual(prosst.enhanced, 0.556);
  assert.strictEqual(prosst.delta, 0.027);
  assert.strictEqual(context.catalogPairs(benchmark).length, 3);
  assert.ok(context.catalogPairs(benchmark).every((row) => row.key !== "prosst_ensemble"));
  assert.strictEqual(context.featuredPair(benchmark).key, "prosst_ensemble");
  assert.ok(context.benchmarkRows(benchmark).every((row) => row.key !== "prosst_ensemble"));
  const saprot = benchmark.pairs.find((row) => row.key === "saprot_wt");
  assert.strictEqual(Object.prototype.hasOwnProperty.call(saprot, "official_reference"), false);
  context.state.benchmarkVariant = "vrh";
  const featured = context.benchmarkPairRow(prosst, 1, "overall", 1, { featured: true });
  assert.ok(featured.includes("VenusREM2"));
  assert.ok(featured.includes("ours"));
  assert.ok(featured.includes("bbio-row"));
  assert.ok(featured.includes("bbio-tip"));
  assert.ok(featured.includes("--bbio:#145246"));
  assert.ok(!featured.includes("ProSST-Ensemble"));
  assert.strictEqual(context.benchmarkFamily(benchmark.pairs[1]), "esm");
  assert.strictEqual(context.benchmarkFamily({ key: "pmpnn_v_48_030" }), "proteinmpnn");
  const enhanced = context.benchmarkPairRow(benchmark.pairs[1], 2, "overall");
  assert.ok(enhanced.includes("esm2_650m_wt"));
  assert.ok(enhanced.includes("+0.054"));
  assert.ok(enhanced.includes("Evo"));
  assert.ok(enhanced.includes("bbio-bar"));
  assert.ok(enhanced.includes("bbio-tip"));
  assert.ok(enhanced.includes(context.familyColor("esm")));
  context.state.benchmarkVariant = "raw";
  const raw = context.benchmarkPairRow(benchmark.pairs[1], 2, "overall");
  assert.ok(!raw.includes("Foundation model"));
  assert.ok(!raw.includes("ProSST-Ensemble"));
  context.state.benchmarkVariant = "vrh";
});

check("benchmark property, input, and score mode keep one row per model", () => {
  const original = context.state.catalog;
  context.state.catalog = { default_benchmark: "proteingym", benchmarks: [{
    id: "proteingym",
    pairs: [
      { key: "alpha", base_name: "Alpha", enhanced_name: "Alpha vrh", inputs: ["seq"], protocol: "paired", properties: { binding: { base: 0.7, vrh: 0.8, delta: 0.1 } }, properties_by_metric: { spearman: { binding: { base: 0.7, vrh: 0.8, delta: 0.1 } }, ndcg: { binding: { base: 0.6, vrh: 0.7, delta: 0.1 } } } },
      { key: "beta", base_name: "Beta", enhanced_name: "Beta vrh", inputs: ["str"], protocol: "paired", properties: { binding: { base: 0.75, vrh: 0.76, delta: 0.01 } }, properties_by_metric: { spearman: { binding: { base: 0.75, vrh: 0.76, delta: 0.01 } }, ndcg: { binding: { base: 0.8, vrh: 0.75, delta: -0.05 } } } },
    ],
  }] };
  context.state.benchmarkProperty = "binding";
  context.state.benchmarkInput = "all";
  context.state.benchmarkVariant = "raw";
  context.state.benchmarkQuery = "";
  context.state.benchmarkMetric = "spearman";
  context.state.benchmarkId = "proteingym";
  assert.deepStrictEqual(context.benchmarkRows(context.currentBenchmark()).map((row) => row.key), ["beta", "alpha"]);
  context.state.benchmarkVariant = "vrh";
  assert.deepStrictEqual(context.benchmarkRows(context.currentBenchmark()).map((row) => row.key), ["alpha", "beta"]);
  context.state.benchmarkMetric = "ndcg";
  assert.deepStrictEqual(context.benchmarkRows(context.currentBenchmark()).map((row) => row.key), ["beta", "alpha"]);
  context.state.benchmarkMetric = "spearman";
  context.state.benchmarkInput = "str";
  const rows = context.benchmarkRows(context.currentBenchmark());
  assert.strictEqual(rows.length, 1);
  assert.strictEqual(rows[0].key, "beta");
  assert.deepStrictEqual(context.benchmarkVisibleInputs(rows[0]), ["str", "evo"]);
  context.state.catalog = original;
  context.state.benchmarkProperty = "overall";
  context.state.benchmarkInput = "all";
  context.state.benchmarkVariant = "vrh";
  context.state.benchmarkQuery = "";
});

check("model picker groups by series", () => {
  assert.strictEqual(context.inferModelSeries("esm2-8m"), "esm");
  assert.strictEqual(context.inferModelSeries("prosst-2048"), "prosst");
  assert.strictEqual(context.inferModelSeries("progen3-1b") || context.inferModelSeries("progen3"), "progen");
  assert.strictEqual(context.inferModelSeries("protein_mpnn-soluble-v_48_020"), "proteinmpnn");
  assert.strictEqual(context.inferModelSeries("rita-s"), "rita");
  assert.strictEqual(context.inferModelSeries("venusrem2"), "venusrem2");
  assert.strictEqual(context.inferModelSeries("venusrem"), "venusrem2");
  assert.strictEqual(context.inferModelSeries("protssn-k20-h512"), "protssn");
  assert.strictEqual(context.inferModelSeries("carp-38m"), "carp");
  assert.strictEqual(context.inferModelSeries("mifst"), "carp");
  assert.strictEqual(context.inferModelSeries("s2f"), "s3f");
  assert.strictEqual(context.inferModelSeries("s3f"), "s3f");
  const groups = context.groupModelSeries();
  const ids = groups.map((g) => g.id);
  assert.ok(ids.includes("venusrem2"));
  assert.ok(ids.includes("esm"));
  assert.ok(ids.includes("prosst"));
  const rem = groups.find((g) => g.id === "venusrem2");
  assert.ok(rem.variants.some((v) => v.name === "venusrem2"));
  assert.ok(rem.variants.some((v) => v.name === "venusrem"));
  const carp = groups.find((g) => g.id === "carp");
  assert.ok(carp && carp.variants.some((v) => v.name === "carp"));
  assert.ok(carp.variants.some((v) => v.name === "mifst"));
  const sf = groups.find((g) => g.id === "s3f");
  assert.ok(sf && sf.variants.some((v) => v.name === "s2f"));
  assert.ok(sf.variants.some((v) => v.name === "s3f"));
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

check("catalog hubs keep filter chrome and honest empty scores", () => {
  const host = { innerHTML: "" };
  const catalog = context.state.catalog;
  const muthub = catalog.benchmarks.find((item) => item.id === "venusmuthub");
  const virohub = catalog.benchmarks.find((item) => item.id === "venusvirohub");
  assert.ok(muthub && muthub.status === "catalog" && muthub.n === 905);
  assert.ok(Array.isArray(muthub.pairs) && muthub.pairs.length === 0);
  assert.ok(virohub && virohub.status === "offline" && virohub.n === 89);
  assert.ok(Array.isArray(virohub.pairs) && virohub.pairs.length === 0);
  assert.ok(virohub.properties.some((item) => item.id === "immune_escape"));
  assert.strictEqual(context.benchmarkTabMeta(muthub), "905 assays");
  const viroMeta = context.benchmarkTabMeta(virohub);
  assert.ok(viroMeta === "89 assays" || viroMeta === "Offline");
  context.renderProductBenchmark(host, muthub, catalog);
  assert.ok(host.innerHTML.includes("is-catalog"));
  assert.ok(host.innerHTML.includes("data/VenusMutHub/assay_manifest.csv"));
  assert.ok(host.innerHTML.includes("Disabled until pairs exist"));
  assert.ok(host.innerHTML.includes("PPI binding"));
  assert.ok(host.innerHTML.includes("905"));
  assert.ok(host.innerHTML.includes("27846"));
  assert.ok(host.innerHTML.includes("Spearman"));
  assert.ok(!host.innerHTML.includes("Scores not packaged."));
  assert.ok(!host.innerHTML.includes("Schema ready"));
  assert.ok(!host.innerHTML.includes("Data pending"));
  assert.ok(!host.innerHTML.includes("unavailable offline"));
  assert.ok(host.innerHTML.includes("data-benchmark-metric"));
  context.renderProductBenchmark(host, virohub, catalog);
  assert.ok(host.innerHTML.includes("Offline"));
  assert.ok(host.innerHTML.includes("Immune escape"));
  assert.ok(host.innerHTML.includes("is-offline"));
  assert.ok(!host.innerHTML.includes("not packaged"));
  assert.ok(!host.innerHTML.includes("catalog ready"));
  assert.ok(!host.innerHTML.includes("Schema ready"));
  assert.ok(!host.innerHTML.includes("Data pending"));
  assert.ok(!host.innerHTML.includes("scores pending to be packaged"));
  assert.ok(css.includes(".benchmark-input-tag.is-seq"));
  assert.ok(css.includes(".bbio-callout"));
  assert.ok(css.includes("z-index: 6"));
  assert.ok(css.includes(".benchmark-awaiting.is-offline"));
  assert.ok(css.includes(".benchmark-awaiting.is-catalog"));
  assert.ok(css.includes(".benchmark-catalog-properties"));
});

check("benchmark filter chrome is two compact rows", () => {
  const host = { innerHTML: "" };
  context.renderProductBenchmark(host, context.currentBenchmark(), context.state.catalog);
  assert.ok(host.innerHTML.includes("benchmark-filter-primary"));
  assert.ok(host.innerHTML.includes("benchmark-filter-secondary"));
  assert.ok(host.innerHTML.includes("benchmark-list-tools-cluster"));
  assert.ok(host.innerHTML.includes("data-benchmark-metric"));
  assert.ok(host.innerHTML.includes("data-benchmark-property"));
  assert.ok(host.innerHTML.includes("data-benchmark-variant"));
  assert.ok(host.innerHTML.includes("data-benchmark-input"));
  const toolsIdx = host.innerHTML.indexOf("benchmark-list-tools-cluster");
  const inputIdx = host.innerHTML.indexOf("data-benchmark-input");
  assert.ok(toolsIdx >= 0 && inputIdx > toolsIdx);
  assert.ok(!host.innerHTML.includes("<span>Model input</span>"));
  assert.ok(!host.innerHTML.includes("<span>Benchmark</span>"));
  assert.ok(css.includes(".benchmark-filter-primary"));
  assert.ok(css.includes("min-height: 38px"));
  assert.ok(!/benchmark-variant-pill\.is-on\s*\{\s*background:\s*var\(--accent\)/.test(css));
  const muthub = context.state.catalog.benchmarks.find((item) => item.id === "venusmuthub");
  context.renderProductBenchmark(host, muthub, context.state.catalog);
  assert.ok(/data-benchmark-metric="[^"]+" disabled/.test(host.innerHTML));
  assert.ok(/data-benchmark-variant="[^"]+" disabled/.test(host.innerHTML));
});

check("VenusREM2 is a separate callout with its true rank", () => {
  const original = context.state.catalog;
  context.state.catalog = { default_benchmark: "venusvirohub", benchmarks: [{
    id: "venusvirohub",
    n: 89,
    status: "ready",
    properties: [{ id: "overall", label: "Overall" }],
    metrics: [{ id: "spearman", label: "Spearman" }],
    pairs: [
      { key: "prosst_ensemble", family: "ProSST-Ensemble (K=all)", base: 0.236, enhanced: 0.297, delta: 0.061, inputs: ["seq", "str"], properties: { overall: { base: 0.236, vrh: 0.297, delta: 0.061 } }, properties_by_metric: { spearman: { overall: { base: 0.236, vrh: 0.297, delta: 0.061 } } } },
      { key: "saprot650m_pdb_mask", family: "SaProt (650M_PDB, mask)", base: 0.306, enhanced: 0.320, delta: 0.014, inputs: ["seq", "str"], properties: { overall: { base: 0.306, vrh: 0.320, delta: 0.014 } }, properties_by_metric: { spearman: { overall: { base: 0.306, vrh: 0.320, delta: 0.014 } } } },
    ],
  }] };
  context.state.benchmarkId = "venusvirohub";
  context.state.benchmarkProperty = "overall";
  context.state.benchmarkVariant = "vrh";
  const bench = context.currentBenchmark();
  assert.strictEqual(context.featuredRank(bench, context.featuredPair(bench)), 2);
  assert.deepStrictEqual(context.benchmarkRows(bench).map((row) => row.key), ["saprot650m_pdb_mask"]);
  const host = { innerHTML: "" };
  context.renderProductBenchmark(host, bench, context.state.catalog);
  assert.ok(host.innerHTML.includes("bbio-callout"));
  assert.ok(host.innerHTML.includes("#2 / 2"));
  assert.ok(host.innerHTML.includes("VenusREM2 #2"));
  const rowMatch = host.innerHTML.match(/<article class="bbio-row[\s\S]*?<\/article>/);
  assert.ok(rowMatch);
  assert.ok(rowMatch[0].includes("SaProt") || rowMatch[0].includes("0.320"));
  assert.ok(!rowMatch[0].includes("VenusREM2"));
  context.state.catalog = original;
  context.state.benchmarkId = "proteingym";
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
