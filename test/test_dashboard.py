"""Dashboard API (no GPU / no real rem2 forward)."""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from rem2.dashboard.app import create_app
from rem2.dashboard.jobs import JobRunner, build_argv, recipe_argv
from rem2.dashboard.recipes import recipe_public
from rem2.dashboard.store import RunStore


def _fake_execute(job, store):
    out = Path(job["out_dir"])
    scores = out / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    (scores / "demo.csv").write_text(
        "mutant,esm2_t6_8M_UR50D__rem2,DMS_score\n"
        "A1C,0.50,0.10\n"
        "A1D,-0.20,0.00\n"
        "A1E,1.25,0.80\n",
        encoding="utf-8",
    )
    fasta = store.inputs_dir(job["id"]) / "query.fasta"
    if fasta.is_file():
        seq = fasta.read_text(encoding="utf-8")
    else:
        fasta.write_text(">p\nACDE\n", encoding="utf-8")
        seq = ">p\nACDE\n"
    (out / "_inputs" / "aa_seq").mkdir(parents=True, exist_ok=True)
    (out / "_inputs" / "aa_seq" / "p.fasta").write_text(seq, encoding="utf-8")
    store.append_log(job["id"], "ok\n")


@pytest.fixture
def client(tmp_path):
    store = RunStore(tmp_path)
    runner = JobRunner(store, execute=_fake_execute)
    app = create_app(tmp_path, runner=runner)
    with TestClient(app) as test_client:
        yield test_client


def test_recipes_match_cli_flags():
    ids = {item["id"] for item in recipe_public()}
    assert ids == {"full", "raw", "msa", "ccd"}
    assert "--alpha" in recipe_argv("raw")
    assert recipe_argv("full") == []
    by_id = {item["id"]: item for item in recipe_public()}
    assert by_id["full"]["msa"] == "optional"
    assert by_id["raw"]["msa"] == "off"


def test_venusrem2_full_argv_does_not_require_msa():
    argv = build_argv(
        model="venusrem2",
        recipe="full",
        out_dir="/tmp/out",
        fasta="/tmp/q.fasta",
    )
    assert "--model" in argv
    assert "venusrem2" in argv
    assert "--aa_seq_aln_dir" not in argv


def test_health_and_models(client):
    health = client.get("/api/health").json()
    assert health["ok"] is True
    assert "version" in health
    models = client.get("/api/models").json()["models"]
    names = {row["name"] for row in models}
    assert models[0]["name"] == "venusrem2"
    assert models[0]["series"] == "venusrem2"
    assert models[0]["label"] == "VenusREM2"
    assert models[0]["needs_pdb"] is True
    assert models[0]["needs_msa"] is False
    assert models[0]["input_kind"] == "structure"
    assert all(row.get("needs_msa") is False for row in models)
    esm = next(row for row in models if row["name"] == "esm2")
    assert esm["input_kind"] == "sequence"
    assert esm["series"] == "esm"
    assert esm["label"] == "ESM-2 650M"
    assert "esm2" in names
    assert "esm2-8m" in names
    by_series = {}
    for row in models:
        by_series.setdefault(row["series"], set()).add(row["name"])
    assert "venusrem2" in by_series["venusrem2"]
    assert {"esm2", "esm2-8m", "esm1b", "esm_if", "esmc"} <= by_series["esm"]
    assert {"prosst", "prosst-20", "prosst-4096"} <= by_series["prosst"]
    assert {"saprot", "saprot-35m-af2"} <= by_series["saprot"]
    assert {"progen2", "progen2-s", "progen3"} <= by_series["progen"]
    assert {"protein_mpnn", "protein_mpnn-soluble-v_48_020"} <= by_series["proteinmpnn"]
    assert {"rita", "rita-s"} <= by_series["rita"]
    assert {"protssn", "auto"} <= by_series["other"]
    recipes = client.get("/api/recipes").json()["recipes"]
    assert any(item["id"] == "full" for item in recipes)
    doctor = client.get("/api/doctor").json()
    assert doctor["version"]
    assert isinstance(doctor["extras"], list)


def test_create_run_scores_and_top(client):
    files = {"fasta": ("prot.fasta", b">p\nACDE\n", "text/plain")}
    data = {
        "model": "esm2-8m",
        "recipe": "full",
        "mutant_mode": "saturation",
        "mutant_sites": "1",
    }
    res = client.post("/api/runs", data=data, files=files)
    assert res.status_code == 200, res.text
    job = res.json()
    run_id = job["id"]
    assert job["status"] == "queued"
    assert job["model"] == "esm2-8m"
    assert "--model" in job["argv"]

    for _ in range(80):
        job = client.get(f"/api/runs/{run_id}").json()
        if job["status"] in {"done", "failed"}:
            break
        import time

        time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["primary_score"] == "esm2_t6_8M_UR50D__rem2"
    assert job["n_mutants"] == 3
    assert job["has_dms"] is True

    scores = client.get(f"/api/runs/{run_id}/scores?sort=-score").json()
    assert scores["total"] == 3
    assert scores["rows"][0]["mutant"] == "A1E"
    assert scores["rows"][0]["rank"] == 1

    filtered = client.get(f"/api/runs/{run_id}/scores?q=A1D").json()
    assert filtered["total"] == 1
    assert filtered["rows"][0]["mutant"] == "A1D"

    top = client.get(f"/api/runs/{run_id}/top?k=2").json()
    assert len(top["rows"]) == 2
    assert top["rows"][0]["mutant"] == "A1E"

    hist = client.get(f"/api/runs/{run_id}/histogram").json()
    assert hist["bins"]
    assert hist["primary_score"] == "esm2_t6_8M_UR50D__rem2"

    feats = client.get(f"/api/runs/{run_id}/features").json()
    assert feats["sequence"] == "ACDE"
    assert "plddt_histogram" in feats
    assert "rsa_histogram" in feats

    fasta = client.get(f"/api/runs/{run_id}/artifact?kind=fasta")
    assert fasta.status_code == 200
    assert "ACDE" in fasta.text

    log = client.get(f"/api/runs/{run_id}/log")
    assert log.status_code == 200
    assert "ok" in log.text


def test_artifacts_fall_back_to_base_dir(tmp_path):
    base = tmp_path / "dataset"
    (base / "aa_seq").mkdir(parents=True)
    (base / "pdbs").mkdir()
    (base / "aa_seq" / "protein.fasta").write_text(">protein\nACDE\n", encoding="utf-8")
    (base / "pdbs" / "protein.pdb").write_text("MODEL        1\nENDMDL\n", encoding="utf-8")

    store = RunStore(tmp_path / "dashboard")
    store.write_job(
        {
            "id": "base-dir-run",
            "status": "done",
            "out_dir": str(store.result_dir("base-dir-run")),
            "argv_spec": {"base_dir": str(base)},
        }
    )
    app = create_app(tmp_path / "dashboard", runner=JobRunner(store, execute=_fake_execute))
    with TestClient(app) as test_client:
        fasta = test_client.get("/api/runs/base-dir-run/artifact?kind=fasta")
        pdb = test_client.get("/api/runs/base-dir-run/artifact?kind=pdb")

    assert fasta.status_code == 200
    assert "ACDE" in fasta.text
    assert pdb.status_code == 200
    assert "MODEL" in pdb.text


def test_features_fall_back_to_base_dir(tmp_path):
    base = Path(__file__).parent / "fixtures" / "trp_cage"
    store = RunStore(tmp_path / "dashboard")
    store.write_job(
        {
            "id": "base-dir-features",
            "status": "done",
            "protein": "trp_cage",
            "out_dir": str(store.result_dir("base-dir-features")),
            "argv_spec": {"base_dir": str(base)},
        }
    )
    app = create_app(tmp_path / "dashboard", runner=JobRunner(store, execute=_fake_execute))
    with TestClient(app) as test_client:
        features = test_client.get("/api/runs/base-dir-features/features").json()

    assert features["sequence"]
    assert features["has_pdb"] is True
    assert len(features["rsa"]) == len(features["sequence"])


def test_proteingym_leaderboard(client):
    from rem2.dashboard.leaderboard import proteingym_board

    board = proteingym_board()
    assert board["rows"][0]["name"] == "VenusREM2"
    assert board["rows"][0]["rank"] == 1
    assert board["rows"][0]["score"] == 0.556
    assert "str" in board["rows"][0]["inputs"]
    assert "evo" in board["rows"][0]["inputs"]
    api = client.get("/api/leaderboard").json()
    boards = {item["id"]: item for item in api["boards"]}
    assert set(boards) >= {"substitutions", "stability", "activity", "ablations"}
    assert boards["substitutions"]["rows"][0]["name"] == "VenusREM2"
    assert boards["substitutions"]["n"] == 217
    assert boards["stability"]["rows"][0]["name"] == "VenusREM2"
    assert boards["stability"]["rows"][0]["score"] == 0.691
    names = {row["name"] for row in boards["substitutions"]["rows"]}
    assert names == {
        "VenusREM2",
        "AIDO Protein-RAG (16B)",
        "VenusREM",
        "ProSST (K=2048)",
        "S3F-MSA",
        "Protriever",
        "ESCOTT",
        "PoET (200M)",
        "ESM3 open (1.4B)",
        "RSALOR",
        "VespaG",
        "SaProt (650M)",
        "TranceptEVE-L",
        "GEMME",
        "ProtSSN ensemble",
    }
    assert boards["ablations"]["rows"][0]["score"] == 0.556
    assert boards["ablations"]["rows"][-1]["score"] == 0.524

    assert api["default_benchmark"] == "proteingym"
    assert api["planned_benchmarks"] == ["VenusMutHub", "VenusViroHub"]
    assert len(api["benchmarks"]) == 1
    benchmark = api["benchmarks"][0]
    assert benchmark["id"] == "proteingym"
    pairs = {row["family"]: row for row in benchmark["pairs"]}
    assert pairs["ProSST ensemble"]["base"] == 0.524
    assert pairs["ProSST ensemble"]["enhanced"] == 0.556
    assert pairs["ProSST ensemble"]["delta"] == 0.032
    assert pairs["ProSST ensemble"]["official_reference"] is None
    assert pairs["SaProt AF-650M"]["comparison"] == "protocol_mismatch"
    assert pairs["SaProt AF-650M"]["official_gap"] == -0.033


def test_create_run_requires_input(client):
    res = client.post("/api/runs", data={"model": "esm2", "recipe": "full"})
    assert res.status_code == 400


def test_create_run_fetches_sequence(client, monkeypatch):
    def fake_fasta(seq_id, dest, **_kwargs):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(">P0A6Y8\nACDE\n", encoding="utf-8")
        assert seq_id == "P0A6Y8"
        return str(dest)

    monkeypatch.setattr("rem2.data.fetch_sequence.fetch_query_fasta", fake_fasta)
    res = client.post(
        "/api/runs",
        data={"model": "esm2-8m", "recipe": "full", "seq_id": "P0A6Y8"},
    )
    assert res.status_code == 200, res.text
    job = res.json()
    assert job["seq_id"] == "P0A6Y8"
    assert job["fetch_mode"] == "none"
    fasta = client.get(f"/api/runs/{job['id']}/artifact?kind=fasta")
    assert fasta.status_code == 200
    assert "ACDE" in fasta.text


def test_create_run_seq_id_fetch_fails(client, monkeypatch):
    def boom(seq_id, dest, **_kwargs):
        raise OSError("uniprot down")

    monkeypatch.setattr("rem2.data.fetch_sequence.fetch_query_fasta", boom)
    res = client.post(
        "/api/runs",
        data={"model": "esm2-8m", "recipe": "full", "seq_id": "P0A6Y8"},
    )
    assert res.status_code == 400
    assert "P0A6Y8" in res.text


def test_create_run_rejects_bad_seq_id(client):
    res = client.post(
        "/api/runs",
        data={"model": "esm2-8m", "recipe": "full", "seq_id": "not-an-id"},
    )
    assert res.status_code == 400


def test_unknown_run_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_static_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"REM2 Dashboard" in res.content
    assert b"Review" in res.content
    assert b"AlphaFold DB" in res.content
    js = client.get("/app.js")
    assert js.status_code == 200
    assert b"ssPyMol" in js.content
    assert b"protein-bench" in js.content
    assert b"aa-strip" in js.content
    assert b"aa-jump" in js.content
    assert b"data-gutter" in js.content
    assert b"no pLDDT" in js.content
    assert b"preferredModel" in js.content
    assert b"modelLockReason" in js.content
    assert b"needs_msa: false" in js.content
    assert "MSA optional (none → α=0)".encode() in js.content
    assert b"progress-bar" in js.content
    assert b"Start scoring" in res.content
    assert b'id="btn-demo"' in res.content
    assert b"Skip the form" not in res.content
    assert b"Full ProteinGym-level scoring needs at least a PDB and an MSA" not in res.content
    assert b'id="pg-board"' in res.content
    assert b"What REM2 adds" in res.content
    assert "skip → α=0".encode() in res.content
    assert b"intake-box" in res.content
    assert b"slot-sequence" in res.content
    assert b"slot-structure" in res.content
    assert b"slot-msa" in res.content
    assert b'name="seq_id"' in res.content
    assert b"New prediction" in res.content
    vendor = client.get("/vendor/3Dmol-min.js")
    assert vendor.status_code == 200
    assert len(vendor.content) > 10000


_ATOM = (
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 91.00           C\n"
    "ATOM      2  CA  GLY A   2       1.000   0.000   0.000  1.00 72.00           C\n"
)
_AF = "TITLE     ALPHAFOLD MONOMER V2.0\nREMARK 99 pLDDT\n" + _ATOM
_CRYSTAL = "HEADER    HYDROLASE\nEXPDTA    X-RAY DIFFRACTION\n" + _ATOM


def test_create_run_fetches_structure(client, monkeypatch):
    def fake_fetch(**kwargs):
        dest = Path(kwargs["inputs"])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "afdb.pdb").write_text(_AF)
        (dest / "rcsb.pdb").write_text(_CRYSTAL)
        if kwargs.get("as_query", True):
            (dest / "query.pdb").write_text(_AF)
        return {
            "pdb_id": "2L6Q",
            "uniprot_id": "P0A6Y8",
            "errors": [],
            "preferred": {"kind": "afdb", "has_plddt": True},
        }

    monkeypatch.setattr("rem2.dashboard.jobs.maybe_fetch_structures", fake_fetch)
    res = client.post(
        "/api/runs",
        data={
            "model": "esm2-8m",
            "recipe": "full",
            "pdb_id": "2L6Q",
            "uniprot_id": "P0A6Y8",
            "fetch_structure": "auto",
        },
    )
    assert res.status_code == 200, res.text
    job = res.json()
    run_id = job["id"]
    assert job["status"] == "queued"
    assert job["progress"]["stage"] == "queued"

    for _ in range(80):
        job = client.get(f"/api/runs/{run_id}").json()
        if job["status"] in {"done", "failed"}:
            break
        import time

        time.sleep(0.05)
    assert job["status"] == "done", job
    assert job["has_pdb"] is True
    assert job["has_plddt"] is True
    assert job["pdb_id"] == "2L6Q"
    assert "afdb" in job["structure_sources"]
    assert job["structure_sources"]["afdb"]["has_plddt"] is True
    assert job["structure_sources"]["rcsb"]["has_plddt"] is False
    assert "--pdb" in job["argv"]
    assert job["progress"]["pct"] == 100

    af = client.get(f"/api/runs/{job['id']}/artifact?kind=pdb&source=afdb")
    assert af.status_code == 200
    assert "ALPHAFOLD" in af.text
    xtal = client.get(f"/api/runs/{job['id']}/artifact?kind=pdb&source=rcsb")
    assert xtal.status_code == 200
    assert "X-RAY" in xtal.text


def test_fetch_structure_on_existing_run(client, monkeypatch):
    files = {"fasta": ("prot.fasta", b">p\nACDE\n", "text/plain")}
    res = client.post(
        "/api/runs",
        data={"model": "esm2-8m", "recipe": "full", "fetch_structure": "none"},
        files=files,
    )
    assert res.status_code == 200, res.text
    run_id = res.json()["id"]
    assert res.json()["has_pdb"] is False

    for _ in range(80):
        job = client.get(f"/api/runs/{run_id}").json()
        if job["status"] in {"done", "failed"}:
            break
        import time

        time.sleep(0.05)
    assert job["status"] == "done", job

    def fake_fetch(**kwargs):
        dest = Path(kwargs["inputs"])
        (dest / "afdb.pdb").write_text(_AF)
        if kwargs.get("as_query", True):
            (dest / "query.pdb").write_text(_AF)
        return {"pdb_id": None, "uniprot_id": "P0A6Y8", "errors": []}

    monkeypatch.setattr("rem2.dashboard.app.maybe_fetch_structures", fake_fetch)
    fetched = client.post(
        f"/api/runs/{run_id}/fetch_structure",
        data={"uniprot_id": "P0A6Y8", "source": "afdb"},
    )
    assert fetched.status_code == 200, fetched.text
    job = fetched.json()
    assert job["has_pdb"] is True
    assert job["has_plddt"] is True
    assert job["uniprot_id"] == "P0A6Y8"


def test_log_tee_isatty(tmp_path):
    from rem2.dashboard.jobs import LogTee
    from rem2.dashboard.store import RunStore

    store = RunStore(tmp_path)
    job = {"id": "abc123", "progress": {"pct": 0}}
    store.write_job({**job, "id": "abc123", "status": "running"})
    tee = LogTee("abc123", store, job)
    assert tee.isatty() is False
    assert tee.write("loading 40%\n") == len("loading 40%\n")
    tee.flush()
    assert "40%" in store.log_path("abc123").read_text(encoding="utf-8")
    from rem2.scoring.run_utils import should_use_color
    import sys

    old = sys.stdout
    sys.stdout = tee
    try:
        assert should_use_color() is False
    finally:
        sys.stdout = old
