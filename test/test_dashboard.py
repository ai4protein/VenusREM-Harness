"""Dashboard API (no GPU / no real rem2 forward)."""

from __future__ import annotations

from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from rem2.dashboard.app import create_app
from rem2.dashboard.jobs import JobRunner, recipe_argv
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


def test_health_and_models(client):
    health = client.get("/api/health").json()
    assert health["ok"] is True
    assert "version" in health
    models = client.get("/api/models").json()["models"]
    names = {row["name"] for row in models}
    assert models[0]["name"] == "venusrem2"
    assert "esm2" in names
    assert "esm2-8m" in names
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

    fasta = client.get(f"/api/runs/{run_id}/artifact?kind=fasta")
    assert fasta.status_code == 200
    assert "ACDE" in fasta.text

    log = client.get(f"/api/runs/{run_id}/log")
    assert log.status_code == 200
    assert "ok" in log.text


def test_create_run_requires_input(client):
    res = client.post("/api/runs", data={"model": "esm2", "recipe": "full"})
    assert res.status_code == 400


def test_unknown_run_404(client):
    assert client.get("/api/runs/nope").status_code == 404


def test_static_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"REM2 Dashboard" in res.content
    assert b"Structure" in res.content
    assert b"AlphaFold DB" in res.content
    js = client.get("/app.js")
    assert js.status_code == 200
    assert b"ssPyMol" in js.content
    assert b"protein-bench" in js.content
    assert b"data-gutter" in js.content
    assert b"no pLDDT" in js.content
    assert b'sel.value = "venusrem2"' in js.content
    assert b"progress-bar" in js.content
    assert b"Start scoring" in res.content
    assert b"1. Input" in res.content
    assert b"New job" in res.content
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
