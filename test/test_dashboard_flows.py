"""Dashboard flows and edge cases (no GPU / no real rem2 forward)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from rem2.dashboard.app import create_app
from rem2.dashboard.jobs import JobRunner, histogram, prepare_run, slice_scores
from rem2.dashboard.store import RunStore
from test.test_dashboard import _AF, _CRYSTAL, _fake_execute

pytest.importorskip("pandas")
import pandas as pd


def _wait(client: TestClient, run_id: str, timeout: float = 4.0) -> dict:
    deadline = time.time() + timeout
    job = {}
    while time.time() < deadline:
        job = client.get(f"/api/runs/{run_id}").json()
        if job.get("status") in {"done", "failed", "cancelled"}:
            return job
        time.sleep(0.04)
    raise AssertionError(f"job {run_id} stuck at {job}")


@pytest.fixture
def store_client(tmp_path):
    store = RunStore(tmp_path)
    runner = JobRunner(store, execute=_fake_execute)
    app = create_app(tmp_path, runner=runner)
    with TestClient(app) as client:
        yield store, runner, client


def test_frontend_logic_and_layout():
    import subprocess

    script = Path(__file__).resolve().parent / "test_dashboard_ui.js"
    proc = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout


def test_default_model_and_list_order(store_client):
    store, _runner, client = store_client
    first = client.post(
        "/api/runs",
        data={"recipe": "full", "fetch_structure": "none"},
        files={"fasta": ("a.fasta", b">one\nACDE\n", "text/plain")},
    )
    assert first.status_code == 200
    assert first.json()["model"] == "venusrem2"
    assert first.json()["recipe"] == "full"
    assert "--aa_seq_aln_dir" not in first.json()["argv"]
    time.sleep(0.02)
    second = client.post(
        "/api/runs",
        data={"model": "esm2-8m", "recipe": "raw", "fetch_structure": "none"},
        files={"fasta": ("b.fasta", b">two\nACDE\n", "text/plain")},
    )
    assert second.status_code == 200
    runs = client.get("/api/runs").json()["runs"]
    names = {row["protein"] for row in runs}
    assert {"one", "two"} <= names
    stamps = [row.get("created_at") or "" for row in runs]
    assert stamps == sorted(stamps, reverse=True)
    _wait(client, first.json()["id"])
    _wait(client, second.json()["id"])


def test_pdb_only_run_preserves_upload_name_after_enrichment(store_client):
    _store, _runner, client = store_client
    response = client.post(
        "/api/runs",
        data={"model": "esm2-8m", "recipe": "full", "fetch_structure": "none"},
        files={"pdb": ("5A71_kcat.pdb", _AF.encode(), "chemical/x-pdb")},
    )
    assert response.status_code == 200
    job = _wait(client, response.json()["id"])
    assert job["protein"] == "5A71_kcat"


def test_pdb_upload_mutants_and_limits(store_client):
    store, _runner, client = store_client
    res = client.post(
        "/api/runs",
        data={
            "model": "esm2-8m",
            "recipe": "full",
            "positions": "1,2",
            "residue_range": "1-2",
            "max_mutants": "50",
            "scoring_strategy": "wt",
            "fetch_structure": "none",
        },
        files={
            "pdb": ("q.pdb", _CRYSTAL.encode(), "chemical/x-pdb"),
            "mutants": ("m.csv", b"mutant\nA1C\nA1D\n", "text/csv"),
            "msa": ("q.a2m", b">q\nACDE\n", "text/plain"),
        },
    )
    assert res.status_code == 200, res.text
    job = _wait(client, res.json()["id"])
    assert job["status"] == "done"
    assert job["has_pdb"] is True
    assert job["has_plddt"] is False
    argv = " ".join(job["argv"])
    assert "--pdb" in job["argv"]
    assert "--positions" in job["argv"]
    assert "--max_mutants" in job["argv"]
    assert "--aa_seq_aln_dir" in job["argv"]
    assert "50" in argv
    assert (store.inputs_dir(job["id"]) / "msa" / "query.a2m").is_file()


def test_fetch_none_does_not_call_network(store_client, monkeypatch):
    _store, _runner, client = store_client

    def boom(**_kwargs):
        raise AssertionError("fetch should be skipped when fetch_structure=none")

    monkeypatch.setattr("rem2.dashboard.jobs.maybe_fetch_structures", boom)
    res = client.post(
        "/api/runs",
        data={"model": "esm2-8m", "recipe": "full", "fetch_structure": "none"},
        files={"fasta": ("p.fasta", b">p\nACDE\n", "text/plain")},
    )
    job = _wait(client, res.json()["id"])
    assert job["status"] == "done"
    stages = (job.get("progress") or {}).get("stage")
    assert stages in {"done", "score"}


def test_id_only_without_fetch_fails(store_client, monkeypatch):
    _store, _runner, client = store_client
    monkeypatch.setattr("rem2.dashboard.jobs.maybe_fetch_structures", lambda **_k: None)
    res = client.post(
        "/api/runs",
        data={
            "model": "esm2-8m",
            "recipe": "full",
            "pdb_id": "2L6Q",
            "fetch_structure": "none",
        },
    )
    assert res.status_code == 200
    job = _wait(client, res.json()["id"])
    assert job["status"] == "failed"
    assert "No FASTA or PDB" in (job.get("error") or "")


def test_failed_execute_stays_failed(tmp_path):
    store = RunStore(tmp_path)

    def boom(job, _store):
        raise RuntimeError("forward exploded")

    app = create_app(tmp_path, runner=JobRunner(store, execute=boom))
    with TestClient(app) as client:
        res = client.post(
            "/api/runs",
            data={"model": "esm2-8m", "recipe": "full", "fetch_structure": "none"},
            files={"fasta": ("p.fasta", b">p\nACDE\n", "text/plain")},
        )
        job = _wait(client, res.json()["id"])
        assert job["status"] == "failed"
        assert "forward exploded" in job["error"]
        scores = client.get(f"/api/runs/{job['id']}/scores").json()
        assert scores["total"] == 0
        top = client.get(f"/api/runs/{job['id']}/top").json()
        assert top["rows"] == []
        hist = client.get(f"/api/runs/{job['id']}/histogram").json()
        assert hist["bins"] == []


def test_cancel_queued_and_running(tmp_path):
    store = RunStore(tmp_path)
    release = threading.Event()

    def blocked(job, inner):
        release.wait(timeout=3)
        _fake_execute(job, inner)

    runner = JobRunner(store, execute=blocked)
    queued = {
        "id": "queued1",
        "status": "queued",
        "progress": {"pct": 0, "stage": "queued", "message": "Queued"},
    }
    store.write_job(queued)
    cancelled = runner.cancel("queued1")
    assert cancelled["status"] == "cancelled"

    app = create_app(tmp_path, runner=runner)
    with TestClient(app) as client:
        res = client.post(
            "/api/runs",
            data={"model": "esm2-8m", "recipe": "full", "fetch_structure": "none"},
            files={"fasta": ("p.fasta", b">p\nACDE\n", "text/plain")},
        )
        run_id = res.json()["id"]
        for _ in range(50):
            job = client.get(f"/api/runs/{run_id}").json()
            if job["status"] == "running":
                break
            time.sleep(0.04)
        assert job["status"] == "running"
        mid = client.get(f"/api/runs/{run_id}/scores").json()
        assert mid["total"] == 0
        stopped = client.post(f"/api/runs/{run_id}/cancel")
        assert stopped.status_code == 200
        assert "cancel" in (stopped.json().get("error") or "").lower()
        release.set()
        final = _wait(client, run_id)
        assert final["status"] == "cancelled"
        assert client.post("/api/runs/nope/cancel").status_code == 404


def test_scores_search_sort_top_bounds(store_client):
    _store, _runner, client = store_client
    res = client.post(
        "/api/runs",
        data={"model": "esm2-8m", "recipe": "full", "fetch_structure": "none"},
        files={"fasta": ("p.fasta", b">p\nACDE\n", "text/plain")},
    )
    run_id = res.json()["id"]
    _wait(client, run_id)

    empty = client.get(f"/api/runs/{run_id}/scores?q=ZZZ").json()
    assert empty["total"] == 0
    assert empty["rows"] == []

    by_mut = client.get(f"/api/runs/{run_id}/scores?sort=mutant").json()
    mutants = [row["mutant"] for row in by_mut["rows"]]
    assert mutants == sorted(mutants)

    page = client.get(f"/api/runs/{run_id}/scores?offset=2&limit=2").json()
    assert page["total"] == 3
    assert len(page["rows"]) == 1

    top = client.get(f"/api/runs/{run_id}/top?k=999").json()
    assert len(top["rows"]) == 3
    tiny = client.get(f"/api/runs/{run_id}/top?k=1").json()
    assert len(tiny["rows"]) == 1
    assert tiny["rows"][0]["mutant"] == "A1E"


def test_artifacts_and_fetch_errors(store_client, monkeypatch):
    _store, _runner, client = store_client
    res = client.post(
        "/api/runs",
        data={"model": "esm2-8m", "recipe": "full", "fetch_structure": "none"},
        files={"fasta": ("p.fasta", b">p\nACDE\n", "text/plain")},
    )
    run_id = res.json()["id"]
    _wait(client, run_id)

    assert client.get(f"/api/runs/{run_id}/artifact?kind=pdb").status_code == 404
    assert client.get(f"/api/runs/{run_id}/artifact?kind=msa").status_code == 400
    assert client.get("/api/runs/nope/scores").status_code == 404
    assert client.get("/api/runs/nope/log").status_code == 404

    missing = client.post(
        f"/api/runs/{run_id}/fetch_structure",
        data={"source": "afdb"},
    )
    assert missing.status_code == 400

    def fake_fetch(**kwargs):
        dest = Path(kwargs["inputs"])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "afdb.pdb").write_text(_AF)
        if kwargs.get("as_query", True):
            (dest / "query.pdb").write_text(_AF)
        return {"pdb_id": None, "uniprot_id": "P0A6Y8", "errors": ["rcsb missed"]}

    monkeypatch.setattr("rem2.dashboard.app.maybe_fetch_structures", fake_fetch)
    fetched = client.post(
        f"/api/runs/{run_id}/fetch_structure",
        data={"uniprot_id": "P0A6Y8", "source": "afdb"},
    )
    assert fetched.status_code == 200
    job = fetched.json()
    assert job["has_pdb"] is True
    assert job["has_plddt"] is True
    assert job["fetch_errors"] == ["rcsb missed"]
    pdb = client.get(f"/api/runs/{run_id}/artifact?kind=pdb&source=afdb")
    assert pdb.status_code == 200
    assert "ALPHAFOLD" in pdb.text
    assert client.post("/api/runs/nope/fetch_structure", data={"uniprot_id": "P0A6Y8"}).status_code == 404


def test_write_job_concurrent(tmp_path):
    store = RunStore(tmp_path)
    store.write_job({"id": "race", "status": "queued", "n": 0})
    errors: list[BaseException] = []

    def writer(i: int) -> None:
        try:
            job = store.load_job("race") or {"id": "race"}
            job["n"] = i
            store.write_job(job)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(24)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    latest = store.load_job("race")
    assert latest is not None
    assert latest["id"] == "race"


def test_prepare_run_skips_fetch_when_none(tmp_path, monkeypatch):
    store = RunStore(tmp_path)
    inputs = store.inputs_dir("p1")
    fasta = inputs / "query.fasta"
    fasta.write_text(">p\nACDE\n")
    job = store.write_job(
        {
            "id": "p1",
            "status": "running",
            "model": "esm2-8m",
            "recipe": "full",
            "fetch_mode": "none",
            "out_dir": str(store.result_dir("p1")),
            "argv_spec": {"fasta": str(fasta), "mutant_sites": "1"},
        }
    )

    monkeypatch.setattr(
        "rem2.dashboard.jobs.maybe_fetch_structures",
        lambda **_k: (_ for _ in ()).throw(AssertionError("no fetch")),
    )
    out = prepare_run(job, store)
    assert "--model" in out["argv"]
    assert out["progress"]["stage"] == "score"


def test_slice_and_histogram_edges():
    frame = pd.DataFrame(
        {
            "mutant": ["A1C", "A1D", "A1E"],
            "esm2__rem2": [0.1, 0.1, 0.1],
        }
    )
    rows, total = slice_scores(frame, primary="esm2__rem2", offset=10, limit=5, sort="-score", q="")
    assert total == 3
    assert rows == []
    hist = histogram(frame, "esm2__rem2")
    assert hist["bins"]
    assert hist["min"] == hist["max"]
    empty = histogram(pd.DataFrame(), "missing")
    assert empty["bins"] == []
