"""FastAPI app for the rem2 local console."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from rem2 import __version__
from rem2.dashboard.inspect import doctor_report, list_model_payload
from rem2.dashboard.jobs import (
    JobRunner,
    attach_structure_meta,
    build_argv,
    enrich_job,
    histogram,
    load_score_frame,
    maybe_fetch_structures,
    resolve_pdb_artifact,
    slice_scores,
)
from rem2.dashboard.recipes import recipe_public
from rem2.dashboard.store import RunStore, utc_now

STATIC_DIR = Path(__file__).resolve().parent / "static"

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, PlainTextResponse
    from fastapi.staticfiles import StaticFiles
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    File = Form = HTTPException = UploadFile = None  # type: ignore[assignment]
    FileResponse = PlainTextResponse = StaticFiles = None  # type: ignore[assignment]


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _blank(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _existing_fasta(inputs: Path) -> Optional[str]:
    for name in ("query.fasta", "query.fa", "query.faa"):
        path = inputs / name
        if path.is_file():
            return str(path)
    return None


def _existing_query_pdb(inputs: Path) -> Optional[str]:
    path = inputs / "query.pdb"
    return str(path) if path.is_file() else None


def _save_upload(upload, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        handle.write(upload.file.read())
    return dest


def create_app(root: Optional[Path] = None, runner: Optional[JobRunner] = None):
    if FastAPI is None:
        raise RuntimeError(
            "rem2 dashboard needs FastAPI. Install with: pip install 'rem2[dashboard]'"
        )

    store = RunStore(root)
    worker = runner or JobRunner(store)

    app = FastAPI(title="rem2 Console", version=__version__, docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.runner = worker

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "runs_root": str(store.runs_dir)}

    @app.get("/api/models")
    def models():
        return {"models": list_model_payload()}

    @app.get("/api/doctor")
    def doctor():
        return doctor_report()

    @app.get("/api/recipes")
    def recipes():
        return {"recipes": recipe_public()}

    @app.get("/api/runs")
    def list_runs():
        return {"runs": store.list_jobs()}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        job = store.load_job(run_id)
        if job is None:
            raise HTTPException(404, "run not found")
        if job.get("status") == "done":
            enrich_job(job, store)
        else:
            attach_structure_meta(job, store)
        return job

    @app.get("/api/runs/{run_id}/scores")
    def get_scores(
        run_id: str,
        offset: int = 0,
        limit: int = 80,
        sort: str = "-score",
        q: str = "",
    ):
        job = store.load_job(run_id)
        if job is None:
            raise HTTPException(404, "run not found")
        enrich_job(job, store)
        frame = load_score_frame(job, store)
        primary = job.get("primary_score")
        rows, total = slice_scores(
            frame, primary=primary, offset=offset, limit=limit, sort=sort, q=q
        )
        return {
            "rows": rows,
            "total": total,
            "columns": list(frame.columns) if not frame.empty else [],
            "primary_score": primary,
            "offset": offset,
            "limit": limit,
        }

    @app.get("/api/runs/{run_id}/top")
    def get_top(run_id: str, k: int = 20):
        job = store.load_job(run_id)
        if job is None:
            raise HTTPException(404, "run not found")
        enrich_job(job, store)
        frame = load_score_frame(job, store)
        primary = job.get("primary_score")
        rows, _total = slice_scores(
            frame, primary=primary, offset=0, limit=max(1, min(k, 500)), sort="-score"
        )
        return {"rows": rows, "k": k, "primary_score": primary}

    @app.get("/api/runs/{run_id}/histogram")
    def get_histogram(run_id: str):
        job = store.load_job(run_id)
        if job is None:
            raise HTTPException(404, "run not found")
        enrich_job(job, store)
        frame = load_score_frame(job, store)
        return histogram(frame, job.get("primary_score"))

    @app.get("/api/runs/{run_id}/log")
    def get_log(run_id: str):
        path = store.log_path(run_id)
        if not path.is_file():
            raise HTTPException(404, "no log")
        return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"))

    @app.get("/api/runs/{run_id}/artifact")
    def get_artifact(run_id: str, kind: str = "fasta", source: Optional[str] = None):
        job = store.load_job(run_id)
        if job is None:
            raise HTTPException(404, "run not found")
        inputs = store.inputs_dir(run_id)
        result = Path(job.get("out_dir") or store.result_dir(run_id))
        if kind == "pdb":
            path = resolve_pdb_artifact(inputs, result / "_inputs" / "pdbs", source)
            media = "chemical/x-pdb"
        elif kind == "fasta":
            path = _find_suffix(inputs, (".fasta", ".fa", ".faa")) or _find_suffix(
                result / "_inputs" / "aa_seq", (".fasta", ".fa", ".faa")
            )
            media = "text/plain"
        else:
            raise HTTPException(400, "kind must be pdb or fasta")
        if path is None or not path.is_file():
            raise HTTPException(404, f"{kind} not found")
        return FileResponse(path, media_type=media, filename=path.name)

    @app.post("/api/runs/{run_id}/fetch_structure")
    async def fetch_structure(
        run_id: str,
        pdb_id: Optional[str] = Form(None),
        uniprot_id: Optional[str] = Form(None),
        source: Optional[str] = Form("auto"),
    ):
        job = store.load_job(run_id)
        if job is None:
            raise HTTPException(404, "run not found")
        inputs = store.inputs_dir(run_id)
        try:
            fetched = maybe_fetch_structures(
                inputs=inputs,
                fasta_path=_existing_fasta(inputs),
                pdb_path=_existing_query_pdb(inputs),
                protein=job.get("protein"),
                pdb_id=_blank(pdb_id) or job.get("pdb_id"),
                uniprot_id=_blank(uniprot_id) or job.get("uniprot_id"),
                fetch_mode=_blank(source) or "auto",
                as_query=not _existing_query_pdb(inputs),
            )
        except FileNotFoundError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc
        if fetched is None:
            raise HTTPException(
                400,
                "Need a PDB id or UniProt accession (from the form, FASTA header, or protein name).",
            )
        job["pdb_id"] = fetched.get("pdb_id") or job.get("pdb_id")
        job["uniprot_id"] = fetched.get("uniprot_id") or job.get("uniprot_id")
        job["fetch_errors"] = fetched.get("errors") or []
        attach_structure_meta(job, store)
        return store.write_job(job)

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        try:
            return worker.cancel(run_id)
        except KeyError:
            raise HTTPException(404, "run not found") from None

    @app.post("/api/runs")
    async def create_run(
        model: str = Form("esm2"),
        recipe: str = Form("full"),
        mutant_mode: str = Form("saturation"),
        demo: Optional[str] = Form(None),
        mutant_sites: Optional[str] = Form("1"),
        positions: Optional[str] = Form(None),
        residue_range: Optional[str] = Form(None),
        max_mutants: Optional[str] = Form(None),
        scoring_strategy: Optional[str] = Form(None),
        fasta: Optional[UploadFile] = File(None),
        pdb: Optional[UploadFile] = File(None),
        mutants: Optional[UploadFile] = File(None),
        msa: Optional[UploadFile] = File(None),
        pdb_id: Optional[str] = Form(None),
        uniprot_id: Optional[str] = Form(None),
        fetch_structure: Optional[str] = Form("auto"),
    ):
        run_id = store.new_id()
        inputs = store.inputs_dir(run_id)
        out_dir = str(store.result_dir(run_id))
        fasta_path = pdb_path = mutants_path = msa_dir = base_dir = None
        protein = "protein"
        has_pdb = False
        fetched = None
        pdb_id = _blank(pdb_id)
        uniprot_id = _blank(uniprot_id)
        is_demo = _truthy(demo) or mutant_mode == "demo"

        if is_demo:
            from rem2.download.example import bundled_example_dir, ensure_demo_dataset
            from rem2.examples import DEMO_ASSAY

            try:
                demo_dir = ensure_demo_dataset()
            except Exception:
                bundled = bundled_example_dir()
                if bundled is None:
                    raise HTTPException(400, "Demo dataset is missing") from None
                demo_dir = bundled
            base_dir = str(demo_dir)
            model = model or "esm2-8m"
            if model == "esm2":
                model = "esm2-8m"
            protein = DEMO_ASSAY
            has_pdb = True
            argv = build_argv(
                model=model,
                recipe=recipe,
                out_dir=out_dir,
                base_dir=base_dir,
            )
        else:
            if fasta is not None and fasta.filename:
                fasta_path = str(_save_upload(fasta, inputs / "query.fasta"))
                from rem2.dashboard.jobs import _read_fasta_sequence

                protein, _seq = _read_fasta_sequence(Path(fasta_path))
                protein = protein or "query"
            if pdb is not None and pdb.filename:
                pdb_path = str(_save_upload(pdb, inputs / "query.pdb"))
                has_pdb = True
                if protein == "protein":
                    protein = Path(pdb.filename).stem
            if mutants is not None and mutants.filename:
                mutants_path = str(_save_upload(mutants, inputs / "mutants.csv"))
            if msa is not None and msa.filename:
                suffix = Path(msa.filename).suffix or ".a2m"
                msa_path = inputs / "msa" / f"{protein}{suffix}"
                _save_upload(msa, msa_path)
                msa_dir = str(msa_path.parent)
            uploaded_pdb = pdb_path
            try:
                fetched = maybe_fetch_structures(
                    inputs=inputs,
                    fasta_path=fasta_path,
                    pdb_path=pdb_path,
                    protein=protein,
                    pdb_id=pdb_id,
                    uniprot_id=uniprot_id,
                    fetch_mode=_blank(fetch_structure) or "auto",
                    as_query=not uploaded_pdb,
                )
            except FileNotFoundError as exc:
                if fasta_path or uploaded_pdb:
                    fetched = {"errors": [str(exc)], "pdb_id": pdb_id, "uniprot_id": uniprot_id}
                else:
                    raise HTTPException(400, str(exc)) from exc
            except Exception as exc:
                if fasta_path or uploaded_pdb:
                    fetched = {"errors": [f"{type(exc).__name__}: {exc}"]}
                else:
                    raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc
            if fetched and fetched.get("pdb_id"):
                pdb_id = fetched.get("pdb_id") or pdb_id
            if fetched and fetched.get("uniprot_id"):
                uniprot_id = fetched.get("uniprot_id") or uniprot_id
            query_pdb = _existing_query_pdb(inputs)
            if query_pdb:
                pdb_path = query_pdb
                has_pdb = True
                if protein == "protein":
                    protein = pdb_id or uniprot_id or Path(query_pdb).stem
            if not fasta_path and not pdb_path:
                raise HTTPException(
                    400,
                    "Provide a FASTA and/or PDB, a PDB/UniProt id, or set demo=true",
                )
            argv = build_argv(
                model=model,
                recipe=recipe,
                out_dir=out_dir,
                fasta=fasta_path,
                pdb=pdb_path,
                mutants=mutants_path,
                msa_dir=msa_dir,
                mutant_sites=None if mutants_path else (mutant_sites or "1"),
                positions=positions,
                residue_range=residue_range,
                max_mutants=max_mutants,
                scoring_strategy=scoring_strategy,
            )

        job = {
            "id": run_id,
            "status": "queued",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "model": model,
            "recipe": recipe,
            "protein": protein,
            "n_mutants": None,
            "n_rows": None,
            "error": None,
            "argv": argv,
            "out_dir": out_dir,
            "has_pdb": has_pdb,
            "has_plddt": False,
            "pdb_origin": None,
            "pdb_id": pdb_id,
            "uniprot_id": uniprot_id,
            "structure_sources": {},
            "preferred_source": None,
            "fetch_errors": (fetched or {}).get("errors") or [],
            "has_dms": False,
            "sequence": "",
            "score_columns": [],
            "primary_score": None,
            "mutant_mode": mutant_mode if not is_demo else "demo",
        }
        attach_structure_meta(job, store)
        store.write_job(job)
        worker.submit(run_id)
        return job

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


def _find_suffix(root: Path, suffixes: tuple[str, ...]) -> Optional[Path]:
    if not root.exists():
        return None
    hits = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    return sorted(hits)[0] if hits else None
