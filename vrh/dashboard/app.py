"""FastAPI app for the vrh local console."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from vrh import __version__
from vrh.dashboard.deps import friendly_import_error, scoring_blocked_message, setup_hints
from vrh.dashboard.inspect import doctor_report, list_model_payload
from vrh.dashboard.leaderboard import proteingym_catalog
from vrh.dashboard.jobs import (
    JobRunner,
    attach_structure_meta,
    build_argv,
    enrich_job,
    existing_fasta,
    existing_query_pdb,
    histogram,
    load_score_frame,
    residue_features,
    maybe_fetch_structures,
    resolve_pdb_artifact,
    slice_scores,
)
from vrh.dashboard.recipes import recipe_public
from vrh.dashboard.store import RunStore, utc_now

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


def _save_upload(upload, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        handle.write(upload.file.read())
    return dest


def create_app(root: Optional[Path] = None, runner: Optional[JobRunner] = None):
    if FastAPI is None:
        raise RuntimeError(
            "vrh dashboard needs FastAPI. Install with: pip install 'vrh[dashboard]'"
        )

    store = RunStore(root)
    worker = runner or JobRunner(store)

    app = FastAPI(title="VRH Dashboard", version=__version__, docs_url=None, redoc_url=None)
    app.state.store = store
    app.state.runner = worker

    @app.get("/api/health")
    def health():
        report = doctor_report()
        problems = list(report.get("problems") or [])
        return {
            "ok": True,
            "version": __version__,
            "runs_root": str(store.runs_dir),
            "ready": not problems,
            "problems": problems,
            "hints": report.get("hints") or setup_hints(problems),
        }

    @app.exception_handler(ModuleNotFoundError)
    async def missing_module(_request, exc: ModuleNotFoundError):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": friendly_import_error(exc)})

    @app.exception_handler(ImportError)
    async def missing_import(_request, exc: ImportError):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": friendly_import_error(exc)})

    @app.get("/api/models")
    def models():
        return {"models": list_model_payload()}

    @app.get("/api/doctor")
    def doctor():
        return doctor_report()

    @app.get("/api/recipes")
    def recipes():
        return {"recipes": recipe_public()}

    @app.get("/api/leaderboard")
    @app.get("/api/catalog")
    def leaderboard():
        return proteingym_catalog()

    @app.get("/api/examples")
    def examples():
        from vrh.examples import demo_example_payload

        demo = demo_example_payload()
        return {"default": demo.get("default_preset") or "full", "examples": [demo]}

    @app.get("/api/examples/{example_id}/{kind}")
    def example_file(example_id: str, kind: str):
        from vrh.examples import DEMO_ASSAY, DEMO_FILE_KINDS, demo_file_path

        alias = str(example_id or "").strip().lower()
        if alias not in {"demo", "2l6q", DEMO_ASSAY.lower()}:
            raise HTTPException(404, "example not found")
        if kind not in DEMO_FILE_KINDS:
            raise HTTPException(404, "example file not found")
        path = demo_file_path(kind)
        if path is None:
            raise HTTPException(404, "example file not found")
        media = {
            "fasta": "text/plain",
            "pdb": "chemical/x-pdb",
            "msa": "text/plain",
            "mutants": "text/csv",
        }.get(kind, "application/octet-stream")
        return FileResponse(path, media_type=media, filename=path.name)

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

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        job = store.load_job(run_id)
        if job is None:
            raise HTTPException(404, "run not found")
        if job.get("status") in {"queued", "running"}:
            raise HTTPException(409, "Cancel this run before deleting it")
        try:
            deleted = store.delete_run(run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not deleted:
            raise HTTPException(404, "run not found")
        return {"deleted": run_id}

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
    def get_top(run_id: str, k: int = 30):
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

    @app.get("/api/runs/{run_id}/features")
    def get_features(run_id: str):
        job = store.load_job(run_id)
        if job is None:
            raise HTTPException(404, "run not found")
        attach_structure_meta(job, store)
        return residue_features(job, store)

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
        try:
            path = store.log_path(run_id)
        except ValueError:
            raise HTTPException(404, "no log") from None
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
        base_value = str((job.get("argv_spec") or {}).get("base_dir") or "").strip()
        base_dir = Path(base_value).expanduser() if base_value else None
        if kind == "pdb":
            path = resolve_pdb_artifact(inputs, result / "_inputs" / "pdbs", source)
            if path is None and base_dir is not None:
                path = _find_suffix(base_dir / "pdbs", (".pdb", ".ent", ".cif"))
            media = "chemical/x-pdb"
        elif kind == "fasta":
            path = _find_suffix(inputs, (".fasta", ".fa", ".faa")) or _find_suffix(
                result / "_inputs" / "aa_seq", (".fasta", ".fa", ".faa")
            )
            if path is None and base_dir is not None:
                path = _find_suffix(base_dir / "aa_seq", (".fasta", ".fa", ".faa"))
            media = "text/plain"
        else:
            raise HTTPException(400, "kind must be pdb or fasta")
        if path is None or not path.is_file():
            raise HTTPException(404, f"{kind} not found")
        return FileResponse(path, media_type=media, filename=path.name)

    @app.post("/api/fetch_sequence")
    async def fetch_sequence(seq_id: Optional[str] = Form(None)):
        seq_id = _blank(seq_id)
        if not seq_id:
            raise HTTPException(400, "Need a UniProt accession or PDB id")
        from vrh.data.fetch_sequence import fetch_query_fasta
        from vrh.data.fetch_structure import normalize_pdb_id, normalize_uniprot

        acc = normalize_uniprot(seq_id) or normalize_pdb_id(seq_id)
        if not acc:
            raise HTTPException(400, f"Not a UniProt accession or PDB id: {seq_id}")
        try:
            with TemporaryDirectory() as tmp:
                path = fetch_query_fasta(seq_id, Path(tmp) / "query.fasta")
                text = Path(path).read_text(encoding="utf-8", errors="replace")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc
        return {"id": acc, "name": f"{acc}.fasta", "fasta": text}

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
                fasta_path=existing_fasta(inputs),
                pdb_path=existing_query_pdb(inputs),
                protein=job.get("protein"),
                pdb_id=_blank(pdb_id) or job.get("pdb_id"),
                uniprot_id=_blank(uniprot_id) or job.get("uniprot_id"),
                fetch_mode=_blank(source) or "auto",
                as_query=not existing_query_pdb(inputs),
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
        model: str = Form("venusrem2"),
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
        seq_id: Optional[str] = Form(None),
        fetch_structure: Optional[str] = Form("auto"),
    ):
        blocked = scoring_blocked_message()
        if blocked:
            raise HTTPException(400, blocked)
        run_id = store.new_id()
        inputs = store.inputs_dir(run_id)
        out_dir = str(store.result_dir(run_id))
        fasta_path = pdb_path = mutants_path = msa_dir = base_dir = None
        protein = "protein"
        has_pdb = False
        pdb_id = _blank(pdb_id)
        uniprot_id = _blank(uniprot_id)
        seq_id = _blank(seq_id)
        is_demo = _truthy(demo) or mutant_mode == "demo"
        argv_spec: dict = {}

        if is_demo:
            from vrh.download.example import bundled_example_dir, ensure_demo_dataset
            from vrh.examples import DEMO_ASSAY

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
            argv_spec = {"base_dir": base_dir}
        else:
            if fasta is not None and fasta.filename:
                fasta_path = str(_save_upload(fasta, inputs / "query.fasta"))
                from vrh.dashboard.jobs import _read_fasta_sequence

                protein, _seq = _read_fasta_sequence(Path(fasta_path))
                protein = protein or "query"
            elif seq_id:
                from vrh.data.fetch_sequence import fetch_query_fasta
                from vrh.dashboard.jobs import _read_fasta_sequence

                try:
                    fasta_path = fetch_query_fasta(seq_id, inputs / "query.fasta")
                except Exception as exc:
                    raise HTTPException(
                        400, f"Could not fetch sequence for {seq_id}: {exc}"
                    ) from exc
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
                # PDB-only single-protein runs are materialized as `query`,
                # regardless of the original upload filename. Keep the MSA
                # stem aligned so entropy-alpha can actually discover it.
                msa_stem = "query" if pdb_path and not fasta_path else protein
                msa_path = inputs / "msa" / f"{msa_stem}{suffix}"
                _save_upload(msa, msa_path)
                msa_dir = str(msa_path.parent)
            if not fasta_path and not pdb_path and not pdb_id and not uniprot_id and not seq_id:
                raise HTTPException(
                    400,
                    "Provide a FASTA and/or PDB, a PDB/UniProt id, or set demo=true",
                )
            argv_spec = {
                "fasta": fasta_path,
                "pdb": pdb_path,
                "mutants": mutants_path,
                "msa_dir": msa_dir,
                "mutant_sites": None if mutants_path else (mutant_sites or "1"),
                "positions": positions,
                "residue_range": residue_range,
                "max_mutants": max_mutants,
                "scoring_strategy": scoring_strategy,
            }
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
            ) if (fasta_path or pdb_path) else []

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
            "seq_id": seq_id,
            "structure_sources": {},
            "preferred_source": None,
            "fetch_mode": (
                "none"
                if is_demo or not (pdb_path or pdb_id or uniprot_id)
                else (_blank(fetch_structure) or "auto")
            ),
            "fetch_errors": [],
            "argv_spec": argv_spec,
            "progress": {"pct": 0, "stage": "queued", "message": "Queued"},
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
        @app.get("/")
        def index_page():
            return FileResponse(
                STATIC_DIR / "index.html",
                media_type="text/html",
                headers={"Cache-Control": "no-store"},
            )

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
