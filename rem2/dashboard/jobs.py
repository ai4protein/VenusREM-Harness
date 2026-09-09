"""Queue and execute rem2 scoring jobs for the dashboard."""

from __future__ import annotations

import io
import math
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from rem2.dashboard.recipes import recipe_argv
from rem2.dashboard.store import RunStore

ExecuteFn = Callable[[dict[str, Any], RunStore], None]


def _score_columns(columns: list[str]) -> list[str]:
    preferred: list[str] = []
    if "VenusREM2" in columns:
        preferred.append("VenusREM2")
    preferred.extend(
        name
        for name in columns
        if name.endswith("__rem2") and name not in preferred
    )
    preferred.extend(
        name
        for name in columns
        if name.startswith("VenusREM2__") and name not in preferred
    )
    return preferred


def _scores_csv(result_dir: Path) -> Optional[Path]:
    scores = result_dir / "scores"
    if not scores.is_dir():
        return None
    tables = sorted(path for path in scores.glob("*.csv") if path.is_file())
    return tables[0] if tables else None


def _read_fasta_sequence(path: Path) -> tuple[str, str]:
    if not path.is_file():
        return "", ""
    name = path.stem
    seq_parts: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if not seq_parts:
                name = line[1:].split()[0] or name
            continue
        seq_parts.append(line)
    return name, "".join(seq_parts)


def _first_file(root: Path, suffixes: tuple[str, ...]) -> Optional[Path]:
    if not root.exists():
        return None
    matches = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    return sorted(matches)[0] if matches else None


_STRUCTURE_FILES = (
    ("query", "query.pdb"),
    ("afdb", "afdb.pdb"),
    ("rcsb", "rcsb.pdb"),
)


def collect_structures(inputs: Path, extra: Optional[Path] = None) -> dict[str, Any]:
    """Describe PDB files sitting in a run inputs directory."""
    from rem2.data.fetch_structure import describe_pdb, public_structure

    sources: dict[str, dict[str, Any]] = {}
    for kind, name in _STRUCTURE_FILES:
        path = inputs / name
        if path.is_file() and path.stat().st_size > 80:
            meta = describe_pdb(path)
            meta.update({"kind": kind, "label": kind.upper(), "id": None})
            sources[kind] = meta
    if not sources:
        found = _first_file(inputs, (".pdb",))
        if found is None and extra is not None:
            found = _first_file(extra, (".pdb",))
        if found is not None:
            meta = describe_pdb(found)
            meta.update({"kind": "query", "label": found.name, "id": None})
            sources["query"] = meta
    public = {key: public_structure(meta) for key, meta in sources.items()}
    preferred_key = next((key for key in ("afdb", "query", "rcsb") if key in sources), None)
    preferred = sources.get(preferred_key) if preferred_key else None
    return {
        "has_pdb": bool(sources),
        "has_plddt": any(bool(item.get("has_plddt")) for item in sources.values()),
        "pdb_origin": (preferred or {}).get("origin"),
        "structure_sources": public,
        "preferred_source": preferred_key,
        "_paths": {key: Path(meta["path"]) for key, meta in sources.items()},
    }


def resolve_pdb_artifact(
    inputs: Path, extra: Optional[Path] = None, source: Optional[str] = None
) -> Optional[Path]:
    info = collect_structures(inputs, extra)
    paths: dict[str, Path] = info.get("_paths") or {}
    if source:
        hit = paths.get(source.strip().lower())
        if hit is not None:
            return hit
    preferred = info.get("preferred_source")
    if preferred and preferred in paths:
        return paths[preferred]
    return None


def attach_structure_meta(job: dict[str, Any], store: RunStore) -> dict[str, Any]:
    run_id = job["id"]
    result = Path(job.get("out_dir") or store.result_dir(run_id))
    info = collect_structures(store.inputs_dir(run_id), result / "_inputs" / "pdbs")
    job["has_pdb"] = bool(info["has_pdb"] or job.get("has_pdb"))
    job["has_plddt"] = bool(info["has_plddt"])
    job["pdb_origin"] = info["pdb_origin"] or job.get("pdb_origin")
    job["structure_sources"] = info["structure_sources"]
    job["preferred_source"] = info["preferred_source"] or job.get("preferred_source")
    return job


def existing_fasta(inputs: Path) -> Optional[str]:
    for name in ("query.fasta", "query.fa", "query.faa"):
        path = inputs / name
        if path.is_file():
            return str(path)
    return None


def existing_query_pdb(inputs: Path) -> Optional[str]:
    path = inputs / "query.pdb"
    return str(path) if path.is_file() else None


def set_progress(job: dict[str, Any], store: RunStore, pct: int, stage: str, message: str) -> dict[str, Any]:
    latest = store.load_job(job["id"]) or job
    latest["progress"] = {
        "pct": max(0, min(100, int(pct))),
        "stage": stage,
        "message": message,
    }
    store.write_job(latest)
    job.update(latest)
    return latest


def maybe_fetch_structures(
    *,
    inputs: Path,
    fasta_path: Optional[str] = None,
    pdb_path: Optional[str] = None,
    protein: Optional[str] = None,
    pdb_id: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    fetch_mode: str = "auto",
    as_query: bool = True,
) -> Optional[dict[str, Any]]:
    """Download RCSB / AFDB models when an accession can be inferred."""
    from rem2.data.fetch_structure import (
        fetch_structures,
        guess_accessions,
        install_structures,
        normalize_pdb_id,
        normalize_uniprot,
    )

    mode = (fetch_mode or "auto").strip().lower() or "auto"
    if mode in {"none", "off", "0", "false"}:
        return None
    hints = [protein, pdb_id, uniprot_id, pdb_path]
    if fasta_path:
        hints.append(Path(fasta_path).read_text(encoding="utf-8", errors="replace")[:4000])
        hints.append(Path(fasta_path).stem)
    if pdb_path:
        hints.append(Path(pdb_path).stem)
    guessed = guess_accessions(*hints)
    pid = normalize_pdb_id(pdb_id or "") or (guessed["pdb_ids"][0] if guessed["pdb_ids"] else None)
    acc = normalize_uniprot(uniprot_id or "") or (
        guessed["uniprot_ids"][0] if guessed["uniprot_ids"] else None
    )
    if not pid and not acc:
        if mode in {"auto", ""}:
            return None
        raise FileNotFoundError("Need a PDB id (2L6Q) or UniProt accession to fetch a structure")
    result = fetch_structures(
        pdb_id=pid,
        uniprot_id=acc,
        hints=hints,
        dest_dir=inputs / "_fetch",
        source=mode,
    )
    return install_structures(result, inputs, as_query=as_query)


def enrich_job(job: dict[str, Any], store: RunStore) -> dict[str, Any]:
    """Fill score metadata from a finished rem2 result directory."""
    run_id = job["id"]
    result = Path(job.get("out_dir") or store.result_dir(run_id))
    csv_path = _scores_csv(result)
    sequence = job.get("sequence") or ""
    protein = job.get("protein") or ""
    if not sequence:
        fasta = _first_file(store.inputs_dir(run_id), (".fasta", ".fa", ".faa"))
        if fasta is None:
            fasta = _first_file(result / "_inputs" / "aa_seq", (".fasta", ".fa", ".faa"))
        if fasta is not None:
            protein, sequence = _read_fasta_sequence(fasta)
            protein = protein or fasta.stem
    job["sequence"] = sequence
    if protein:
        job["protein"] = protein
    attach_structure_meta(job, store)
    if csv_path is None:
        job["n_rows"] = job.get("n_rows") or 0
        job["n_mutants"] = job.get("n_mutants") or 0
        job["score_columns"] = job.get("score_columns") or []
        job["primary_score"] = job.get("primary_score")
        job["has_dms"] = bool(job.get("has_dms"))
        return job
    frame = pd.read_csv(csv_path)
    columns = [str(c) for c in frame.columns]
    scores = _score_columns(columns)
    primary = scores[0] if scores else None
    job["n_rows"] = int(len(frame))
    job["n_mutants"] = int(len(frame))
    job["score_columns"] = scores
    job["primary_score"] = primary
    job["has_dms"] = "DMS_score" in frame.columns
    job["proteins"] = [{"name": job.get("protein") or csv_path.stem, "sequence": sequence}]
    meta_path = result / "run_meta.json"
    if meta_path.is_file():
        try:
            import json

            job["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return job


def load_score_frame(job: dict[str, Any], store: RunStore) -> pd.DataFrame:
    result = Path(job.get("out_dir") or store.result_dir(job["id"]))
    csv_path = _scores_csv(result)
    if csv_path is None:
        return pd.DataFrame()
    frame = pd.read_csv(csv_path)
    primary = job.get("primary_score") or ( _score_columns(list(frame.columns)) or [None] )[0]
    if primary and primary in frame.columns:
        ranked = frame.sort_values(primary, ascending=False, kind="mergesort")
        ranks = {idx: i + 1 for i, idx in enumerate(ranked.index)}
        frame = frame.copy()
        frame["rank"] = frame.index.map(ranks)
        frame["score"] = frame[primary]
    return frame


def slice_scores(
    frame: pd.DataFrame,
    *,
    primary: Optional[str],
    offset: int = 0,
    limit: int = 80,
    sort: str = "-score",
    q: str = "",
) -> tuple[list[dict[str, Any]], int]:
    work = frame
    if q:
        needle = q.strip().lower()
        if "mutant" in work.columns:
            work = work[work["mutant"].astype(str).str.lower().str.contains(needle, na=False)]
    descending = sort.startswith("-")
    key = sort[1:] if descending else sort
    if key == "score" and primary and primary in work.columns:
        key = primary
    if key in work.columns:
        work = work.sort_values(key, ascending=not descending, kind="mergesort")
    total = int(len(work))
    chunk = work.iloc[max(offset, 0) : max(offset, 0) + max(limit, 1)]
    rows = chunk.where(pd.notnull(chunk), None).to_dict(orient="records")
    for row in rows:
        for name, value in list(row.items()):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                row[name] = None
    return rows, total


def histogram(frame: pd.DataFrame, primary: Optional[str], bins: int = 24) -> dict[str, Any]:
    if frame.empty or not primary or primary not in frame.columns:
        return {"bins": [], "min": None, "max": None, "primary_score": primary}
    values = pd.to_numeric(frame[primary], errors="coerce").dropna()
    if values.empty:
        return {"bins": [], "min": None, "max": None, "primary_score": primary}
    lo = float(values.min())
    hi = float(values.max())
    if lo == hi:
        return {
            "bins": [{"lo": lo, "hi": hi, "count": int(len(values))}],
            "min": lo,
            "max": hi,
            "primary_score": primary,
        }
    counts = pd.cut(values, bins=bins, include_lowest=True)
    hist = []
    grouped = counts.value_counts(sort=False)
    for interval, count in grouped.items():
        hist.append(
            {
                "lo": float(interval.left),
                "hi": float(interval.right),
                "count": int(count),
            }
        )
    return {"bins": hist, "min": lo, "max": hi, "primary_score": primary}


_PCT_RE = re.compile(r"(?<!\d)(\d{1,3})\s*%")


class LogTee:
    """File-like stdout/stderr that tqdm and rem2 status can write to."""

    encoding = "utf-8"
    errors = "replace"
    closed = False
    name = "<rem2-dashboard>"
    mode = "w"

    def __init__(self, run_id: str, store: RunStore, job: dict[str, Any]):
        self.run_id = run_id
        self.store = store
        self.job = job
        self._chunks: list[str] = []
        self._lock = threading.Lock()
        self._last_flush = 0.0
        self._last_progress = 0.0

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return False

    def fileno(self) -> int:
        raise io.UnsupportedOperation("LogTee has no fileno")

    @property
    def buffer(self) -> "LogTee":
        return self

    def write(self, text) -> int:
        if text is None:
            return 0
        if isinstance(text, bytes):
            text = text.decode(self.encoding, self.errors)
        text = str(text).replace("\r", "\n")
        if not text:
            return 0
        with self._lock:
            self._chunks.append(text)
            now = time.monotonic()
            if now - self._last_flush >= 0.2 or sum(len(c) for c in self._chunks) >= 4096:
                self._flush_unlocked()
            if now - self._last_progress >= 0.6:
                self._maybe_progress_unlocked(text)
                self._last_progress = now
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        if not self._chunks:
            return
        blob = "".join(self._chunks)
        self._chunks.clear()
        self._last_flush = time.monotonic()
        self.store.append_log(self.run_id, blob)

    def _maybe_progress_unlocked(self, text: str) -> None:
        match = _PCT_RE.search(text)
        if not match:
            return
        pct = int(match.group(1))
        if pct > 100:
            return
        mapped = 20 + int(pct * 0.7)
        latest = self.store.load_job(self.run_id) or self.job
        prev = (latest.get("progress") or {}).get("pct") or 0
        if mapped <= prev:
            return
        latest["progress"] = {
            "pct": mapped,
            "stage": "score",
            "message": text.strip().split("\n")[-1][:160],
        }
        self.store.write_job(latest)


def prepare_run(job: dict[str, Any], store: RunStore) -> dict[str, Any]:
    """Fetch structures and rebuild argv before scoring. Must stay off the request thread."""
    spec = dict(job.get("argv_spec") or {})
    if spec.get("base_dir") or job.get("mutant_mode") == "demo":
        set_progress(job, store, 12, "score", "Starting…")
        return store.load_job(job["id"]) or job
    inputs = store.inputs_dir(job["id"])
    uploaded = spec.get("pdb")
    fetch_mode = (job.get("fetch_mode") or "auto").strip().lower() or "auto"
    skip_fetch = fetch_mode in {"none", "off", "0", "false"}
    fetched = None
    if not skip_fetch:
        set_progress(job, store, 6, "fetch", "Looking up RCSB / AlphaFold…")
        try:
            fetched = maybe_fetch_structures(
                inputs=inputs,
                fasta_path=spec.get("fasta"),
                pdb_path=uploaded,
                protein=job.get("protein"),
                pdb_id=job.get("pdb_id"),
                uniprot_id=job.get("uniprot_id"),
                fetch_mode=fetch_mode,
                as_query=not uploaded,
            )
        except Exception as exc:
            fetched = {"errors": [f"{type(exc).__name__}: {exc}"]}
            if not spec.get("fasta") and not uploaded:
                raise
    job = store.load_job(job["id"]) or job
    if fetched:
        job["pdb_id"] = fetched.get("pdb_id") or job.get("pdb_id")
        job["uniprot_id"] = fetched.get("uniprot_id") or job.get("uniprot_id")
        job["fetch_errors"] = fetched.get("errors") or []
    attach_structure_meta(job, store)
    query = uploaded or existing_query_pdb(inputs)
    fasta = spec.get("fasta") or existing_fasta(inputs)
    if not fasta and not query and not spec.get("base_dir"):
        raise FileNotFoundError("No FASTA or PDB to score. Fetch failed or no id was provided.")
    job["argv"] = build_argv(
        model=job.get("model") or "venusrem2",
        recipe=job.get("recipe") or "full",
        out_dir=job.get("out_dir") or str(store.result_dir(job["id"])),
        fasta=fasta,
        pdb=query,
        mutants=spec.get("mutants"),
        msa_dir=spec.get("msa_dir"),
        base_dir=spec.get("base_dir"),
        mutant_sites=spec.get("mutant_sites"),
        positions=spec.get("positions"),
        residue_range=spec.get("residue_range"),
        max_mutants=spec.get("max_mutants"),
        scoring_strategy=spec.get("scoring_strategy"),
    )
    store.write_job(job)
    set_progress(job, store, 18, "score", "Scoring…")
    return store.load_job(job["id"]) or job


def default_execute(job: dict[str, Any], store: RunStore) -> None:
    from rem2.cli import main

    argv = list(job.get("argv") or [])
    if not argv:
        raise RuntimeError("Job has no command line; structure fetch may have failed.")
    os.environ.setdefault("REM2_NO_SPINNER", "1")
    import sys

    tee = LogTee(job["id"], store, job)
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = tee  # type: ignore[assignment]
        sys.stderr = tee  # type: ignore[assignment]
        main(argv)
        tee.flush()
    finally:
        try:
            tee.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = old_out, old_err


class JobRunner:
    def __init__(self, store: RunStore, execute: Optional[ExecuteFn] = None):
        self.store = store
        self.execute = execute or default_execute
        self._queue: queue.Queue[str] = queue.Queue()
        self._cancel: set[str] = set()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def submit(self, run_id: str) -> None:
        self._queue.put(run_id)

    def cancel(self, run_id: str) -> dict[str, Any]:
        job = self.store.load_job(run_id)
        if job is None:
            raise KeyError(run_id)
        with self._lock:
            if job.get("status") in {"queued"}:
                job["status"] = "cancelled"
                job["error"] = "cancelled before start"
                return self.store.write_job(job)
            self._cancel.add(run_id)
        if job.get("status") == "running":
            job["error"] = "cancel requested; the current rem2 process may finish"
            self.store.write_job(job)
        return job

    def _loop(self) -> None:
        while True:
            run_id = self._queue.get()
            job = self.store.load_job(run_id)
            if job is None:
                continue
            with self._lock:
                if run_id in self._cancel or job.get("status") == "cancelled":
                    self._cancel.discard(run_id)
                    job["status"] = "cancelled"
                    self.store.write_job(job)
                    continue
                job["status"] = "running"
                self.store.write_job(job)
            try:
                job = prepare_run(job, self.store)
                self.execute(job, self.store)
                latest = self.store.load_job(run_id) or job
                with self._lock:
                    was_cancel = run_id in self._cancel or latest.get("status") == "cancelled"
                    self._cancel.discard(run_id)
                if was_cancel:
                    latest["status"] = "cancelled"
                    latest["error"] = latest.get("error") or "cancelled"
                    latest["progress"] = {
                        "pct": (latest.get("progress") or {}).get("pct") or 0,
                        "stage": "failed",
                        "message": "Cancelled",
                    }
                    self.store.write_job(latest)
                    continue
                latest["status"] = "done"
                latest["error"] = None
                latest["progress"] = {"pct": 100, "stage": "done", "message": "Done"}
                enrich_job(latest, self.store)
                self.store.write_job(latest)
            except SystemExit as exc:
                latest = self.store.load_job(run_id) or job
                latest["status"] = "failed"
                latest["error"] = str(exc) or "rem2 exited"
                latest["progress"] = {
                    "pct": (latest.get("progress") or {}).get("pct") or 0,
                    "stage": "failed",
                    "message": latest["error"],
                }
                self.store.write_job(latest)
            except Exception as exc:
                latest = self.store.load_job(run_id) or job
                latest["status"] = "failed"
                latest["error"] = f"{type(exc).__name__}: {exc}"
                latest["progress"] = {
                    "pct": (latest.get("progress") or {}).get("pct") or 0,
                    "stage": "failed",
                    "message": latest["error"],
                }
                self.store.write_job(latest)


def build_argv(
    *,
    model: str,
    recipe: str,
    out_dir: str,
    fasta: Optional[str] = None,
    pdb: Optional[str] = None,
    mutants: Optional[str] = None,
    msa_dir: Optional[str] = None,
    base_dir: Optional[str] = None,
    mutant_sites: Optional[str] = None,
    positions: Optional[str] = None,
    residue_range: Optional[str] = None,
    max_mutants: Optional[str] = None,
    scoring_strategy: Optional[str] = None,
) -> list[str]:
    from rem2.api import build_score_argv

    extra: list[str] = list(recipe_argv(recipe))
    extra += ["--auto_download", "--disable_tqdm"]
    if scoring_strategy:
        extra += ["--scoring_strategy", scoring_strategy]
    if mutant_sites and not mutants and not base_dir:
        extra += ["--mutant_sites", mutant_sites]
    if positions:
        extra += ["--positions", positions]
    if residue_range:
        extra += ["--residue_range", residue_range]
    if max_mutants:
        extra += ["--max_mutants", str(max_mutants)]
    if msa_dir:
        extra += ["--aa_seq_aln_dir", msa_dir]
    return build_score_argv(
        fasta=fasta,
        pdb=pdb,
        mutants=mutants,
        base_dir=base_dir,
        model=model,
        out_dir=out_dir,
        extra_argv=extra,
    )
