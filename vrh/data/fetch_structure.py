"""Download experimental PDBs (RCSB) and AlphaFold models (AFDB).

Crystal / NMR / cryo-EM files have no pLDDT — the B column is a temperature
factor. vrh already skips pLDDT decay on those. This helper fetches an
AlphaFold model when a UniProt accession can be inferred so the dashboard can
color by pLDDT and scoring can use it.
"""

from __future__ import annotations

import json
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from vrh.models.weights import default_cache_dir
from vrh.scoring.structure_weights import classify_pdb_origin, plddt_skip_reason

RCSB_PDB = "https://files.rcsb.org/download/{id}.pdb"
PDBE_UNIPROT = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{id}"
AFDB_PDB = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_{ver}.pdb"
AFDB_VERSIONS = ("v4", "v6", "v3")

# Official UniProt accession pattern (not a 4-char PDB id).
_UNIPROT = re.compile(
    r"(?:AF-)?([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-F\d)?",
    re.IGNORECASE,
)
_SP_TR = re.compile(r"\b(?:sp|tr)\|([A-Z0-9]{6,10})\|", re.IGNORECASE)
_PDB_TOKEN = re.compile(r"(?:^|[\s:/=_|])([0-9][A-Za-z0-9]{3})(?:$|[\s,;|])")
_YEAR = re.compile(r"^20[0-2][0-9]$")


def structure_cache_dir(cache_dir: Optional[str] = None) -> Path:
    root = Path(default_cache_dir(cache_dir)) / "structures"
    root.mkdir(parents=True, exist_ok=True)
    return root


def normalize_pdb_id(value: str) -> Optional[str]:
    raw = (value or "").strip()
    if raw.lower().endswith(".pdb"):
        raw = Path(raw).stem
    if "_" in raw:
        raw = raw.rsplit("_", 1)[-1]
    raw = raw.strip()
    if len(raw) != 4 or not raw[0].isdigit() or _YEAR.match(raw):
        return None
    if not re.fullmatch(r"[0-9][A-Za-z0-9]{3}", raw):
        return None
    return raw.upper()


def normalize_uniprot(value: str) -> Optional[str]:
    raw = (value or "").strip()
    if raw.upper().startswith("AF-"):
        raw = raw[3:]
    raw = raw.split("-")[0]
    m = _UNIPROT.fullmatch(raw)
    if not m:
        m = _UNIPROT.search(raw)
    if not m:
        return None
    return m.group(1).upper()


def guess_accessions(*texts: Optional[str]) -> dict[str, list[str]]:
    """Pull PDB / UniProt ids from FASTA headers, filenames, and assay names."""
    pdb_ids: list[str] = []
    uniprot_ids: list[str] = []
    blob = "\n".join(str(item) for item in texts if item)
    for m in _SP_TR.finditer(blob):
        acc = normalize_uniprot(m.group(1))
        if acc and acc not in uniprot_ids:
            uniprot_ids.append(acc)
    for m in _UNIPROT.finditer(blob):
        acc = normalize_uniprot(m.group(1))
        if acc and acc not in uniprot_ids:
            uniprot_ids.append(acc)
    for line in blob.splitlines():
        token = line[1:].split()[0] if line.startswith(">") else line.strip()
        tail = normalize_pdb_id(token)
        if tail and tail not in pdb_ids:
            pdb_ids.append(tail)
        for m in _PDB_TOKEN.finditer(token.replace("-", "_")):
            pid = normalize_pdb_id(m.group(1))
            if pid and pid not in pdb_ids:
                pdb_ids.append(pid)
    return {"pdb_ids": pdb_ids, "uniprot_ids": uniprot_ids}


def _download(url: str, dest: Path, timeout: float = 40) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "vrh-dashboard/0.2"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = response.read()
    if not data or len(data) < 80:
        raise OSError(f"empty download from {url}")
    text = data[:200].decode("utf-8", errors="replace").lstrip()
    looks_html = text.startswith("<") or text.lower().startswith("<!doctype")
    looks_json = text.startswith("{") and b"ATOM" not in data[:4000] and b"HETATM" not in data[:4000]
    if looks_html or looks_json:
        raise OSError(f"not a coordinate file: {url}")
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def fetch_rcsb_pdb(
    pdb_id: str,
    dest: Optional[Path] = None,
    *,
    cache_dir: Optional[str] = None,
    opener: Callable = _download,
) -> Path:
    pid = normalize_pdb_id(pdb_id)
    if not pid:
        raise ValueError(f"Not a PDB id: {pdb_id!r}")
    dest = Path(dest) if dest else structure_cache_dir(cache_dir) / "rcsb" / f"{pid}.pdb"
    if dest.is_file() and dest.stat().st_size > 80:
        return dest
    return opener(RCSB_PDB.format(id=pid), dest)


def fetch_afdb_pdb(
    uniprot: str,
    dest: Optional[Path] = None,
    *,
    cache_dir: Optional[str] = None,
    opener: Callable = _download,
) -> Path:
    acc = normalize_uniprot(uniprot)
    if not acc:
        raise ValueError(f"Not a UniProt accession: {uniprot!r}")
    dest = Path(dest) if dest else structure_cache_dir(cache_dir) / "afdb" / f"AF-{acc}-F1.pdb"
    if dest.is_file() and dest.stat().st_size > 80:
        return dest
    last: Optional[BaseException] = None
    for ver in AFDB_VERSIONS:
        try:
            return opener(AFDB_PDB.format(acc=acc, ver=ver), dest)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError) as exc:
            last = exc
            continue
    raise OSError(f"AlphaFold model not found for {acc}") from last


def uniprot_for_pdb(
    pdb_id: str,
    *,
    timeout: float = 20,
) -> Optional[str]:
    pid = normalize_pdb_id(pdb_id)
    if not pid:
        return None
    url = PDBE_UNIPROT.format(id=pid.lower())
    req = urllib.request.Request(url, headers={"User-Agent": "vrh-dashboard/0.2"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None
    block = payload.get(pid.lower()) or payload.get(pid) or next(iter(payload.values()), None)
    if not isinstance(block, dict):
        return None
    mapping = (block.get("UniProt") or block.get("uniprot") or {})
    if isinstance(mapping, dict) and mapping:
        return normalize_uniprot(next(iter(mapping.keys())))
    return None


def describe_pdb(path: Path) -> dict:
    kind, detail = classify_pdb_origin(str(path))
    skip = plddt_skip_reason(str(path))
    return {
        "path": str(path),
        "origin": kind,
        "detail": detail,
        "has_plddt": skip is None and kind != "experimental",
    }


def fetch_structures(
    *,
    pdb_id: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    hints: Optional[list[str]] = None,
    dest_dir: Optional[Path] = None,
    source: str = "auto",
    cache_dir: Optional[str] = None,
    opener: Callable = _download,
    map_uniprot: Callable = uniprot_for_pdb,
) -> dict:
    """Download RCSB and/or AFDB models. ``source``: auto | rcsb | afdb | both."""
    guessed = guess_accessions(*(hints or []), pdb_id, uniprot_id)
    pid = normalize_pdb_id(pdb_id or "") or (guessed["pdb_ids"][0] if guessed["pdb_ids"] else None)
    acc = normalize_uniprot(uniprot_id or "") or (
        guessed["uniprot_ids"][0] if guessed["uniprot_ids"] else None
    )
    source = (source or "auto").strip().lower()
    want_rcsb = source in {"auto", "rcsb", "both", "pdb"}
    want_afdb = source in {"auto", "afdb", "both", "alphafold"}
    dest_dir = Path(dest_dir) if dest_dir else structure_cache_dir(cache_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    fetched: dict[str, dict] = {}
    errors: list[str] = []

    if want_rcsb and pid:
        try:
            path = fetch_rcsb_pdb(pid, dest_dir / f"{pid}.pdb", cache_dir=cache_dir, opener=opener)
            meta = describe_pdb(path)
            meta.update({"id": pid, "kind": "rcsb", "label": f"RCSB {pid}"})
            fetched["rcsb"] = meta
        except Exception as exc:
            errors.append(f"RCSB {pid}: {exc}")

    if want_afdb and not acc and pid:
        acc = map_uniprot(pid)

    if want_afdb and acc:
        try:
            path = fetch_afdb_pdb(
                acc, dest_dir / f"AF-{acc}-F1.pdb", cache_dir=cache_dir, opener=opener
            )
            meta = describe_pdb(path)
            meta.update({"id": acc, "kind": "afdb", "label": f"AFDB {acc}"})
            fetched["afdb"] = meta
        except Exception as exc:
            errors.append(f"AFDB {acc}: {exc}")

    if not fetched:
        raise FileNotFoundError(
            "Could not download a structure. "
            + ("; ".join(errors) if errors else "Pass a PDB id (2L6Q) or UniProt accession.")
        )

    preferred = fetched.get("afdb") or fetched.get("rcsb")
    return {
        "pdb_id": pid,
        "uniprot_id": acc,
        "guessed": guessed,
        "sources": fetched,
        "preferred": preferred,
        "errors": errors,
    }


def _copy_if_needed(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def install_structures(result: dict, dest_dir: Path, *, as_query: bool = True) -> dict:
    """Copy fetched RCSB / AFDB files into a run ``inputs/`` directory."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    installed: dict[str, dict] = {}
    names = {"rcsb": "rcsb.pdb", "afdb": "afdb.pdb"}
    for key, meta in (result.get("sources") or {}).items():
        src = Path(meta["path"])
        if not src.is_file():
            continue
        target = _copy_if_needed(src, dest_dir / names.get(key, f"{key}.pdb"))
        item = describe_pdb(target)
        item.update(
            {
                "id": meta.get("id"),
                "kind": key,
                "label": meta.get("label") or key.upper(),
            }
        )
        installed[key] = item
    if as_query:
        preferred = result.get("preferred") or installed.get("afdb") or installed.get("rcsb")
        key = (preferred or {}).get("kind")
        src_name = names.get(key or "", "")
        src = dest_dir / src_name if src_name else None
        if src is not None and src.is_file():
            query = _copy_if_needed(src, dest_dir / "query.pdb")
            item = describe_pdb(query)
            item.update(
                {
                    "id": preferred.get("id") if preferred else None,
                    "kind": "query",
                    "label": (preferred or {}).get("label") or "query",
                }
            )
            installed["query"] = item
    out = dict(result)
    out["installed"] = installed
    return out


def public_structure(meta: dict) -> dict:
    return {
        "kind": meta.get("kind"),
        "id": meta.get("id"),
        "label": meta.get("label"),
        "origin": meta.get("origin"),
        "detail": meta.get("detail"),
        "has_plddt": bool(meta.get("has_plddt")),
        "name": Path(str(meta.get("path") or "")).name or None,
    }
