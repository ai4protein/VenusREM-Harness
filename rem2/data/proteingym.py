"""Download ProteinGym substitutions into the rem2 --base_dir layout.

v1 listed these Hugging Face archives under ``AI4Protein/VenusREM``::

    aa_seq.tar.gz  aa_seq_aln_a2m.tar.gz  pdbs.tar.gz
    struc_seq.tar.gz  substitutions.tar.gz

Only the MSA tarballs are public today. Missing pieces fall back to
ProteinGym v1.3 (substitutions + AF2 PDBs). Sequences are written from the
reference table. ProSST tokens are optional: rem2 builds them from PDB.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import stat
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional

HF_DATASET = "AI4Protein/VenusREM"
HF_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"
HF_ARCHIVES = {
    "aa_seq": "aa_seq.tar.gz",
    "substitutions": "substitutions.tar.gz",
    "pdbs": "pdbs.tar.gz",
    "struc_seq": "struc_seq.tar.gz",
    "aa_seq_aln_a2m": "aa_seq_aln_a2m.tar.gz",
    "aa_seq_aln_a3m": "aa_seq_aln_a3m.tar.gz",
}
PG_VERSION = "v1.3"
PG_BASE = f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{PG_VERSION}"
PG_SUBSTITUTIONS = f"{PG_BASE}/DMS_ProteinGym_substitutions.zip"
PG_STRUCTURES = f"{PG_BASE}/ProteinGym_AF2_structures.zip"
PG_REFERENCE = (
    "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/"
    "main/reference_files/DMS_substitutions.csv"
)
EXPECTED_ASSAYS = 217


def bundled_reference_csv() -> Optional[Path]:
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "data" / "proteingym_v1" / "DMS_substitutions.csv"
        if cand.is_file():
            return cand
    return None


def default_dest() -> Path:
    return Path.cwd() / "data" / "proteingym_v1"


def count_files(directory: Path, suffixes: tuple[str, ...]) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for name in os.listdir(directory) if name.endswith(suffixes))


def load_reference_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_aa_seq_from_reference(reference: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in load_reference_rows(reference):
        name = (row.get("DMS_id") or "").strip()
        seq = (row.get("target_seq") or "").strip()
        if not name or not seq:
            continue
        fasta = dest / f"{name}.fasta"
        fasta.write_text(f">{name}\n{seq}\n", encoding="utf-8")
        written += 1
    return written


def map_pdbs_to_assays(extracted: Path, reference: Path, dest: Path) -> int:
    """Copy AF2 PDBs onto assay stems used by substitutions/ and aa_seq/."""
    dest.mkdir(parents=True, exist_ok=True)
    by_name = {path.name: path for path in _iter_files(extracted, (".pdb",))}
    written = 0
    for row in load_reference_rows(reference):
        assay = (row.get("DMS_id") or "").strip()
        if not assay:
            continue
        pdb_name = (row.get("pdb_file") or "").strip() or f"{assay}.pdb"
        src = by_name.get(f"{assay}.pdb") or by_name.get(pdb_name)
        if src is None:
            src = by_name.get(Path(pdb_name).name)
        if src is None:
            continue
        target = dest / f"{assay}.pdb"
        if not target.exists() or not target.samefile(src):
            shutil.copy2(src, target)
        written += 1
    return written


def copy_named_files(extracted: Path, dest: Path, suffixes: tuple[str, ...]) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    written = 0
    for src in _iter_files(extracted, suffixes):
        target = dest / src.name
        if target.resolve() != src.resolve():
            shutil.copy2(src, target)
        written += 1
    return written


def _iter_files(root: Path, suffixes: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(suffixes):
                files.append(Path(dirpath) / name)
    return sorted(files)


def safe_extract_tar(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()
    with tarfile.open(archive, "r:*") as handle:
        for member in handle.getmembers():
            _assert_safe_member(dest, member.name)
            if member.issym() or member.islnk():
                raise ValueError(f"Refusing archive symlink: {member.name}")
        kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
        handle.extractall(dest, **kwargs)
    _reject_escaping_symlinks(dest)
    return dest


def safe_extract_zip(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            _assert_safe_member(dest, info.filename)
            if _zip_is_symlink(info):
                raise ValueError(f"Refusing archive symlink: {info.filename}")
        handle.extractall(dest)
    _reject_escaping_symlinks(dest)
    return dest


def _assert_safe_member(dest: Path, name: str) -> None:
    dest = dest.resolve()
    raw = str(name).replace("\\", "/")
    if not raw or raw.startswith("/") or raw.startswith("\\"):
        raise ValueError(f"Refusing archive member outside dest: {name}")
    if len(raw) >= 2 and raw[1] == ":":
        raise ValueError(f"Refusing archive member outside dest: {name}")
    target = Path(os.path.normpath(dest / raw))
    if dest != target and dest not in target.parents:
        raise ValueError(f"Refusing archive member outside dest: {name}")


def _zip_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = info.external_attr >> 16
    return bool(mode and stat.S_ISLNK(mode))


def _reject_escaping_symlinks(dest: Path) -> None:
    dest = dest.resolve()
    for dirpath, dirnames, filenames in os.walk(dest, followlinks=False):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            if not path.is_symlink():
                continue
            try:
                resolved = path.resolve()
            except OSError as exc:
                raise ValueError(f"Refusing broken archive symlink: {path}") from exc
            if dest != resolved and dest not in resolved.parents:
                raise ValueError(f"Refusing archive symlink outside dest: {path}")


def download_url(url: str, dest: Path, force: bool = False, *, desc: Optional[str] = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    from rem2.download.progress import download_url_with_progress

    return download_url_with_progress(url, dest, desc=desc or dest.name, force=force)


def download_hf_archive(filename: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        return dest
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return download_url(f"{HF_BASE}/{filename}", dest, force=force)
    path = hf_hub_download(
        repo_id=HF_DATASET,
        filename=filename,
        repo_type="dataset",
        local_dir=str(dest.parent),
        force_download=force,
    )
    downloaded = Path(path)
    if downloaded.resolve() != dest.resolve():
        shutil.copy2(downloaded, dest)
    return dest


def hf_archive_available(filename: str) -> bool:
    req = urllib.request.Request(f"{HF_BASE}/{filename}", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return 200 <= getattr(response, "status", 200) < 300
    except urllib.error.HTTPError as exc:
        return exc.code == 200
    except Exception:
        return False


def ensure_reference(dest: Path, force: bool = False) -> Path:
    local = dest / "DMS_substitutions.csv"
    if local.is_file() and not force:
        return local
    bundled = bundled_reference_csv()
    if bundled is not None and not force:
        if local.resolve() != bundled.resolve():
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundled, local)
        return local
    download_url(PG_REFERENCE, local, force=force)
    return local


def _skip_dir(path: Path, suffixes: tuple[str, ...], force: bool) -> bool:
    return (not force) and count_files(path, suffixes) >= EXPECTED_ASSAYS


def download_proteingym(
    dest: Path,
    msa: str = "a2m",
    force: bool = False,
    dry_run: bool = False,
    log=print,
) -> dict[str, int]:
    dest = Path(dest)
    cache = dest / "_downloads"
    counts: dict[str, int] = {}
    plan: list[str] = []

    want_a2m = msa in {"a2m", "both"}
    want_a3m = msa in {"a3m", "both"}

    if dry_run:
        plan.append(f"dest: {dest}")
        plan.append(f"HF dataset: {HF_DATASET} (v1 MSA + optional archives)")
        plan.append(f"fallback substitutions: {PG_SUBSTITUTIONS}")
        plan.append(f"fallback AF2 PDBs: {PG_STRUCTURES}")
        if want_a2m:
            plan.append(f"MSA a2m: {HF_BASE}/{HF_ARCHIVES['aa_seq_aln_a2m']}")
        if want_a3m:
            plan.append(f"MSA a3m: {HF_BASE}/{HF_ARCHIVES['aa_seq_aln_a3m']}")
        log("\n".join(plan))
        return counts

    dest.mkdir(parents=True, exist_ok=True)
    reference = ensure_reference(dest, force=force)
    log(f"Reference table: {reference}")

    counts["substitutions"] = _fill_substitutions(dest, cache, force, log)
    counts["aa_seq"] = _fill_aa_seq(dest, cache, reference, force, log)
    counts["pdbs"] = _fill_pdbs(dest, cache, reference, force, log)
    if want_a2m:
        counts["aa_seq_aln_a2m"] = _fill_hf_folder(
            dest, cache, "aa_seq_aln_a2m", (".a2m", ".a3m", ".fasta"), force, log
        )
    if want_a3m:
        counts["aa_seq_aln_a3m"] = _fill_hf_folder(
            dest, cache, "aa_seq_aln_a3m", (".a3m", ".a2m", ".fasta"), force, log
        )
    if hf_archive_available(HF_ARCHIVES["struc_seq"]) or (
        dest / "struc_seq"
    ).is_dir():
        counts["struc_seq"] = _fill_hf_folder(
            dest, cache, "struc_seq", (".fasta",), force, log
        )
    else:
        log("struc_seq/ not on Hugging Face; rem2 will build ProSST tokens from pdbs/")

    log("ProteinGym layout:")
    for key, n in counts.items():
        log(f"  {key:<16} {n} files")
    log(f"Next: rem2 --model esm2 --base_dir {dest}")
    return counts


def _fill_substitutions(dest: Path, cache: Path, force: bool, log) -> int:
    out = dest / "substitutions"
    if _skip_dir(out, (".csv",), force):
        log(f"substitutions/ already has {count_files(out, ('.csv',))} CSVs")
        return count_files(out, (".csv",))
    archive_name = HF_ARCHIVES["substitutions"]
    if hf_archive_available(archive_name):
        archive = download_hf_archive(archive_name, cache / archive_name, force=force)
        log(f"Extracting {archive_name}")
        safe_extract_tar(archive, dest)
        return count_files(out, (".csv",))
    log("Hugging Face substitutions.tar.gz is missing; using ProteinGym v1.3")
    archive = download_url(PG_SUBSTITUTIONS, cache / "DMS_ProteinGym_substitutions.zip", force=force)
    extracted = cache / "DMS_ProteinGym_substitutions"
    if extracted.exists() and force:
        shutil.rmtree(extracted)
    if not extracted.exists() or force:
        safe_extract_zip(archive, extracted)
    return copy_named_files(extracted, out, (".csv",))


def _fill_aa_seq(dest: Path, cache: Path, reference: Path, force: bool, log) -> int:
    out = dest / "aa_seq"
    if _skip_dir(out, (".fasta", ".fa"), force):
        log(f"aa_seq/ already has {count_files(out, ('.fasta', '.fa'))} FASTA files")
        return count_files(out, (".fasta", ".fa"))
    archive_name = HF_ARCHIVES["aa_seq"]
    if hf_archive_available(archive_name):
        archive = download_hf_archive(archive_name, cache / archive_name, force=force)
        log(f"Extracting {archive_name}")
        safe_extract_tar(archive, dest)
        return count_files(out, (".fasta", ".fa"))
    log("Hugging Face aa_seq.tar.gz is missing; writing FASTA from the reference table")
    return write_aa_seq_from_reference(reference, out)


def _fill_pdbs(dest: Path, cache: Path, reference: Path, force: bool, log) -> int:
    out = dest / "pdbs"
    if _skip_dir(out, (".pdb",), force):
        log(f"pdbs/ already has {count_files(out, ('.pdb',))} PDBs")
        return count_files(out, (".pdb",))
    archive_name = HF_ARCHIVES["pdbs"]
    if hf_archive_available(archive_name):
        archive = download_hf_archive(archive_name, cache / archive_name, force=force)
        log(f"Extracting {archive_name}")
        safe_extract_tar(archive, dest)
        return count_files(out, (".pdb",))
    log("Hugging Face pdbs.tar.gz is missing; using ProteinGym AF2 structures")
    archive = download_url(PG_STRUCTURES, cache / "ProteinGym_AF2_structures.zip", force=force)
    extracted = cache / "ProteinGym_AF2_structures"
    if extracted.exists() and force:
        shutil.rmtree(extracted)
    if not extracted.exists() or force:
        safe_extract_zip(archive, extracted)
    n = map_pdbs_to_assays(extracted, reference, out)
    if n == 0:
        n = copy_named_files(extracted, out, (".pdb",))
    return n


def _fill_hf_folder(
    dest: Path,
    cache: Path,
    folder: str,
    suffixes: tuple[str, ...],
    force: bool,
    log,
) -> int:
    out = dest / folder
    if _skip_dir(out, suffixes, force):
        log(f"{folder}/ already has {count_files(out, suffixes)} files")
        return count_files(out, suffixes)
    archive_name = HF_ARCHIVES[folder]
    if not hf_archive_available(archive_name):
        log(f"Hugging Face {archive_name} is not available")
        return count_files(out, suffixes)
    archive = download_hf_archive(archive_name, cache / archive_name, force=force)
    log(f"Extracting {archive_name} (~this can take a few minutes)")
    safe_extract_tar(archive, dest)
    return count_files(out, suffixes)


def build_download_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rem2 download",
        description=(
            "Fetch ProteinGym substitutions into data/proteingym_v1. "
            "MSAs come from Hugging Face (AI4Protein/VenusREM); "
            "substitutions and AF2 PDBs fall back to ProteinGym v1.3."
        ),
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="proteingym",
        help="only proteingym is supported",
    )
    parser.add_argument(
        "--dest",
        default=str(default_dest()),
        help="output directory (default: data/proteingym_v1)",
    )
    parser.add_argument(
        "--msa",
        choices=["a2m", "a3m", "both", "none"],
        default="a2m",
        help="which VenusREM MSA archive to fetch (default: a2m)",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--dry-run", action="store_true", help="print sources and exit")
    return parser


def run_download(argv: Optional[Iterable[str]] = None) -> int:
    from rem2.data.download import run_download as run_hub_download

    return run_hub_download(argv)
