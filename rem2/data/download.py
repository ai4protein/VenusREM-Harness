"""``rem2 download``: ProteinGym, VenusMutHub, VenusViroHub.

Archives are fetched from ``AI4Protein/VenusREM2``, then
``tyang816/VenusREM2``. ProteinGym still falls back to
``AI4Protein/VenusREM`` and official ProteinGym v1.3.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from rem2.data.mirrors import (
    LEGACY_PROTEINGYM_REPO,
    VENUSREM2_REPOS,
    download_from_venusrem2,
)
from rem2.data import proteingym as pg

ALIASES = {
    "proteingym": "proteingym",
    "proteingymv1": "proteingym",
    "pg": "proteingym",
    "muthub": "muthub",
    "venusmuthub": "muthub",
    "vmh": "muthub",
    "virohub": "virohub",
    "venusvirohub": "virohub",
    "vvh": "virohub",
    "all": "all",
}

DATASETS = {
    "proteingym": {
        "dest": "data/proteingym_v1",
        "folder": "ProteinGym",
        "expected": 217,
        "archives": {
            "aa_seq": ("aa_seq.tar.gz", (".fasta", ".fa")),
            "substitutions": ("substitutions.tar.gz", (".csv",)),
            "pdbs": ("pdbs.tar.gz", (".pdb",)),
            "aa_seq_aln_a2m": ("aa_seq_aln_a2m.tar.gz", (".a2m", ".a3m", ".fasta")),
            "aa_seq_aln_a3m": ("aa_seq_aln_a3m.tar.gz", (".a3m", ".a2m", ".fasta")),
        },
        "sidecar": (),
    },
    "muthub": {
        "dest": "data/VenusMutHub",
        "folder": "VenusMutHub",
        "expected": 905,
        "archives": {
            "aa_seq": ("aa_seq.tar.gz", (".fasta", ".fa")),
            "substitutions": ("substitutions.tar.gz", (".csv",)),
            "pdbs": ("pdbs.tar.gz", (".pdb",)),
            "aa_seq_aln_a2m": ("aa_seq_aln_a2m.tar.gz", (".a2m", ".a3m", ".fasta")),
        },
        "sidecar": ("assay_manifest.csv",),
    },
    "virohub": {
        "dest": "data/venusvirohub",
        "folder": "VenusViroHub",
        "expected": 89,
        "archives": {
            "aa_seq": ("aa_seq.tar.gz", (".fasta", ".fa")),
            "substitutions": ("substitutions.tar.gz", (".csv",)),
            "pdbs": ("pdbs.tar.gz", (".pdb",)),
            "aa_seq_aln_a2m": ("aa_seq_aln_a2m.tar.gz", (".a2m", ".a3m", ".fasta")),
        },
        "sidecar": ("DMS_substitutions.csv",),
    },
}


def _fold_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def normalize_dataset(name: str) -> str:
    key = _fold_name(name)
    if key not in ALIASES:
        known = "ProteinGym | VenusMutHub | VenusViroHub | all"
        raise SystemExit(f"Unknown dataset {name!r}. Use: rem2 download [{known}]")
    return ALIASES[key]


def default_dest(dataset: str) -> Path:
    return Path.cwd() / DATASETS[dataset]["dest"]


def venusrem2_path(dataset: str, filename: str) -> str:
    return f"{DATASETS[dataset]['folder']}/{filename}"


def _extract_archive(archive: Path, dest: Path, log) -> None:
    log(f"Extracting {archive.name}")
    pg.safe_extract_tar(archive, dest)


def _fill_from_venusrem2(
    dataset: str,
    dest: Path,
    cache: Path,
    folder: str,
    archive_name: str,
    suffixes: tuple[str, ...],
    expected: int,
    force: bool,
    log,
) -> Optional[int]:
    out = dest / folder
    if (not force) and pg.count_files(out, suffixes) >= expected:
        log(f"{folder}/ already has {pg.count_files(out, suffixes)} files")
        return pg.count_files(out, suffixes)
    remote = venusrem2_path(dataset, archive_name)
    archive = download_from_venusrem2(remote, cache / archive_name, force=force, log=log)
    if archive is None:
        return None
    _extract_archive(archive, dest, log)
    return pg.count_files(out, suffixes)


def download_hub_dataset(
    dataset: str,
    dest: Path,
    force: bool = False,
    dry_run: bool = False,
    msa: str = "a2m",
    log=print,
) -> dict[str, int]:
    spec = DATASETS[dataset]
    dest = Path(dest)
    cache = dest / "_downloads"
    counts: dict[str, int] = {}
    expected = int(spec["expected"])
    archives = dict(spec["archives"])

    if dataset == "proteingym":
        if msa == "none":
            archives.pop("aa_seq_aln_a2m", None)
            archives.pop("aa_seq_aln_a3m", None)
        elif msa == "a2m":
            archives.pop("aa_seq_aln_a3m", None)
        elif msa == "a3m":
            archives.pop("aa_seq_aln_a2m", None)

    if dry_run:
        log(f"dataset: {dataset}  dest: {dest}")
        log("VenusREM2 mirrors (first that has the file):")
        for repo in VENUSREM2_REPOS:
            log(f"  {repo}")
        for folder, (archive_name, _suf) in archives.items():
            log(f"  {venusrem2_path(dataset, archive_name)}")
        for name in spec["sidecar"]:
            log(f"  {venusrem2_path(dataset, name)}")
        if dataset == "proteingym":
            log(f"legacy ProteinGym MSA: {LEGACY_PROTEINGYM_REPO}")
            log(f"fallback substitutions: {pg.PG_SUBSTITUTIONS}")
            log(f"fallback AF2 PDBs: {pg.PG_STRUCTURES}")
        return counts

    dest.mkdir(parents=True, exist_ok=True)
    for folder, (archive_name, suffixes) in archives.items():
        n = _fill_from_venusrem2(
            dataset, dest, cache, folder, archive_name, suffixes, expected, force, log
        )
        if n is None:
            if dataset == "proteingym":
                continue
            raise SystemExit(
                f"Could not download {venusrem2_path(dataset, archive_name)} "
                f"from {' or '.join(VENUSREM2_REPOS)}. "
                "Private repos need HF_TOKEN."
            )
        counts[folder] = n

    for name in spec["sidecar"]:
        target = dest / name
        remote = venusrem2_path(dataset, name)
        got = download_from_venusrem2(remote, target, force=force, log=log)
        if got is None and dataset != "proteingym":
            log(f"optional sidecar missing: {remote}")
        elif got is not None:
            counts[name] = 1

    if dataset == "proteingym":
        _fill_proteingym_gaps(dest, cache, force, msa, counts, log)

    log(f"{dataset} layout:")
    for key, n in counts.items():
        log(f"  {key:<20} {n} files")
    log(f"Next: rem2 --model esm2 --base_dir {dest}")
    return counts


def _fill_proteingym_gaps(
    dest: Path,
    cache: Path,
    force: bool,
    msa: str,
    counts: dict[str, int],
    log,
) -> None:
    reference = pg.ensure_reference(dest, force=force)
    log(f"Reference table: {reference}")
    if counts.get("substitutions", 0) < pg.EXPECTED_ASSAYS:
        counts["substitutions"] = pg._fill_substitutions(dest, cache, force, log)
    if counts.get("aa_seq", 0) < pg.EXPECTED_ASSAYS:
        counts["aa_seq"] = pg._fill_aa_seq(dest, cache, reference, force, log)
    if counts.get("pdbs", 0) < pg.EXPECTED_ASSAYS:
        counts["pdbs"] = pg._fill_pdbs(dest, cache, reference, force, log)
    if msa in {"a2m", "both"} and counts.get("aa_seq_aln_a2m", 0) < pg.EXPECTED_ASSAYS:
        counts["aa_seq_aln_a2m"] = pg._fill_hf_folder(
            dest, cache, "aa_seq_aln_a2m", (".a2m", ".a3m", ".fasta"), force, log
        )
    if msa in {"a3m", "both"} and counts.get("aa_seq_aln_a3m", 0) < pg.EXPECTED_ASSAYS:
        counts["aa_seq_aln_a3m"] = pg._fill_hf_folder(
            dest, cache, "aa_seq_aln_a3m", (".a3m", ".a2m", ".fasta"), force, log
        )


def build_download_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rem2 download",
        description=(
            "Fetch ProteinGym, VenusMutHub, or VenusViroHub into a rem2 --base_dir. "
            "Looks in AI4Protein/VenusREM2 then tyang816/VenusREM2. "
            "Private datasets need HF_TOKEN. ProteinGym still falls back to "
            "AI4Protein/VenusREM and official ProteinGym v1.3."
        ),
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="proteingym",
        help=(
            "ProteinGym | VenusMutHub | VenusViroHub | all "
            "(case-insensitive; muthub / virohub aliases; default: ProteinGym)"
        ),
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="output directory (default depends on dataset)",
    )
    parser.add_argument(
        "--msa",
        choices=["a2m", "a3m", "both", "none"],
        default="a2m",
        help="ProteinGym MSA archive (default: a2m); ignored for other datasets",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--dry-run", action="store_true", help="print sources and exit")
    return parser


def run_download(argv: Optional[Iterable[str]] = None) -> int:
    args = build_download_parser().parse_args(list(argv or []))
    names = normalize_dataset(args.dataset)
    selected = list(DATASETS) if names == "all" else [names]
    if args.dest and len(selected) > 1:
        raise SystemExit("--dest cannot be used with rem2 download all")
    for name in selected:
        dest = Path(args.dest) if args.dest else default_dest(name)
        download_hub_dataset(
            name,
            dest,
            force=args.force,
            dry_run=args.dry_run,
            msa=args.msa,
        )
    return 0
