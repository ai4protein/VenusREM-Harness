"""``rem2 download``: benchmarks and model weights.

Archives are fetched from ``AI4Protein/VenusREM2``, then
``tyang816/VenusREM2``. ProteinGym still falls back to
``AI4Protein/VenusREM`` and official ProteinGym v1.3.

Model checkpoints go to the Hugging Face hub cache and
``~/.cache/rem2/weights``.
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
from rem2.download.progress import print_plan

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
    "benchmarkall": "all",
    "benchmarks": "all",
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
        known = "ProteinGym | VenusMutHub | VenusViroHub | benchmark-all"
        raise SystemExit(f"Unknown dataset {name!r}. Use: rem2 download [{known}]")
    return ALIASES[key]


def _known_targets_text() -> str:
    return (
        "benchmarks: ProteinGym | VenusMutHub | VenusViroHub | benchmark-all\n"
        "example:    rem2 download example  (ProteinGym HCP_LAMBD_Tsuboyama_2023_2L6Q)\n"
        "models:     esm2 | venusrem2 | saprot | … | model-all\n"
        "            rem2 download --help"
    )


def resolve_download_target(name: str) -> tuple[str, str]:
    """Return ``(\"benchmark\"|\"model\", key)``."""
    key = _fold_name(name)
    if key in {"modelall", "models"}:
        return "model", "all"
    if key == "model":
        return "model", "list"
    from rem2.download.example import EXAMPLE_ALIASES, EXAMPLE_NAME

    if key in EXAMPLE_ALIASES:
        return "example", EXAMPLE_NAME
    if key in ALIASES:
        return "benchmark", ALIASES[key]
    from rem2.download.models import resolve_model_key

    model = resolve_model_key(name)
    if model:
        return "model", model
    raise SystemExit(f"Unknown download target {name!r}.\n{_known_targets_text()}")


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
        n = pg.count_files(out, suffixes)
        from rem2.download.progress import tqdm_bar

        with tqdm_bar(f"{folder}/ (cached, {n} files)", 1, unit="file") as bar:
            bar.update(1)
        log(f"{folder}/ already has {n} files")
        return n
    remote = venusrem2_path(dataset, archive_name)
    archive = download_from_venusrem2(
        remote,
        cache / archive_name,
        force=force,
        log=log,
        progress=True,
        desc=archive_name,
    )
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
    jobs = [(folder, archive_name, suffixes) for folder, (archive_name, suffixes) in archives.items()]
    sidecars = list(spec["sidecar"])
    n_jobs = len(jobs) + len(sidecars)
    log(f"{dataset} → {dest}  ({n_jobs} file(s), {expected} assays)")
    print_plan(
        f"{dataset} files",
        [
            (archive_name, "archive", venusrem2_path(dataset, archive_name))
            for _folder, archive_name, _suf in jobs
        ]
        + [(name, "sidecar", venusrem2_path(dataset, name)) for name in sidecars],
        log=log,
    )
    for i, (folder, archive_name, suffixes) in enumerate(jobs, 1):
        log(f"[{i}/{n_jobs}] {dataset}  {archive_name}")
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

    for j, name in enumerate(sidecars, len(jobs) + 1):
        log(f"[{j}/{n_jobs}] {dataset}  {name}")
        target = dest / name
        remote = venusrem2_path(dataset, name)
        got = download_from_venusrem2(
            remote, target, force=force, log=log, progress=True, desc=name
        )
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Fetch a benchmark into a rem2 --base_dir, or prefetch model weights. "
            "Benchmarks: rem2 download benchmark-all "
            "(AI4Protein/VenusREM2 then tyang816/VenusREM2; "
            "private repos need HF_TOKEN; ProteinGym still falls back to "
            "AI4Protein/VenusREM and official ProteinGym v1.3). "
            "Models: rem2 download model-all "
            "(Hugging Face hub cache + ~/.cache/rem2/weights)."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        default="proteingym",
        help=(
            "ProteinGym | VenusMutHub | VenusViroHub | benchmark-all | "
            "example | esm2 | venusrem2 | saprot | … | model-all "
            "(default: ProteinGym)"
        ),
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="benchmark output directory, or rem2 weight cache for models",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="rem2 weight cache for model downloads (default: ~/.cache/rem2/weights)",
    )
    parser.add_argument(
        "--msa",
        choices=["a2m", "a3m", "both", "none"],
        default="a2m",
        help="ProteinGym MSA archive (default: a2m); ignored for other datasets",
    )
    parser.add_argument("--force", action="store_true", help="re-download even if files exist")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    parser.epilog = (
        "examples:\n"
        "  rem2 download\n"
        "  rem2 download example\n"
        "  rem2 download benchmark-all\n"
        "  rem2 download esm2\n"
        "  rem2 download venusrem2\n"
        "  rem2 download model-all"
    )
    return parser


def _run_benchmarks(
    names: list[str],
    *,
    dest: Optional[str],
    force: bool,
    dry_run: bool,
    msa: str,
    log=print,
) -> int:
    if dest and len(names) > 1:
        raise SystemExit("--dest cannot be used with rem2 download benchmark-all")
    rows = []
    for name in names:
        spec = DATASETS[name]
        n_files = len(spec["archives"]) + len(spec["sidecar"])
        rows.append((name, f"{spec['expected']} assays", f"{n_files} files → {dest or default_dest(name)}"))
    print_plan(f"Will download {len(names)} benchmark(s)", rows, log=log)
    for i, name in enumerate(names, 1):
        log(f"benchmark [{i}/{len(names)}] {name}")
        download_hub_dataset(
            name,
            Path(dest) if dest else default_dest(name),
            force=force,
            dry_run=dry_run,
            msa=msa,
            log=log,
        )
    return 0


def _run_models(
    key: str,
    *,
    dest: Optional[str],
    cache_dir: Optional[str],
    force: bool,
    dry_run: bool,
) -> int:
    from rem2.download.models import download_models, list_downloadable_models

    if key == "list":
        print("rem2 download <model> | rem2 download model-all")
        print("Downloadable models:")
        for name in list_downloadable_models():
            print(f"  {name}")
        return 0
    keys = list_downloadable_models() if key == "all" else [key]
    download_models(
        keys,
        cache_dir=cache_dir or dest,
        force=force,
        dry_run=dry_run,
    )
    return 0


def run_download(argv: Optional[Iterable[str]] = None) -> int:
    args = build_download_parser().parse_args(list(argv or []))
    raw = getattr(args, "target", None) or getattr(args, "dataset", None) or "proteingym"
    kind, key = resolve_download_target(raw)
    if kind == "example":
        from rem2.download.example import ensure_demo_dataset

        ensure_demo_dataset(
            dest=args.dest,
            force=args.force,
            dry_run=args.dry_run,
        )
        return 0
    if kind == "model":
        return _run_models(
            key,
            dest=args.dest,
            cache_dir=getattr(args, "cache_dir", None),
            force=args.force,
            dry_run=args.dry_run,
        )
    selected = list(DATASETS) if key == "all" else [key]
    return _run_benchmarks(
        selected,
        dest=args.dest,
        force=args.force,
        dry_run=args.dry_run,
        msa=args.msa,
    )
