"""Download a query FASTA from UniProt or RCSB. Does not invent sequences."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from rem2.data.fetch_structure import normalize_pdb_id, normalize_uniprot

UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
RCSB_FASTA = "https://www.rcsb.org/fasta/entry/{pdb}"


def _download_fasta(url: str, dest: Path, timeout: float = 40) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "rem2-dashboard/0.2"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise OSError(f"sequence fetch failed: {exc}") from exc
    if not data or len(data) < 8:
        raise OSError(f"empty FASTA from {url}")
    dest.write_bytes(data)
    return dest


def fetch_query_fasta(
    seq_id: str,
    dest: Path,
    *,
    opener: Optional[Callable] = None,
) -> str:
    """Write ``dest`` with UniProt or RCSB FASTA. Raise if the id or download is bad."""
    acc = normalize_uniprot(seq_id or "")
    pid = normalize_pdb_id(seq_id or "")
    if acc:
        url = UNIPROT_FASTA.format(acc=acc)
    elif pid:
        url = RCSB_FASTA.format(pdb=pid)
    else:
        raise ValueError(f"Not a UniProt accession or PDB id: {seq_id!r}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    writer = opener or _download_fasta
    writer(url, dest)
    text = dest.read_text(encoding="utf-8", errors="replace")
    if not text.lstrip().startswith(">"):
        dest.unlink(missing_ok=True)
        raise OSError(f"not a FASTA from {url}")
    return str(dest)
