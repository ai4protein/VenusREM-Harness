"""Batch single-chain protein folding via ESMFold v1 and/or ESMFold2.

Backends
--------
v1: ESM Atlas free HTTP API (no token, POST https://api.esmatlas.com/foldSequence/v1/pdb/).
    Returns PDB with per-residue pLDDT in the B-factor column. Hard limit ~400 residues.
v2: Biohub ESMFold2 API (requires $BIOHUB_TOKEN). Returns mmCIF + pLDDT tensor + pTM.

Reads `<aa_seq_dir>/<name>.fasta` (one sequence per file, matching the project
convention used by compute_fitness.py) and writes:
    <out_dir>/v1/<name>.pdb
    <out_dir>/v1/<name>.plddt.json
    <out_dir>/v2/<name>.cif
    <out_dir>/v2/<name>.pdb
    <out_dir>/v2/<name>.plddt.json

Install (one-time, only if v2 is enabled):
    pip install 'esm@git+https://github.com/Biohub/esm.git@main'

Run:
    export BIOHUB_TOKEN=<your-token>          # only needed for v2
    python src/esmfold2.py \\
        --aa_seq_dir data/case/aa_seq \\
        --out_dir   data/case/esmfold_pdbs \\
        --backends  v1,v2 \\
        --workers   4
"""

import argparse
import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

ESM_ATLAS_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
ESM_ATLAS_MAX_LEN = 400  # API hard limit


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--aa_seq_dir", required=True, help="Directory of <name>.fasta files (single sequence each).")
    p.add_argument("--out_dir", required=True, help="Output directory; backends write to <out_dir>/<backend>/.")
    p.add_argument("--backends", default="v2",
                   help="Comma-separated subset of {v1, v2}. Default: v2.")
    # v2 (Biohub) options
    p.add_argument("--token", default=os.environ.get("BIOHUB_TOKEN"),
                   help="Biohub API token (v2 only). Defaults to $BIOHUB_TOKEN.")
    p.add_argument("--model", default="esmfold2-fast-2026-05",
                   help="Biohub model name. Use 'esmfold2-2026-05' for the larger variant.")
    p.add_argument("--url", default="https://biohub.ai", help="Biohub API base URL.")
    p.add_argument("--num_loops", type=int, default=3)
    p.add_argument("--num_sampling_steps", type=int, default=32)
    p.add_argument("--include_pae", action="store_true",
                   help="(v2) Also save PAE matrix as <name>.pae.npy.")
    p.add_argument("--keep_cif_only", action="store_true",
                   help="(v2) Skip CIF→PDB conversion.")
    # Shared
    p.add_argument("--workers", type=int, default=4, help="Concurrent API requests across all backends.")
    p.add_argument("--max_proteins", type=int, default=None)
    p.add_argument("--protein_list", default=None, help="Comma-separated subset of protein names.")
    p.add_argument("--overwrite", action="store_true", help="Re-fold even if outputs exist.")
    p.add_argument("--retries", type=int, default=3, help="Per-protein retry budget.")
    p.add_argument("--v1_timeout", type=int, default=180, help="ESM Atlas request timeout (seconds).")

    args = p.parse_args()
    args.backend_list = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = set(args.backend_list) - {"v1", "v2"}
    if unknown:
        p.error(f"Unknown backends: {sorted(unknown)}; expected subset of v1,v2.")
    if "v2" in args.backend_list and not args.token:
        p.error("v2 requires a Biohub token: pass --token or set BIOHUB_TOKEN.")
    return args


def read_fasta(path):
    """Return (header, sequence) for the first record; warn on extras."""
    header, seq_lines, records = None, [], 0
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                records += 1
                if header is None:
                    header = line[1:].strip()
                else:
                    break
            elif header is not None and records == 1:
                seq_lines.append(line)
    if header is None or not seq_lines:
        raise ValueError(f"No sequence found in {path}")
    if records > 1:
        print(f"[warn] {path} has >1 record; using only the first.", file=sys.stderr)
    return header, "".join(seq_lines).upper()


def extract_plddt_from_pdb(pdb_text):
    """ESM Atlas writes pLDDT (0–100) into the B-factor column on every atom."""
    plddt = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                plddt.append(float(line[60:66]))
            except ValueError:
                pass
    return plddt


def cif_to_pdb(cif_text: str, pdb_path: str) -> None:
    from biotite.structure.io.pdb import PDBFile
    from biotite.structure.io.pdbx import CIFFile, get_structure
    cif = CIFFile.read(io.StringIO(cif_text))
    structure = get_structure(cif, model=1)
    pdb = PDBFile()
    pdb.set_structure(structure)
    pdb.write(pdb_path)


def fold_v1(name, seq, out_dir, overwrite, retries, timeout):
    pdb_path = os.path.join(out_dir, f"{name}.pdb")
    meta_path = os.path.join(out_dir, f"{name}.plddt.json")
    if not overwrite and os.path.exists(pdb_path) and os.path.exists(meta_path):
        return "skip", None
    if len(seq) > ESM_ATLAS_MAX_LEN:
        return "skip_long", str(len(seq))

    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(
                ESM_ATLAS_URL,
                data=seq,
                headers={"Content-Type": "text/plain"},
                timeout=timeout,
            )
            if r.status_code == 200 and r.text.lstrip().startswith(("HEADER", "ATOM", "MODEL")):
                pdb_text = r.text
                break
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    else:
        return "fail", last_err

    with open(pdb_path, "w") as f:
        f.write(pdb_text)
    plddt = extract_plddt_from_pdb(pdb_text)
    meta = {
        "name": name,
        "length": len(plddt),
        "mean_plddt": float(sum(plddt) / len(plddt)) if plddt else None,
        "ptm": None,  # not provided by ESM Atlas API
        "per_residue_plddt": [round(p, 3) for p in plddt],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    return "ok", None


def fold_v2(client, config, name, seq, out_dir, overwrite, keep_cif_only,
            include_pae, retries):
    from esm.utils.structure import input_builder

    cif_path = os.path.join(out_dir, f"{name}.cif")
    pdb_path = os.path.join(out_dir, f"{name}.pdb")
    meta_path = os.path.join(out_dir, f"{name}.plddt.json")
    done = os.path.exists(cif_path) and os.path.exists(meta_path) and (
        keep_cif_only or os.path.exists(pdb_path)
    )
    if done and not overwrite:
        return "skip", None

    request = input_builder.StructurePredictionInput(
        sequences=[input_builder.ProteinInput(id="A", sequence=seq)]
    )

    last_err = None
    for attempt in range(retries):
        try:
            result = client.fold_all_atom(request, config=config)
            break
        except Exception as e:  # noqa: BLE001 - API errors come in many flavors
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    else:
        return "fail", f"{type(last_err).__name__}: {last_err}"

    cif_text = result.complex.to_mmcif()
    with open(cif_path, "w") as f:
        f.write(cif_text)
    if not keep_cif_only:
        try:
            cif_to_pdb(cif_text, pdb_path)
        except Exception as e:  # noqa: BLE001
            return "fail", f"CIF→PDB conversion failed: {type(e).__name__}: {e}"

    plddt = result.plddt.cpu().numpy()
    meta = {
        "name": name,
        "length": int(plddt.shape[0]),
        "mean_plddt": float(plddt.mean()),
        "ptm": float(result.ptm) if getattr(result, "ptm", None) is not None else None,
        "per_residue_plddt": [round(float(x), 3) for x in plddt.tolist()],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    if include_pae and getattr(result, "pae", None) is not None:
        import numpy as np
        np.save(os.path.join(out_dir, f"{name}.pae.npy"), result.pae.cpu().numpy())
    return "ok", None


def main():
    args = parse_args()
    for b in args.backend_list:
        os.makedirs(os.path.join(args.out_dir, b), exist_ok=True)

    # v2 client + config built lazily (skip the heavy import when v2 is disabled)
    v2_client = v2_config = None
    if "v2" in args.backend_list:
        from esm.sdk import esmfold2_client
        from esm.sdk.api import FoldingConfig
        v2_client = esmfold2_client(model=args.model, url=args.url, token=args.token)
        v2_config = FoldingConfig(
            num_loops=args.num_loops,
            num_sampling_steps=args.num_sampling_steps,
            include_pae=args.include_pae,
        )

    names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(args.aa_seq_dir)
        if f.endswith(".fasta")
    )
    if args.protein_list:
        allowed = set(args.protein_list.split(","))
        names = [n for n in names if n in allowed]
    if args.max_proteins:
        names = names[: args.max_proteins]

    proteins = []
    for name in names:
        try:
            _, seq = read_fasta(os.path.join(args.aa_seq_dir, f"{name}.fasta"))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {name}: {e}", file=sys.stderr)
            continue
        proteins.append((name, seq))

    print(f"Folding {len(proteins)} proteins; backends={args.backend_list}; workers={args.workers}")

    stats = {b: {"ok": 0, "skip": 0, "skip_long": 0, "fail": 0} for b in args.backend_list}
    fails = []
    skip_long_by_backend = {b: [] for b in args.backend_list}

    def submit(pool):
        for name, seq in proteins:
            for backend in args.backend_list:
                bdir = os.path.join(args.out_dir, backend)
                if backend == "v1":
                    fut = pool.submit(fold_v1, name, seq, bdir,
                                      args.overwrite, args.retries, args.v1_timeout)
                else:  # v2
                    fut = pool.submit(fold_v2, v2_client, v2_config, name, seq, bdir,
                                      args.overwrite, args.keep_cif_only,
                                      args.include_pae, args.retries)
                yield fut, name, backend

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        meta = {}
        futures = []
        for fut, name, backend in submit(pool):
            meta[fut] = (name, backend)
            futures.append(fut)
        for fut in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True):
            name, backend = meta[fut]
            try:
                status, msg = fut.result()
            except Exception as e:  # noqa: BLE001
                status, msg = "fail", f"{type(e).__name__}: {e}"
            stats[backend][status] += 1
            if status == "fail":
                fails.append((backend, name, msg))
            elif status == "skip_long":
                skip_long_by_backend[backend].append((name, msg))

    for b in args.backend_list:
        s = stats[b]
        print(f"[{b}] ok={s['ok']}  skip={s['skip']}  skip_long={s['skip_long']}  fail={s['fail']}")
        if skip_long_by_backend[b]:
            manifest = os.path.join(args.out_dir, b, "skipped_too_long.tsv")
            with open(manifest, "w") as f:
                f.write("name\tlength\n")
                for name, length in sorted(skip_long_by_backend[b]):
                    f.write(f"{name}\t{length}\n")
            print(f"  manifest: {manifest}")
    for backend, name, msg in fails:
        print(f"  [fail][{backend}] {name}: {msg}", file=sys.stderr)
    sys.exit(1 if any(s["fail"] for s in stats.values()) else 0)


if __name__ == "__main__":
    main()
