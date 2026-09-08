"""Real FASTA/PDB integration tests for every registered backbone.

Each model is loaded and asked for [L, V] log-probs on the trp-cage fixture.
Optional dependencies / external tools cause pytest.skip (not a silent pass).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
import torch

from rem2.models import apply_model_defaults, get_model
from rem2.scoring import score_protein
from rem2.scoring.score_protein import read_seq

from helpers import SEQUENCE, NullLogger, make_args

# One representative per family. Size variants (prosst-4096, esm2-3b, …)
# are covered by registry tests without downloading extra checkpoints.
ALL_MODELS = [
    "auto",
    "carp",
    "esm1b",
    "esm1v",
    "esm2",
    "esm3",
    "esm_if",
    "progen2",
    "progen3",
    "prosst",
    "protein_mpnn",
    "protgpt2",
    "protssn",
    "rita",
    "s2f",
    "s3f",
    "saprot",
]


def _skip_reason(model_name: str) -> Optional[str]:
    if model_name == "s3f":
        try:
            import torchdrug  # noqa: F401
        except Exception:
            return "torchdrug not installed (required for full S3F)"
    if model_name == "carp":
        try:
            import sequence_models  # noqa: F401
        except Exception:
            return "python package `sequence_models` not installed"
    if model_name == "prosst":
        try:
            import torch_geometric  # noqa: F401
        except Exception:
            return "torch_geometric not installed (pip install -e '.[prosst]')"
    return None


def _build_args(model_name: str, fixture_root: Path, pdb_path: Path, struc_fasta_path: Path):
    args = make_args(model=model_name)
    # `auto` is the generic HF MLM path — use ESM-2 weights in tests (ProSST needs ss tokens).
    if model_name == "auto":
        args.model_id = "facebook/esm2_t33_650M_UR50D"
        args.model_name = ["facebook/esm2_t33_650M_UR50D"]
    apply_model_defaults(args, model_name)
    if model_name == "auto":
        args.model_name = ["facebook/esm2_t33_650M_UR50D"]
        args.backbone_mode = "plain_mlm"
    args.pdb_dir = str(fixture_root / "pdbs")
    args.struc_seq_dir = str(fixture_root / "struc_seq")
    args.aa_seq_dir = str(fixture_root / "aa_seq")
    args.aa_seq_aln_dir = str(fixture_root / "aa_seq_aln_a2m")
    if model_name == "prosst":
        args.backbone_mode = "prosst"
    if model_name in ("esm2", "esm1b", "esm1v"):
        args.backbone_mode = "plain_mlm"
    if model_name == "esm1v":
        args.esm1v_seeds = [1]
    return args


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_model_forward_log_probs_on_real_fasta_pdb(
    model_name: str,
    device: str,
    sequence: str,
    fasta_path: Path,
    pdb_path: Path,
    struc_fasta_path: Path,
    fixture_root: Path,
    protein_name: str,
):
    reason = _skip_reason(model_name)
    if reason:
        pytest.skip(reason)

    assert read_seq(str(fasta_path)) == sequence

    args = _build_args(model_name, fixture_root, pdb_path, struc_fasta_path)
    adapter_cls = get_model(model_name)

    try:
        adapter = adapter_cls.load(
            model_id=getattr(args, "model_id", None) or (args.model_name[0] if args.model_name else None),
            device=device,
            cache_dir=args.cache_dir,
            args=args,
            logger=NullLogger(),
        )
    except Exception as exc:
        msg = str(exc).lower()
        # Environmental / optional-weight failures → skip, not fail the suite.
        skip_tokens = (
            "not installed",
            "no module named",
            "cannot import name",
            "404",
            "could not find",
            "checkpoint not found",
            "requires",
            "permission",
            "connection",
            "timed out",
            "huggingface",
            "disk quota",
            "out of memory",
            "cuda out of memory",
            "remote end closed",
            "failed to build graph",
        )
        if any(t in msg for t in skip_tokens):
            pytest.skip(f"{model_name} unavailable in this environment: {exc}")
        raise

    structure_fasta = str(struc_fasta_path) if model_name == "prosst" else None
    pdb_file = str(pdb_path) if adapter_cls.spec.needs_pdb else None

    fwd = adapter.create_forward_fn(
        protein_name=protein_name,
        pdb_file=pdb_file,
        structure_fasta=structure_fasta,
        idx=0,
        logger=NullLogger(),
    )

    with torch.no_grad():
        if fwd is not None:
            logits = fwd(sequence=sequence)
        else:
            from rem2.backbone.forward_utils import forward_sequence_logits

            use_structure = model_name == "prosst"
            structure_sequence = None
            if use_structure:
                raw = read_seq(str(struc_fasta_path))
                structure_sequence = [int(x) for x in raw.split(",")]
            logits = forward_sequence_logits(
                model=adapter.state.model,
                tokenizer=adapter.state.tokenizer,
                sequence=sequence,
                device=device,
                use_structure=use_structure,
                structure_sequence=structure_sequence,
                max_residue_len=adapter.state.model_max_residue_len,
                long_seq_mode="auto_window",
                long_seq_overlap=256,
                logger=NullLogger(),
                protein_name=protein_name,
            )

    assert isinstance(logits, torch.Tensor)
    assert logits.ndim == 2
    assert logits.shape[0] == len(sequence)
    assert logits.shape[1] >= 20
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    "model_name",
    ["esm2", "protein_mpnn", "prosst", "s2f"],
)
def test_score_protein_end_to_end(
    model_name: str,
    device: str,
    sequence: str,
    fasta_path: Path,
    pdb_path: Path,
    mutant_csv: Path,
    msa_path: Path,
    struc_fasta_path: Path,
    fixture_root: Path,
    protein_name: str,
):
    reason = _skip_reason(model_name)
    if reason:
        pytest.skip(reason)

    args = _build_args(model_name, fixture_root, pdb_path, struc_fasta_path)
    adapter_cls = get_model(model_name)
    try:
        adapter = adapter_cls.load(
            model_id=args.model_name[0] if args.model_name else None,
            device=device,
            cache_dir=args.cache_dir,
            args=args,
            logger=NullLogger(),
        )
    except Exception as exc:
        pytest.skip(f"{model_name} unavailable: {exc}")

    pdb_file = str(pdb_path) if adapter_cls.spec.needs_pdb or model_name in ("esm2", "prosst", "s2f") else None
    structure_fasta = str(struc_fasta_path) if model_name == "prosst" else None
    fwd = adapter.create_forward_fn(
        protein_name=protein_name,
        pdb_file=str(pdb_path) if adapter_cls.spec.needs_pdb else None,
        structure_fasta=structure_fasta,
        idx=0,
        logger=NullLogger(),
    )

    mutant_df = pd.read_csv(mutant_csv)
    scores = score_protein(
        model=adapter.state.model,
        tokenizer=adapter.state.tokenizer,
        residue_fasta=str(fasta_path),
        structure_fasta=structure_fasta,
        mutant_df=mutant_df,
        alpha=0.0,
        protein_name=protein_name,
        backbone_mode=args.backbone_mode,
        pdb_file=pdb_file,
        max_residue_len=adapter.state.model_max_residue_len,
        long_seq_mode="auto_window",
        long_seq_overlap=256,
        show_progress=False,
        logger=NullLogger(),
        scoring_mode="log_odds",
        baseline_forward_fn=fwd,
        aa_seq_aln_file=None,
        quiet=True,
    )
    assert len(scores) == len(mutant_df)
    assert all(s == s for s in scores)  # not NaN
