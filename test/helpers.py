"""Shared helpers for Venus-Orbit integration tests."""

from __future__ import annotations

import argparse
from pathlib import Path

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "trp_cage"
PROTEIN = "trp_cage"
SEQUENCE = "NLYIQWLKDGGPSSGRPPPS"


def make_args(**overrides):
    """Minimal Namespace compatible with load_baseline / score_protein."""
    defaults = dict(
        model=None,
        model_id=None,
        cache_dir=None,
        model_name=["AI4Protein/ProSST-2048"],
        model_out_name=["test"],
        baseline_type="auto",
        backbone_mode="auto",
        max_residue_len=None,
        long_seq_mode="auto_window",
        long_seq_overlap=256,
        scoring_strategy="wt-marginals",
        esm1v_seeds=[1],
        foldseek_bin=None,
        protssn_model_dir=None,
        protssn_norm_dir=None,
        protssn_no_ensemble=True,
        esm_if_chain="A",
        protein_mpnn_checkpoint=None,
        protein_mpnn_chain="A",
        protein_mpnn_scoring_mode="teacher_force",
        protein_mpnn_random_orders=1,
        progen2_model_name_or_path="hugohrban/progen2-small",
        progen2_fp16=False,
        progen3_model_name_or_path="Profluent-Bio/progen3-112m",
        progen3_fp16=False,
        protgpt2_model_name_or_path="nferruz/ProtGPT2",
        rita_model_name_or_path="lightonai/RITA_s",
        esm3_model_name="esmc_300m",
        tranception_checkpoint="OATML-Markslab/Tranception_Small",
        tranception_no_mirror=False,
        carp_model_name="carp_600k",
        s2f_config=None,
        s2f_checkpoint=None,
        structure_vocab_subdir=None,
        pdb_dir=None,
        struc_seq_dir=None,
        aa_seq_dir=None,
        aa_seq_aln_dir=None,
        show_progress=False,
        disable_tqdm=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def section(self, *args, **kwargs):
        pass

    def success(self, *args, **kwargs):
        pass