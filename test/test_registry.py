"""Registry / public API tests (no model weights required)."""

from __future__ import annotations

from venus_orbit.models import (
    apply_model_defaults,
    get_model,
    list_models,
    resolve_model_name,
)
from venus_orbit.config import create_parser

from helpers import make_args


EXPECTED_MODELS = {
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
    "tranception",
}


def test_list_models_covers_builtins():
    names = {s.name for s in list_models()}
    assert EXPECTED_MODELS.issubset(names)
    assert len(names) >= len(EXPECTED_MODELS)


def test_get_model_roundtrip():
    for name in sorted(EXPECTED_MODELS):
        cls = get_model(name)
        assert cls.spec.name == name or name in cls.spec.aliases


def test_resolve_model_prefers_model_flag():
    args = make_args(model="esm2", baseline_type="auto")
    assert resolve_model_name(args) == "esm2"


def test_apply_model_defaults_esm2():
    args = make_args(model="esm2")
    apply_model_defaults(args, "esm2")
    assert args.baseline_type == "esm2"
    assert args.model_name == ["facebook/esm2_t33_650M_UR50D"]
    assert args.cache_dir


def test_apply_model_defaults_auto_cache_for_mpnn_and_protssn():
    args = make_args(model="protein_mpnn")
    apply_model_defaults(args, "protein_mpnn")
    assert args.protein_mpnn_checkpoint is not None

    args = make_args(model="protssn")
    apply_model_defaults(args, "protssn")
    assert args.protssn_model_dir is not None


def test_cli_parser_list_models_flag():
    parser = create_parser()
    args = parser.parse_args(["--list-models"])
    assert args.list_models is True


def test_fixture_files_exist(fasta_path, pdb_path, mutant_csv, msa_path, struc_fasta_path, sequence):
    assert fasta_path.read_text().splitlines()[1].strip() == sequence
    assert "ATOM" in pdb_path.read_text()
    assert "mutant" in mutant_csv.read_text()
    assert sequence in msa_path.read_text()
    assert "," in struc_fasta_path.read_text().splitlines()[1]
