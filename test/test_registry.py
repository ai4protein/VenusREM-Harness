"""Registry / public API tests (no model weights required)."""

from __future__ import annotations

from venusrem2.models import (
    apply_model_defaults,
    get_model,
    list_models,
    resolve_model_name,
)
from venusrem2.config import create_parser

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
    assert get_model("venusrem2").spec.name == "prosst"


def test_resolve_model_prefers_model_flag():
    args = make_args(model="esm2", baseline_type="auto")
    assert resolve_model_name(args) == "esm2"


def test_resolve_model_defaults_to_esm2():
    args = make_args()
    assert resolve_model_name(args) == "esm2"


def test_resolve_model_env_override(monkeypatch):
    monkeypatch.setenv("VENUSREM2_MODEL", "esm2-8m")
    args = make_args()
    assert resolve_model_name(args) == "esm2-8m"


def test_resolve_explicit_prosst_name_still_prosst():
    args = make_args(model_name=["AI4Protein/ProSST-512"])
    assert resolve_model_name(args) == "prosst"


def test_apply_model_defaults_esm2():
    args = make_args(model="esm2")
    apply_model_defaults(args, "esm2")
    assert args.baseline_type == "esm2"
    assert args.model_name == ["facebook/esm2_t33_650M_UR50D"]
    assert args.cache_dir
    assert args.model_out_name == ["esm2_t33_650M_UR50D__rem2"]


def test_venusrem2_expands_prosst_ensemble():
    from venusrem2.naming import PROSST_ENSEMBLE_IDS, is_official_venusrem2

    args = make_args(model="venusrem2")
    apply_model_defaults(args, "venusrem2")
    assert list(args.model_name) == list(PROSST_ENSEMBLE_IDS)
    assert is_official_venusrem2("venusrem2", args)
    assert args.model_out_name[0].startswith("VenusREM2__")


def test_score_label_rem2_vs_venusrem2_ensemble():
    from venusrem2.naming import (
        default_score_label,
        is_official_venusrem2,
        run_banner,
    )

    esm = make_args(model="esm2", model_name=["facebook/esm2_t33_650M_UR50D"])
    assert not is_official_venusrem2("esm2", esm)
    assert default_score_label("esm2", esm, "facebook/esm2_t33_650M_UR50D").endswith("__rem2")
    assert run_banner("esm2", esm).startswith("rem2")

    single = make_args(model="prosst", model_name=["AI4Protein/ProSST-2048"])
    assert not is_official_venusrem2("prosst", single)
    assert default_score_label("prosst", single, "AI4Protein/ProSST-2048") == "ProSST-2048__rem2"

    ens = make_args(
        model="venusrem2",
        model_name=["AI4Protein/ProSST-512", "AI4Protein/ProSST-2048"],
    )
    assert is_official_venusrem2("venusrem2", ens)
    assert default_score_label("venusrem2", ens, "AI4Protein/ProSST-2048") == "VenusREM2__ProSST-2048"
    assert "VenusREM2" in run_banner("venusrem2", ens)


def test_apply_model_defaults_auto_cache_for_mpnn_and_protssn():
    args = make_args(model="protein_mpnn")
    apply_model_defaults(args, "protein_mpnn")
    assert args.protein_mpnn_checkpoint is not None

    args = make_args(model="protssn")
    apply_model_defaults(args, "protssn")
    assert args.protssn_model_dir is not None


def test_mask_capability_and_refuse():
    from venusrem2.models.scoring_strategy import (
        UnsupportedScoringStrategy,
        models_supporting_mask,
        require_scoring_strategy,
        spec_supports_mask,
    )

    masked = set(models_supporting_mask())
    assert {"esm2", "esm1b", "esm1v", "prosst", "auto", "saprot", "protssn", "carp", "esm3", "s3f"} <= masked
    for name in ("progen2", "progen3", "protgpt2", "rita", "tranception", "protein_mpnn", "esm_if", "mifst", "s2f"):
        assert not spec_supports_mask(name)
        require_scoring_strategy(name, "wt-marginals")
        try:
            require_scoring_strategy(name, "masked-marginals")
        except UnsupportedScoringStrategy as exc:
            assert "refusing" in str(exc).lower() or "not supported" in str(exc).lower()
        else:
            raise AssertionError(f"{name} should refuse masked-marginals")
    require_scoring_strategy("esm2", "masked-marginals")
    require_scoring_strategy("venusrem2", "masked-marginals")


def test_cli_parser_list_models_flag():
    parser = create_parser()
    args = parser.parse_args(["--list-models"])
    assert args.list_models is True


def test_cli_parser_rem2_defaults():
    parser = create_parser()
    args = parser.parse_args([])
    assert args.alpha == "entropy"
    assert args.background_weight == "one_minus_alpha"
    assert args.wt_confidence_weight == 0.0
    assert args.scoring_mode == "calibrated_margin"
    assert args.calibrate_on_raw is True
    assert args.use_rsa_decay is True
    assert args.use_plddt_decay is True
    assert args.rsa_decay_mode == "above_mean"
    assert args.plddt_decay_mode == "above_mean"
    off = parser.parse_args(["--no_rsa_decay", "--no_plddt_decay", "--no_calibrate_on_raw", "--alpha", "0.8"])
    assert off.use_rsa_decay is False
    assert off.use_plddt_decay is False
    assert off.calibrate_on_raw is False
    assert off.alpha == "0.8"


def test_fixture_files_exist(fasta_path, pdb_path, mutant_csv, msa_path, struc_fasta_path, sequence):
    assert fasta_path.read_text().splitlines()[1].strip() == sequence
    assert "ATOM" in pdb_path.read_text()
    assert "mutant" in mutant_csv.read_text()
    assert sequence in msa_path.read_text()
    assert "," in struc_fasta_path.read_text().splitlines()[1]


def test_resolve_existing_weight_falls_back_to_orbit_cache(tmp_path, monkeypatch):
    from venusrem2.models.weights import resolve_existing_weight

    rem2 = tmp_path / "venusrem2" / "weights"
    orbit = tmp_path / "venus_orbit" / "weights"
    (orbit / "protein_mpnn").mkdir(parents=True)
    ckpt = orbit / "protein_mpnn" / "v_48_020.pt"
    ckpt.write_bytes(b"orbit-weight")
    rem2.mkdir(parents=True)
    monkeypatch.setattr(
        "venusrem2.models.weights.legacy_weight_cache_dirs",
        lambda: [str(orbit)],
    )
    found = resolve_existing_weight(
        "protein_mpnn", "v_48_020.pt", cache_dir=str(rem2)
    )
    assert found == str(ckpt)
