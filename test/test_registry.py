"""Registry / public API tests (no model weights required)."""

from __future__ import annotations

from vrh.models import (
    apply_model_defaults,
    get_model,
    list_models,
    resolve_model_name,
)
from vrh.config import create_parser

from helpers import make_args


EXPECTED_MODELS = {
    "auto",
    "carp",
    "esm1b",
    "esm1v",
    "esm2",
    "esm2-8m",
    "esm2-35m",
    "esm2-150m",
    "esm2-3b",
    "esm3",
    "esmc",
    "esmc-600m",
    "esm_if",
    "mifst",
    "progen2",
    "progen2-s",
    "progen2-m",
    "progen2-b",
    "progen2-xl",
    "progen3",
    "progen3-112m",
    "progen3-219m",
    "progen3-339m",
    "progen3-762m",
    "progen3-3b",
    "prosst",
    "prosst-20",
    "prosst-128",
    "prosst-512",
    "prosst-1024",
    "prosst-2048",
    "prosst-4096",
    "protein_mpnn",
    "protein_mpnn-v_48_002",
    "protein_mpnn-v_48_010",
    "protein_mpnn-v_48_030",
    "protein_mpnn-soluble-v_48_002",
    "protein_mpnn-soluble-v_48_010",
    "protein_mpnn-soluble-v_48_020",
    "protein_mpnn-soluble-v_48_030",
    "protgpt2",
    "protssn",
    "rita",
    "rita-s",
    "rita-m",
    "rita-l",
    "s2f",
    "s3f",
    "saprot",
    "saprot-35m-af2",
    "saprot-650m-pdb",
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
    assert get_model("prosst_ensemble").spec.name == "prosst"
    assert get_model("prosst-2048").spec.name == "prosst-2048"
    assert get_model("prosst-4096").spec.name == "prosst-4096"
    assert get_model("prosst_k4096").spec.name == "prosst-4096"
    assert get_model("prosst-k4096").spec.name == "prosst-4096"
    assert get_model("esmif").spec.name == "esm_if"
    assert get_model("pmpnn_soluble_v_48_002").spec.name == "protein_mpnn-soluble-v_48_002"
    assert get_model("protssn-ensemble").spec.name == "protssn"
    assert get_model("prosst-ensemble").spec.name == "prosst"
    assert get_model("proteinmpnn-020").spec.name == "protein_mpnn"
    assert get_model("proteinmpnn-v_48_002").spec.name == "protein_mpnn-v_48_002"
    assert get_model("esm2-650m").spec.name == "esm2"


def test_resolve_model_prefers_model_flag():
    args = make_args(model="esm2", baseline_type="auto")
    assert resolve_model_name(args) == "esm2"


def test_resolve_model_defaults_to_esm2(monkeypatch):
    monkeypatch.delenv("VRH_MODEL", raising=False)
    monkeypatch.delenv("REM2_MODEL", raising=False)
    monkeypatch.delenv("VENUSREM2_MODEL", raising=False)
    args = make_args()
    assert resolve_model_name(args) == "esm2"


def test_resolve_model_env_override(monkeypatch):
    monkeypatch.delenv("REM2_MODEL", raising=False)
    monkeypatch.delenv("VENUSREM2_MODEL", raising=False)
    monkeypatch.setenv("VRH_MODEL", "esm2-8m")
    args = make_args()
    assert resolve_model_name(args) == "esm2-8m"


def test_resolve_model_legacy_rem2_env(monkeypatch):
    monkeypatch.delenv("VRH_MODEL", raising=False)
    monkeypatch.delenv("VENUSREM2_MODEL", raising=False)
    monkeypatch.setenv("REM2_MODEL", "esm2-8m")
    args = make_args()
    assert resolve_model_name(args) == "esm2-8m"


def test_resolve_model_legacy_venusrem2_env(monkeypatch):
    monkeypatch.delenv("VRH_MODEL", raising=False)
    monkeypatch.delenv("REM2_MODEL", raising=False)
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
    assert args.model_out_name == ["esm2_t33_650M_UR50D__vrh"]


def test_apply_model_defaults_prosst_2048():
    args = make_args(model="prosst-2048")
    apply_model_defaults(args, "prosst-2048")
    assert args.backbone_mode == "prosst"
    assert args.model_name == ["AI4Protein/ProSST-2048"]
    assert args.model_out_name == ["ProSST-2048__vrh"]


def test_apply_model_defaults_size_specific_keys():
    args = make_args(model="prosst-4096")
    apply_model_defaults(args, "prosst-4096")
    assert args.backbone_mode == "prosst"
    assert args.model_name == ["AI4Protein/ProSST-4096"]
    assert args.model_out_name == ["ProSST-4096__vrh"]

    args = make_args(model="prosst_k4096")
    apply_model_defaults(args, "prosst_k4096")
    assert args.model_name == ["AI4Protein/ProSST-4096"]

    args = make_args(model="progen2-xl")
    apply_model_defaults(args, "progen2-xl")
    assert args.progen2_model_name_or_path == "hugohrban/progen2-xlarge"

    args = make_args(model="esm3")
    apply_model_defaults(args, "esm3")
    assert args.esm3_model_name == "esm3_sm_open_v1"

    args = make_args(model="esmc-600m")
    apply_model_defaults(args, "esmc-600m")
    assert args.esm3_model_name == "esmc_600m"

    args = make_args(model="protein_mpnn-soluble-v_48_002")
    apply_model_defaults(args, "protein_mpnn-soluble-v_48_002")
    assert args.protein_mpnn_checkpoint.endswith("soluble_v_48_002.pt")

    args = make_args(model="proteinmpnn-020", scoring_strategy="tf")
    args.protein_mpnn_scoring_mode = "random_order"
    apply_model_defaults(args, "proteinmpnn-020")
    assert args.scoring_strategy == "teacher-force"
    assert args.protein_mpnn_scoring_mode == "teacher_force"


def test_venusrem2_expands_prosst_ensemble():
    from vrh.naming import PROSST_ENSEMBLE_IDS, is_official_venusrem2

    args = make_args(model="venusrem2")
    apply_model_defaults(args, "venusrem2")
    assert list(args.model_name) == list(PROSST_ENSEMBLE_IDS)
    assert is_official_venusrem2("venusrem2", args)
    assert args.model_out_name[0].startswith("VenusREM2__")

    args = make_args(model="prosst_ensemble")
    apply_model_defaults(args, "prosst_ensemble")
    assert list(args.model_name) == list(PROSST_ENSEMBLE_IDS)
    assert is_official_venusrem2("prosst_ensemble", args)

    args = make_args(model="prosst-ensemble")
    apply_model_defaults(args, "prosst-ensemble")
    assert list(args.model_name) == list(PROSST_ENSEMBLE_IDS)


def test_experiment_csv_backbone_keys_resolve():
    """ProteinGym ablation keys (mask/wt stripped) resolve as --model."""
    from vrh.models.scoring_strategy import spec_supports_mask

    # Unique model_key stems from staged_ablation_59.csv (not _mask/_wt).
    experiment_keys = [
        "prosst_ensemble",
        "carp_640m",
        "esm1b",
        "esm1v",
        "esm2_8m",
        "esm2_35m",
        "esm2_150m",
        "esm2_650m",
        "esm2_3b",
        "esm3",
        "esmc",
        "esmc_600m",
        "esmif",
        "mifst",
        "progen2",
        "progen2_s",
        "progen2_m",
        "progen2_b",
        "progen2_xl",
        "progen3",
        "progen3_112m",
        "progen3_219m",
        "progen3_339m",
        "progen3_762m",
        "progen3_3b",
        "prosst_k20",
        "prosst_k128",
        "prosst_k512",
        "prosst_k1024",
        "prosst_k2048",
        "prosst_k4096",
        "proteinmpnn",
        "pmpnn_v_48_002",
        "pmpnn_v_48_010",
        "pmpnn_v_48_030",
        "pmpnn_soluble_v_48_002",
        "pmpnn_soluble_v_48_010",
        "pmpnn_soluble_v_48_020",
        "pmpnn_soluble_v_48_030",
        "protssn",
        "rita_s",
        "rita_m",
        "rita_l",
        "rita_xl",
        "s3f",
        "saprot",
        "saprot35m_af2",
        "saprot650m_pdb",
    ]
    for key in experiment_keys:
        spec = get_model(key).spec
        assert spec.name
        if key in {"esmif", "mifst", "proteinmpnn"} or key.startswith("pmpnn") or key.startswith("progen") or key.startswith("rita"):
            assert not spec_supports_mask(key)
        if key.startswith("esm2") or key.startswith("saprot") or key in {"esm1b", "esm1v", "s3f"}:
            assert spec_supports_mask(key)


def test_score_label_vrh_vs_venusrem2_ensemble():
    from vrh.naming import (
        default_score_label,
        is_official_venusrem2,
        run_banner,
    )

    esm = make_args(model="esm2", model_name=["facebook/esm2_t33_650M_UR50D"])
    assert not is_official_venusrem2("esm2", esm)
    assert default_score_label("esm2", esm, "facebook/esm2_t33_650M_UR50D").endswith("__vrh")
    assert run_banner("esm2", esm).startswith("vrh")

    single = make_args(model="prosst", model_name=["AI4Protein/ProSST-2048"])
    assert not is_official_venusrem2("prosst", single)
    assert default_score_label("prosst", single, "AI4Protein/ProSST-2048") == "ProSST-2048__vrh"

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
    from vrh.models.scoring_strategy import (
        UnsupportedScoringStrategy,
        models_supporting_mask,
        require_scoring_strategy,
        spec_supports_mask,
    )

    masked = set(models_supporting_mask())
    assert {"esm2", "esm1b", "esm1v", "auto", "saprot", "protssn", "carp", "esm3", "s3f"} <= masked
    assert "prosst" not in masked
    for name in ("progen2", "progen3", "protgpt2", "rita", "protein_mpnn", "esm_if", "mifst", "s2f", "prosst", "prosst-4096"):
        assert not spec_supports_mask(name)
        require_scoring_strategy(name, "wt")
        try:
            require_scoring_strategy(name, "mask")
        except UnsupportedScoringStrategy as exc:
            assert "refusing" in str(exc).lower() or "not supported" in str(exc).lower()
        else:
            raise AssertionError(f"{name} should refuse mask")
    require_scoring_strategy("esm2", "mask")
    require_scoring_strategy("esm2", "wt")
    require_scoring_strategy("protein_mpnn", "tf")
    require_scoring_strategy("proteinmpnn-020", "teacher-force")
    try:
        require_scoring_strategy("venusrem2", "mask")
    except UnsupportedScoringStrategy:
        pass
    else:
        raise AssertionError("venusrem2 / ProSST should refuse mask")
    try:
        require_scoring_strategy("esm2", "tf")
    except UnsupportedScoringStrategy as exc:
        assert "tf" in str(exc).lower()
    else:
        raise AssertionError("esm2 should refuse tf")


def test_cli_parser_list_models_flag():
    parser = create_parser()
    args = parser.parse_args(["--list-models"])
    assert args.list_models is True


def test_cli_parser_vrh_defaults():
    parser = create_parser()
    args = parser.parse_args([])
    assert args.alpha == "entropy"
    assert args.scoring_strategy == "wt-marginals"
    assert args.background_weight == "one_minus_alpha"
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
    from vrh.models.weights import resolve_existing_weight

    vrh = tmp_path / "venusrem2" / "weights"
    orbit = tmp_path / "venus_orbit" / "weights"
    (orbit / "protein_mpnn").mkdir(parents=True)
    ckpt = orbit / "protein_mpnn" / "v_48_020.pt"
    ckpt.write_bytes(b"orbit-weight")
    vrh.mkdir(parents=True)
    monkeypatch.setattr(
        "vrh.models.weights.legacy_weight_cache_dirs",
        lambda: [str(orbit)],
    )
    found = resolve_existing_weight(
        "protein_mpnn", "v_48_020.pt", cache_dir=str(vrh)
    )
    assert found == str(ckpt)
