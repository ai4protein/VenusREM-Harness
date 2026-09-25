"""Model / benchmark target resolution (no network)."""

from vrh.data.download import resolve_download_target
from vrh.download.models import (
    list_downloadable_models,
    model_artifacts,
    resolve_model_key,
)
from vrh.download.progress import format_bytes, format_hint


def test_format_bytes():
    assert format_bytes(512) == "512 B"
    assert format_hint(2_500_000_000).startswith("~")


def test_resolve_benchmark_and_model_targets():
    assert resolve_download_target("ProteinGym") == ("benchmark", "proteingym")
    assert resolve_download_target("benchmark-all") == ("benchmark", "all")
    assert resolve_download_target("all") == ("benchmark", "all")
    assert resolve_download_target("model-all") == ("model", "all")
    assert resolve_download_target("esm2") == ("model", "esm2")
    assert resolve_download_target("venusrem2") == ("model", "venusrem2")
    assert resolve_download_target("proteinmpnn-020") == ("model", "protein_mpnn")
    from vrh.download.example import EXAMPLE_NAME

    assay = EXAMPLE_NAME
    assert resolve_download_target("example") == ("example", assay)
    assert resolve_download_target("demo") == ("example", assay)
    assert resolve_download_target(assay) == ("example", assay)


def test_resolve_model_key_keeps_ensemble():
    assert resolve_model_key("venusrem2") == "venusrem2"
    assert resolve_model_key("venusrem") == "venusrem"
    assert resolve_model_key("prosst") == "prosst"
    assert resolve_model_key("prosst-4096") == "prosst-4096"
    assert resolve_model_key("auto") is None
    assert resolve_model_key("s2f") is None


def test_venusrem_v1_downloads_prosst_2048_only():
    arts = model_artifacts("venusrem")
    repos = [a.source for a in arts if a.kind == "hf_repo"]
    assert repos == ["AI4Protein/ProSST-2048"]
    assert "AI4Protein/ProSST-20" not in repos


def test_venusrem2_artifacts_cover_six_prosst(monkeypatch):
    monkeypatch.delenv("VRH_PROSST_STATIC_REPO", raising=False)
    arts = model_artifacts("venusrem2")
    repos = [a.source for a in arts if a.kind == "hf_repo"]
    assert "AI4Protein/ProSST-20" in repos
    assert "AI4Protein/ProSST-4096" in repos
    assert len(repos) == 6
    assert [a for a in arts if a.kind == "prosst_static"] == []

    monkeypatch.setenv("VRH_PROSST_STATIC_REPO", "review/prosst-static")
    names = [a.extra.get("name") for a in model_artifacts("venusrem2") if a.kind == "prosst_static"]
    assert "AE.pt" in names
    assert "2048.joblib" in names


def test_s3f_artifact_uses_zenodo():
    arts = model_artifacts("s3f")
    assert arts[0].kind == "url"
    assert "zenodo.org" in arts[0].source
    assert arts[0].source.endswith("s3f.pth?download=1") or "s3f.pth" in arts[0].source


def test_protssn_single_downloads_one_gnn(monkeypatch):
    monkeypatch.delenv("VRH_PROTSSN_REPO", raising=False)
    assert [a.label for a in model_artifacts("protssn-k20-h512")] == [
        "ESM-2 650M (ProtSSN encoder)"
    ]

    monkeypatch.setenv("VRH_PROTSSN_REPO", "review/protssn")
    arts = model_artifacts("protssn-k20-h512")
    labels = [a.label for a in arts]
    assert "protssn_k20_h512.pt" in labels
    assert "protssn_k10_h512.pt" not in labels
    ens = [a.label for a in model_artifacts("protssn")]
    assert "protssn_k10_h512.pt" in ens
    assert "protssn_k30_h1280.pt" in ens


def test_carp_small_artifact():
    arts = model_artifacts("carp-38m")
    assert arts[0].kind == "url"
    assert "carp_38M.pt" in arts[0].source


def test_model_all_list_includes_core_backbones():
    names = list_downloadable_models()
    assert "venusrem2" in names
    assert "esm2" in names
    assert "saprot" in names
    assert "auto" not in names
    assert "s2f" not in names
