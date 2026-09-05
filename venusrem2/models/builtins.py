"""Built-in ModelAdapter registrations (wrap legacy baseline_dispatch)."""

from __future__ import annotations

from typing import Any, Optional

from venus_orbit.models.base import ModelAdapter, ModelSpec
from venus_orbit.models.registry import register_model


def _load_via_dispatch(
    cls,
    baseline_type: str,
    model_id: Optional[str],
    device: str,
    args: Any,
    logger: Any,
) -> ModelAdapter:
    from venus_orbit.backbone.baseline_dispatch import load_baseline

    names = list(getattr(args, "model_name", None) or [])
    if model_id:
        model_name = model_id
    elif names:
        model_name = names[0]
    else:
        model_name = cls.spec.default_model_id or baseline_type
    # Keep args.model_name in sync for multi-model loops.
    if model_id:
        args.model_name = [model_id]
    state = load_baseline(baseline_type, model_name, args, device, logger)
    return cls(state=state, args=args, device=device)


def _make_adapter(
    name: str,
    baseline_type: str,
    *,
    description: str = "",
    default_model_id: Optional[str] = None,
    needs_pdb: bool = False,
    auto_download: bool = True,
    extras: str = "",
    notes: str = "",
    aliases: tuple = (),
):
    @register_model
    class _Adapter(ModelAdapter):
        spec = ModelSpec(
            name=name,
            baseline_type=baseline_type,
            description=description,
            default_model_id=default_model_id,
            needs_pdb=needs_pdb,
            auto_download=auto_download,
            extras=extras,
            notes=notes,
            aliases=aliases,
        )

        @classmethod
        def load(cls, model_id, device, cache_dir, args, logger):
            return _load_via_dispatch(cls, baseline_type, model_id, device, args, logger)

    _Adapter.__name__ = f"{name.title().replace('_', '')}Adapter"
    _Adapter.__qualname__ = _Adapter.__name__
    return _Adapter


_make_adapter(
    "prosst",
    "auto",
    description="ProSST structure-aware MLM (default Orbit backbone)",
    default_model_id="AI4Protein/ProSST-2048",
    extras="prosst",
    notes="Uses structure token FASTA under struc_seq/",
    aliases=("venusrem",),
)
_make_adapter(
    "auto",
    "auto",
    description="Any HuggingFace AutoModelForMaskedLM (pass --model_id)",
    default_model_id=None,
    notes="Generic HF MLM path",
)
_make_adapter(
    "esm2",
    "esm2",
    description="ESM-2 masked language model",
    default_model_id="facebook/esm2_t33_650M_UR50D",
)
_make_adapter(
    "esm1b",
    "esm1b",
    description="ESM-1b masked language model",
    default_model_id="facebook/esm1b_t33_650M_UR50S",
)
_make_adapter(
    "esm1v",
    "esm1v",
    description="ESM-1v 5-seed ensemble",
    default_model_id="facebook/esm1v_t33_650M_UR90S_1",
    notes="Uses --esm1v_seeds (default 1-5); ignores --model_id for weights",
)
_make_adapter(
    "saprot",
    "saprot",
    description="SaProt structure-aware PLM",
    default_model_id="westlake-repl/SaProt_650M_AF2",
    needs_pdb=True,
    notes="Foldseek auto-downloaded from HF if missing",
)
_make_adapter(
    "protssn",
    "protssn",
    description="ProtSSN structure GNN ensemble",
    needs_pdb=True,
    notes="Weights auto-download to cache/protssn",
)
_make_adapter(
    "esm_if",
    "esm_if",
    description="ESM-IF1 inverse folding",
    needs_pdb=True,
    notes="fair-esm sources isolated from ESM3; weights auto-download",
)
_make_adapter(
    "protein_mpnn",
    "protein_mpnn",
    description="ProteinMPNN",
    needs_pdb=True,
    notes="Checkpoint auto-download to cache/protein_mpnn",
)
_make_adapter(
    "progen2",
    "progen2",
    description="ProGen2 causal LM",
    default_model_id="hugohrban/progen2-large",
)
_make_adapter(
    "progen3",
    "progen3",
    description="ProGen3 causal LM",
    default_model_id="Profluent-Bio/progen3-1b",
    notes="May require extra flash-attn / megablocks deps",
)
_make_adapter(
    "protgpt2",
    "protgpt2",
    description="ProtGPT2 causal LM",
    default_model_id="nferruz/ProtGPT2",
)
_make_adapter(
    "rita",
    "rita",
    description="RITA causal LM",
    default_model_id="lightonai/RITA_xl",
)
_make_adapter(
    "esm3",
    "esm3",
    description="ESM3 / ESM-C",
    default_model_id="esmc_300m",
    extras="baselines",
    notes="Requires `pip install esm`",
)
_make_adapter(
    "tranception",
    "tranception",
    description="Tranception AR model",
    default_model_id="OATML-Markslab/Tranception_Large",
)
_make_adapter(
    "carp",
    "carp",
    description="CARP (Zenodo weights)",
    default_model_id="carp_640M",
    notes="Requires sequence_models; downloads from Zenodo",
)
_make_adapter(
    "s2f",
    "s2f",
    description="S2F (lightweight ESM2 if no checkpoint)",
    auto_download=True,
    notes="Full TorchDrug mode needs --s2f_checkpoint; config auto-bundled",
)
_make_adapter(
    "s3f",
    "s3f",
    description="S3F structure+sequence",
    needs_pdb=True,
    auto_download=True,
    notes="Uses data/s3f_weights/s3f.pth or cache/HF; config auto-bundled",
)
