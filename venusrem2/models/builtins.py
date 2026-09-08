"""Built-in ModelAdapter registrations (wrap legacy baseline_dispatch)."""

from __future__ import annotations

from typing import Any, Optional

from venusrem2.models.base import ModelAdapter, ModelSpec
from venusrem2.models.registry import register_model


def _load_via_dispatch(
    cls,
    baseline_type: str,
    model_id: Optional[str],
    device: str,
    args: Any,
    logger: Any,
) -> ModelAdapter:
    from venusrem2.backbone.baseline_dispatch import load_baseline

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
    supports_mask: bool = False,
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
            supports_mask=supports_mask,
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
    description="ProSST structure-aware MLM (VenusREM2 = rem2 on a ProSST ensemble)",
    default_model_id="AI4Protein/ProSST-2048",
    extras="prosst",
    notes="Single ProSST + rem2 is not VenusREM2; pass 2+ --model_name ProSST-* for the official ensemble",
    aliases=("venusrem", "venusrem2"),
    supports_mask=True,
)
_make_adapter(
    "auto",
    "auto",
    description="Any HuggingFace AutoModelForMaskedLM (pass --model_id)",
    default_model_id=None,
    notes="Generic HF MLM path; refuses masked-marginals if the tokenizer has no mask token",
    supports_mask=True,
)
_make_adapter(
    "esm2",
    "esm2",
    description="ESM-2 masked language model",
    default_model_id="facebook/esm2_t33_650M_UR50D",
    supports_mask=True,
)
_make_adapter(
    "esm2-8m",
    "esm2",
    description="ESM-2 8M (smoke / rem2 demo)",
    default_model_id="facebook/esm2_t6_8M_UR50D",
    notes="Small checkpoint for rem2 demo and install checks",
    aliases=("esm2_8m",),
    supports_mask=True,
)
_make_adapter(
    "esm1b",
    "esm1b",
    description="ESM-1b masked language model",
    default_model_id="facebook/esm1b_t33_650M_UR50S",
    supports_mask=True,
)
_make_adapter(
    "esm1v",
    "esm1v",
    description="ESM-1v 5-seed ensemble",
    default_model_id="facebook/esm1v_t33_650M_UR90S_1",
    notes="Uses --esm1v_seeds (default 1-5); ignores --model_id for weights",
    supports_mask=True,
)
_make_adapter(
    "saprot",
    "saprot",
    description="SaProt structure-aware PLM",
    default_model_id="westlake-repl/SaProt_650M_AF2",
    needs_pdb=True,
    notes="Foldseek auto-downloaded from HF if missing",
    supports_mask=True,
)
_make_adapter(
    "protssn",
    "protssn",
    description="ProtSSN structure GNN ensemble",
    needs_pdb=True,
    notes="Weights auto-download to cache/protssn",
    supports_mask=True,
)
_make_adapter(
    "esm_if",
    "esm_if",
    description="ESM-IF1 inverse folding",
    needs_pdb=True,
    notes="Inverse folding (no mask); masked-marginals is refused",
)
_make_adapter(
    "protein_mpnn",
    "protein_mpnn",
    description="ProteinMPNN",
    needs_pdb=True,
    notes="Inverse folding (no mask); masked-marginals is refused",
)
_make_adapter(
    "progen2",
    "progen2",
    description="ProGen2 causal LM",
    default_model_id="hugohrban/progen2-large",
    notes="Causal LM (no mask); masked-marginals is refused",
)
_make_adapter(
    "progen3",
    "progen3",
    description="ProGen3 causal LM",
    default_model_id="Profluent-Bio/progen3-1b",
    notes="Causal LM (no mask); masked-marginals is refused. May need flash-attn / megablocks",
)
_make_adapter(
    "protgpt2",
    "protgpt2",
    description="ProtGPT2 causal LM",
    default_model_id="nferruz/ProtGPT2",
    notes="Causal LM (no mask); masked-marginals is refused",
)
_make_adapter(
    "rita",
    "rita",
    description="RITA causal LM",
    default_model_id="lightonai/RITA_xl",
    notes="Causal LM (no mask); masked-marginals is refused",
)
_make_adapter(
    "esm3",
    "esm3",
    description="ESM3 / ESM-C",
    default_model_id="esmc_300m",
    extras="baselines",
    notes="Requires `pip install esm`",
    supports_mask=True,
)
_make_adapter(
    "tranception",
    "tranception",
    description="Tranception AR model",
    default_model_id="OATML-Markslab/Tranception_Large",
    notes="Autoregressive (no mask); masked-marginals is refused",
)
_make_adapter(
    "carp",
    "carp",
    description="CARP (Zenodo weights)",
    default_model_id="carp_640M",
    notes="Requires sequence_models; downloads from Zenodo; scores via mask",
    supports_mask=True,
)
_make_adapter(
    "mifst",
    "mifst",
    description="MIF-ST masked inverse folding (+ CARP-640M sequence transfer)",
    default_model_id="mifst",
    needs_pdb=True,
    extras="carp",
    notes="Teacher-force / unmasked (ProteinGym name is misleading); masked-marginals is refused",
    aliases=("mif_st", "mif-st"),
)
_make_adapter(
    "s2f",
    "s2f",
    description="S2F (lightweight ESM2 if no checkpoint)",
    auto_download=True,
    notes="No masked-marginals path (use esm2 / s3f); config auto-bundled",
)
_make_adapter(
    "s3f",
    "s3f",
    description="S3F structure+sequence",
    needs_pdb=True,
    auto_download=True,
    notes="Uses data/s3f_weights/s3f.pth or cache/HF; config auto-bundled",
    supports_mask=True,
)
