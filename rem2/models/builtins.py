"""Built-in ModelAdapter registrations (wrap legacy baseline_dispatch)."""

from __future__ import annotations

from typing import Any, Optional

from rem2.models.base import ModelAdapter, ModelSpec
from rem2.models.registry import register_model


def _load_via_dispatch(
    cls,
    baseline_type: str,
    model_id: Optional[str],
    device: str,
    args: Any,
    logger: Any,
) -> ModelAdapter:
    from rem2.backbone.baseline_dispatch import load_baseline

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
    supports_tf: bool = False,
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
            supports_tf=supports_tf,
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
    notes="wt-marginals only; single ProSST + rem2 is not VenusREM2",
    aliases=("venusrem", "venusrem2", "prosst_ensemble", "prosst-ensemble"),
)
for _k in (20, 128, 512, 1024, 2048, 4096):
    _v1 = "wt only; VenusREM v1 backbone" if _k == 2048 else "wt-marginals only"
    _make_adapter(
        f"prosst-{_k}",
        "auto",
        description=f"ProSST-K{_k}",
        default_model_id=f"AI4Protein/ProSST-{_k}",
        extras="prosst",
        notes=_v1,
        aliases=(f"prosst_{_k}", f"prosst_k{_k}", f"prosst-k{_k}", f"prosst{_k}"),
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
    description="ESM-2 650M (default)",
    default_model_id="facebook/esm2_t33_650M_UR50D",
    aliases=("esm2-650m", "esm2_650m"),
    supports_mask=True,
)
for _name, _hf, _desc, _notes, _aliases in (
    ("esm2-8m", "facebook/esm2_t6_8M_UR50D", "ESM-2 8M", "rem2 demo / smoke", ("esm2_8m",)),
    ("esm2-35m", "facebook/esm2_t12_35M_UR50D", "ESM-2 35M", "", ("esm2_35m",)),
    ("esm2-150m", "facebook/esm2_t30_150M_UR50D", "ESM-2 150M", "", ("esm2_150m",)),
    ("esm2-3b", "facebook/esm2_t36_3B_UR50D", "ESM-2 3B", "", ("esm2_3b",)),
):
    _make_adapter(
        _name,
        "esm2",
        description=_desc,
        default_model_id=_hf,
        notes=_notes,
        aliases=_aliases,
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
    description="SaProt 650M AF2",
    default_model_id="westlake-repl/SaProt_650M_AF2",
    needs_pdb=True,
    notes="Foldseek auto-downloaded from HF if missing",
    aliases=("saprot-650m-af2", "saprot_650m_af2"),
    supports_mask=True,
)
_make_adapter(
    "saprot-35m-af2",
    "saprot",
    description="SaProt 35M AF2",
    default_model_id="westlake-repl/SaProt_35M_AF2",
    needs_pdb=True,
    aliases=("saprot_35m_af2", "saprot35m_af2"),
    supports_mask=True,
)
_make_adapter(
    "saprot-650m-pdb",
    "saprot",
    description="SaProt 650M PDB",
    default_model_id="westlake-repl/SaProt_650M_PDB",
    needs_pdb=True,
    aliases=("saprot_650m_pdb", "saprot650m_pdb"),
    supports_mask=True,
)
_make_adapter(
    "protssn",
    "protssn",
    description="ProtSSN structure GNN ensemble",
    needs_pdb=True,
    notes="9-model ensemble by default; --protssn_no_ensemble for one GNN",
    aliases=("protssn-ensemble", "protssn_ensemble"),
    supports_mask=True,
)
_make_adapter(
    "esm_if",
    "esm_if",
    description="ESM-IF1 inverse folding",
    needs_pdb=True,
    notes="Inverse folding (no mask); masked-marginals is refused",
    aliases=("esmif",),
)
_make_adapter(
    "protein_mpnn",
    "protein_mpnn",
    description="ProteinMPNN v_48_020",
    default_model_id="v_48_020.pt",
    needs_pdb=True,
    notes="teacher-force (tf); no mask",
    aliases=("proteinmpnn", "proteinmpnn-v_48_020", "proteinmpnn-020", "protein_mpnn-v_48_020"),
    supports_tf=True,
)
for _name, _ckpt, _aliases in (
    ("protein_mpnn-v_48_002", "v_48_002.pt", ("pmpnn_v_48_002", "proteinmpnn-v_48_002", "proteinmpnn-002")),
    ("protein_mpnn-v_48_010", "v_48_010.pt", ("pmpnn_v_48_010", "proteinmpnn-v_48_010", "proteinmpnn-010")),
    ("protein_mpnn-v_48_030", "v_48_030.pt", ("pmpnn_v_48_030", "proteinmpnn-v_48_030", "proteinmpnn-030")),
    ("protein_mpnn-soluble-v_48_002", "soluble_v_48_002.pt", ("pmpnn_soluble_v_48_002", "proteinmpnn-soluble-v_48_002")),
    ("protein_mpnn-soluble-v_48_010", "soluble_v_48_010.pt", ("pmpnn_soluble_v_48_010", "proteinmpnn-soluble-v_48_010")),
    ("protein_mpnn-soluble-v_48_020", "soluble_v_48_020.pt", ("pmpnn_soluble_v_48_020", "proteinmpnn-soluble-v_48_020")),
    ("protein_mpnn-soluble-v_48_030", "soluble_v_48_030.pt", ("pmpnn_soluble_v_48_030", "proteinmpnn-soluble-v_48_030")),
):
    _make_adapter(
        _name,
        "protein_mpnn",
        description=f"ProteinMPNN {_ckpt.replace('.pt', '')}",
        default_model_id=_ckpt,
        needs_pdb=True,
        notes="teacher-force (tf); no mask",
        aliases=_aliases,
        supports_tf=True,
    )
_make_adapter(
    "progen2",
    "progen2",
    description="ProGen2-L",
    default_model_id="hugohrban/progen2-large",
    notes="Causal LM (no mask); masked-marginals is refused",
    aliases=("progen2-l", "progen2_l"),
)
for _name, _hf, _aliases in (
    ("progen2-s", "hugohrban/progen2-small", ("progen2_s",)),
    ("progen2-m", "hugohrban/progen2-medium", ("progen2_m",)),
    ("progen2-b", "hugohrban/progen2-base", ("progen2_b",)),
    ("progen2-xl", "hugohrban/progen2-xlarge", ("progen2_xl",)),
):
    _make_adapter(
        _name,
        "progen2",
        description=_hf.split("/")[-1],
        default_model_id=_hf,
        notes="Causal LM (no mask)",
        aliases=_aliases,
    )
_make_adapter(
    "progen3",
    "progen3",
    description="ProGen3-1B",
    default_model_id="Profluent-Bio/progen3-1b",
    notes="Causal LM (no mask); masked-marginals is refused. May need flash-attn / megablocks",
    aliases=("progen3-1b", "progen3_1b"),
)
for _name, _hf, _aliases in (
    ("progen3-112m", "Profluent-Bio/progen3-112m", ("progen3_112m",)),
    ("progen3-219m", "Profluent-Bio/progen3-219m", ("progen3_219m",)),
    ("progen3-339m", "Profluent-Bio/progen3-339m", ("progen3_339m",)),
    ("progen3-762m", "Profluent-Bio/progen3-762m", ("progen3_762m",)),
    ("progen3-3b", "Profluent-Bio/progen3-3b", ("progen3_3b",)),
):
    _make_adapter(
        _name,
        "progen3",
        description=_hf.split("/")[-1],
        default_model_id=_hf,
        notes="Causal LM (no mask)",
        aliases=_aliases,
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
    description="RITA-XL",
    default_model_id="lightonai/RITA_xl",
    notes="Causal LM (no mask); masked-marginals is refused",
    aliases=("rita-xl", "rita_xl"),
)
for _name, _hf, _aliases in (
    ("rita-s", "lightonai/RITA_s", ("rita_s",)),
    ("rita-m", "lightonai/RITA_m", ("rita_m",)),
    ("rita-l", "lightonai/RITA_l", ("rita_l",)),
):
    _make_adapter(
        _name,
        "rita",
        description=_hf.split("/")[-1],
        default_model_id=_hf,
        notes="Causal LM (no mask)",
        aliases=_aliases,
    )
_make_adapter(
    "esm3",
    "esm3",
    description="ESM3 small open",
    default_model_id="esm3_sm_open_v1",
    extras="esm3",
    notes="Requires `pip install esm`",
    supports_mask=True,
)
_make_adapter(
    "esmc",
    "esm3",
    description="ESM-C 300M",
    default_model_id="esmc_300m",
    extras="esm3",
    aliases=("esmc-300m", "esmc_300m"),
    supports_mask=True,
)
_make_adapter(
    "esmc-600m",
    "esm3",
    description="ESM-C 600M",
    default_model_id="esmc_600m",
    extras="esm3",
    aliases=("esmc_600m",),
    supports_mask=True,
)
_make_adapter(
    "carp",
    "carp",
    description="CARP-640M",
    default_model_id="carp_640M",
    extras="carp",
    notes="Requires sequence_models; downloads from Zenodo; scores via mask",
    aliases=("carp-640m", "carp_640m"),
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
