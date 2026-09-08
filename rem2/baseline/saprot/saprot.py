import os
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from rem2.baseline.saprot.foldseek_util import get_struc_seq

FOLDSEEK_STRUC_VOCAB = "pynwrqhgdlvtmfsaeikc#"
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"


def get_saprot_3di(foldseek_bin: str, pdb_file: str, process_id: int = 0) -> str:
    result = get_struc_seq(
        foldseek_bin, pdb_file, chains=["A"],
        plddt_mask=True, plddt_threshold=70.0, process_id=process_id,
    )
    if "A" not in result:
        chains = list(result.keys())
        if not chains:
            raise ValueError(f"No chains found in PDB: {pdb_file}")
        chain = chains[0]
    else:
        chain = "A"
    return result[chain][1].lower()


def build_saprot_joint_sequence(aa_sequence: str, struc_3di: str) -> str:
    if len(aa_sequence) != len(struc_3di):
        min_len = min(len(aa_sequence), len(struc_3di))
        aa_sequence = aa_sequence[:min_len]
        struc_3di = struc_3di[:min_len]
    return "".join(aa + ss for aa, ss in zip(aa_sequence, struc_3di))


def build_projection_map(saprot_vocab: dict, esm_vocab: dict):
    n_struc = len(FOLDSEEK_STRUC_VOCAB)
    aa_to_saprot_start = {}
    for aa in AA_LIST:
        key = aa + FOLDSEEK_STRUC_VOCAB[0]
        if key in saprot_vocab:
            aa_to_saprot_start[aa] = saprot_vocab[key]

    aa_to_esm_idx = {}
    for aa in AA_LIST:
        if aa in esm_vocab:
            aa_to_esm_idx[aa] = esm_vocab[aa]

    return aa_to_saprot_start, aa_to_esm_idx, n_struc


@torch.no_grad()
def forward_saprot_masked_marginal(
    model,
    saprot_tokenizer: AutoTokenizer,
    sequence: str,
    pdb_file: str,
    foldseek_bin: str,
    device: torch.device,
    esm_tokenizer: AutoTokenizer,
    process_id: int = 0,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    struc_3di = get_saprot_3di(foldseek_bin, pdb_file, process_id=process_id)
    joint_seq = build_saprot_joint_sequence(sequence, struc_3di)
    tokens = saprot_tokenizer.tokenize(joint_seq)
    seq_len = len(tokens)

    if logger is not None:
        logger.debug(
            f"SaProt masked marginal: AA={len(sequence)}, 3Di={len(struc_3di)}, tokens={seq_len}",
            protein=protein_name,
        )

    saprot_vocab = saprot_tokenizer.get_vocab()
    esm_vocab = esm_tokenizer.get_vocab()
    esm_vocab_size = esm_tokenizer.vocab_size
    aa_to_saprot_start, aa_to_esm_idx, n_struc = build_projection_map(saprot_vocab, esm_vocab)

    projected = torch.full((seq_len, esm_vocab_size), -1e9, device=device)

    for i in range(seq_len):
        masked_tokens = list(tokens)
        masked_tokens[i] = "#" + tokens[i][-1]
        mask_seq = " ".join(masked_tokens)
        inputs = saprot_tokenizer(mask_seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        probs = outputs.logits[0, i + 1].softmax(dim=-1)

        for aa in AA_LIST:
            if aa not in aa_to_saprot_start or aa not in aa_to_esm_idx:
                continue
            start = aa_to_saprot_start[aa]
            aa_prob = probs[start : start + n_struc].sum()
            projected[i, aa_to_esm_idx[aa]] = torch.log(aa_prob.clamp_min(1e-12))

    return projected


@torch.no_grad()
def forward_saprot_wt_marginal(
    model,
    saprot_tokenizer: AutoTokenizer,
    sequence: str,
    pdb_file: str,
    foldseek_bin: str,
    device: torch.device,
    esm_tokenizer: AutoTokenizer,
    process_id: int = 0,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    struc_3di = get_saprot_3di(foldseek_bin, pdb_file, process_id=process_id)
    joint_seq = build_saprot_joint_sequence(sequence, struc_3di)
    tokens = saprot_tokenizer.tokenize(joint_seq)
    seq_len = len(tokens)

    if logger is not None:
        logger.debug(
            f"SaProt WT marginal: AA={len(sequence)}, 3Di={len(struc_3di)}, tokens={seq_len}",
            protein=protein_name,
        )

    saprot_vocab = saprot_tokenizer.get_vocab()
    esm_vocab = esm_tokenizer.get_vocab()
    esm_vocab_size = esm_tokenizer.vocab_size
    aa_to_saprot_start, aa_to_esm_idx, n_struc = build_projection_map(saprot_vocab, esm_vocab)

    inputs = saprot_tokenizer(joint_seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    all_logits = outputs.logits[0, 1 : seq_len + 1]
    probs = all_logits.softmax(dim=-1)

    projected = torch.full((seq_len, esm_vocab_size), -1e9, device=device)
    for aa in AA_LIST:
        if aa not in aa_to_saprot_start or aa not in aa_to_esm_idx:
            continue
        start = aa_to_saprot_start[aa]
        aa_probs = probs[:, start : start + n_struc].sum(dim=-1)
        projected[:, aa_to_esm_idx[aa]] = torch.log(aa_probs.clamp_min(1e-12))

    return projected


def load_saprot_model_and_tokenizers(model_name: str, device: torch.device):
    model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device).eval()
    saprot_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    esm_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esm2_t6_8M_UR50D", trust_remote_code=True
    )
    return model, saprot_tokenizer, esm_tokenizer
