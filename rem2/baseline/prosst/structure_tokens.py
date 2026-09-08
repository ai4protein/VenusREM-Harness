import inspect
import os

import torch


def tokenize_structure_sequence(structure_sequence):
    shift_structure_sequence = [i + 3 for i in structure_sequence]
    shift_structure_sequence = [1, *shift_structure_sequence, 2]
    return torch.tensor([shift_structure_sequence], dtype=torch.long)


def backbone_supports_structure_tokens(model):
    if model is None:
        return False
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return "ss_input_ids" in signature.parameters


def infer_structure_vocab_subdir(model_name):
    name = model_name.split("/")[-1]
    if "ProSST-" in name:
        return name.split("-")[-1]
    return None


def resolve_structure_fasta_path(args, protein_name, model_name):
    if args.struc_seq_dir is None:
        return None
    if args.backbone_mode == "plain_mlm":
        return None

    subdir = args.structure_vocab_subdir
    if subdir is None:
        subdir = infer_structure_vocab_subdir(model_name)

    if subdir is not None:
        candidate = f"{args.struc_seq_dir}/{subdir}/{protein_name}.fasta"
        if os.path.exists(candidate):
            return candidate
    candidate = f"{args.struc_seq_dir}/{protein_name}.fasta"
    if os.path.exists(candidate):
        return candidate
    return None
