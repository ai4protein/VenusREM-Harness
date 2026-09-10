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


_PROSST_VOCABS = {20, 64, 128, 512, 1024, 2048, 4096}


def needed_structure_vocab_sizes(model_key, args=None):
    from rem2.naming import PROSST_ENSEMBLE_SIZES, is_ensemble_model_key

    if is_ensemble_model_key(model_key):
        return list(PROSST_ENSEMBLE_SIZES)
    sizes = []
    for mid in getattr(args, "model_name", None) or []:
        sub = infer_structure_vocab_subdir(str(mid))
        if sub and str(sub).isdigit() and int(sub) in _PROSST_VOCABS:
            sizes.append(int(sub))
    if not sizes:
        for token in str(model_key or "").lower().replace("_", "-").split("-"):
            if token.isdigit() and int(token) in _PROSST_VOCABS:
                sizes.append(int(token))
    return sizes or [2048]


def write_struc_seq_fasta(path, name, tokens):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f">{name}\n")
        handle.write(",".join(str(i) for i in tokens))
        handle.write("\n")


def _token_paths(dest_dir, protein_name, vocab, vocab_sizes):
    nested = os.path.join(dest_dir, str(vocab), f"{protein_name}.fasta")
    flat = os.path.join(dest_dir, f"{protein_name}.fasta")
    return nested, flat if len(vocab_sizes) == 1 else None


def generate_struc_seq_from_pdbs(items, dest_dir, vocab_sizes):
    """Build ProSST tokens for ``[(pdb_path, protein_name), ...]``. Returns dest_dir."""
    from rem2.baseline.prosst.get_sst_seq import SSTPredictor

    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    vocab_sizes = list(vocab_sizes)
    for vocab in vocab_sizes:
        todo = []
        for pdb_path, protein_name in items:
            nested, flat = _token_paths(dest_dir, protein_name, vocab, vocab_sizes)
            if os.path.exists(nested) or (flat and os.path.exists(flat)):
                continue
            todo.append((pdb_path, protein_name))
        if not todo:
            continue
        predictor = SSTPredictor(
            structure_vocab_size=vocab,
            # One means sequential preprocessing and in-process DataLoader
            # collation; see get_sst_seq.graph_conventer.
            num_processes=1,
            num_threads=1,
        )
        results = predictor.predict_from_pdb([path for path, _ in todo])
        if len(results) != len(todo):
            raise RuntimeError(
                f"ProSST tokenizer returned {len(results)} results for {len(todo)} PDBs (K={vocab})"
            )
        for (_, protein_name), result in zip(todo, results):
            tokens = result[f"{vocab}_sst_seq"]
            nested, flat = _token_paths(dest_dir, protein_name, vocab, vocab_sizes)
            write_struc_seq_fasta(nested, protein_name, tokens)
            if flat:
                write_struc_seq_fasta(flat, protein_name, tokens)
    return dest_dir


def generate_struc_seq_from_pdb(pdb_path, dest_dir, protein_name, vocab_sizes):
    """Build ProSST structure-token FASTAs from a PDB. Returns dest_dir."""
    return generate_struc_seq_from_pdbs([(str(pdb_path), protein_name)], dest_dir, vocab_sizes)
