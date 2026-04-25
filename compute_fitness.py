
import torch
import os
import json
import random
import inspect
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForMaskedLM
from argparse import ArgumentParser, Namespace
from src.orbit.pipeline import maybe_fuse_logits_with_orbit

amino_acid_properties = {
    'A': {'hydrophobicity': 1.8,  'charge':  0, 'polarity':  0, 'molecular_weight':  89.09, 'volume':  88.6},
    'R': {'hydrophobicity': -4.5, 'charge': +1, 'polarity':  1, 'molecular_weight': 174.20, 'volume': 173.4},
    'N': {'hydrophobicity': -3.5, 'charge':  0, 'polarity':  1, 'molecular_weight': 132.12, 'volume': 114.1},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'polarity':  1, 'molecular_weight': 133.10, 'volume': 111.1},
    'C': {'hydrophobicity': 2.5,  'charge':  0, 'polarity':  0, 'molecular_weight': 121.15, 'volume': 108.5},
    'Q': {'hydrophobicity': -3.5, 'charge':  0, 'polarity':  1, 'molecular_weight': 146.15, 'volume': 143.8},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'polarity':  1, 'molecular_weight': 147.13, 'volume': 138.4},
    'G': {'hydrophobicity': -0.4, 'charge':  0, 'polarity':  0, 'molecular_weight':  75.07, 'volume':  60.1},
    'H': {'hydrophobicity': -3.2, 'charge':  0, 'polarity':  1, 'molecular_weight': 155.16, 'volume': 153.2},
    'I': {'hydrophobicity': 4.5,  'charge':  0, 'polarity':  0, 'molecular_weight': 131.17, 'volume': 166.7},
    'L': {'hydrophobicity': 3.8,  'charge':  0, 'polarity':  0, 'molecular_weight': 131.17, 'volume': 166.7},
    'K': {'hydrophobicity': -3.9, 'charge': +1, 'polarity':  1, 'molecular_weight': 146.19, 'volume': 168.6},
    'M': {'hydrophobicity': 1.9,  'charge':  0, 'polarity':  0, 'molecular_weight': 149.21, 'volume': 162.9},
    'F': {'hydrophobicity': 2.8,  'charge':  0, 'polarity':  0, 'molecular_weight': 165.19, 'volume': 189.9},
    'P': {'hydrophobicity': -1.6, 'charge':  0, 'polarity':  0, 'molecular_weight': 115.13, 'volume': 112.7},
    'S': {'hydrophobicity': -0.8, 'charge':  0, 'polarity':  1, 'molecular_weight': 105.09, 'volume':  89.0},
    'T': {'hydrophobicity': -0.7, 'charge':  0, 'polarity':  1, 'molecular_weight': 119.12, 'volume': 116.1},
    'W': {'hydrophobicity': -0.9, 'charge':  0, 'polarity':  0, 'molecular_weight': 204.23, 'volume': 227.8},
    'Y': {'hydrophobicity': -1.3, 'charge':  0, 'polarity':  1, 'molecular_weight': 181.19, 'volume': 193.6},
    'V': {'hydrophobicity': 4.2,  'charge':  0, 'polarity':  0, 'molecular_weight': 117.15, 'volume': 140.0},
}
device = "cuda" if torch.cuda.is_available() else "cpu"


class CliLogger:
    LEVEL_ORDER = {"debug": 10, "info": 20, "warn": 30, "error": 40}
    RESET = "\033[0m"
    COLORS = {
        "debug": "\033[36m",
        "info": "\033[34m",
        "warn": "\033[33m",
        "error": "\033[31m",
        "success": "\033[32m",
        "section": "\033[35m",
        "protein": "\033[96m",
    }

    def __init__(self, level="info", use_color=True):
        self.level = level if level in self.LEVEL_ORDER else "info"
        self.use_color = use_color

    def _allowed(self, level):
        return self.LEVEL_ORDER.get(level, 20) >= self.LEVEL_ORDER[self.level]

    def _paint(self, text, key):
        if not self.use_color:
            return text
        return f"{self.COLORS.get(key, '')}{text}{self.RESET}"

    def log(self, level, message, protein=None):
        if not self._allowed(level):
            return
        ts = datetime.now().strftime("%H:%M:%S")
        level_tag = self._paint(level.upper().ljust(5), level)
        protein_tag = ""
        if protein:
            protein_tag = f" {self._paint(f'[{protein}]', 'protein')}"
        print(f"[{ts}] {level_tag}{protein_tag} {message}")

    def debug(self, message, protein=None):
        self.log("debug", message, protein=protein)

    def info(self, message, protein=None):
        self.log("info", message, protein=protein)

    def warn(self, message, protein=None):
        self.log("warn", message, protein=protein)

    def error(self, message, protein=None):
        self.log("error", message, protein=protein)

    def success(self, message, protein=None):
        ts = datetime.now().strftime("%H:%M:%S")
        tag = self._paint("OK   ", "success")
        protein_tag = ""
        if protein:
            protein_tag = f" {self._paint(f'[{protein}]', 'protein')}"
        print(f"[{ts}] {tag}{protein_tag} {message}")

    def section(self, title):
        divider = "=" * 72
        if self.use_color:
            divider = self._paint(divider, "section")
            title = self._paint(title, "section")
        print(divider)
        print(title)
        print(divider)


def should_use_color(no_color=False):
    if no_color:
        return False
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("FORCE_COLOR"):
        return True
    return sys.stdout.isatty()


def read_multi_fasta(file_path):
    """
    params:
        file_path: path to a fasta file
    return:
        a dictionary of sequences
    """
    sequences = {}
    current_sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences[header] = current_sequence.upper().replace('-', '<pad>').replace('.', '<pad>')
                    current_sequence = ''
                header = line
            else:
                current_sequence += line
        if current_sequence:
            sequences[header] = current_sequence
    return sequences


def read_seq(fasta):
    for record in SeqIO.parse(fasta, "fasta"):
        return str(record.seq)

def count_matrix_from_residue_alignment(tokenizer, alignment_dict, verbose=True, logger=None, protein_name=None):
    alignment_seqs = list(alignment_dict.values())
    try:
        aln_start, aln_end = list(alignment_dict.keys())[0].split('/')[-1].split('-')
    except:
        aln_start, aln_end = 1, len(alignment_seqs[0])
    if verbose:
        msg1 = f"Alignment span: start={aln_start}, end={aln_end}"
        msg2 = f"Tokenizing residue alignments: n={len(alignment_seqs)}"
        if logger is not None:
            logger.debug(msg1, protein=protein_name)
            logger.debug(msg2, protein=protein_name)
        else:
            print(f">>> {msg1}")
            print(f">>> {msg2}")
    tokenized_results = tokenizer(alignment_seqs, return_tensors="pt", padding=True)
    alignment_ids = tokenized_results["input_ids"][:,1:-1]
    return alignment_ids, int(aln_start)-1, int(aln_end)
    # count distribution of each column, [seq_len, vocab_size]
    count_matrix = torch.zeros(alignment_ids.size(1), tokenizer.vocab_size)
    for i in tqdm(range(alignment_ids.size(1))):
        count_matrix[i] = torch.bincount(alignment_ids[:,i], minlength=tokenizer.vocab_size)
    # calculate coverage of each column and normalize count matrix
    # coverage = (1.0 - (count_matrix == tokenizer.pad_token_id).float().mean(dim=-1)).unsqueeze(-1).to(device)
    
    count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
    # count_matrix = count_matrix * coverage
    return count_matrix, int(aln_start)-1, int(aln_end)


def count_matrix_from_structure_alignment(tokenizer, alignment_dict, verbose=True, logger=None, protein_name=None):
    alignment_seqs = list(alignment_dict.values())
    if verbose:
        msg = f"Tokenizing structure alignments: n={len(alignment_seqs)}"
        if logger is not None:
            logger.debug(msg, protein=protein_name)
        else:
            print(f">>> {msg}")
    if len(alignment_seqs) == 0:
        return None
    tokenized_results = tokenizer(alignment_seqs, return_tensors="pt", padding=True)
    alignment_ids = tokenized_results["input_ids"][:,1:-1]
    return alignment_ids
    # count distribution of each column, [seq_len, vocab_size]
    count_matrix = torch.zeros(alignment_ids.size(1), tokenizer.vocab_size)
    for i in tqdm(range(alignment_ids.size(1))):
        count_matrix[i] = torch.bincount(alignment_ids[:,i], minlength=tokenizer.vocab_size)
    return count_matrix
    count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
    return count_matrix
    


def calculate_property_difference(wild_aa, mutant_aa, weights=None):
    properties = amino_acid_properties[wild_aa].keys()
    if weights is None:
        weights = {prop: 1 for prop in properties}
    differences = []
    for prop in properties:
        wild_value = amino_acid_properties[wild_aa][prop]
        mutant_value = amino_acid_properties[mutant_aa][prop]
        difference = abs(mutant_value - wild_value)
        weighted_diff = weights.get(prop, 1) * difference
        differences.append(weighted_diff)
    return differences

def tokenize_structure_sequence(structure_sequence):
    shift_structure_sequence = [i + 3 for i in structure_sequence]
    shift_structure_sequence = [1, *shift_structure_sequence, 2]
    return torch.tensor([shift_structure_sequence,], dtype=torch.long)


def backbone_supports_structure_tokens(model):
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


def infer_model_max_residue_len(model, tokenizer):
    """
    Infer the maximum residue length (without BOS/EOS) supported by a backbone.
    """
    config_candidates = []
    config = getattr(model, "config", None)
    for attr in ["max_position_embeddings", "n_positions", "max_sequence_length"]:
        value = getattr(config, attr, None) if config is not None else None
        if isinstance(value, int) and value > 2:
            config_candidates.append(value - 2)

    # Prefer model config. Tokenizer max length can be conservative and may not
    # reflect the actual backbone limit for some remote-code models.
    if config_candidates:
        return min(config_candidates)

    tok_max = getattr(tokenizer, "model_max_length", None)
    # HuggingFace often uses huge sentinels when max length is unknown.
    if isinstance(tok_max, int) and 2 < tok_max < 1_000_000:
        return tok_max - 2

    return None


def force_config_max_residue_len(model, residue_len):
    """
    Force backbone config max length to a target residue length.
    We store token-level length in config (residue + BOS/EOS).
    """
    if residue_len is None or residue_len <= 0:
        return
    token_len = int(residue_len) + 2
    config = getattr(model, "config", None)
    if config is None:
        return
    for attr in ["max_position_embeddings", "n_positions", "max_sequence_length"]:
        if hasattr(config, attr):
            setattr(config, attr, token_len)


@torch.no_grad()
def forward_sequence_logits(
    model,
    tokenizer,
    sequence,
    use_structure=False,
    structure_sequence=None,
    max_residue_len=None,
    long_seq_mode="auto_window",
    long_seq_overlap=256,
    logger=None,
    protein_name=None,
):
    """
    Run backbone forward and return per-residue log-softmax logits [L, vocab].
    For long sequences, optionally use overlapping windows and average overlaps.
    """
    seq_len = len(sequence)
    if max_residue_len is None or seq_len <= max_residue_len:
        tokenized_results = tokenizer([sequence], return_tensors="pt")
        input_ids = tokenized_results["input_ids"].to(device)
        attention_mask = tokenized_results["attention_mask"].to(device)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }
        if use_structure:
            if structure_sequence is None:
                raise ValueError("structure_sequence is required when use_structure=True")
            ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
            model_inputs["ss_input_ids"] = ss_input_ids
        outputs = model(**model_inputs)
        return torch.log_softmax(outputs.logits[0][1:-1, :], dim=-1)

    if long_seq_mode == "error":
        raise ValueError(
            f"Sequence length {seq_len} exceeds model limit {max_residue_len}. "
            "Use --long_seq_mode auto_window (default) or set --max_residue_len."
        )

    if max_residue_len < 1:
        raise ValueError(f"Invalid max_residue_len={max_residue_len}")

    if use_structure and structure_sequence is None:
        raise ValueError("structure_sequence is required when use_structure=True")
    if use_structure and len(structure_sequence) != seq_len:
        raise ValueError(
            f"Structure sequence length mismatch: {len(structure_sequence)} vs residue length {seq_len}"
        )

    overlap = max(0, int(long_seq_overlap))
    if overlap >= max_residue_len:
        overlap = max(0, max_residue_len // 2)
    step = max(1, max_residue_len - overlap)

    msg = (
        f"Long sequence detected: L={seq_len} > {max_residue_len}; "
        f"using sliding-window inference (window={max_residue_len}, overlap={overlap})"
    )
    if logger is not None:
        logger.warn(msg, protein=protein_name)
    else:
        print(f">>> {msg}")

    aggregated_logits = None
    counts = None
    start = 0
    while start < seq_len:
        end = min(seq_len, start + max_residue_len)
        window_seq = sequence[start:end]
        tokenized_results = tokenizer([window_seq], return_tensors="pt")
        input_ids = tokenized_results["input_ids"].to(device)
        attention_mask = tokenized_results["attention_mask"].to(device)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }
        if use_structure:
            window_structure = structure_sequence[start:end]
            ss_input_ids = tokenize_structure_sequence(window_structure).to(device)
            model_inputs["ss_input_ids"] = ss_input_ids

        outputs = model(**model_inputs)
        window_logits = torch.log_softmax(outputs.logits[0][1:-1, :], dim=-1)
        window_len = min(window_logits.size(0), end - start)
        window_logits = window_logits[:window_len]

        if aggregated_logits is None:
            aggregated_logits = torch.zeros(
                seq_len,
                window_logits.size(-1),
                device=window_logits.device,
                dtype=window_logits.dtype,
            )
            counts = torch.zeros(seq_len, 1, device=window_logits.device, dtype=window_logits.dtype)

        aggregated_logits[start : start + window_len, :] += window_logits
        counts[start : start + window_len, :] += 1.0

        if end >= seq_len:
            break
        start += step

    counts = counts.clamp_min(1.0)
    return aggregated_logits / counts


@torch.no_grad()
def score_protein(model, tokenizer, residue_fasta, structure_fasta, mutant_df, 
                  alpha=0.7, aa_seq_aln_file=None, struc_seq_aln_file=None,
                  sample_size=None, sample_ratio=1.0, sample_times=1,
                  orbit_args=None, protein_name=None, backbone_mode="auto", pdb_file=None,
                  max_residue_len=None, long_seq_mode="auto_window", long_seq_overlap=256,
                  quiet=False, show_progress=True, logger=None):
    def log_local(msg):
        if not quiet:
            if logger is not None:
                logger.info(msg, protein=protein_name)
            elif protein_name:
                print(f"[{protein_name}] {msg}")
            else:
                print(msg)

    sequence = read_seq(residue_fasta)

    supports_structure = backbone_supports_structure_tokens(model)
    if backbone_mode == "prosst":
        use_structure = True
    elif backbone_mode == "plain_mlm":
        use_structure = False
    else:
        use_structure = supports_structure and structure_fasta is not None

    if use_structure and not supports_structure:
        raise ValueError(
            "Current backbone does not support structure tokens (ss_input_ids). "
            "Use --backbone_mode plain_mlm or auto for this model."
        )

    if use_structure:
        if structure_fasta is None or not os.path.exists(structure_fasta):
            raise FileNotFoundError(
                f"Structure sequence is required for backbone_mode={backbone_mode}, missing: {structure_fasta}"
            )
        structure_sequence = read_seq(structure_fasta)
        structure_sequence = [int(i) for i in structure_sequence.split(",")]
    else:
        structure_sequence = None

    logits = forward_sequence_logits(
        model=model,
        tokenizer=tokenizer,
        sequence=sequence,
        use_structure=use_structure,
        structure_sequence=structure_sequence,
        max_residue_len=max_residue_len,
        long_seq_mode=long_seq_mode,
        long_seq_overlap=long_seq_overlap,
        logger=logger,
        protein_name=protein_name,
    )
    
    if orbit_args is not None and orbit_args.orbit_enable and alpha != 0:
        logits = maybe_fuse_logits_with_orbit(
            args=orbit_args,
            tokenizer=tokenizer,
            plm_logits=logits,
            aa_seq_aln_file=aa_seq_aln_file,
            struc_seq_aln_file=struc_seq_aln_file,
            protein_name=protein_name,
            residue_fasta=residue_fasta,
            structure_fasta=structure_fasta,
            pdb_file=pdb_file,
        )
    elif alpha != 0:
        if aa_seq_aln_file is not None and struc_seq_aln_file is None:
            log_local("Using residue sequence alignment matrix")
            alignment_dict = read_multi_fasta(aa_seq_aln_file)
            alignment_matrix, aln_start, aln_end = count_matrix_from_residue_alignment(
                tokenizer, alignment_dict, verbose=not quiet, logger=logger, protein_name=protein_name
            )
            for sample in range(sample_times):
                if sample_ratio < 1.0:
                    log_local(f"Sample {sample+1}/{sample_times} with ratio {sample_ratio}")
                    sample_size = int(len(alignment_matrix) * sample_ratio)
                    sample_indices = random.sample(range(len(alignment_matrix)), sample_size)
                    alignment_matrix_sample = alignment_matrix[sample_indices]
                else:
                    alignment_matrix_sample = alignment_matrix
                
                count_matrix = torch.zeros(alignment_matrix_sample.size(1), tokenizer.vocab_size)
                for i in tqdm(
                    range(alignment_matrix_sample.size(1)),
                    disable=(not show_progress) or quiet,
                    leave=False,
                ):
                    count_matrix[i] = torch.bincount(alignment_matrix_sample[:,i], minlength=tokenizer.vocab_size)
                
                count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
                count_matrix = torch.log_softmax(count_matrix, dim=-1)
                aln_modify_logits = (1-alpha) * logits[aln_start: aln_end, :] + alpha * count_matrix
                logits = torch.cat([logits[:aln_start], aln_modify_logits, logits[aln_end:]], dim=0)

        if struc_seq_aln_file is not None and aa_seq_aln_file is None:
            log_local("Using structure sequence alignment matrix")
            alignment_dict = read_multi_fasta(struc_seq_aln_file)
            alignment_matrix = count_matrix_from_structure_alignment(
                tokenizer, alignment_dict, verbose=not quiet, logger=logger, protein_name=protein_name
            )
            if alignment_matrix is not None:
                count_matrix = torch.zeros(alignment_matrix.size(1), tokenizer.vocab_size)
                for i in tqdm(
                    range(alignment_matrix.size(1)),
                    disable=(not show_progress) or quiet,
                    leave=False,
                ):
                    count_matrix[i] = torch.bincount(alignment_matrix[:,i], minlength=tokenizer.vocab_size)
                count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
                count_matrix = torch.log_softmax(count_matrix, dim=-1)
                logits = (1-alpha) * logits + alpha * count_matrix
                
        if aa_seq_aln_file is not None and struc_seq_aln_file is not None:
            log_local("Using both residue and structure sequence alignment matrix")
            plm_logits = logits.clone()
            
            alignment_dict = read_multi_fasta(struc_seq_aln_file)
            structure_alignment_matrix = count_matrix_from_structure_alignment(
                tokenizer, alignment_dict, verbose=not quiet, logger=logger, protein_name=protein_name
            )
            if structure_alignment_matrix is not None:
                count_matrix = torch.zeros(structure_alignment_matrix.size(1), tokenizer.vocab_size)
                for i in tqdm(
                    range(structure_alignment_matrix.size(1)),
                    disable=(not show_progress) or quiet,
                    leave=False,
                ):
                    count_matrix[i] = torch.bincount(structure_alignment_matrix[:,i], minlength=tokenizer.vocab_size)
                
                count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
                count_matrix = torch.log_softmax(count_matrix, dim=-1)
                logits = (1-alpha) * plm_logits + alpha * count_matrix
            
            alignment_dict = read_multi_fasta(aa_seq_aln_file)
            residue_alignment_matrix, aln_start, aln_end = count_matrix_from_residue_alignment(
                tokenizer, alignment_dict, verbose=not quiet, logger=logger, protein_name=protein_name
            )
            count_matrix = torch.zeros(residue_alignment_matrix.size(1), tokenizer.vocab_size)
            for i in tqdm(
                range(residue_alignment_matrix.size(1)),
                disable=(not show_progress) or quiet,
                leave=False,
            ):
                count_matrix[i] = torch.bincount(residue_alignment_matrix[:,i], minlength=tokenizer.vocab_size)
            
            count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
            count_matrix = torch.log_softmax(count_matrix, dim=-1)
            aln_modify_logits = (1-alpha) * logits[aln_start: aln_end, :] + alpha * count_matrix
            logits = torch.cat([plm_logits[:aln_start], aln_modify_logits, plm_logits[aln_end:]], dim=0)
    else:
        log_local("No alignment matrix used")
    
    
    mutants = mutant_df["mutant"].tolist()
    scores = []
    vocab = tokenizer.get_vocab()
    log_local("Scoring mutants")
    mutant_desc = f"Mutants[{protein_name}]" if protein_name else "Mutants"
    for mutant in tqdm(
        mutants,
        desc=mutant_desc,
        disable=(not show_progress) or quiet,
        leave=True,
        dynamic_ncols=True,
    ):
        pred_score = 0
        for sub_mutant in mutant.split(":"):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            assert sequence[idx] == wt, f"Wild type mismatch: {sequence[idx]} != {wt}, idx {idx}"
            score = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
            pred_score += score.item()
        scores.append(pred_score)

    return scores
    


def read_names(fasta_dir):
    files = Path(fasta_dir).glob("*.fasta")
    names = [file.stem for file in files]
    return names


def clone_args_with_overrides(args, **kwargs):
    copied = Namespace(**vars(args))
    for key, value in kwargs.items():
        setattr(copied, key, value)
    return copied


def format_name_preview(names, max_items=8):
    if len(names) <= max_items:
        return ", ".join(names)
    head_n = max(1, max_items // 2)
    tail_n = max_items - head_n
    head = ", ".join(names[:head_n])
    tail = ", ".join(names[-tail_n:])
    return f"{head}, ..., {tail}"


def _fit_text(text, width):
    text = str(text)
    if len(text) <= width:
        return text.ljust(width)
    return (text[: width - 1] + "…") if width > 1 else text[:width]


def print_compare_table_header(
    logger,
    include_venus=True,
    raw_label="Raw",
    venus_label="VenusREM",
    orbit_label="Orbit",
    current_label="Current",
):
    protein_w = 40
    metric_w = 9
    delta_w = 8
    if include_venus:
        raw_h = _fit_text(raw_label, metric_w)
        venus_h = _fit_text(venus_label, metric_w)
        orbit_h = _fit_text(orbit_label, metric_w)
        header = (
            f"| {'Protein'.ljust(protein_w)} | {raw_h.rjust(metric_w)} | "
            f"{venus_h.rjust(metric_w)} | {orbit_h.rjust(metric_w)} | {'Delta'.rjust(delta_w)} |"
        )
        bar = (
            f"+-{'-' * protein_w}-+-{'-' * metric_w}-+-{'-' * metric_w}-+-{'-' * metric_w}-+-{'-' * delta_w}-+"
        )
    else:
        raw_h = _fit_text(raw_label, metric_w)
        current_h = _fit_text(current_label, metric_w)
        header = (
            f"| {'Protein'.ljust(protein_w)} | {raw_h.rjust(metric_w)} | "
            f"{current_h.rjust(metric_w)} | {'Delta'.rjust(delta_w)} |"
        )
        bar = f"+-{'-' * protein_w}-+-{'-' * metric_w}-+-{'-' * metric_w}-+-{'-' * delta_w}-+"
    logger.info(bar)
    logger.info(header)
    logger.info(bar)


def print_compare_table_row(logger, protein_name, raw_corr, orbit_corr, venus_corr=None):
    protein_w = 40
    metric_w = 9
    delta_w = 8
    raw_s = f"{raw_corr:.4f}".rjust(metric_w)
    orbit_s = f"{orbit_corr:.4f}".rjust(metric_w)
    protein_s = _fit_text(protein_name, protein_w)
    if venus_corr is not None:
        venus_s = f"{venus_corr:.4f}".rjust(metric_w)
        delta = orbit_corr - venus_corr
        delta_s = f"{delta:+.4f}".rjust(delta_w)
        if logger.use_color:
            delta_color = "success" if delta >= 0 else "error"
            delta_s = logger._paint(delta_s, delta_color)
        row = f"| {protein_s} | {raw_s} | {venus_s} | {orbit_s} | {delta_s} |"
    else:
        delta = orbit_corr - raw_corr
        delta_s = f"{delta:+.4f}".rjust(delta_w)
        if logger.use_color:
            delta_color = "success" if delta >= 0 else "error"
            delta_s = logger._paint(delta_s, delta_color)
        row = f"| {protein_s} | {raw_s} | {orbit_s} | {delta_s} |"
    logger.info(row)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default=["AI4Protein/ProSST-2048"], nargs="+", help="Model name",)
    parser.add_argument("--model_out_name", type=str, default=["VenusREM"], nargs="+", help="Output model name",)
    
    # data directories
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory containing all data",)
    parser.add_argument("--aa_seq_dir", type=str, default=None, help="Directory containing FASTA files of residue sequences",)
    parser.add_argument("--struc_seq_dir", type=str, default=None, help="Directory containing FASTA files of structure sequences",)
    parser.add_argument("--mutant_dir", type=str, default=None, help="Directory containing CSV files with mutants",)
    
    # retrieval and logits mode
    parser.add_argument("--logit_mode", type=str, default="aa_seq_aln", choices=["aa_seq_aln", "struc_seq_aln", "aa_seq_aln+struc_seq_aln", "struc_seq_aln+aa_seq_aln"], help="Mode to retrieve data",)
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha value for combining logits",)
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use",)
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Ratio of samples to use",)
    parser.add_argument("--sample_times", type=int, default=1, help="Number of times to sample",)
    parser.add_argument("--aa_seq_aln_dir", type=str, default=None, help="Directory containing a2m files of residue alignments",)
    parser.add_argument("--struc_seq_aln_dir", type=str, default=None, help="Directory containing fasta files of foldseek structure alignments",)
    
    # orbit plugin options
    parser.add_argument("--orbit_enable", action="store_true", help="Enable VenusREM-Orbit plugin pipeline")
    parser.add_argument("--retriever", type=str, default="msa", choices=["msa", "hits", "psalor_exact", "psalor_variant"], help="Retriever plugin used in Orbit mode")
    parser.add_argument("--retriever2", type=str, default="none", choices=["none", "msa", "hits", "psalor_exact", "psalor_variant"], help="Second-layer retriever plugin")
    parser.add_argument("--fusion", type=str, default="linear_alpha", choices=["linear_alpha", "adaptive_gate", "two_stage_learnable_gate"], help="Primary fusion plugin used in Orbit mode")
    parser.add_argument("--fusion2", type=str, default="none", choices=["none", "two_stage_learnable_gate"], help="Second-stage fusion plugin")
    parser.add_argument("--hits_dir", type=str, default=None, help="Directory containing homolog hits metadata files")
    parser.add_argument("--hits_file_suffix", type=str, default="_tblout.txt", help="Suffix of hits metadata files")
    parser.add_argument("--psalor_mode", type=str, default="none", choices=["none", "exact", "variant", "both"], help="Automatic PSALOR layer2 mode when retriever2 is not set")
    parser.add_argument("--psalor_mix", type=float, default=0.5, help="Mix ratio for PSALOR variant between aa and structure channels")
    parser.add_argument("--disable_psalor_variant_center_wt_lor", action="store_true", help="Disable wt-centered LOR conversion in PSALOR variant retriever")
    parser.add_argument("--psalor_exact_weight", type=float, default=0.7, help="Mix weight for PSALOR exact layer2 when combining with variant")
    parser.add_argument("--psalor_variant_weight", type=float, default=0.3, help="Mix weight for PSALOR variant layer2 when combining with exact")
    parser.add_argument("--plddt_dir", type=str, default=None, help="Directory containing residue-level pLDDT csv files")
    parser.add_argument("--protein_info_file", type=str, default=None, help="Protein metadata table with global pLDDT")
    parser.add_argument("--layer2_weight", type=float, default=1.0, help="Layer2 scalar weight in two-stage fusion")
    parser.add_argument("--gate_temperature", type=float, default=1.0, help="Gate temperature in two-stage fusion")
    parser.add_argument("--alpha_family", type=float, default=1.0, help="Family-level scaling factor for layer2 gate")
    parser.add_argument("--adaptive_min_gate", type=float, default=0.0, help="Minimum gate for adaptive fusion")
    parser.add_argument("--adaptive_max_gate", type=float, default=0.95, help="Maximum gate for adaptive fusion")
    parser.add_argument("--enable_gate_diagnostics", action="store_true", help="Print gate diagnostics for two-stage fusion")
    parser.add_argument("--print_compare_spearman", action="store_true", help="Print raw backbone vs VenusREM vs VenusREM-Orbit Spearman per protein")
    parser.add_argument("--backbone_mode", type=str, default="auto", choices=["auto", "prosst", "plain_mlm"], help="Backbone forward mode: auto/prosst/plain_mlm")
    parser.add_argument("--structure_vocab_subdir", type=str, default=None, help="Optional structure-seq subdir under struc_seq_dir (e.g. 2048)")
    parser.add_argument("--pdb_dir", type=str, default=None, help="Directory containing pdb files for RSA computation")
    parser.add_argument("--rsa_mode", type=str, default="auto", choices=["auto", "rsa", "plddt"], help="Accessibility source for PSALOR: true RSA or pLDDT fallback")
    parser.add_argument("--disable_sequence_dedup", action="store_true", help="Disable sequence identity clustering weights in PSALOR exact retriever")
    parser.add_argument("--dedup_identity_threshold", type=float, default=0.8, help="Identity threshold for MSA de-redundancy clustering")
    parser.add_argument("--max_sequences_for_clustering", type=int, default=2000, help="Skip exact clustering when MSA rows exceed this threshold")
    parser.add_argument("--max_residue_len", type=int, default=None, help="Override max residue length for backbone forward; defaults to model limit")
    parser.add_argument("--long_seq_mode", type=str, default="auto_window", choices=["auto_window", "error"], help="How to handle proteins longer than model limit")
    parser.add_argument("--long_seq_overlap", type=int, default=256, help="Overlap size for long-sequence sliding-window inference")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bars for cleaner logs")
    parser.add_argument("--max_proteins", type=int, default=None, help="Only score first N proteins (debug/subset validation)")
    parser.add_argument("--no_color", action="store_true", help="Disable ANSI colors in logs")
    parser.add_argument("--log_level", type=str, default="info", choices=["debug", "info", "warn", "error"], help="Console log verbosity")
    
    # output directory
    parser.add_argument("--out_scores_dir", default=None, help="Directory to save scores")
    args = parser.parse_args()
    args.enable_sequence_dedup = not args.disable_sequence_dedup
    args.psalor_variant_center_wt_lor = not args.disable_psalor_variant_center_wt_lor
    args.show_progress = not args.disable_tqdm
    logger = CliLogger(level=args.log_level, use_color=should_use_color(args.no_color))

    logger.section("VenusREM / Orbit scoring run")
    logger.info("Scoring proteins")
    os.makedirs(args.out_scores_dir, exist_ok=True)
    os.makedirs(f"{args.out_scores_dir}/scores", exist_ok=True)
    if args.base_dir:
        if args.aa_seq_dir is None:
            args.aa_seq_dir = f"{args.base_dir}/aa_seq"
        else:
            args.aa_seq_dir = f"{args.base_dir}/{args.aa_seq_dir}"
            
        if args.struc_seq_dir is None:
            args.struc_seq_dir = f"{args.base_dir}/struc_seq"
        else:
            args.struc_seq_dir = f"{args.base_dir}/{args.struc_seq_dir}"
        
        if args.mutant_dir is None:
            args.mutant_dir = f"{args.base_dir}/substitutions"
        else:
            args.mutant_dir = f"{args.base_dir}/{args.mutant_dir}"
        if args.pdb_dir is None:
            args.pdb_dir = f"{args.base_dir}/pdbs"
        elif not os.path.isabs(args.pdb_dir):
            args.pdb_dir = f"{args.base_dir}/{args.pdb_dir}"
        
        if args.aa_seq_aln_dir is None:
            args.aa_seq_aln_dir = f"{args.base_dir}/aa_seq_aln_a2m"
        else:
            args.aa_seq_aln_dir = f"{args.base_dir}/{args.aa_seq_aln_dir}"
            
        if args.struc_seq_aln_dir is None:
            args.struc_seq_aln_dir = f"{args.base_dir}/struc_seq_aln_foldseek"
        else:
            args.struc_seq_aln_dir = f"{args.base_dir}/{args.struc_seq_aln_dir}"

        if args.protein_info_file is None:
            default_info = f"{args.base_dir}/protein_info.csv"
            args.protein_info_file = default_info if os.path.exists(default_info) else None
        elif not os.path.isabs(args.protein_info_file):
            args.protein_info_file = f"{args.base_dir}/{args.protein_info_file}"
            
            
    protein_names = sorted(read_names(args.aa_seq_dir))
    if args.max_proteins is not None and args.max_proteins > 0:
        protein_names = protein_names[: args.max_proteins]
    logger.info(f"Total proteins: {len(protein_names)}")
    logger.debug(f"Protein preview: {format_name_preview(protein_names)}")
    
    for model_idx, model_name in enumerate(args.model_name):
        corrs = []
        compare_table_printed = False
        logger.section(f"Model {model_idx+1}/{len(args.model_name)}: {model_name}")
        logger.info(f"Loading model: {model_name}")
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        # User-requested override: force config max residue length to 4096.
        force_config_max_residue_len(model, residue_len=4096)
        model_max_residue_len = args.max_residue_len
        if model_max_residue_len is None:
            model_max_residue_len = infer_model_max_residue_len(model, tokenizer)
        logger.info(f"Max residue length for forward: {model_max_residue_len}")
        
        protein_progress = tqdm(
            protein_names,
            desc=f"Proteins[{model_name.split('/')[-1]}]",
            disable=not args.show_progress,
            leave=True,
            dynamic_ncols=True,
        )
        for idx, protein_name in enumerate(protein_progress):
            logger.info(f"Scoring protein {idx+1}/{len(protein_names)}", protein=protein_name)
            # load data
            residue_fasta = f"{args.aa_seq_dir}/{protein_name}.fasta"
            structure_fasta = resolve_structure_fasta_path(args, protein_name, model_name)
            pdb_file = (
                f"{args.pdb_dir}/{protein_name}.pdb"
                if args.pdb_dir and os.path.exists(f"{args.pdb_dir}/{protein_name}.pdb")
                else None
            )
            mutant_file = f"{args.mutant_dir}/{protein_name}.csv"
            aa_seq_aln_file = None
            struc_seq_aln_file = None
            if args.logit_mode is not None:
                if "aa_seq_aln" in args.logit_mode:
                    if os.path.exists(f"{args.aa_seq_aln_dir}/{protein_name}.a2m"):
                        aa_seq_aln_file = f"{args.aa_seq_aln_dir}/{protein_name}.a2m"
                    elif os.path.exists(f"{args.aa_seq_aln_dir}/{protein_name}.a3m"):
                        aa_seq_aln_file = f"{args.aa_seq_aln_dir}/{protein_name}.a3m"
                    elif os.path.exists(f"{args.aa_seq_aln_dir}/{protein_name}.fasta"):
                        aa_seq_aln_file = f"{args.aa_seq_aln_dir}/{protein_name}.fasta"
                else:
                    aa_seq_aln_file = None
                
                if "struc_seq_aln" in args.logit_mode:
                    struc_seq_aln_file = f"{args.struc_seq_aln_dir}/{protein_name}.fasta"
                else:
                    struc_seq_aln_file = None
            else:
                aa_seq_aln_file = None
                struc_seq_aln_file = None
                    
            if os.path.exists(f"{args.out_scores_dir}/scores/{protein_name}.csv"):
                mutant_file = f"{args.out_scores_dir}/scores/{protein_name}.csv"
            mutant_df = pd.read_csv(mutant_file)
            
            if args.model_out_name:
                model_out_name = args.model_out_name[model_idx]
            else:
                model_out_name = model_name.split("/")[-1]

            backbone_name = model_name.split("/")[-1]
            raw_col = f"{backbone_name}__raw_backbone"
            venus_col = f"{backbone_name}__venusrem"
            orbit_col = model_out_name
                
            if args.print_compare_spearman:
                if raw_col not in mutant_df.columns:
                    raw_scores = score_protein(
                        model=model,
                        tokenizer=tokenizer,
                        residue_fasta=residue_fasta,
                        structure_fasta=structure_fasta,
                        mutant_df=mutant_df,
                        alpha=0.0,
                        aa_seq_aln_file=None,
                        struc_seq_aln_file=None,
                        sample_size=args.sample_size,
                        sample_ratio=args.sample_ratio,
                        sample_times=args.sample_times,
                        orbit_args=None,
                        protein_name=protein_name,
                        backbone_mode=args.backbone_mode,
                        pdb_file=pdb_file,
                        max_residue_len=model_max_residue_len,
                        long_seq_mode=args.long_seq_mode,
                        long_seq_overlap=args.long_seq_overlap,
                        quiet=True,
                        show_progress=args.show_progress,
                        logger=logger,
                    )
                    mutant_df[raw_col] = raw_scores
                raw_corr = spearmanr(mutant_df["DMS_score"], mutant_df[raw_col]).correlation

                if args.orbit_enable:
                    if venus_col not in mutant_df.columns:
                        venus_args = clone_args_with_overrides(args, orbit_enable=False)
                        venus_scores = score_protein(
                            model=model,
                            tokenizer=tokenizer,
                            residue_fasta=residue_fasta,
                            structure_fasta=structure_fasta,
                            mutant_df=mutant_df,
                            alpha=args.alpha,
                            aa_seq_aln_file=aa_seq_aln_file,
                            struc_seq_aln_file=struc_seq_aln_file,
                            sample_size=args.sample_size,
                            sample_ratio=args.sample_ratio,
                            sample_times=args.sample_times,
                            orbit_args=venus_args,
                            protein_name=protein_name,
                            backbone_mode=args.backbone_mode,
                            pdb_file=pdb_file,
                            max_residue_len=model_max_residue_len,
                            long_seq_mode=args.long_seq_mode,
                            long_seq_overlap=args.long_seq_overlap,
                            quiet=True,
                            show_progress=args.show_progress,
                            logger=logger,
                        )
                        mutant_df[venus_col] = venus_scores
                    venus_corr = spearmanr(mutant_df["DMS_score"], mutant_df[venus_col]).correlation

            if orbit_col not in mutant_df.columns:
                scores = score_protein(
                        model=model,
                        tokenizer=tokenizer,
                        residue_fasta=residue_fasta,
                        structure_fasta=structure_fasta,
                        mutant_df=mutant_df,
                        alpha=args.alpha,
                        aa_seq_aln_file=aa_seq_aln_file,
                        struc_seq_aln_file=struc_seq_aln_file,
                        sample_size=args.sample_size,
                        sample_ratio=args.sample_ratio,
                        sample_times=args.sample_times,
                        orbit_args=args,
                        protein_name=protein_name,
                        backbone_mode=args.backbone_mode,
                        pdb_file=pdb_file,
                        max_residue_len=model_max_residue_len,
                        long_seq_mode=args.long_seq_mode,
                        long_seq_overlap=args.long_seq_overlap,
                        quiet=False,
                        show_progress=args.show_progress,
                        logger=logger,
                    )
                mutant_df[orbit_col] = scores
        
            corr = spearmanr(mutant_df["DMS_score"], mutant_df[orbit_col]).correlation
            corrs.append(corr)
            if args.print_compare_spearman and args.orbit_enable:
                if not compare_table_printed:
                    logger.info("Compare Spearman table (Orbit - VenusREM as Delta)")
                    print_compare_table_header(
                        logger,
                        include_venus=True,
                        raw_label=backbone_name,
                        venus_label="VenusREM",
                        orbit_label=orbit_col,
                    )
                    compare_table_printed = True
                print_compare_table_row(
                    logger=logger,
                    protein_name=protein_name,
                    raw_corr=raw_corr,
                    venus_corr=venus_corr,
                    orbit_corr=corr,
                )
            elif args.print_compare_spearman:
                if not compare_table_printed:
                    logger.info("Compare Spearman table (Current - Raw as Delta)")
                    print_compare_table_header(
                        logger,
                        include_venus=False,
                        raw_label=backbone_name,
                        current_label=orbit_col,
                    )
                    compare_table_printed = True
                print_compare_table_row(
                    logger=logger,
                    protein_name=protein_name,
                    raw_corr=raw_corr,
                    orbit_corr=corr,
                    venus_corr=None,
                )
            else:
                logger.success(f"{model_out_name} Spearman={corr:.4f}", protein=protein_name)
            mutant_df.to_csv(f"{args.out_scores_dir}/scores/{protein_name}.csv", index=False)
        
        if compare_table_printed:
            if args.orbit_enable:
                logger.info("+" + "-" * 42 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 10 + "+")
            else:
                logger.info("+" + "-" * 42 + "+" + "-" * 11 + "+" + "-" * 11 + "+" + "-" * 10 + "+")
        logger.section(f"{model_out_name} average Spearman: {sum(corrs)/len(corrs):.4f}")
        summary_df_path = f"{args.out_scores_dir}/summary_performance.csv"
        if os.path.exists(summary_df_path):
            summary_df = pd.read_csv(summary_df_path)
            summary_df[model_out_name] = corrs
        else:
            summary_df = pd.DataFrame({'protein': protein_names, model_out_name: corrs})
        summary_df.to_csv(f"{args.out_scores_dir}/summary_performance.csv", index=False)