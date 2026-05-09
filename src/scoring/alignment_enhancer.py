import random

import torch
from tqdm import tqdm


def read_multi_fasta(file_path):
    sequences = {}
    current_sequence = ""
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_sequence:
                    sequences[header] = (
                        current_sequence.upper().replace("-", "<pad>").replace(".", "<pad>")
                    )
                    current_sequence = ""
                header = line
            else:
                current_sequence += line
        if current_sequence:
            sequences[header] = current_sequence
    return sequences


def count_matrix_from_residue_alignment(
    tokenizer, alignment_dict, verbose=True, logger=None, protein_name=None
):
    alignment_seqs = list(alignment_dict.values())
    try:
        aln_start, aln_end = list(alignment_dict.keys())[0].split("/")[-1].split("-")
    except Exception:
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
    alignment_ids = tokenized_results["input_ids"][:, 1:-1]
    return alignment_ids, int(aln_start) - 1, int(aln_end)


def count_matrix_from_structure_alignment(
    tokenizer, alignment_dict, verbose=True, logger=None, protein_name=None
):
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
    alignment_ids = tokenized_results["input_ids"][:, 1:-1]
    return alignment_ids


def apply_alignment_prior(
    logits,
    tokenizer,
    alpha,
    aa_seq_aln_file=None,
    struc_seq_aln_file=None,
    sample_ratio=1.0,
    sample_times=1,
    show_progress=True,
    quiet=False,
    logger=None,
    protein_name=None,
):
    if alpha == 0:
        return logits

    device = logits.device

    if aa_seq_aln_file is not None and struc_seq_aln_file is None:
        if logger is not None:
            logger.info("Using residue sequence alignment matrix", protein=protein_name)
        alignment_dict = read_multi_fasta(aa_seq_aln_file)
        alignment_matrix, aln_start, aln_end = count_matrix_from_residue_alignment(
            tokenizer, alignment_dict, verbose=not quiet, logger=logger, protein_name=protein_name
        )
        for _ in range(sample_times):
            if sample_ratio < 1.0:
                curr_size = int(len(alignment_matrix) * sample_ratio)
                sample_indices = random.sample(range(len(alignment_matrix)), curr_size)
                alignment_matrix_sample = alignment_matrix[sample_indices]
            else:
                alignment_matrix_sample = alignment_matrix

            count_matrix = torch.zeros(alignment_matrix_sample.size(1), tokenizer.vocab_size)
            for i in tqdm(
                range(alignment_matrix_sample.size(1)),
                disable=(not show_progress) or quiet,
                leave=False,
            ):
                count_matrix[i] = torch.bincount(
                    alignment_matrix_sample[:, i], minlength=tokenizer.vocab_size
                )

            count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
            count_matrix = torch.log_softmax(count_matrix, dim=-1)
            aln_modify_logits = (1 - alpha) * logits[aln_start:aln_end, :] + alpha * count_matrix
            logits = torch.cat([logits[:aln_start], aln_modify_logits, logits[aln_end:]], dim=0)

    if struc_seq_aln_file is not None and aa_seq_aln_file is None:
        if logger is not None:
            logger.info("Using structure sequence alignment matrix", protein=protein_name)
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
                count_matrix[i] = torch.bincount(
                    alignment_matrix[:, i], minlength=tokenizer.vocab_size
                )
            count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
            count_matrix = torch.log_softmax(count_matrix, dim=-1)
            logits = (1 - alpha) * logits + alpha * count_matrix

    if aa_seq_aln_file is not None and struc_seq_aln_file is not None:
        if logger is not None:
            logger.info("Using both residue and structure sequence alignment matrix", protein=protein_name)
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
                count_matrix[i] = torch.bincount(
                    structure_alignment_matrix[:, i], minlength=tokenizer.vocab_size
                )

            count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
            count_matrix = torch.log_softmax(count_matrix, dim=-1)
            logits = (1 - alpha) * plm_logits + alpha * count_matrix

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
            count_matrix[i] = torch.bincount(
                residue_alignment_matrix[:, i], minlength=tokenizer.vocab_size
            )

        count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
        count_matrix = torch.log_softmax(count_matrix, dim=-1)
        aln_modify_logits = (1 - alpha) * logits[aln_start:aln_end, :] + alpha * count_matrix
        logits = torch.cat([plm_logits[:aln_start], aln_modify_logits, plm_logits[aln_end:]], dim=0)

    return logits
