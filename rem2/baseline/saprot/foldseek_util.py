import os
import re
import shutil
import subprocess
import tempfile

import numpy as np


def extract_plddt(pdb_path):
    with open(pdb_path, "r") as r:
        plddt_dict = {}
        for line in r:
            line = re.sub(" +", " ", line).strip()
            splits = line.split(" ")
            if splits[0] == "ATOM":
                if len(splits[4]) == 1:
                    pos = int(splits[5])
                else:
                    pos = int(splits[4][1:])
                plddt = float(splits[-2])
                if pos not in plddt_dict:
                    plddt_dict[pos] = [plddt]
                else:
                    plddt_dict[pos].append(plddt)
    plddts = np.array([np.mean(v) for v in plddt_dict.values()])
    return plddts


def get_struc_seq(foldseek, path, chains=None, process_id=0, plddt_mask=False, plddt_threshold=70.0):
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"PDB file not found: {path}"

    tmp_dir = tempfile.mkdtemp(prefix="rem2_foldseek_")
    tmp_save_path = os.path.join(tmp_dir, f"get_struc_seq_{process_id}.tsv")
    try:
        subprocess.run(
            [
                foldseek,
                "structureto3didescriptor",
                "-v",
                "0",
                "--threads",
                "1",
                "--chain-name-mode",
                "1",
                path,
                tmp_save_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        seq_dict = {}
        name = os.path.basename(path)
        with open(tmp_save_path, "r") as r:
            for line in r:
                desc, seq, struc_seq = line.split("\t")[:3]
                if plddt_mask:
                    plddts = extract_plddt(path)
                    if len(plddts) == len(struc_seq):
                        indices = np.where(plddts < plddt_threshold)[0]
                        np_seq = np.array(list(struc_seq))
                        np_seq[indices] = "#"
                        struc_seq = "".join(np_seq)
                name_chain = desc.split(" ")[0]
                chain = name_chain.replace(name, "").split("_")[-1]
                if chains is None or chain in chains:
                    if chain not in seq_dict:
                        combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                        seq_dict[chain] = (seq, struc_seq, combined_seq)
        return seq_dict
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip() or str(exc)
        raise RuntimeError(f"Foldseek failed on {path}: {detail}") from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
