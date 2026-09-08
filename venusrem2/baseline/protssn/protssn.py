"""
ProtSSN baseline adapter for VenusREM2.

ProtSSN = frozen ESM2 embeddings + trainable EGNN on protein structure kNN graph.
Outputs [L, 20] logits over amino acids, projected to ESM2 vocab for VenusREM2 compatibility.

Reference: Tan et al., "Semantical and geometrical protein encoding toward enhanced
bioactivity and thermostability", eLife 2025.
"""

import gc
import math
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.spatial as spa
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from torch_geometric.data import Batch, Data
from transformers import AutoTokenizer, EsmModel

from venusrem2.baseline.protssn.protssn_egnn import EGNN

PROTSSN_AA_LIST = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I",
                   "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

POSSIBLE_AMINO_ACIDS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]

THREE_TO_ONE = {
    "VAL": "V", "ILE": "I", "LEU": "L", "GLU": "E", "GLN": "Q",
    "ASP": "D", "ASN": "N", "HIS": "H", "TRP": "W", "PHE": "F",
    "TYR": "Y", "ARG": "R", "LYS": "K", "SER": "S", "THR": "T",
    "MET": "M", "ALA": "A", "GLY": "G", "PRO": "P", "CYS": "C",
}

ALLOWABLE_AAS = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "HIP", "HIE", "TPO", "HID", "LEV", "MEU", "PTR", "GLV", "CYT", "SEP",
    "HIZ", "CYM", "GLM", "ASQ", "TYS", "CYX", "GLZ", "misc",
]

DEFAULT_CONFIG_DIR = Path(__file__).parent / "protssn_config"

ROUND_ERROR = 1e-14


# ---------------------------------------------------------------------------
# Math utilities (from VenusFactory)
# ---------------------------------------------------------------------------

def _safe_index(lst, e):
    try:
        return lst.index(e)
    except ValueError:
        return len(lst) - 1


def _one_hot_res(type_idx, num_residue_type=20):
    feat = [0] * num_residue_type
    if type_idx < num_residue_type:
        feat[type_idx] = 1
        return feat
    return False


def _create_vector(vec):
    return np.array([vec[0], vec[1], vec[2]])


def _norm(a):
    return np.sqrt(np.sum((a * a).flat))


def _angle(v1, v2):
    length_product = _norm(v1) * _norm(v2)
    if length_product == 0:
        return 0.0
    cosine = np.sum(v1 * v2) / length_product
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.arccos(cosine)


def _dihedral(vec1, vec2, vec3, vec4):
    v1, v2, v3, v4 = map(_create_vector, [vec1, vec2, vec3, vec4])
    v12 = v2 - v1
    v23 = v3 - v2
    v34 = v4 - v3
    normal1 = np.cross(v12, v23)
    normal2 = np.cross(v23, v34)
    n1_norm = _norm(normal1)
    n2_norm = _norm(normal2)
    if n1_norm == 0 or n2_norm == 0:
        return 0.0
    normal1 = normal1 / n1_norm
    normal2 = normal2 / n2_norm
    torsion = _angle(normal1, normal2) * 180.0 / np.pi
    if np.sum(normal1 * v34) >= 0:
        return torsion
    torsion = 360 - torsion
    return 0.0 if torsion == 360 else torsion


# ---------------------------------------------------------------------------
# NormalizeProtein transform
# ---------------------------------------------------------------------------

class NormalizeProtein:
    def __init__(self, filename, skip_x=20, skip_edge_attr=64, safe_domi=1e-10):
        dic = torch.load(filename, map_location="cpu", weights_only=True)
        self.skip_x = skip_x
        self.skip_edge_attr = skip_edge_attr
        self.safe_domi = safe_domi
        self.x_mean = dic["x_mean"]
        self.x_std = dic["x_std"]
        self.pos_mean = dic["pos_mean"]
        self.pos_std = torch.mean(dic["pos_std"])
        self.edge_attr_mean = dic["edge_attr_mean"]
        self.edge_attr_std = dic["edge_attr_std"]

    def __call__(self, data):
        data.x[:, self.skip_x:] = (
            data.x[:, self.skip_x:] - self.x_mean[self.skip_x:]
        ).div_(self.x_std[self.skip_x:] + self.safe_domi)
        data.pos = data.pos - data.pos.mean(dim=-2, keepdim=False)
        data.pos = data.pos.div_(self.pos_std + self.safe_domi)
        data.edge_attr[:, self.skip_edge_attr:] = (
            data.edge_attr[:, self.skip_edge_attr:]
            - self.edge_attr_mean[self.skip_edge_attr:]
        ).div_(self.edge_attr_std[self.skip_edge_attr:] + self.safe_domi)
        return data


# ---------------------------------------------------------------------------
# Protein graph building (from VenusFactory ProtSSN class)
# ---------------------------------------------------------------------------

class ProteinGraphBuilder:
    def __init__(
        self,
        c_alpha_max_neighbors: int = 10,
        cutoff: int = 30,
        seq_dist_cut: int = 64,
        num_residue_type: int = 20,
        pre_transform=None,
    ):
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.cutoff = cutoff
        self.seq_dist_cut = seq_dist_cut
        self.num_residue_type = num_residue_type
        self.pre_transform = pre_transform
        self.sr = ShrakeRupley(probe_radius=1.4, n_points=100)
        self.parser = PDBParser(QUIET=True)

    def build(self, pdb_file: str) -> Data:
        rec, _, c_alpha_coords, n_coords, c_coords, seq = self._get_receptor(pdb_file)
        graph = self._build_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, seq)
        if graph is False or graph is None:
            raise ValueError(f"Failed to build graph for {pdb_file}")
        if self.pre_transform is not None:
            graph = self.pre_transform(graph)
        del graph["distances"]
        del graph["edge_dist"]
        del graph["mu_r_norm"]
        return graph

    def _get_receptor(self, pdb_file):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PDBConstructionWarning)
            structure = self.parser.get_structure("protein", pdb_file)
            rec = structure[0]

        c_alpha_coords, n_coords, c_coords = [], [], []
        coords = []
        valid_chain_ids = []
        lengths = []
        seq = []

        for chain in rec:
            chain_ca, chain_n, chain_c, chain_coords = [], [], [], []
            count = 0
            invalid_res_ids = []
            for residue in chain:
                if residue.get_resname() == "HOH":
                    invalid_res_ids.append(residue.get_id())
                    continue
                c_alpha = n = c = None
                residue_coords = []
                for atom in residue:
                    if atom.name == "CA":
                        c_alpha = list(atom.get_vector())
                        seq.append(str(residue).split(" ")[1])
                    if atom.name == "N":
                        n = list(atom.get_vector())
                    if atom.name == "C":
                        c = list(atom.get_vector())
                    residue_coords.append(list(atom.get_vector()))
                if c_alpha is not None and n is not None and c is not None:
                    chain_ca.append(c_alpha)
                    chain_n.append(n)
                    chain_c.append(c)
                    chain_coords.append(np.array(residue_coords))
                    count += 1
                else:
                    invalid_res_ids.append(residue.get_id())
            for res_id in invalid_res_ids:
                chain.detach_child(res_id)
            lengths.append(count)
            coords.append(chain_coords)
            c_alpha_coords.append(np.array(chain_ca) if chain_ca else np.zeros((0, 3)))
            n_coords.append(np.array(chain_n) if chain_n else np.zeros((0, 3)))
            c_coords.append(np.array(chain_c) if chain_c else np.zeros((0, 3)))
            if len(chain_coords) > 0:
                valid_chain_ids.append(chain.get_id())

        valid_ca, valid_n, valid_c, valid_coords = [], [], [], []
        invalid_chain_ids = []
        for i, chain in enumerate(rec):
            if chain.get_id() in valid_chain_ids:
                valid_ca.append(c_alpha_coords[i])
                valid_n.append(n_coords[i])
                valid_c.append(c_coords[i])
                valid_coords.append(coords[i])
            else:
                invalid_chain_ids.append(chain.get_id())
        for cid in invalid_chain_ids:
            rec.detach_child(cid)

        coords_flat = [item for sublist in valid_coords for item in sublist]
        c_alpha_coords = np.concatenate(valid_ca, axis=0)
        n_coords = np.concatenate(valid_n, axis=0)
        c_coords = np.concatenate(valid_c, axis=0)
        return rec, coords_flat, c_alpha_coords, n_coords, c_coords, seq

    def _rec_residue_featurizer(self, rec, add_feature):
        num_res = len([_ for _ in rec.get_residues()])
        num_feature = 2
        if add_feature is not None and add_feature.any():
            num_feature += add_feature.shape[1]
        res_feature = torch.zeros(num_res, self.num_residue_type + num_feature)
        count = 0
        self.sr.compute(rec, level="R")
        for residue in rec.get_residues():
            sasa = residue.sasa
            bfactor = 0.0
            for atom in residue:
                if atom.name == "CA":
                    bfactor = atom.bfactor
            residx = _safe_index(ALLOWABLE_AAS, residue.get_resname())
            res_feat = _one_hot_res(residx, num_residue_type=self.num_residue_type)
            if not res_feat:
                return False
            res_feat.append(sasa)
            res_feat.append(bfactor)
            if num_feature > 2 and add_feature is not None:
                res_feat.extend(list(add_feature[count, :]))
            res_feature[count, :] = torch.tensor(res_feat, dtype=torch.float32)
            count += 1
        for k in range(self.num_residue_type, self.num_residue_type + 2):
            mean = res_feature[:, k].mean()
            std = res_feature[:, k].std()
            res_feature[:, k] = (res_feature[:, k] - mean) / (std + 1e-9)
        return res_feature

    def _get_node_features(self, n_coords, c_coords, c_alpha_coords):
        num_res = n_coords.shape[0]
        angles = np.zeros((num_res, 2))
        for i in range(num_res - 1):
            angles[i, 0] = _dihedral(c_coords[i], n_coords[i], c_alpha_coords[i], n_coords[i + 1])
            angles[i, 1] = _dihedral(n_coords[i], c_alpha_coords[i], c_coords[i], n_coords[i + 1])
        node_features = np.zeros((num_res, 4))
        for i in range(2):
            node_features[:, 2 * i] = np.sin(angles[:, i])
            node_features[:, 2 * i + 1] = np.cos(angles[:, i])
        return node_features

    def _distance_featurizer(self, dist_list, divisor=4):
        length_scale_list = [1.5 ** x for x in range(15)]
        dist_arr = np.array(dist_list)
        transformed = np.array([
            np.exp(-((dist_arr / divisor) ** 2) / ls) for ls in length_scale_list
        ]).T
        return torch.from_numpy(transformed.astype(np.float32))

    def _get_edge_features(self, src_list, dst_list, dist_list, divisor=4):
        seq_edge = torch.abs(torch.tensor(src_list) - torch.tensor(dst_list)).reshape(-1, 1)
        seq_edge = torch.where(seq_edge > self.seq_dist_cut, self.seq_dist_cut, seq_edge)
        seq_edge = F.one_hot(seq_edge.long(), num_classes=self.seq_dist_cut + 1).reshape(-1, self.seq_dist_cut + 1)
        contact_sig = torch.where(torch.tensor(dist_list) <= 8, 1, 0).reshape(-1, 1)
        dist_fea = self._distance_featurizer(dist_list, divisor=divisor)
        return torch.cat([seq_edge.float(), dist_fea, contact_sig.float()], dim=-1)

    def _build_calpha_graph(self, rec, c_alpha_coords, n_coords, c_coords, seq):
        scalar_feature = self._get_node_features(n_coords, c_coords, c_alpha_coords)
        num_residues = len(c_alpha_coords)
        if num_residues <= 1:
            return False

        n_i_list, u_i_list, v_i_list = [], [], []
        residue_locs = []
        for i, residue in enumerate(rec.get_residues()):
            n_coord, ca_coord, c_coord = n_coords[i], c_alpha_coords[i], c_coords[i]
            u_i = (n_coord - ca_coord) / np.linalg.norm(n_coord - ca_coord)
            t_i = (c_coord - ca_coord) / np.linalg.norm(c_coord - ca_coord)
            n_i = np.cross(u_i, t_i)
            n_i_norm = np.linalg.norm(n_i)
            if n_i_norm < 1e-8:
                n_i = np.array([1.0, 0.0, 0.0])
            else:
                n_i = n_i / n_i_norm
            v_i = np.cross(n_i, u_i)
            n_i_list.append(n_i)
            u_i_list.append(u_i)
            v_i_list.append(v_i)
            residue_locs.append(ca_coord)

        residue_locs = np.stack(residue_locs, axis=0)
        n_i_feat = np.stack(n_i_list, axis=0)
        u_i_feat = np.stack(u_i_list, axis=0)
        v_i_feat = np.stack(v_i_list, axis=0)

        distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
        src_list, dst_list, dist_list, mean_norm_list = [], [], [], []

        for i in range(num_residues):
            dst = list(np.where(distances[i, :] < self.cutoff)[0])
            if i in dst:
                dst.remove(i)
            if self.c_alpha_max_neighbors is not None and len(dst) > self.c_alpha_max_neighbors:
                dst = list(np.argsort(distances[i, :]))[1:self.c_alpha_max_neighbors + 1]
            if len(dst) == 0:
                dst = list(np.argsort(distances[i, :]))[1:2]

            src = [i] * len(dst)
            src_list.extend(src)
            dst_list.extend(dst)
            valid_dist = list(distances[i, dst])
            dist_list.extend(valid_dist)
            valid_dist_np = distances[i, dst]

            sigma = np.array([1.0, 2.0, 5.0, 10.0, 30.0]).reshape((-1, 1))
            from scipy.special import softmax
            weights = softmax(-valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)
            diff_vecs = residue_locs[src, :] - residue_locs[dst, :]
            mean_vec = weights.dot(diff_vecs)
            denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))
            mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / (denominator + 1e-10)
            mean_norm_list.append(mean_vec_ratio_norm)

        pos_feat = torch.from_numpy(residue_locs.astype(np.float32))
        x = self._rec_residue_featurizer(rec, add_feature=np.array(scalar_feature))
        if isinstance(x, bool) and not x:
            return False

        edge_attr = self._get_edge_features(src_list, dst_list, dist_list, divisor=4)

        graph = Data(
            x=x,
            pos=pos_feat,
            edge_attr=edge_attr,
            edge_index=torch.tensor([src_list, dst_list]),
            edge_dist=torch.tensor(dist_list),
            distances=torch.tensor(distances),
            mu_r_norm=torch.from_numpy(np.array(mean_norm_list).astype(np.float32)),
            seq=seq,
        )

        edge_feat_ori_list = []
        for idx in range(len(dist_list)):
            src = src_list[idx]
            dst = dst_list[idx]
            basis_matrix = np.stack(
                (n_i_feat[dst, :], u_i_feat[dst, :], v_i_feat[dst, :]), axis=0
            )
            p_ij = np.matmul(basis_matrix, residue_locs[src, :] - residue_locs[dst, :])
            q_ij = np.matmul(basis_matrix, n_i_feat[src, :])
            k_ij = np.matmul(basis_matrix, u_i_feat[src, :])
            t_ij = np.matmul(basis_matrix, v_i_feat[src, :])
            s_ij = np.concatenate((p_ij, q_ij, k_ij, t_ij), axis=0)
            edge_feat_ori_list.append(s_ij)

        edge_feat_ori = np.stack(edge_feat_ori_list, axis=0)
        edge_feat_ori = torch.from_numpy(edge_feat_ori.astype(np.float32))
        graph.edge_attr = torch.cat([graph.edge_attr, edge_feat_ori], dim=1)
        return graph


# ---------------------------------------------------------------------------
# PLM (ESM2) wrapper for ProtSSN
# ---------------------------------------------------------------------------

class PLMForProtSSN(nn.Module):
    def __init__(self, esm_model, tokenizer):
        super().__init__()
        self.model = esm_model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def forward(self, graphs: List[Data], device: torch.device) -> Batch:
        seqs = []
        for g in graphs:
            one_hot = g.x[:, :20].argmax(1).tolist()
            seq = "".join(THREE_TO_ONE[POSSIBLE_AMINO_ACIDS[idx]] for idx in one_hot)
            seqs.append(seq)

        inputs = self.tokenizer(seqs, return_tensors="pt", padding=True).to(device)
        batch_lens = (inputs["attention_mask"] == 1).sum(1) - 2
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state

        for i, (hs, seq_len) in enumerate(zip(hidden_states, batch_lens)):
            graphs[i].esm_rep = hs[1:1 + seq_len]
            if hasattr(graphs[i], "seq"):
                del graphs[i].seq

        graphs = [g.to(device) for g in graphs]
        return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# Forward and loading functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_protssn(
    sequence: str,
    pdb_file: str,
    device: torch.device,
    esm_tokenizer: AutoTokenizer,
    protssn_pack: dict,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    """
    Forward ProtSSN ensemble, return [L, V_esm] logits projected to ESM2 vocab.

    protssn_pack contains:
        - "plm": PLMForProtSSN instance
        - "models": list of (gnn_model, graph_builder, k, h) tuples
        - "esm_vocab": tokenizer vocab dict
        - "esm_vocab_size": int
    """
    plm = protssn_pack["plm"]
    models = protssn_pack["models"]
    esm_vocab = protssn_pack["esm_vocab"]
    esm_vocab_size = protssn_pack["esm_vocab_size"]

    if logger is not None:
        logger.debug(
            f"ProtSSN forward: {len(models)} model(s), seq_len={len(sequence)}",
            protein=protein_name,
        )

    graphs_by_k: Dict[int, Data] = {}
    all_logits = []

    for gnn_model, graph_builder, k, h in models:
        if k not in graphs_by_k:
            graph = graph_builder.build(pdb_file)
            graphs_by_k[k] = graph

        graph = graphs_by_k[k].clone()
        batch_graph = plm([graph], device)
        logits, _ = gnn_model(batch_graph)
        all_logits.append(logits.cpu())

        if logger is not None:
            logger.debug(f"  ProtSSN k={k} h={h}: logits shape {logits.shape}", protein=protein_name)

    avg_logits = torch.stack(all_logits).mean(dim=0).to(device)
    seq_len = avg_logits.shape[0]

    projected = torch.full((seq_len, esm_vocab_size), -1e9, device=device)
    for i, aa in enumerate(PROTSSN_AA_LIST):
        if aa in esm_vocab:
            projected[:, esm_vocab[aa]] = avg_logits[:, i]

    return projected


@torch.no_grad()
def forward_protssn_masked_marginal(
    sequence: str,
    pdb_file: str,
    device: torch.device,
    esm_tokenizer: AutoTokenizer,
    protssn_pack: dict,
    logger=None,
    protein_name: Optional[str] = None,
) -> torch.Tensor:
    plm = protssn_pack["plm"]
    models = protssn_pack["models"]
    esm_vocab = protssn_pack["esm_vocab"]
    esm_vocab_size = protssn_pack["esm_vocab_size"]
    seq_len = len(sequence)

    if logger is not None:
        logger.debug(
            f"ProtSSN masked marginal: {len(models)} model(s), seq_len={seq_len}",
            protein=protein_name,
        )

    graphs_by_k: Dict[int, Data] = {}
    for _, graph_builder, k, _ in models:
        if k not in graphs_by_k:
            graphs_by_k[k] = graph_builder.build(pdb_file)

    esm_model = plm.model
    mask_token_id = esm_tokenizer.mask_token_id

    wt_inputs = esm_tokenizer([sequence], return_tensors="pt", padding=False).to(device)
    wt_input_ids = wt_inputs["input_ids"]

    all_pos_logits = torch.zeros(seq_len, 20, device="cpu")

    for pos in range(seq_len):
        masked_ids = wt_input_ids.clone()
        masked_ids[0, pos + 1] = mask_token_id
        outputs = esm_model(input_ids=masked_ids, attention_mask=wt_inputs["attention_mask"])
        esm_rep = outputs.last_hidden_state[0, 1:1 + seq_len]

        pos_logits_sum = torch.zeros(20, device=device)
        for gnn_model, _, k, _ in models:
            graph = graphs_by_k[k].clone()
            graph.esm_rep = esm_rep
            graph = graph.to(device)
            batch_graph = Batch.from_data_list([graph])
            logits, _ = gnn_model(batch_graph)
            pos_logits_sum += logits[pos]
        all_pos_logits[pos] = (pos_logits_sum / len(models)).cpu()

    all_pos_logits = all_pos_logits.to(device)
    projected = torch.full((seq_len, esm_vocab_size), -1e9, device=device)
    for i, aa in enumerate(PROTSSN_AA_LIST):
        if aa in esm_vocab:
            projected[:, esm_vocab[aa]] = all_pos_logits[:, i]

    return projected


def _ensure_protssn_weights(model_dir: str, configs: List[Tuple[int, int]], logger=None):
    """Resolve ProtSSN weights from cache, or prompt to download from HuggingFace."""
    os.makedirs(model_dir, exist_ok=True)
    from venusrem2.models.weights import log_cache_hit, resolve_existing_weight

    missing = []
    for k, h in configs:
        fname = f"protssn_k{k}_h{h}.pt"
        path = os.path.join(model_dir, fname)
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            continue
        cached = resolve_existing_weight("protssn", fname)
        if cached:
            if os.path.abspath(cached) != os.path.abspath(path):
                import shutil

                shutil.copy2(cached, path)
            log_cache_hit(fname, cached, logger)
            continue
        missing.append((k, h, fname))
    if not missing:
        return
    names = [m[2] for m in missing]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(
            f"Missing ProtSSN weights: {names}. Install huggingface_hub, "
            "place files in --protssn_model_dir / --cache_dir, or allow download."
        )
    from venusrem2.models.download_policy import confirm_download

    confirm_download(
        name="ProtSSN (" + ", ".join(names) + ")",
        dest=model_dir,
        source="hf://tyang816/ProtSSN",
        looked_in=[model_dir],
        logger=logger,
    )
    for k, h, fname in missing:
        hf_hub_download(repo_id="tyang816/ProtSSN", filename=fname, local_dir=model_dir)


def load_protssn_models(
    model_dir: str,
    device: torch.device,
    norm_dir: Optional[str] = None,
    gnn_config_path: Optional[str] = None,
    use_ensemble: bool = True,
    k: int = 20,
    h: int = 512,
    esm_model_name: str = "facebook/esm2_t33_650M_UR50D",
    logger=None,
) -> Tuple[dict, AutoTokenizer]:
    """
    Load ProtSSN models and return (protssn_pack, esm_tokenizer).

    Args:
        model_dir: directory containing protssn_k{k}_h{h}.pt weight files.
                   Missing weights are auto-downloaded from HuggingFace.
        device: torch device
        norm_dir: directory containing cath_k{k}_mean_attr.pt files (defaults to bundled)
        gnn_config_path: path to egnn.yaml (defaults to bundled)
        use_ensemble: if True, load all 9 models; otherwise load single (k, h)
        k: k-neighbors for single model mode
        h: hidden dim for single model mode
        esm_model_name: ESM2 model name for embeddings
    """
    if norm_dir is None:
        norm_dir = str(DEFAULT_CONFIG_DIR / "norm")
    if gnn_config_path is None:
        gnn_config_path = str(DEFAULT_CONFIG_DIR / "egnn.yaml")

    with open(gnn_config_path) as f:
        gnn_config = yaml.safe_load(f)["egnn"]

    if logger is not None:
        logger.info(f"Loading ESM2 for ProtSSN: {esm_model_name}")
    esm_model = EsmModel.from_pretrained(esm_model_name, trust_remote_code=True).to(device).eval()
    esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name, trust_remote_code=True)
    plm = PLMForProtSSN(esm_model, esm_tokenizer)

    if use_ensemble:
        configs = [(kk, hh) for kk in [10, 20, 30] for hh in [512, 768, 1280]]
    else:
        configs = [(k, h)]

    _ensure_protssn_weights(model_dir, configs, logger=logger)

    models = []
    for kk, hh in configs:
        cfg = dict(gnn_config)
        cfg["hidden_channels"] = hh

        gnn = EGNN(cfg, input_dim=1280, out_dim=20).to(device)
        weight_path = os.path.join(model_dir, f"protssn_k{kk}_h{hh}.pt")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"ProtSSN weight not found: {weight_path}")
        state_dict = torch.load(weight_path, map_location=device, weights_only=True)
        stripped = {}
        for key, val in state_dict.items():
            new_key = key.replace("GNN_model.", "", 1) if key.startswith("GNN_model.") else key
            stripped[new_key] = val
        gnn.load_state_dict(stripped)
        gnn.eval()

        norm_file = os.path.join(norm_dir, f"cath_k{kk}_mean_attr.pt")
        if not os.path.exists(norm_file):
            raise FileNotFoundError(f"ProtSSN norm file not found: {norm_file}")
        pre_transform = NormalizeProtein(filename=norm_file)

        graph_builder = ProteinGraphBuilder(
            c_alpha_max_neighbors=kk,
            pre_transform=pre_transform,
        )
        models.append((gnn, graph_builder, kk, hh))

        if logger is not None:
            logger.info(f"  Loaded ProtSSN k={kk} h={hh}")

    esm_vocab = esm_tokenizer.get_vocab()
    protssn_pack = {
        "plm": plm,
        "models": models,
        "esm_vocab": esm_vocab,
        "esm_vocab_size": esm_tokenizer.vocab_size,
    }

    if logger is not None:
        label = f"ensemble ({len(models)} models)" if use_ensemble else f"single (k={k}, h={h})"
        logger.info(f"ProtSSN ready: {label}")

    return protssn_pack, esm_tokenizer
