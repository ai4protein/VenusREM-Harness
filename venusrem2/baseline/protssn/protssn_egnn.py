"""
Vendored EGNN network for ProtSSN inference.

Minimal extraction from:
- VenusFactory/src/mutation/models/egnn/egnn_pytorch.py
- VenusFactory/src/mutation/models/egnn/egnn_pytorch_geometric.py
- VenusFactory/src/mutation/models/egnn/network.py
- VenusFactory/src/mutation/models/egnn/utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj, Size, OptTensor, Tensor


def exists(val):
    return val is not None


SiLU = nn.SiLU


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.0):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


def fourier_encode_dist(x, num_encodings=4, include_self=True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    if include_self:
        x = torch.cat((x, orig_x), dim=-1)
    return x


def get_node_feature_dims():
    return [20, 1, 1, 4, 5, 640]


def get_edge_feature_dims():
    return [65, 1, 15, 12]


class EGNN_Sparse(MessagePassing):
    def __init__(
        self,
        feats_dim,
        pos_dim=3,
        edge_attr_dim=0,
        m_dim=16,
        fourier_features=0,
        soft_edge=0,
        norm_feats=False,
        norm_coors=False,
        norm_coors_scale_init=1e-2,
        update_feats=True,
        update_coors=False,
        dropout=0.0,
        coor_weights_clamp_value=None,
        aggr="add",
        mlp_num=2,
        **kwargs,
    ):
        assert aggr in {"add", "sum", "max", "mean"}
        assert update_feats or update_coors
        kwargs.setdefault("aggr", aggr)
        super(EGNN_Sparse, self).__init__(**kwargs)
        self.fourier_features = fourier_features
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.soft_edge = soft_edge
        self.norm_feats = norm_feats
        self.norm_coors = norm_coors
        self.update_coors = update_coors
        self.update_feats = update_feats
        self.coor_weights_clamp_value = None
        self.mlp_num = mlp_num
        self.edge_input_dim = (fourier_features * 2) + edge_attr_dim + 1 + (feats_dim * 2)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.mlp_num > 2:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.edge_input_dim, self.edge_input_dim * 8),
                self.dropout, SiLU(),
                nn.Linear(self.edge_input_dim * 8, self.edge_input_dim * 4),
                self.dropout, SiLU(),
                nn.Linear(self.edge_input_dim * 4, self.edge_input_dim * 2),
                self.dropout, SiLU(),
                nn.Linear(self.edge_input_dim * 2, m_dim), SiLU(),
            ) if update_feats else None
        else:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.edge_input_dim, self.edge_input_dim * 2),
                self.dropout, SiLU(),
                nn.Linear(self.edge_input_dim * 2, m_dim), SiLU(),
            )

        self.edge_weight = (
            nn.Sequential(nn.Linear(m_dim, 1), nn.Sigmoid()) if soft_edge else None
        )

        import torch_geometric
        self.node_norm = torch_geometric.nn.norm.LayerNorm(feats_dim) if norm_feats else None
        self.coors_norm = CoorsNorm(scale_init=norm_coors_scale_init) if norm_coors else nn.Identity()

        if self.mlp_num > 2:
            self.node_mlp = nn.Sequential(
                nn.Linear(feats_dim + m_dim, feats_dim * 8),
                self.dropout, SiLU(),
                nn.Linear(feats_dim * 8, feats_dim * 4),
                self.dropout, SiLU(),
                nn.Linear(feats_dim * 4, feats_dim * 2),
                self.dropout, SiLU(),
                nn.Linear(feats_dim * 2, feats_dim),
            ) if update_feats else None
        else:
            self.node_mlp = nn.Sequential(
                nn.Linear(feats_dim + m_dim, feats_dim * 2),
                self.dropout, SiLU(),
                nn.Linear(feats_dim * 2, feats_dim),
            ) if update_feats else None

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4), self.dropout, SiLU(),
            nn.Linear(self.m_dim * 4, 1),
        ) if update_coors else None

        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr: OptTensor = None, batch: Adj = None,
                angle_data=None, size: Size = None) -> Tensor:
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist = (rel_coors ** 2).sum(dim=-1, keepdim=True)

        if self.fourier_features > 0:
            from einops import rearrange
            rel_dist = fourier_encode_dist(rel_dist, num_encodings=self.fourier_features)
            rel_dist = rearrange(rel_dist, "n () d -> n d")

        if exists(edge_attr):
            edge_attr_feats = torch.cat([edge_attr, rel_dist], dim=-1)
        else:
            edge_attr_feats = rel_dist

        hidden_out, coors_out = self.propagate(
            edge_index, x=feats, edge_attr=edge_attr_feats,
            coors=coors, rel_coors=rel_coors, batch=batch,
        )
        return torch.cat([coors_out, hidden_out], dim=-1)

    def message(self, x_i, x_j, edge_attr) -> Tensor:
        return self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self._check_input(edge_index, size)
        coll_dict = self._collect(self._user_args, edge_index, size, kwargs)
        msg_kwargs = self.inspector.collect_param_data("message", coll_dict)
        aggr_kwargs = self.inspector.collect_param_data("aggregate", coll_dict)
        update_kwargs = self.inspector.collect_param_data("update", coll_dict)

        m_ij = self.message(**msg_kwargs)

        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])
            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]

        if self.update_feats:
            if self.soft_edge:
                m_ij = m_ij * self.edge_weight(m_ij)
            m_i = self.aggregate(m_ij, **aggr_kwargs)
            hidden_feats = self.node_norm(kwargs["x"], kwargs["batch"]) if self.node_norm else kwargs["x"]
            hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
            hidden_out = kwargs["x"] + hidden_out
        else:
            hidden_out = kwargs["x"]

        return self.update((hidden_out, coors_out), **update_kwargs)


class nodeEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.atom_embedding_list = nn.ModuleList()
        self.node_feature_dim = get_node_feature_dims()
        for dim in self.node_feature_dim:
            emb = nn.Linear(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        feature_dim_count = 0
        for i in range(len(self.node_feature_dim)):
            x_embedding += self.atom_embedding_list[i](
                x[:, feature_dim_count:feature_dim_count + self.node_feature_dim[i]]
            )
            feature_dim_count += self.node_feature_dim[i]
        return x_embedding


class edgeEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.atom_embedding_list = nn.ModuleList()
        self.edge_feature_dims = get_edge_feature_dims()
        for dim in self.edge_feature_dims:
            emb = nn.Linear(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        feature_dim_count = 0
        for i in range(len(self.edge_feature_dims)):
            x_embedding += self.atom_embedding_list[i](
                x[:, feature_dim_count:feature_dim_count + self.edge_feature_dims[i]]
            )
            feature_dim_count += self.edge_feature_dims[i]
        return x_embedding


class EGNN(nn.Module):
    def __init__(self, gnn_config, input_dim, out_dim):
        super().__init__()
        self.gnn_config = gnn_config
        self.mpnn_layes = nn.ModuleList([
            EGNN_Sparse(
                input_dim,
                m_dim=int(gnn_config["hidden_channels"]),
                edge_attr_dim=int(gnn_config["edge_attr_dim"]),
                dropout=int(gnn_config["dropout"]),
                mlp_num=int(gnn_config["mlp_num"]),
            )
            for _ in range(int(gnn_config["n_layers"]))
        ])

        if gnn_config.get("embedding", False):
            self.node_embedding = nodeEncoder(input_dim)
            self.edge_embedding = edgeEncoder(input_dim)

        self.lin = nn.Linear(input_dim, out_dim)
        self.droplayer = nn.Dropout(int(gnn_config["dropout"]))

    def forward(self, data):
        input_x = data.esm_rep
        input_x = torch.cat([data.pos, input_x], dim=1)

        if self.gnn_config.get("embedding", False):
            input_x = self.node_embedding(input_x)
            data.edge_attr = self.edge_embedding(data.edge_attr)

        for layer in self.mpnn_layes:
            h = layer(input_x, data.edge_index, data.edge_attr, batch=data.batch)
            if self.gnn_config.get("residual", False):
                input_x = input_x + h
            else:
                input_x = h

        x = input_x[:, 3:]
        x = self.droplayer(x)
        logits = self.lin(x)
        return logits, x
