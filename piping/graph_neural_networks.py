#!/usr/bin/env python3
"""Graph-aware policy modules for GRL-DEACO.

The public training stack uses a 27-dimensional connection state, 24-dimensional
device-node features, and a 29-dimensional DEACO action vector. This module
keeps those external tensor shapes stable while providing a GAT scene encoder,
cross-attention fusion, and graph-aware actor/critic networks.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)

NODE_FEATURE_DIM = 24
STATE_DIM = 27
ACTION_DIM = 29

__all__ = [
    "ACTION_DIM",
    "NODE_FEATURE_DIM",
    "STATE_DIM",
    "CrossAttentionFusion",
    "GAT",
    "GraphAttentionLayer",
    "GraphAwareActorNetwork",
    "GraphAwareCriticNetwork",
    "SceneGraphEncoder",
    "build_scene_graph",
]

class GraphAttentionLayer(nn.Module):
    """Multi-head graph attention layer with weighted-edge masking."""

    def __init__(self, in_features: int, out_features: int, num_heads: int = 4,
                 dropout: float = 0.1, alpha: float = 0.2, concat: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(num_heads, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(num_heads, 2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply graph attention.

        Args:
            h: Node features with shape ``[batch, num_nodes, in_features]``.
            adj: Weighted adjacency with shape ``[batch, num_nodes, num_nodes]``.

        Returns:
            Updated node features. The final dimension is
            ``out_features * num_heads`` when ``concat=True`` and
            ``out_features`` otherwise.
        """
        batch_size, num_nodes, _ = h.size()

        h_transformed = torch.einsum('bni,hio->bnho', h, self.W)

        h_i = h_transformed.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        h_j = h_transformed.unsqueeze(1).expand(-1, num_nodes, -1, -1, -1)
        h_concat = torch.cat([h_i, h_j], dim=-1)

        e = torch.einsum('bnmhf,hfo->bnmh', h_concat, self.a)
        e = self.leakyrelu(e)

        adj = adj.to(device=e.device, dtype=e.dtype)
        edge_mask = adj > 0
        edge_bias = torch.log(torch.clamp(adj, min=1e-6)).unsqueeze(-1)
        e = e + edge_bias
        e_masked = e.masked_fill(~edge_mask.unsqueeze(-1), -1e9)

        attention = F.softmax(e_masked, dim=2)
        attention = self.dropout_layer(attention)

        h_prime = torch.einsum('bnmh,bmhf->bnhf', attention, h_transformed)

        if self.concat:
            h_prime = h_prime.reshape(batch_size, num_nodes, -1)
        else:
            h_prime = h_prime.mean(dim=2)

        return h_prime


class GAT(nn.Module):
    """Device-scene graph attention encoder."""

    def __init__(self, node_features: int = NODE_FEATURE_DIM, hidden_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.layers = nn.ModuleList()

        first_concat = num_layers > 1
        self.layers.append(
            GraphAttentionLayer(
                in_features=node_features,
                out_features=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                concat=first_concat,
            )
        )

        for i in range(1, num_layers - 1):
            self.layers.append(
                GraphAttentionLayer(
                    in_features=hidden_dim * num_heads,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
            )

        if num_layers > 1:
            self.layers.append(
                GraphAttentionLayer(
                    in_features=hidden_dim * num_heads,
                    out_features=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=False,
                )
            )

        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a padded batch of device graphs.

        ``adj_matrix`` keeps the existing public shape ``[B, N, N]``. Padding
        nodes are inferred from missing diagonal self-loops and ignored during
        graph-level pooling.
        """
        h = node_features

        for i, layer in enumerate(self.layers):
            h = layer(h, adj_matrix)
            if i < self.num_layers - 1:
                h = F.elu(h)

        node_embeddings = h
        node_mask = torch.diagonal(adj_matrix, dim1=1, dim2=2) > 0
        mask = node_mask.unsqueeze(-1).to(dtype=node_embeddings.dtype, device=node_embeddings.device)
        valid_counts = mask.sum(dim=1).clamp(min=1.0)

        graph_mean = (node_embeddings * mask).sum(dim=1) / valid_counts

        masked_embeddings = node_embeddings.masked_fill(~node_mask.unsqueeze(-1), torch.finfo(node_embeddings.dtype).min)
        graph_max, _ = torch.max(masked_embeddings, dim=1)
        has_nodes = node_mask.any(dim=1, keepdim=True)
        graph_max = torch.where(has_nodes, graph_max, torch.zeros_like(graph_mean))

        graph_embedding = self.global_pool(graph_mean + graph_max)

        return node_embeddings, graph_embedding


SceneGraphEncoder = GAT


class CrossAttentionFusion(nn.Module):
    """Let the current connection representation attend to device nodes."""

    def __init__(self, connection_dim: int = 128, graph_dim: int = 128, num_heads: int = 4,
                 attention_temperature: float = 0.5):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = graph_dim // num_heads
        self.temperature = attention_temperature
        assert graph_dim % num_heads == 0, "graph_dim must be divisible by num_heads"

        self.W_q = nn.Linear(connection_dim, graph_dim)
        self.W_k = nn.Linear(graph_dim, graph_dim)
        self.W_v = nn.Linear(graph_dim, graph_dim)

        self.W_o = nn.Linear(graph_dim, graph_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        connection_feature: torch.Tensor,
        node_embeddings: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse connection and graph features.

        Args:
            connection_feature: Tensor with shape ``[B, connection_dim]``.
            node_embeddings: Tensor with shape ``[B, N, graph_dim]``.
            node_mask: Optional boolean tensor with shape ``[B, N]``.

        Returns:
            The attended graph feature ``[B, graph_dim]`` and node attention
            weights ``[B, num_heads, N]``.
        """
        batch_size = connection_feature.size(0)
        num_nodes = node_embeddings.size(1)

        Q = self.W_q(connection_feature).view(batch_size, 1, self.num_heads, self.head_dim)
        K = self.W_k(node_embeddings).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        V = self.W_v(node_embeddings).view(batch_size, num_nodes, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if node_mask is not None:
            scores = scores.masked_fill(~node_mask[:, None, None, :], -1e9)

        attention_weights = F.softmax(scores / self.temperature, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)

        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, -1)

        output = self.W_o(context)

        return output, attention_weights.squeeze(2)


class GraphAwareActorNetwork(nn.Module):
    """Graph-aware actor for the 29-dimensional DEACO action vector."""

    def __init__(self,
                 connection_state_dim: int = STATE_DIM,
                 node_feature_dim: int = NODE_FEATURE_DIM,
                 action_dim: int = ACTION_DIM,
                 gat_hidden_dim: int = 128,
                 mlp_hidden_dim: int = 512,
                 num_gat_layers: int = 2,
                 num_attention_heads: int = 4,
                 dropout: float = 0.1,
                 attention_temperature: float = 0.5):
        super().__init__()

        self.gat = GAT(
            node_features=node_feature_dim,
            hidden_dim=gat_hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        self.connection_encoder = nn.Sequential(
            nn.Linear(connection_state_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, gat_hidden_dim),
            nn.ReLU()
        )

        self.cross_attention = CrossAttentionFusion(
            connection_dim=gat_hidden_dim,
            graph_dim=gat_hidden_dim,
            num_heads=num_attention_heads,
            attention_temperature=attention_temperature
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(gat_hidden_dim * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(mlp_hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(mlp_hidden_dim, action_dim)

    def forward(self, connection_state: torch.Tensor,
                node_features: torch.Tensor,
                adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Gaussian action parameters and node attention weights."""
        node_mask = torch.diagonal(adj_matrix, dim1=1, dim2=2) > 0

        node_embeddings, graph_embedding = self.gat(node_features, adj_matrix)

        connection_embedding = self.connection_encoder(connection_state)

        attended_graph, attention_weights = self.cross_attention(
            connection_embedding,
            node_embeddings,
            node_mask=node_mask,
        )

        local_context = attended_graph + connection_embedding
        fused_feature = torch.cat([
            graph_embedding,
            local_context,
        ], dim=-1)

        features = self.fusion_mlp(fused_feature)

        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        std = torch.exp(torch.clamp(log_std, -20, 2))

        return mean, std, attention_weights


class GraphAwareCriticNetwork(nn.Module):
    """Graph-aware critic that estimates V(s)."""

    def __init__(self,
                 connection_state_dim: int = STATE_DIM,
                 node_feature_dim: int = NODE_FEATURE_DIM,
                 gat_hidden_dim: int = 128,
                 mlp_hidden_dim: int = 512,
                 num_gat_layers: int = 2,
                 num_attention_heads: int = 4,
                 dropout: float = 0.1,
                 attention_temperature: float = 0.5):
        super().__init__()

        self.gat = GAT(
            node_features=node_feature_dim,
            hidden_dim=gat_hidden_dim,
            num_layers=num_gat_layers,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        self.connection_encoder = nn.Sequential(
            nn.Linear(connection_state_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, gat_hidden_dim),
            nn.ReLU()
        )

        self.cross_attention = CrossAttentionFusion(
            connection_dim=gat_hidden_dim,
            graph_dim=gat_hidden_dim,
            num_heads=num_attention_heads,
            attention_temperature=attention_temperature
        )

        self.value_network = nn.Sequential(
            nn.Linear(gat_hidden_dim * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, connection_state: torch.Tensor,
                node_features: torch.Tensor,
                adj_matrix: torch.Tensor) -> torch.Tensor:
        """Return value estimates with shape ``[B, 1]``."""
        node_mask = torch.diagonal(adj_matrix, dim1=1, dim2=2) > 0
        node_embeddings, graph_embedding = self.gat(node_features, adj_matrix)

        connection_embedding = self.connection_encoder(connection_state)

        attended_graph, _ = self.cross_attention(
            connection_embedding,
            node_embeddings,
            node_mask=node_mask,
        )

        local_context = attended_graph + connection_embedding
        fused_feature = torch.cat([graph_embedding, local_context], dim=-1)

        value = self.value_network(fused_feature)

        return value


def _stable_type_id(device_name: str, bins: int = 16) -> int:
    """Return a deterministic feature bucket for a device name."""
    digest = hashlib.blake2b(str(device_name).encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(digest, byteorder="big", signed=False) % bins


def build_scene_graph(placed_devices: List[Dict],
                     connections: List[Dict],
                     completed_connections: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the device graph consumed by the graph-aware policy.

    The returned node feature matrix keeps the public 24-dimensional layout:
    center xyz, device extents xyz, port count, stable 16-bin device-type
    encoding, and connected-port ratio.
    """
    num_nodes = len(placed_devices)

    if num_nodes == 0:
        logger.debug("Scene graph has no devices; returning one dummy node.")
        node_features = torch.zeros(1, NODE_FEATURE_DIM)
        adj_matrix = torch.eye(1)
        return node_features, adj_matrix

    node_features_list = []

    for device in placed_devices:
        center = np.asarray(device.get('center', [0, 0, 0]), dtype=np.float32)

        size = device.get('size', [[0, 0, 0], [1, 1, 1]])
        width = abs(size[1][0] - size[0][0])
        height = abs(size[1][1] - size[0][1])
        depth = abs(size[1][2] - size[0][2])

        num_ports = len(device.get('ports', []))

        device_name = device.get('name', 'unknown')
        device_type_id = _stable_type_id(device_name, bins=16)
        device_type_vec = np.zeros(16, dtype=np.float32)
        device_type_vec[device_type_id] = 1.0

        connected_ports = sum(1 for p in device.get('ports', []) if p.get('has_connection', False))
        connection_ratio = connected_ports / num_ports if num_ports > 0 else 0

        node_feat = np.concatenate([
            center,
            np.asarray([width, height, depth], dtype=np.float32),
            np.asarray([num_ports], dtype=np.float32),
            device_type_vec,
            np.asarray([connection_ratio], dtype=np.float32),
        ]).astype(np.float32)

        node_features_list.append(node_feat)

    node_features = torch.FloatTensor(np.array(node_features_list))

    adj_matrix = torch.zeros(num_nodes, num_nodes)

    if len(connections) == 0:
        logger.debug("Scene graph has devices but no process connections.")

    def find_device_index(port_name: str, devices: List[Dict]) -> int:
        device_name = port_name.split('.')[0]
        for idx, dev in enumerate(devices):
            if dev.get('name') == device_name or dev.get('id') == device_name:
                return idx
        return -1

    for i, conn in enumerate(connections):
        from_device_idx = find_device_index(conn.get('from', ''), placed_devices)
        to_device_idx = find_device_index(conn.get('to', ''), placed_devices)

        if from_device_idx >= 0 and to_device_idx >= 0:
            adj_matrix[from_device_idx, to_device_idx] = 1.0
            adj_matrix[to_device_idx, from_device_idx] = 1.0

            if i < completed_connections:
                adj_matrix[from_device_idx, to_device_idx] = 2.0
                adj_matrix[to_device_idx, from_device_idx] = 2.0

    adj_matrix += torch.eye(num_nodes)

    return node_features, adj_matrix
