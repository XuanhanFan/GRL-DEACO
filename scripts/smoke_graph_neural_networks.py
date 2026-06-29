#!/usr/bin/env python3
"""Smoke checks for the graph-aware GRL-DEACO policy modules."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIPING_DIR = REPO_ROOT / "piping"
sys.path.insert(0, str(PIPING_DIR))

try:
    import torch
except ImportError as exc:  # pragma: no cover - developer environment guard
    raise SystemExit("PyTorch is required for graph neural network smoke checks.") from exc

from graph_neural_networks import (  # noqa: E402
    ACTION_DIM,
    NODE_FEATURE_DIM,
    STATE_DIM,
    GAT,
    GraphAttentionLayer,
    GraphAwareActorNetwork,
    GraphAwareCriticNetwork,
    build_scene_graph,
)


def _assert_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape) != expected:
        raise AssertionError(f"{name} shape mismatch: got {tuple(tensor.shape)}, expected {expected}")


def check_attention_heads() -> None:
    h = torch.randn(2, 3, NODE_FEATURE_DIM)
    adj = torch.eye(3).repeat(2, 1, 1)
    adj[:, 0, 1] = 1.0
    adj[:, 1, 0] = 1.0

    one_head = GraphAttentionLayer(NODE_FEATURE_DIM, 8, num_heads=1, dropout=0.0)
    four_heads = GraphAttentionLayer(NODE_FEATURE_DIM, 8, num_heads=4, dropout=0.0)
    _assert_shape(one_head(h, adj), (2, 3, 8), "one-head GAT")
    _assert_shape(four_heads(h, adj), (2, 3, 32), "four-head GAT")


def check_actor_critic_shapes() -> None:
    batch_size, num_nodes = 2, 4
    state = torch.randn(batch_size, STATE_DIM)
    node_features = torch.randn(batch_size, num_nodes, NODE_FEATURE_DIM)
    adj = torch.eye(num_nodes).repeat(batch_size, 1, 1)
    adj[:, 0, 1] = 1.0
    adj[:, 1, 0] = 1.0

    actor = GraphAwareActorNetwork(connection_state_dim=STATE_DIM, action_dim=ACTION_DIM)
    critic = GraphAwareCriticNetwork(connection_state_dim=STATE_DIM)
    actor.eval()
    critic.eval()

    with torch.no_grad():
        mean, std, attention = actor(state, node_features, adj)
        value = critic(state, node_features, adj)

    _assert_shape(mean, (batch_size, ACTION_DIM), "actor mean")
    _assert_shape(std, (batch_size, ACTION_DIM), "actor std")
    _assert_shape(attention, (batch_size, 4, num_nodes), "actor attention")
    _assert_shape(value, (batch_size, 1), "critic value")
    if not torch.all(std > 0):
        raise AssertionError("actor std must be strictly positive")


def check_scene_graph_builder() -> None:
    empty_features, empty_adj = build_scene_graph([], [])
    _assert_shape(empty_features, (1, NODE_FEATURE_DIM), "empty node features")
    _assert_shape(empty_adj, (1, 1), "empty adjacency")

    devices = [
        {"name": "PumpA", "center": [0, 0, 0], "size": [[0, 0, 0], [1, 1, 1]], "ports": [{"name": "out"}]},
        {"name": "CoolerB", "center": [2, 0, 0], "size": [[0, 0, 0], [1, 1, 1]], "ports": [{"name": "in"}]},
    ]
    features, no_edge_adj = build_scene_graph(devices, [])
    _assert_shape(features, (2, NODE_FEATURE_DIM), "node features")
    _assert_shape(no_edge_adj, (2, 2), "no-edge adjacency")

    connections = [{"from": "PumpA.out", "to": "CoolerB.in"}]
    _, adj = build_scene_graph(devices, connections, completed_connections=1)
    if adj[0, 1].item() != 2.0 or adj[1, 0].item() != 2.0:
        raise AssertionError("completed connection edge weights must be preserved")


def check_padding_invariance() -> None:
    torch.manual_seed(7)
    encoder = GAT(node_features=NODE_FEATURE_DIM, hidden_dim=32, num_layers=2, num_heads=2, dropout=0.0)
    encoder.eval()

    small_nodes = torch.randn(1, 2, NODE_FEATURE_DIM)
    small_adj = torch.eye(2).unsqueeze(0)
    small_adj[:, 0, 1] = 1.0
    small_adj[:, 1, 0] = 1.0

    padded_nodes = torch.zeros(2, 4, NODE_FEATURE_DIM)
    padded_adj = torch.zeros(2, 4, 4)
    padded_nodes[0, :2] = small_nodes[0]
    padded_adj[0, :2, :2] = small_adj[0]
    padded_nodes[1] = torch.randn(4, NODE_FEATURE_DIM)
    padded_adj[1] = torch.eye(4)

    with torch.no_grad():
        _, base_graph = encoder(small_nodes, small_adj)
        _, padded_graph = encoder(padded_nodes, padded_adj)

    if not torch.allclose(base_graph[0], padded_graph[0], atol=1e-6):
        raise AssertionError("graph embedding changed after padding")


def check_weighted_edges_change_attention() -> None:
    torch.manual_seed(11)
    layer = GraphAttentionLayer(NODE_FEATURE_DIM, 8, num_heads=1, dropout=0.0, concat=False)
    layer.eval()
    h = torch.randn(1, 2, NODE_FEATURE_DIM)
    adj_regular = torch.eye(2).unsqueeze(0)
    adj_completed = torch.eye(2).unsqueeze(0)
    adj_regular[:, 0, 1] = 1.0
    adj_regular[:, 1, 0] = 1.0
    adj_completed[:, 0, 1] = 2.0
    adj_completed[:, 1, 0] = 2.0

    with torch.no_grad():
        out_regular = layer(h, adj_regular)
        out_completed = layer(h, adj_completed)

    if torch.allclose(out_regular, out_completed):
        raise AssertionError("weighted completed edges did not affect the GAT output")


def check_stable_device_encoding() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(PIPING_DIR)!r})
from graph_neural_networks import build_scene_graph
features, _ = build_scene_graph([{{'name': 'StableDevice', 'center': [0, 0, 0], 'size': [[0, 0, 0], [1, 1, 1]], 'ports': []}}], [])
print(int(features[0, 7:23].argmax().item()))
"""
    env = os.environ.copy()
    env.pop("PYTHONHASHSEED", None)
    first = subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip()
    second = subprocess.check_output([sys.executable, "-c", code], text=True, env=env).strip()
    if first != second:
        raise AssertionError("device type encoding is not stable across Python processes")


def main() -> None:
    torch.manual_seed(2026)
    check_attention_heads()
    check_actor_critic_shapes()
    check_scene_graph_builder()
    check_padding_invariance()
    check_weighted_edges_change_attention()
    check_stable_device_encoding()
    print("graph neural network smoke checks passed")


if __name__ == "__main__":
    main()
