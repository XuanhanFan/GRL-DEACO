#!/usr/bin/env python3
"""PPO policy, critic, action space, and trainer for DEACO-Green tuning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - used by CLI/help smoke checks
    torch = None
    optim = None
    Normal = None
    TORCH_AVAILABLE = False

    class _MissingTorchModule:
        Module = object

    nn = _MissingTorchModule()

try:
    from graph_neural_networks import (
        GraphAwareActorNetwork,
        GraphAwareCriticNetwork,
        build_scene_graph,
    )

    GNN_AVAILABLE = True
    GNN_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - optional policy backend
    GraphAwareActorNetwork = None
    GraphAwareCriticNetwork = None
    GNN_AVAILABLE = False
    GNN_IMPORT_ERROR = exc

    def build_scene_graph(*args, **kwargs):
        raise RuntimeError(f"Graph policy utilities are unavailable: {GNN_IMPORT_ERROR}")


logger = logging.getLogger(__name__)


class ActorNetwork(nn.Module):
    """MLP actor that emits Gaussian parameters for the DEACO action vector."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to instantiate ActorNetwork. Install project requirements first.")
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        features = self.network(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        std = torch.exp(torch.clamp(log_std, -2.3, 2.0))
        return mean, std


class CriticNetwork(nn.Module):
    """MLP critic that estimates V(s)."""

    def __init__(self, state_dim: int, hidden_dim: int = 512):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to instantiate CriticNetwork. Install project requirements first.")
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: "torch.Tensor") -> "torch.Tensor":
        return self.network(state)


@dataclass
class DEACOActionSpace:
    """Continuous 29-dimensional DEACO-Green hyperparameter action space."""

    param_names: Optional[List[str]] = None
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None

    def __post_init__(self) -> None:
        self.param_names = [
            "w_emb",
            "w_L",
            "w_bend",
            "w_alt",
            "w_clear",
            "omega_Length",
            "omega_Bend",
            "omega_Energy",
            "omega_Install",
            "omega_direction_reward",
            "omega_bend_penalty",
            "omega_height_penalty",
            "kappa_y",
            "delta_xz",
            "s_sigmoid",
            "alpha",
            "beta",
            "M_ants",
            "K_iterations",
            "max_steps",
            "early_stop_patience",
            "tau_0",
            "tau_max0",
            "tau_min0",
            "A_q0",
            "B_q0",
            "delta_gamma",
            "rho",
            "Q",
        ]

        public_ranges = {
            "w_emb": (0.1, 5.0),
            "w_L": (0.1, 5.0),
            "w_bend": (0.1, 6.0),
            "w_alt": (0.1, 5.0),
            "w_clear": (0.1, 8.0),
            "omega_Length": (0.1, 5.0),
            "omega_Bend": (0.5, 8.0),
            "omega_Energy": (0.0, 3.0),
            "omega_Install": (0.0, 5.0),
            "omega_direction_reward": (0.0, 5.0),
            "omega_bend_penalty": (0.1, 5.0),
            "omega_height_penalty": (0.0, 5.0),
            "kappa_y": (0.0, 3.0),
            "delta_xz": (3.0, 30.0),
            "s_sigmoid": (1.0, 10.0),
            "alpha": (0.5, 3.0),
            "beta": (1.0, 6.0),
            "M_ants": (10, 80),
            "K_iterations": (20, 120),
            "max_steps": (5000, 40000),
            "early_stop_patience": (5, 40),
            "tau_0": (0.01, 1.0),
            "tau_max0": (1.0, 30.0),
            "tau_min0": (0.001, 0.1),
            "A_q0": (0.1, 0.9),
            "B_q0": (0.0, 0.3),
            "delta_gamma": (0.5, 0.99),
            "rho": (0.02, 0.4),
            "Q": (10.0, 300.0),
        }

        merged_ranges = public_ranges.copy()
        for name, value in (self.param_ranges or {}).items():
            if name not in merged_ranges or value is None:
                continue
            if len(value) != 2:
                raise ValueError(f"Action range for {name} must be [min, max], got {value}")
            low, high = float(value[0]), float(value[1])
            if low >= high:
                raise ValueError(f"Action range for {name} must satisfy min < max, got {value}")
            merged_ranges[name] = (low, high)
        self.param_ranges = merged_ranges

    @property
    def dim(self) -> int:
        return len(self.param_names)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map an unconstrained Gaussian action to [0, 1]."""
        return (np.tanh(action) + 1.0) / 2.0

    def action_to_params(self, action: np.ndarray) -> dict:
        """Project a normalized action vector into DEACO hyperparameters."""
        params = {}
        for idx, name in enumerate(self.param_names):
            low, high = self.param_ranges[name]
            value = low + float(action[idx]) * (high - low)
            if name in {"M_ants", "K_iterations", "max_steps", "early_stop_patience"}:
                params[name] = int(round(value))
            else:
                params[name] = float(value)
        return params

    @staticmethod
    def validate_and_clamp_params(action_params: dict) -> dict:
        """Return JSON-serializable DEACO parameters.

        Range validation is handled by the action-space projection. This method
        preserves the previous public behavior by normalizing NumPy scalar types
        before parameters are passed to DEACO and written to logs.
        """
        validated = action_params.copy()
        for key, value in list(validated.items()):
            if isinstance(value, (np.integer, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                validated[key] = int(value)
            elif isinstance(value, (np.floating, np.float16, np.float32, np.float64)):
                validated[key] = float(value)
            elif isinstance(value, np.ndarray):
                validated[key] = value.tolist()
            elif isinstance(value, np.bool_):
                validated[key] = bool(value)
            elif isinstance(value, (float, int)):
                validated[key] = value
        return validated


class PPOTrainer:
    """PPO trainer for DEACO-Green hyperparameter policies."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        node_feature_dim: int = 24,
        use_gnn: bool = True,
        use_attention: bool = True,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
        epsilon_clip: float = 0.2,
        k_epochs: int = 4,
        entropy_coef: float = 0.01,
        min_batch_size: int = 4,
        device: Optional[str] = None,
        use_scheduler: bool = False,
        total_episodes: int = 1,
        scheduler_config: Optional[Dict] = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for PPO training. Install project requirements first.")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.epsilon_clip = epsilon_clip
        self.epsilon_clip_init = epsilon_clip
        self.k_epochs = k_epochs
        self.k_epochs_init = k_epochs
        self.entropy_coef = entropy_coef
        self.min_batch_size = min_batch_size
        self.lr_actor_init = lr_actor
        self.lr_critic_init = lr_critic
        self.use_scheduler = use_scheduler

        requested_gnn = bool(use_gnn)
        self.use_gnn = requested_gnn and GNN_AVAILABLE
        self.use_attention = bool(use_attention)
        if requested_gnn and not GNN_AVAILABLE:
            logger.warning("GNN modules are unavailable; using the MLP policy. Import error: %s", GNN_IMPORT_ERROR)

        if self.use_gnn and self.use_attention:
            logger.info("Using graph-aware actor and critic with attention.")
            self.actor = GraphAwareActorNetwork(
                connection_state_dim=state_dim,
                node_feature_dim=node_feature_dim,
                action_dim=action_dim,
                gat_hidden_dim=128,
                mlp_hidden_dim=512,
                num_gat_layers=2,
                num_attention_heads=4,
            ).to(self.device)
            self.critic = GraphAwareCriticNetwork(
                connection_state_dim=state_dim,
                node_feature_dim=node_feature_dim,
                gat_hidden_dim=128,
                mlp_hidden_dim=512,
                num_gat_layers=2,
                num_attention_heads=4,
            ).to(self.device)
        else:
            if self.use_gnn and not self.use_attention:
                logger.info("Graph policy without attention is not implemented; using the MLP ablation policy.")
                self.use_gnn = False
            else:
                logger.info("Using the MLP policy.")
            self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
            self.critic = CriticNetwork(state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.scheduler = None
        if self.use_scheduler:
            from training_scheduler import ParameterScheduler

            self.scheduler = ParameterScheduler(
                total_episodes=max(int(total_episodes), 1),
                config=scheduler_config,
            )
            logger.info("PPO parameter scheduler enabled.")

        self.buffer = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "node_features": [],
            "adj_matrices": [],
        }

    def apply_schedule(self, episode: Optional[int] = None) -> Dict[str, float]:
        """Apply scheduled lr, clip, and epoch parameters for the episode."""
        if self.scheduler is None:
            return {}
        if episode is not None:
            self.scheduler.current_episode = int(episode)
        scheduled = self.scheduler.get_all_params()
        for group in self.actor_optimizer.param_groups:
            group["lr"] = float(scheduled["lr_actor"])
        for group in self.critic_optimizer.param_groups:
            group["lr"] = float(scheduled["lr_critic"])
        self.epsilon_clip = float(scheduled["epsilon_clip"])
        self.k_epochs = int(scheduled["k_epochs"])
        return scheduled

    def select_action(
        self,
        state: np.ndarray,
        node_features: Optional["torch.Tensor"] = None,
        adj_matrix: Optional["torch.Tensor"] = None,
    ) -> Tuple[np.ndarray, float, Optional["torch.Tensor"]]:
        """Sample one action from the current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.use_gnn and node_features is not None and adj_matrix is not None:
                node_tensor = node_features.unsqueeze(0).to(self.device)
                adj_tensor = adj_matrix.unsqueeze(0).to(self.device)
                mean, std, attention = self.actor(state_tensor, node_tensor, adj_tensor)
            else:
                mean, std = self.actor(state_tensor)
                attention = None
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).mean(dim=-1)
        return action.cpu().numpy().flatten(), float(log_prob.item()), attention

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        node_features: Optional["torch.Tensor"] = None,
        adj_matrix: Optional["torch.Tensor"] = None,
    ) -> None:
        """Append one transition to the on-policy PPO buffer."""
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["rewards"].append(reward)
        self.buffer["dones"].append(done)
        if self.use_gnn:
            self.buffer["node_features"].append(node_features.cpu().numpy() if node_features is not None else None)
            self.buffer["adj_matrices"].append(adj_matrix.cpu().numpy() if adj_matrix is not None else None)

    def update(self) -> dict:
        """Run the PPO update and return scalar diagnostics."""
        min_batch_size = getattr(self, "min_batch_size", 4)
        if len(self.buffer["states"]) < min_batch_size:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "kl_divergence": 0.0,
                "entropy": 0.0,
                "policy_ratio": 1.0,
                "clip_fraction": 0.0,
                "n_epochs_completed": 0,
                "skipped": True,
                "buffer_size": len(self.buffer["states"]),
            }

        states = torch.FloatTensor(np.asarray(self.buffer["states"])).to(self.device)
        actions = torch.FloatTensor(np.asarray(self.buffer["actions"])).to(self.device)
        old_log_probs = torch.FloatTensor(np.asarray(self.buffer["log_probs"])).to(self.device)
        rewards = np.asarray(self.buffer["rewards"], dtype=np.float32)
        dones = np.asarray(self.buffer["dones"], dtype=np.float32)
        node_features, adj_matrices = self._prepare_graph_batch()

        with torch.no_grad():
            if self.use_gnn and node_features is not None:
                values = self.critic(states, node_features, adj_matrices).squeeze().cpu().numpy()
            else:
                values = self.critic(states).squeeze().cpu().numpy()

        advantages_np, returns_np = self._compute_gae(rewards, values, dones, self.gamma, self.lambda_gae)
        returns = torch.FloatTensor(returns_np).to(self.device)
        advantages = torch.FloatTensor(advantages_np).to(self.device)

        if len(returns) > 1:
            returns_std = returns.std()
            returns_mean = returns.mean()
            relative_std = returns_std / (torch.abs(returns_mean) + 1e-6)
            if returns_std > 0.1 and relative_std > 0.05:
                returns = (returns - returns_mean) / (returns_std + 1e-8)
            else:
                returns = returns / 20.0

        kl_divs, entropies, ratios, clip_fractions = [], [], [], []
        actor_loss = torch.tensor(0.0, device=self.device)
        critic_loss = torch.tensor(0.0, device=self.device)

        for epoch in range(self.k_epochs):
            if self.use_gnn and node_features is not None:
                mean, std, _ = self.actor(states, node_features, adj_matrices)
                predicted_values = self.critic(states, node_features, adj_matrices).squeeze()
            else:
                mean, std = self.actor(states)
                predicted_values = self.critic(states).squeeze()

            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).mean(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratio = torch.exp(new_log_probs - old_log_probs)
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - (new_log_probs - old_log_probs)).mean()
                kl_divs.append(float(approx_kl.item()))
                entropies.append(float(entropy.item()))
                ratios.append(float(ratio.mean().item()))
                clipped = ((ratio < 1.0 - self.epsilon_clip) | (ratio > 1.0 + self.epsilon_clip)).float()
                clip_fractions.append(float(clipped.mean().item()))

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_loss = nn.SmoothL1Loss(beta=0.1)(predicted_values, returns)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.3)
            self.critic_optimizer.step()

            if approx_kl > 0.03:
                break

        self.buffer = {key: [] for key in self.buffer.keys()}
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "kl_divergence": float(np.mean(kl_divs)) if kl_divs else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "policy_ratio": float(np.mean(ratios)) if ratios else 1.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "n_epochs_completed": len(kl_divs),
        }

    def _prepare_graph_batch(self) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        if not self.use_gnn:
            return None, None
        node_features_list = self.buffer["node_features"]
        adj_matrices_list = self.buffer["adj_matrices"]
        if not node_features_list or node_features_list[0] is None:
            return None, None

        max_nodes = max(nf.shape[0] for nf in node_features_list)
        feature_dim = node_features_list[0].shape[1]
        batch_size = len(node_features_list)
        padded_node_features = np.zeros((batch_size, max_nodes, feature_dim), dtype=np.float32)
        padded_adj_matrices = np.zeros((batch_size, max_nodes, max_nodes), dtype=np.float32)
        for idx, (node_features, adj_matrix) in enumerate(zip(node_features_list, adj_matrices_list)):
            num_nodes = node_features.shape[0]
            padded_node_features[idx, :num_nodes, :] = node_features
            padded_adj_matrices[idx, :num_nodes, :num_nodes] = adj_matrix
        return (
            torch.FloatTensor(padded_node_features).to(self.device),
            torch.FloatTensor(padded_adj_matrices).to(self.device),
        )

    @staticmethod
    def _compute_gae(rewards, values, dones, gamma=0.99, lambda_gae=0.95):
        advantages = []
        returns = []
        gae = 0.0
        for idx in reversed(range(len(rewards))):
            if idx == len(rewards) - 1:
                next_value = 0.0 if dones[idx] else values[idx]
            else:
                next_value = values[idx + 1]
            delta = rewards[idx] + gamma * next_value - values[idx]
            gae = delta + gamma * lambda_gae * (1.0 - dones[idx]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[idx])
        return advantages, returns

    def save(self, path: str) -> None:
        """Save actor, critic, and optimizer states."""
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )
        logger.info("Model saved to: %s", path)

    def load(self, path: str) -> None:
        """Load actor, critic, and optimizer states."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        logger.info("Model loaded from: %s", path)
