#!/usr/bin/env python3
"""Training-parameter schedules for PPO-based DEACO tuning.

The scheduler is intentionally narrow: it only emits parameters that are
currently consumed by the PPO training loop. This keeps the public
reproduction code explicit and avoids unused schedules for reward terms or
exploration variables that are not applied downstream.
"""

from dataclasses import dataclass, fields, replace
from math import cos, pi
from typing import Any, Dict, Mapping, Optional, Tuple, Union


ConfigInput = Union["SchedulerConfig", Mapping[str, Any]]


@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration for deterministic PPO schedule generation."""

    lr_actor_init: float = 3e-4
    lr_actor_min: float = 3e-5
    lr_critic_init: float = 1e-3
    lr_critic_min: float = 1e-4
    lr_decay_type: str = "cosine"
    lr_decay_rate: float = 0.96
    lr_decay_steps: int = 100
    lr_milestones: Tuple[int, ...] = ()
    lr_gamma: float = 0.1
    epsilon_clip_init: float = 0.2
    epsilon_clip_end: float = 0.1
    k_epochs_min: int = 4
    k_epochs_max: int = 4
    k_epochs_peak: float = 0.5


class ParameterScheduler:
    """Deterministic scheduler for the PPO parameters used by training."""

    def __init__(
        self,
        total_episodes: int,
        config: Optional[ConfigInput] = None,
    ):
        """Create a scheduler.

        Args:
            total_episodes: Number of training episodes used to normalize
                schedule progress.
            config: Optional default schedule configuration. A dictionary can
                override any field in ``SchedulerConfig``.
        """
        if total_episodes <= 0:
            raise ValueError("total_episodes must be positive")

        self.total_episodes = int(total_episodes)
        self.current_episode = 0
        self.config = self._merge_config(SchedulerConfig(), config)

    def get_progress(self) -> float:
        """Return normalized training progress in the closed interval [0, 1]."""
        progress = self.current_episode / self.total_episodes
        return self._clamp(progress, 0.0, 1.0)

    def step(self, episodes: int = 1) -> None:
        """Advance the scheduler by a number of completed episodes."""
        if episodes < 0:
            raise ValueError("episodes must be non-negative")
        self.current_episode += int(episodes)

    def get_all_params(self, config: Optional[ConfigInput] = None) -> Dict[str, Any]:
        """Return scheduled PPO parameters for the current episode.

        Args:
            config: Optional per-call overrides. This keeps backward
                compatibility with callers that pass a dictionary to
                ``get_all_params``.
        """
        cfg = self._merge_config(self.config, config)
        progress = self.get_progress()

        return {
            "episode": self.current_episode,
            "progress": progress,
            "lr_actor": self._learning_rate(
                cfg.lr_actor_init,
                cfg.lr_actor_min,
                cfg,
            ),
            "lr_critic": self._learning_rate(
                cfg.lr_critic_init,
                cfg.lr_critic_min,
                cfg,
            ),
            "epsilon_clip": self._linear(
                cfg.epsilon_clip_init,
                cfg.epsilon_clip_end,
                progress,
            ),
            "k_epochs": self._scheduled_k_epochs(cfg, progress),
        }

    def _learning_rate(
        self,
        initial: float,
        minimum: float,
        cfg: SchedulerConfig,
    ) -> float:
        """Compute a learning rate using the configured decay policy."""
        decay_type = cfg.lr_decay_type.lower()
        progress = self.get_progress()

        if decay_type == "constant":
            return float(initial)

        if decay_type == "cosine":
            cosine_decay = 0.5 * (1.0 + cos(pi * progress))
            return minimum + (initial - minimum) * cosine_decay

        if decay_type == "exponential":
            decay_steps = max(int(cfg.lr_decay_steps), 1)
            decay_count = self.current_episode // decay_steps
            return max(float(minimum), initial * (cfg.lr_decay_rate ** decay_count))

        if decay_type == "step":
            lr = float(initial)
            for milestone in cfg.lr_milestones:
                if self.current_episode >= milestone:
                    lr *= cfg.lr_gamma
            return max(float(minimum), lr)

        raise ValueError(f"Unknown lr_decay_type: {cfg.lr_decay_type}")

    def _scheduled_k_epochs(
        self,
        cfg: SchedulerConfig,
        progress: float,
    ) -> int:
        """Return a fixed or triangular PPO epoch schedule."""
        k_min = max(int(cfg.k_epochs_min), 1)
        k_max = max(int(cfg.k_epochs_max), k_min)

        if k_min == k_max:
            return k_min

        peak = self._clamp(cfg.k_epochs_peak, 1e-6, 1.0 - 1e-6)
        if progress <= peak:
            ratio = progress / peak
            value = k_min + (k_max - k_min) * ratio
        else:
            ratio = (progress - peak) / (1.0 - peak)
            value = k_max - (k_max - k_min) * ratio

        return max(int(round(value)), 1)

    @staticmethod
    def _linear(start: float, end: float, progress: float) -> float:
        """Linearly interpolate between two values."""
        return start + (end - start) * progress

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        """Clamp a scalar into a closed interval."""
        return max(lower, min(upper, float(value)))

    @staticmethod
    def _merge_config(
        base: SchedulerConfig,
        overrides: Optional[ConfigInput],
    ) -> SchedulerConfig:
        """Merge dictionary or dataclass overrides into a scheduler config."""
        if overrides is None:
            return base

        if isinstance(overrides, SchedulerConfig):
            return overrides

        valid_fields = {field.name for field in fields(SchedulerConfig)}
        updates = {
            key: value
            for key, value in overrides.items()
            if key in valid_fields
        }

        if "lr_milestones" in updates and updates["lr_milestones"] is not None:
            updates["lr_milestones"] = tuple(
                int(milestone) for milestone in updates["lr_milestones"]
            )

        return replace(base, **updates)
