#!/usr/bin/env python3
"""Configurable reward calculation for DEACO-Green PPO training."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ppo_trainer import DEACOActionSpace


logger = logging.getLogger(__name__)


DEFAULT_REWARD_CONFIG: Dict[str, Any] = {
    "normalizer": {
        "window_size": 200,
        "warmup_size": 50,
        "initial_mean": 500.0,
        "initial_std": 200.0,
        "ema_alpha": 0.1,
        "fallback_std": 50.0,
        "inf_penalty": -15.0,
        "output_clip": [-12.0, 0.0],
    },
    "task_normalization": {
        "enabled": True,
        "log_scale": 2.0,
        "clip": [-5.0, 5.0],
        "distance_bucket": 20,
        "height_bucket": 10,
        "history_min_exact": 3,
        "history_min_neighbor": 5,
        "history_limit": 100,
        "baseline_min": 0.3,
        "baseline_max": 10.0,
        "baseline_coefficients": {
            "intercept": 0.5,
            "distance": 0.003,
            "height": 0.015,
            "interaction": 0.002,
        },
    },
    "component_weights": {
        "base": 0.35,
        "length": 0.12,
        "height": 0.05,
        "correction": 0.05,
        "green": 0.05,
        "diversity": 0.02,
        "adaptation": 0.20,
        "utilization": 0.01,
        "time": 0.15,
    },
    "component_scales": {
        "base": 1.0,
        "length": 1.2,
        "height": 1.2,
        "correction": 1.0,
        "green": 4.0,
        "diversity": 2.0,
        "adaptation": 2.0,
        "utilization": 2.0,
        "time": 1.0,
    },
    "component_clips": {
        "base": [-8.0, 8.0],
        "length": [-10.0, 6.0],
        "height": [-10.0, 5.0],
        "correction": [-8.0, 0.0],
        "green": [-20.0, 15.0],
        "diversity": [0.0, 5.0],
        "adaptation": [0.0, 8.0],
        "utilization": [0.0, 3.0],
        "time": [-5.0, 1.0],
    },
    "path_length": {
        "max_expected_length_multiplier": 2.0,
        "reward_scale": 3.0,
        "clip": [-10.0, 3.0],
    },
    "height": {
        "expected_height_multiplier": 1.5,
        "penalty_scale": 2.0,
        "bonus_scale": 0.5,
        "clip": [-10.0, 2.0],
    },
    "green_components": {
        "energy_coefficient": 2.0,
        "energy_clip": [-2.0, 2.0],
        "carbon_coefficient": 1.5,
        "carbon_clip": [-1.5, 1.5],
        "violation_coefficient": 1.5,
        "green_clip": [-5.0, 4.0],
    },
    "time_efficiency": {
        "base_seconds": 20.0,
        "complexity_seconds": 80.0,
        "reference_iterations": 100.0,
        "reference_ants": 50.0,
        "param_factor_clip": [0.3, 5.0],
        "timeout_log_clip": [-5.0, 0.0],
        "early_bonus_scale": 0.3,
        "early_bonus_clip": [0.0, 1.0],
    },
    "failure": {
        "partial_base": -20.0,
        "partial_progress_coefficient": 10.0,
        "complete": -30.0,
        "lower_bound": -40.0,
        "correction_extra_threshold": 0.1,
        "max_extra_penalty": 10.0,
    },
    "success_scaling": {
        "enabled": True,
        "coefficient": 2.5,
        "center": 0.5,
    },
    "adaptive_shaping": {
        "window_size": 50,
        "diversity_clip": [0.0, 1.0],
        "adaptation_clip": [0.0, 1.0],
        "utilization_std_scale": 2.0,
        "utilization_clip": [0.0, 1.0],
    },
    "correction_penalty_coefficient": 5.0,
}


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _clip(value: float, bounds) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    return float(np.clip(value, low, high))


@dataclass
class RewardConfig:
    """Configuration wrapper for reward formulas and ablation overrides."""

    values: Dict[str, Any]

    @classmethod
    def default(cls) -> "RewardConfig":
        return cls.from_dict({})

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "RewardConfig":
        merged = deepcopy(DEFAULT_REWARD_CONFIG)
        _deep_update(merged, data or {})
        return cls(merged)

    def get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.values
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def component_weights(self) -> Dict[str, float]:
        return {key: float(value) for key, value in self.values["component_weights"].items()}

    @property
    def component_scales(self) -> Dict[str, float]:
        return {key: float(value) for key, value in self.values["component_scales"].items()}

    @property
    def component_clips(self) -> Dict[str, Tuple[float, float]]:
        return {key: (float(value[0]), float(value[1])) for key, value in self.values["component_clips"].items()}


class RewardNormalizer:
    """Moving fitness normalizer used when task normalization is disabled."""

    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig.default()
        normalizer = self.config.get("normalizer")
        self.window_size = int(normalizer["window_size"])
        self.warmup_size = int(normalizer["warmup_size"])
        self.fitness_mean = float(normalizer["initial_mean"])
        self.fitness_std = float(normalizer["initial_std"])
        self.ema_alpha = float(normalizer["ema_alpha"])
        self.fallback_std = float(normalizer["fallback_std"])
        self.inf_penalty = float(normalizer["inf_penalty"])
        self.output_clip = normalizer["output_clip"]
        self.fitness_history = []

    def update_statistics(self, fitness: float) -> None:
        """Update moving statistics from a finite fitness value."""
        if np.isinf(fitness):
            return
        self.fitness_history.append(float(fitness))
        if len(self.fitness_history) > self.window_size:
            self.fitness_history.pop(0)

        if len(self.fitness_history) >= self.warmup_size:
            recent = self.fitness_history[-self.warmup_size :]
            recent_mean = float(np.mean(recent))
            recent_std = float(np.std(recent))
            self.fitness_mean = self.ema_alpha * recent_mean + (1.0 - self.ema_alpha) * self.fitness_mean
            self.fitness_std = self.ema_alpha * recent_std + (1.0 - self.ema_alpha) * self.fitness_std
        elif len(self.fitness_history) >= 10:
            self.fitness_mean = float(np.mean(self.fitness_history))
            self.fitness_std = float(np.std(self.fitness_history))

        if self.fitness_std < 1e-6:
            self.fitness_std = self.fallback_std

    def normalize_fitness(self, fitness: float) -> float:
        """Map fitness to a bounded reward where lower fitness is better."""
        if np.isinf(fitness):
            return self.inf_penalty
        z_score = (float(fitness) - self.fitness_mean) / (self.fitness_std + 1e-6)
        normalized = -5.0 * (1.0 + np.tanh(z_score))
        return _clip(normalized, self.output_clip)


class AdaptiveRewardFunction:
    """Bounded parameter-shaping terms for public reproduction runs."""

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.config = config or RewardConfig.default()
        shaping = self.config.get("adaptive_shaping")
        self.window_size = max(int(shaping["window_size"]), 1)
        action_space = DEACOActionSpace(param_ranges=param_ranges)
        self.param_names = action_space.param_names
        self.param_ranges = action_space.param_ranges
        self.param_history = []

    def reset(self) -> None:
        self.param_history.clear()

    def _normalized_param_vector(self, params: dict) -> np.ndarray:
        values = []
        for name in self.param_names:
            low, high = self.param_ranges[name]
            raw_value = float(params.get(name, (low + high) / 2.0))
            values.append(np.clip((raw_value - low) / (high - low + 1e-12), 0.0, 1.0))
        return np.asarray(values, dtype=np.float32)

    def compute_diversity_bonus(self, params: dict) -> float:
        vector = self._normalized_param_vector(params)
        if not self.param_history:
            self.param_history.append(vector)
            return 0.0
        recent_mean = np.mean(self.param_history, axis=0)
        distance = np.linalg.norm(vector - recent_mean) / np.sqrt(len(vector))
        self.param_history.append(vector)
        if len(self.param_history) > self.window_size:
            self.param_history.pop(0)
        return _clip(distance, self.config.get("adaptive_shaping", "diversity_clip"))

    def compute_adaptation_bonus(self, params: dict, state_vector: np.ndarray) -> float:
        task_complexity = float(np.clip(state_vector[-1], 0.0, 1.0)) if len(state_vector) else 0.5
        effort_names = ("M_ants", "K_iterations", "max_steps", "beta")
        effort_values = []
        for name in effort_names:
            low, high = self.param_ranges[name]
            raw_value = float(params.get(name, (low + high) / 2.0))
            effort_values.append(np.clip((raw_value - low) / (high - low + 1e-12), 0.0, 1.0))
        effort = float(np.mean(effort_values))
        return _clip(1.0 - abs(effort - task_complexity), self.config.get("adaptive_shaping", "adaptation_clip"))

    def compute_parameter_utilization_bonus(self, params: dict) -> float:
        shaping = self.config.get("adaptive_shaping")
        vector = self._normalized_param_vector(params)
        utilization = np.std(vector) * float(shaping["utilization_std_scale"])
        return _clip(utilization, shaping["utilization_clip"])


def calculate_correction_penalty(
    original_params: dict,
    corrected_params: dict,
    coefficient: float = DEFAULT_REWARD_CONFIG["correction_penalty_coefficient"],
) -> Tuple[float, float, dict]:
    """Compute a bounded penalty for parameters changed by validation."""
    total_correction = 0.0
    correction_details = {}
    comparable_keys = [key for key in original_params.keys() if not str(key).startswith("_")]

    for name in comparable_keys:
        original = original_params[name]
        corrected = corrected_params.get(name, original)
        if original == corrected:
            continue
        if abs(original) > 1e-6:
            relative_diff = abs(corrected - original) / abs(original)
        else:
            relative_diff = abs(corrected - original)
        correction_details[name] = {
            "original": float(original),
            "corrected": float(corrected),
            "relative_diff": float(relative_diff),
        }
        total_correction += relative_diff

    avg_correction_ratio = total_correction / len(comparable_keys) if comparable_keys else 0.0
    correction_penalty = -avg_correction_ratio * float(coefficient)
    return float(correction_penalty), float(avg_correction_ratio), correction_details


class RewardCalculator:
    """Compute training rewards and structured reward components."""

    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        scene_max_getter: Optional[Callable[[str, float], float]] = None,
        grid_res: float = 0.1,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        verbose: bool = False,
    ):
        self.config = config or RewardConfig.default()
        self.scene_max_getter = scene_max_getter or (lambda _key, fallback: fallback)
        self.grid_res = float(grid_res)
        self.verbose = verbose
        self.normalizer = RewardNormalizer(self.config)
        self.adaptive_reward = AdaptiveRewardFunction(self.config, param_ranges=param_ranges)
        self.task_baseline_fitness_history: Dict[Any, list] = {}

    def reset_episode(self) -> None:
        self.adaptive_reward.reset()

    def estimate_baseline_fitness(self, start_world, goal_world) -> float:
        """Estimate a task baseline fitness from distance and vertical complexity."""
        task_cfg = self.config.get("task_normalization")
        manhattan_dist = np.sum(np.abs(np.asarray(goal_world) - np.asarray(start_world)))
        height_diff = abs(goal_world[1] - start_world[1])
        distance_complexity = manhattan_dist / self.grid_res
        height_complexity = height_diff / self.grid_res / 2.0
        distance_level = int(distance_complexity / float(task_cfg["distance_bucket"]))
        height_level = int(height_complexity / float(task_cfg["height_bucket"]))
        complexity_key = (distance_level, height_level)

        history = self.task_baseline_fitness_history.get(complexity_key, [])
        if len(history) >= int(task_cfg["history_min_exact"]):
            return float(np.median(history))

        neighbor_data = []
        for key in [
            (distance_level - 1, height_level),
            (distance_level + 1, height_level),
            (distance_level, height_level - 1),
            (distance_level, height_level + 1),
        ]:
            neighbor_data.extend(self.task_baseline_fitness_history.get(key, []))
        if len(neighbor_data) >= int(task_cfg["history_min_neighbor"]):
            return float(np.median(neighbor_data))

        coeffs = task_cfg["baseline_coefficients"]
        baseline = (
            float(coeffs["intercept"])
            + float(coeffs["distance"]) * distance_complexity
            + float(coeffs["height"]) * height_complexity
            + float(coeffs["interaction"]) * distance_complexity * height_complexity
        )
        return float(np.clip(baseline, float(task_cfg["baseline_min"]), float(task_cfg["baseline_max"])))

    def update_baseline_fitness(self, complexity_key, actual_fitness: float) -> None:
        history = self.task_baseline_fitness_history.setdefault(complexity_key, [])
        history.append(float(actual_fitness))
        history_limit = int(self.config.get("task_normalization", "history_limit"))
        if len(history) > history_limit:
            self.task_baseline_fitness_history[complexity_key] = history[-history_limit:]

    def _complexity_key(self, start_world, goal_world):
        task_cfg = self.config.get("task_normalization")
        manhattan_dist = np.sum(np.abs(np.asarray(goal_world) - np.asarray(start_world)))
        height_diff = abs(goal_world[1] - start_world[1])
        distance_complexity = manhattan_dist / self.grid_res
        height_complexity = height_diff / self.grid_res / 2.0
        return (
            int(distance_complexity / float(task_cfg["distance_bucket"])),
            int(height_complexity / float(task_cfg["height_bucket"])),
        )

    def compute_time_efficiency_penalty(self, elapsed_time: float, params: dict, task_complexity: float) -> float:
        time_cfg = self.config.get("time_efficiency")
        base_time = float(time_cfg["base_seconds"]) + float(task_complexity) * float(time_cfg["complexity_seconds"])
        iterations = float(params.get("K_iterations", time_cfg["reference_iterations"]))
        ants = float(params.get("M_ants", time_cfg["reference_ants"]))
        param_factor = (iterations / float(time_cfg["reference_iterations"])) * (ants / float(time_cfg["reference_ants"]))
        param_factor = _clip(param_factor, time_cfg["param_factor_clip"])
        expected_time = base_time * param_factor
        time_ratio = float(elapsed_time) / (expected_time + 1e-6)

        if time_ratio > 1.0:
            return _clip(-np.log(1.0 + time_ratio - 1.0), time_cfg["timeout_log_clip"])
        bonus = float(time_cfg["early_bonus_scale"]) * (1.0 / max(time_ratio, 1e-6) - 1.0)
        return _clip(bonus, time_cfg["early_bonus_clip"])

    def calculate(
        self,
        *,
        full_path,
        fitness: float,
        fitness_data,
        deaco_params,
        validated_action_params: dict,
        start_world,
        goal_world,
        correction_penalty: float,
        correction_ratio: float,
        correction_details: dict,
        stability_cv: float,
        elapsed_time: float,
        state_vector: np.ndarray,
        task_complexity: float,
    ) -> dict:
        """Return reward, diagnostics, and structured components."""
        if full_path is not None and len(full_path) > 1 and not np.isinf(fitness):
            return self._success_reward(
                full_path=full_path,
                fitness=float(fitness),
                fitness_data=fitness_data,
                deaco_params=deaco_params,
                params=validated_action_params,
                start_world=start_world,
                goal_world=goal_world,
                correction_penalty=correction_penalty,
                correction_ratio=correction_ratio,
                correction_details=correction_details,
                stability_cv=stability_cv,
                elapsed_time=elapsed_time,
                state_vector=state_vector,
                task_complexity=task_complexity,
            )
        return self._failure_reward(
            full_path=full_path,
            start_world=start_world,
            goal_world=goal_world,
            correction_ratio=correction_ratio,
            correction_details=correction_details,
        )

    def _success_reward(self, **kwargs) -> dict:
        full_path = kwargs["full_path"]
        fitness = kwargs["fitness"]
        params = kwargs["params"]
        start_world = kwargs["start_world"]
        goal_world = kwargs["goal_world"]
        correction_penalty = kwargs["correction_penalty"]
        elapsed_time = kwargs["elapsed_time"]
        state_vector = kwargs["state_vector"]
        task_complexity = kwargs["task_complexity"]

        self.normalizer.update_statistics(fitness)
        base_reward, task_info = self._base_reward(fitness, start_world, goal_world)
        length_bonus, path_length = self._length_bonus(full_path, start_world, goal_world)
        height_penalty, total_height_change, height_ratio = self._height_penalty(full_path, start_world, goal_world, params)
        green_bonus, green_metrics = self._green_bonus(kwargs["fitness_data"], kwargs["deaco_params"])
        diversity = self.adaptive_reward.compute_diversity_bonus(params)
        adaptation = self.adaptive_reward.compute_adaptation_bonus(params, state_vector)
        utilization = self.adaptive_reward.compute_parameter_utilization_bonus(params)
        time_penalty = self.compute_time_efficiency_penalty(elapsed_time, params, task_complexity)

        raw_components = {
            "base": base_reward,
            "length": length_bonus,
            "height": height_penalty,
            "correction": correction_penalty,
            "green": green_bonus,
            "diversity": diversity,
            "adaptation": adaptation,
            "utilization": utilization,
            "time": time_penalty,
        }
        scaled_components = {}
        weights = self.config.component_weights
        for name, raw_value in raw_components.items():
            scaled = raw_value * self.config.component_scales[name]
            scaled_components[name] = _clip(scaled, self.config.component_clips[name])

        reward_before_scaling = float(sum(weights[name] * scaled_components[name] for name in weights))
        reward_multiplier = 1.0
        success_cfg = self.config.get("success_scaling")
        if bool(success_cfg["enabled"]):
            fitness_normalized = 1.0 / (1.0 + fitness)
            reward_multiplier = float(np.exp(float(success_cfg["coefficient"]) * (fitness_normalized - float(success_cfg["center"]))))
        reward = reward_before_scaling * reward_multiplier

        reward_components = {
            "raw": {key: float(value) for key, value in raw_components.items()},
            "scaled": {key: float(value) for key, value in scaled_components.items()},
            "weights": weights,
            "weighted": {key: float(weights[key] * scaled_components[key]) for key in weights},
            "reward_before_success_scaling": float(reward_before_scaling),
            "success_multiplier": float(reward_multiplier),
            "task_normalization": task_info,
            "correction_ratio": float(kwargs["correction_ratio"]),
            "correction_details": kwargs["correction_details"],
            "stability_cv": float(kwargs["stability_cv"]),
        }

        if self.verbose:
            logger.info("Path found: fitness=%.2f, path_points=%s, reward=%.2f", fitness, path_length, reward)

        return {
            "reward": float(reward),
            "success": True,
            "reward_components": reward_components,
            "green_metrics": green_metrics,
            "green_bonus": float(green_bonus),
            "time_penalty": float(time_penalty),
            "path_length": int(path_length),
            "total_height_change": float(total_height_change),
            "height_ratio": float(height_ratio),
        }

    def _failure_reward(self, *, full_path, start_world, goal_world, correction_ratio: float, correction_details: dict) -> dict:
        failure_cfg = self.config.get("failure")
        if full_path is not None and len(full_path) > 0:
            manhattan_dist = np.sum(np.abs(np.asarray(goal_world) - np.asarray(start_world)))
            expected_length = int(manhattan_dist / self.grid_res) * 2
            progress_ratio = min(1.0, len(full_path) / max(expected_length, 1))
            reward = float(failure_cfg["partial_base"]) + progress_ratio * float(failure_cfg["partial_progress_coefficient"])
            failure_type = "partial"
        else:
            progress_ratio = 0.0
            reward = float(failure_cfg["complete"])
            failure_type = "complete"
            if correction_ratio > float(failure_cfg["correction_extra_threshold"]):
                extra_penalty = min(correction_ratio * float(failure_cfg["max_extra_penalty"]), float(failure_cfg["max_extra_penalty"]))
                reward = max(float(failure_cfg["lower_bound"]), reward - extra_penalty)

        reward_components = {
            "failure_type": failure_type,
            "progress_ratio": float(progress_ratio),
            "correction_ratio": float(correction_ratio),
            "correction_details": correction_details,
            "raw": {},
            "scaled": {},
            "weights": self.config.component_weights,
            "weighted": {},
        }
        return {
            "reward": float(reward),
            "success": False,
            "reward_components": reward_components,
            "green_metrics": None,
            "green_bonus": 0.0,
            "time_penalty": 0.0,
            "path_length": int(len(full_path) if full_path is not None else 0),
            "total_height_change": 0.0,
            "height_ratio": 0.0,
        }

    def _base_reward(self, fitness: float, start_world, goal_world) -> Tuple[float, dict]:
        task_cfg = self.config.get("task_normalization")
        if bool(task_cfg["enabled"]):
            baseline_fitness = self.estimate_baseline_fitness(start_world, goal_world)
            relative_performance = baseline_fitness / (fitness + 1e-6)
            reward = np.log(relative_performance + 1e-6) * float(task_cfg["log_scale"])
            reward = _clip(reward, task_cfg["clip"])
            self.update_baseline_fitness(self._complexity_key(start_world, goal_world), fitness)
            return float(reward), {
                "enabled": True,
                "baseline_fitness": float(baseline_fitness),
                "relative_performance": float(relative_performance),
            }
        return self.normalizer.normalize_fitness(fitness), {"enabled": False}

    def _length_bonus(self, full_path, start_world, goal_world) -> Tuple[float, int]:
        cfg = self.config.get("path_length")
        manhattan_dist = np.sum(np.abs(np.asarray(goal_world) - np.asarray(start_world)))
        expected = max(int(manhattan_dist / self.grid_res) * float(cfg["max_expected_length_multiplier"]), 1.0)
        path_length = len(full_path)
        if path_length < expected:
            value = (1.0 - path_length / expected) * float(cfg["reward_scale"])
        else:
            value = -((path_length - expected) / expected) * float(cfg["reward_scale"])
        return _clip(value, cfg["clip"]), int(path_length)

    def _height_penalty(self, full_path, start_world, goal_world, params: dict) -> Tuple[float, float, float]:
        cfg = self.config.get("height")
        total_height_change = 0.0
        for idx in range(1, len(full_path)):
            total_height_change += abs(full_path[idx][1] - full_path[idx - 1][1])

        start_pos = np.asarray(start_world)
        goal_pos = np.asarray(goal_world)
        horizontal_dist = abs(start_pos[0] - goal_pos[0]) + abs(start_pos[2] - goal_pos[2])
        height_dist = abs(start_pos[1] - goal_pos[1])
        omega_height = float(params.get("omega_height_penalty", 1.0))
        expected_height_change = height_dist * float(cfg["expected_height_multiplier"])

        if total_height_change > expected_height_change:
            ratio = (total_height_change - expected_height_change) / max(horizontal_dist + 1e-6, 1.0)
            value = -omega_height * float(cfg["penalty_scale"]) * ratio
        else:
            ratio = (expected_height_change - total_height_change) / max(horizontal_dist + 1e-6, 1.0)
            value = omega_height * float(cfg["bonus_scale"]) * ratio
        height_ratio = total_height_change / (horizontal_dist + 1e-6) if horizontal_dist + 1e-6 > 0 else 0.0
        return _clip(value, cfg["clip"]), float(total_height_change), float(height_ratio)

    def _green_bonus(self, fitness_data, deaco_params) -> Tuple[float, Optional[dict]]:
        if fitness_data is None:
            return 0.0, None
        cfg = self.config.get("green_components")
        ref_energy = max(float(getattr(deaco_params, "reference_energy", 1e-6)), 1e-6)
        energy_ratio = float(fitness_data.E_op) / ref_energy
        energy_coeff = float(cfg["energy_coefficient"])
        energy_component = (1.0 - energy_ratio) * energy_coeff if energy_ratio <= 1.0 else -(energy_ratio - 1.0) * energy_coeff
        energy_component = _clip(energy_component, cfg["energy_clip"])

        carbon_ref = max(float(getattr(deaco_params, "pipe_carbon_factor", 1.0)) * self.scene_max_getter("L", 100.0), 1e-6)
        carbon_ratio = float(fitness_data.CO2_emb) / carbon_ref
        carbon_coeff = float(cfg["carbon_coefficient"])
        carbon_component = (1.0 - carbon_ratio) * carbon_coeff if carbon_ratio <= 1.0 else -(carbon_ratio - 1.0) * carbon_coeff
        carbon_component = _clip(carbon_component, cfg["carbon_clip"])

        viol_ref = max(self.scene_max_getter("viol", 10.0), 1e-6)
        viol_ratio = float(fitness_data.viol) / viol_ref
        viol_component = -np.clip(viol_ratio, 0.0, 1.0) * float(cfg["violation_coefficient"])

        green_bonus = _clip(energy_component + carbon_component + viol_component, cfg["green_clip"])
        return green_bonus, {
            "energy_ratio": float(energy_ratio),
            "carbon_ratio": float(carbon_ratio),
            "viol_ratio": float(viol_ratio),
            "energy_component": float(energy_component),
            "carbon_component": float(carbon_component),
            "viol_component": float(viol_component),
        }
