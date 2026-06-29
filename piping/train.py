#!/usr/bin/env python3
"""Training entry point for graph-aware PPO tuning of DEACO-Green."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - handled at config load time
    yaml = None

from environment import DEACOEnvironment
from ppo_trainer import DEACOActionSpace, PPOTrainer, TORCH_AVAILABLE, build_scene_graph, torch
from reward import RewardConfig
from state import STATE_DIM, ConnectionState


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _cfg_value(section: dict, key: str, default):
    value = section.get(key, default) if isinstance(section, dict) else default
    return default if value is None else value


def load_reproduction_config(config_path: Optional[str] = None) -> dict:
    """Load a YAML or JSON training configuration."""
    if config_path is None:
        config_path = str(_repo_root() / "configs" / "paper_reproduction_config.yaml")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Reproduction config not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs. Install project requirements first.")
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def set_reproducibility_seed(seed: Optional[int]) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if not TORCH_AVAILABLE:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainingLogger:
    """Simple file and console logger for training runs."""

    def __init__(self, log_dir: str, log_name: str = "training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = self.log_dir / f"{log_name}_{timestamp}.log"

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        self.logger = logging.getLogger("GRL_DEACO_Trainer")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def get_log_file(self) -> str:
        return str(self.log_file_path)


class TrainingStats:
    """Training metrics and per-connection records."""

    def __init__(self):
        self.episode_rewards = []
        self.episode_avg_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.episode_times = []
        self.episode_steps = []
        self.episode_avg_height_change = []
        self.episode_avg_height_ratio = []
        self.kl_divergences = []
        self.entropies = []
        self.policy_ratios = []
        self.clip_fractions = []
        self.connection_records = []
        self.start_time = None
        self.total_steps = 0

    def start_training(self) -> None:
        self.start_time = time.time()

    @staticmethod
    def _convert_to_json_serializable(obj):
        if isinstance(obj, (np.integer, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {key: TrainingStats._convert_to_json_serializable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [TrainingStats._convert_to_json_serializable(item) for item in obj]
        if isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        if hasattr(obj, "item"):
            try:
                return TrainingStats._convert_to_json_serializable(obj.item())
            except Exception:
                return str(obj)
        return str(obj)

    def log_connection(
        self,
        episode: int,
        connection_idx: int,
        state: np.ndarray,
        action_params: dict,
        reward: float,
        fitness: Optional[float],
        success: bool,
        path_length: int,
        total_height_change: float = 0.0,
        height_ratio: float = 0.0,
        correction_ratio: float = 0.0,
        reward_components: Optional[dict] = None,
    ) -> None:
        if fitness is None:
            fitness_value = None
        else:
            fitness_float = float(fitness)
            fitness_value = None if np.isinf(fitness_float) else fitness_float

        self.connection_records.append(
            {
                "episode": int(episode),
                "connection_idx": int(connection_idx),
                "state": self._convert_to_json_serializable(state),
                "action_params": self._convert_to_json_serializable(action_params),
                "reward": float(reward),
                "fitness": fitness_value,
                "success": bool(success),
                "path_length": int(path_length),
                "total_height_change": float(total_height_change),
                "height_ratio": float(height_ratio),
                "correction_ratio": float(correction_ratio),
                "reward_components": self._convert_to_json_serializable(reward_components or {}),
                "timestamp": datetime.now().isoformat(),
            }
        )

    def log_episode(
        self,
        episode_reward: float,
        avg_reward: float,
        actor_loss: float,
        critic_loss: float,
        episode_time: float,
        step_count: int,
        avg_height_change: Optional[float] = None,
        avg_height_ratio: Optional[float] = None,
        kl_divergence: Optional[float] = None,
        entropy: Optional[float] = None,
        policy_ratio: Optional[float] = None,
        clip_fraction: Optional[float] = None,
    ) -> None:
        self.episode_rewards.append(float(episode_reward))
        self.episode_avg_rewards.append(float(avg_reward))
        self.actor_losses.append(float(actor_loss))
        self.critic_losses.append(float(critic_loss))
        self.episode_times.append(float(episode_time))
        self.episode_steps.append(int(step_count))
        self.episode_avg_height_change.append(float(avg_height_change) if avg_height_change is not None else 0.0)
        self.episode_avg_height_ratio.append(float(avg_height_ratio) if avg_height_ratio is not None else 0.0)
        self.kl_divergences.append(float(kl_divergence) if kl_divergence is not None else 0.0)
        self.entropies.append(float(entropy) if entropy is not None else 0.0)
        self.policy_ratios.append(float(policy_ratio) if policy_ratio is not None else 1.0)
        self.clip_fractions.append(float(clip_fraction) if clip_fraction is not None else 0.0)
        self.total_steps += int(step_count)

    def get_elapsed_time(self) -> float:
        return 0.0 if self.start_time is None else time.time() - self.start_time

    def get_eta(self, current_episode: int, total_episodes: int) -> Optional[float]:
        if not self.episode_times:
            return None
        return float(np.mean(self.episode_times)) * (total_episodes - current_episode)

    def print_progress(self, episode: int, total_episodes: int) -> None:
        elapsed = self.get_elapsed_time()
        eta_seconds = self.get_eta(episode, total_episodes)
        print("\n" + "=" * 70)
        print(f"Training progress: Episode {episode}/{total_episodes} ({100 * episode / total_episodes:.1f}%)")
        print("=" * 70)
        print(f"Elapsed: {self.format_time(elapsed)}")
        if eta_seconds is not None:
            print(f"ETA: {self.format_time(eta_seconds)}")
        if self.episode_rewards:
            recent = min(5, len(self.episode_rewards))
            print(f"Recent average reward: {np.mean(self.episode_avg_rewards[-recent:]):.2f}")
            print(f"Recent actor loss: {np.mean(self.actor_losses[-recent:]):.4f}")
            print(f"Recent critic loss: {np.mean(self.critic_losses[-recent:]):.4f}")
            print(f"Recent KL: {np.mean(self.kl_divergences[-recent:]):.6f}")
        print("=" * 70)

    def save_to_file(self, filepath: str) -> None:
        stats = {
            "summary": self._summary(),
            "connection_records": self._convert_to_json_serializable(self.connection_records),
        }
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, ensure_ascii=False)

    def save_summary_only(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self._summary(), handle, indent=2, ensure_ascii=False)

    def _summary(self) -> dict:
        return self._convert_to_json_serializable(
            {
                "episode_rewards": self.episode_rewards,
                "episode_avg_rewards": self.episode_avg_rewards,
                "actor_losses": self.actor_losses,
                "critic_losses": self.critic_losses,
                "episode_times": self.episode_times,
                "episode_steps": self.episode_steps,
                "episode_avg_height_change": self.episode_avg_height_change,
                "episode_avg_height_ratio": self.episode_avg_height_ratio,
                "kl_divergences": self.kl_divergences,
                "entropies": self.entropies,
                "policy_ratios": self.policy_ratios,
                "clip_fractions": self.clip_fractions,
                "total_steps": self.total_steps,
                "total_time": self.get_elapsed_time(),
                "total_connections": len(self.connection_records),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{seconds / 60:.1f}min"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes}min"


def _collect_training_scenarios(train_dir: str) -> list:
    scenarios = []
    for difficulty in ["simple", "medium", "complex"]:
        difficulty_dir = os.path.join(train_dir, difficulty)
        if not os.path.isdir(difficulty_dir):
            continue
        json_files = sorted(
            filename
            for filename in os.listdir(difficulty_dir)
            if filename.startswith("layout_") and filename.endswith(".json")
        )
        for json_file in json_files:
            scenarios.append(
                {
                    "difficulty": difficulty,
                    "file": json_file,
                    "path": os.path.join(difficulty_dir, json_file),
                }
            )
    return scenarios


def main(
    checkpoint_path: Optional[str] = None,
    resume_episode: Optional[int] = None,
    num_episodes: int = 100,
    export_interval: int = 0,
    debug_export_interval: int = 0,
    config_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    glb_directory: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> dict:
    """Run PPO training over the configured DEACO-Green scenario split."""
    config = load_reproduction_config(config_path)
    path_config = config.get("paths", {})
    ppo_config = config.get("ppo", {})
    training_config = config.get("training", {})
    physical_config = config.get("physical_parameters", {})
    deaco_config = config.get("deaco", {})
    deaco_geometry_config = deaco_config.get("geometry", {}) if isinstance(deaco_config, dict) else {}
    action_ranges = config.get("deaco_action_ranges", {})
    reward_config = RewardConfig.from_dict(config.get("reward", {}))

    weights_sum = sum(reward_config.component_weights.values())
    if not np.isclose(weights_sum, 1.0):
        raise ValueError(f"Reward component weights must sum to 1.0, got {weights_sum:.6f}")

    config_seed = _cfg_value(training_config, "seed", None)
    active_seed = seed if seed is not None else config_seed
    set_reproducibility_seed(active_seed)

    script_dir = Path(__file__).parent
    scenario_dataset_dir = dataset_dir or _cfg_value(path_config, "dataset_dir", str(script_dir / "scenarios_rl_dataset"))
    glb_directory = glb_directory or _cfg_value(path_config, "glb_directory", str(script_dir.parent / "static" / "glb"))
    output_dir = output_dir or _cfg_value(path_config, "output_dir", None)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_file = Path(checkpoint_path)
        model_dir = checkpoint_file.parent
        base_output_dir = model_dir.parent
        if resume_episode is None:
            resume_episode = _infer_episode_from_checkpoint(checkpoint_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = Path(output_dir) if output_dir else Path(os.path.dirname(scenario_dataset_dir)) / "rl_training"
        base_output_dir = output_root / f"run_{timestamp}"
        resume_episode = None
        base_output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = base_output_dir / "logs"
    model_dir = base_output_dir / "models"
    stats_dir = base_output_dir / "stats"
    results_dir = base_output_dir / "results"
    debug_dir = base_output_dir / "debug"
    for directory in [log_dir, model_dir, stats_dir, results_dir, debug_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger(str(log_dir), "training")
    logger.info("Graph-aware PPO training for DEACO-Green hyperparameter tuning")
    logger.info(f"Output directory: {base_output_dir}")

    try:
        from deaco.glb_cache import preload_all_glb_devices

        logger.info("Preloading GLB voxel cache...")
        preload_all_glb_devices(glb_directory)
    except ImportError:
        logger.warning("GLB voxel cache module is unavailable; using standard loading.")

    action_space = DEACOActionSpace(param_ranges=action_ranges)
    state_dim = STATE_DIM
    action_dim = action_space.dim
    logger.info(f"State dimension: {state_dim}")
    logger.info(f"Action dimension: {action_dim}")
    logger.info(f"Reward weights: {reward_config.component_weights}")

    ppo_trainer = PPOTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        node_feature_dim=24,
        use_gnn=bool(_cfg_value(ppo_config, "use_gnn", True)),
        use_attention=bool(_cfg_value(ppo_config, "use_attention", True)),
        lr_actor=float(_cfg_value(ppo_config, "lr_actor", 3e-4)),
        lr_critic=float(_cfg_value(ppo_config, "lr_critic", 1e-3)),
        gamma=float(_cfg_value(ppo_config, "gamma", 0.99)),
        lambda_gae=float(_cfg_value(ppo_config, "lambda_gae", 0.95)),
        epsilon_clip=float(_cfg_value(ppo_config, "epsilon_clip", 0.2)),
        k_epochs=int(_cfg_value(ppo_config, "k_epochs", 4)),
        entropy_coef=float(_cfg_value(ppo_config, "entropy_coef", 0.01)),
        min_batch_size=int(_cfg_value(ppo_config, "min_batch_size", 4)),
        use_scheduler=bool(_cfg_value(training_config, "use_scheduler", False)),
        total_episodes=num_episodes,
        scheduler_config=training_config,
    )

    logger.info("Reproduction config:")
    logger.info(f"   config: {config_path or (_repo_root() / 'configs' / 'paper_reproduction_config.yaml')}")
    logger.info(f"   dataset_dir: {scenario_dataset_dir}")
    logger.info(f"   glb_directory: {glb_directory}")
    logger.info(f"   output_dir: {base_output_dir}")
    logger.info(f"   seed: {active_seed}")

    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        ppo_trainer.load(checkpoint_path)

    stats = TrainingStats()
    if checkpoint_path and os.path.exists(checkpoint_path) and resume_episode is not None:
        _load_previous_stats(stats, base_output_dir / "stats", resume_episode, logger)

    start_episode = resume_episode if resume_episode is not None else 0
    train_scenarios = _collect_training_scenarios(os.path.join(scenario_dataset_dir, "train"))
    if not train_scenarios:
        logger.error("No training scenarios found. Generate or provide a dataset with train/simple|medium|complex splits.")
        return {}

    logger.info(f"Training scenarios: {len(train_scenarios)}")
    logger.info(f"   simple: {len([s for s in train_scenarios if s['difficulty'] == 'simple'])}")
    logger.info(f"   medium: {len([s for s in train_scenarios if s['difficulty'] == 'medium'])}")
    logger.info(f"   complex: {len([s for s in train_scenarios if s['difficulty'] == 'complex'])}")

    if resume_episode is not None and stats.episode_times:
        previous_time = sum(stats.episode_times)
        stats.start_time = time.time() - previous_time
    else:
        stats.start_training()

    scenario_usage_count = {scenario["path"]: 0 for scenario in train_scenarios}
    for episode in range(start_episode, num_episodes):
        scheduled_params = ppo_trainer.apply_schedule(episode)
        if scheduled_params and (episode == start_episode or (episode + 1) % 10 == 0):
            logger.info(
                "Scheduled PPO params: "
                f"lr_actor={scheduled_params['lr_actor']:.6g}, "
                f"lr_critic={scheduled_params['lr_critic']:.6g}, "
                f"epsilon_clip={scheduled_params['epsilon_clip']:.3f}, "
                f"k_epochs={scheduled_params['k_epochs']}"
            )

        min_usage = min(scenario_usage_count.values())
        candidates = [scenario for scenario in train_scenarios if scenario_usage_count[scenario["path"]] == min_usage]
        scenario_info = random.choice(candidates)
        scenario_usage_count[scenario_info["path"]] += 1
        scenario_file = scenario_info["path"]
        scenario_name = f"{scenario_info['difficulty']}/{scenario_info['file']}"
        logger.info(f"\nEpisode {episode + 1}/{num_episodes}: {scenario_name}")

        episode_start_time = time.time()
        env = DEACOEnvironment(
            scenario_file,
            glb_directory,
            grid_res=float(
                _cfg_value(
                    deaco_geometry_config,
                    "grid_spacing_m",
                    _cfg_value(physical_config, "grid_spacing_m", 0.1),
                )
            ),
            reward_config=reward_config,
            reproduction_config=config,
        )
        state = env.reset()

        if ppo_trainer.use_gnn:
            node_features, adj_matrix = build_scene_graph(env.placed_devices, env.connections, completed_connections=0)
            logger.info(f"Scene graph: {len(env.placed_devices)} nodes, adjacency {adj_matrix.shape}")
        else:
            node_features = None
            adj_matrix = None

        episode_reward = 0.0
        step_count = 0
        while state is not None:
            state_vector = state.to_vector()
            action_raw, log_prob, _attention = ppo_trainer.select_action(state_vector, node_features, adj_matrix)
            action_normalized = action_space.normalize_action(action_raw)
            action_params = action_space.action_to_params(action_normalized)
            next_state, reward, done, info = env.step(action_params)

            stats.log_connection(
                episode=episode + 1,
                connection_idx=info["connection_idx"],
                state=state_vector,
                action_params=action_params,
                reward=reward,
                fitness=info["fitness"],
                success=info["success"],
                path_length=info["path_length"],
                total_height_change=info.get("total_height_change", 0.0),
                height_ratio=info.get("height_ratio", 0.0),
                correction_ratio=info.get("correction_ratio", 0.0),
                reward_components=info.get("reward_components", {}),
            )

            ppo_trainer.store_transition(
                state=state_vector,
                action=action_raw,
                log_prob=log_prob,
                reward=reward,
                done=done,
                node_features=node_features,
                adj_matrix=adj_matrix,
            )

            episode_reward += float(reward)
            step_count += 1
            fitness_str = "inf" if info["fitness"] is None else f"{info['fitness']:.2f}"
            logger.info(
                f"   connection {step_count}: reward={reward:.2f}, fitness={fitness_str}, "
                f"success={info['success']}, path_len={info['path_length']}"
            )

            if done:
                break
            state = next_state
            if ppo_trainer.use_gnn:
                node_features, adj_matrix = build_scene_graph(
                    env.placed_devices,
                    env.connections,
                    completed_connections=env.current_connection_idx,
                )

        update_stats = ppo_trainer.update()
        episode_time = time.time() - episode_start_time
        avg_reward = episode_reward / max(step_count, 1)
        episode_connections = [record for record in stats.connection_records if record["episode"] == episode + 1]
        avg_height_change = float(np.mean([record["total_height_change"] for record in episode_connections])) if episode_connections else 0.0
        avg_height_ratio = float(np.mean([record["height_ratio"] for record in episode_connections])) if episode_connections else 0.0

        stats.log_episode(
            episode_reward,
            avg_reward,
            update_stats["actor_loss"],
            update_stats["critic_loss"],
            episode_time,
            step_count,
            avg_height_change=avg_height_change,
            avg_height_ratio=avg_height_ratio,
            kl_divergence=update_stats["kl_divergence"],
            entropy=update_stats["entropy"],
            policy_ratio=update_stats["policy_ratio"],
            clip_fraction=update_stats["clip_fraction"],
        )

        logger.info(
            f"Episode {episode + 1} done: total_reward={episode_reward:.2f}, "
            f"avg_reward={avg_reward:.2f}, steps={step_count}, time={stats.format_time(episode_time)}"
        )
        logger.info(
            f"   actor_loss={update_stats['actor_loss']:.4f}, critic_loss={update_stats['critic_loss']:.4f}, "
            f"KL={update_stats['kl_divergence']:.6f}, entropy={update_stats['entropy']:.4f}"
        )
        stats.print_progress(episode + 1, num_episodes)

        episode_name = f"episode_{episode + 1:03d}_{scenario_info['difficulty']}"
        if export_interval > 0 and (episode + 1) % export_interval == 0:
            env.export_episode_results(str(results_dir), episode_name)
        if debug_export_interval > 0 and (episode + 1) % debug_export_interval == 0:
            env.export_debug_visualization(str(debug_dir), episode_name)

        if (episode + 1) % 5 == 0:
            _save_checkpoint_and_stats(ppo_trainer, stats, model_dir, stats_dir, episode + 1, logger)

    logger.info("Training complete.")
    final_model_path = str(model_dir / "rl_deaco_tuner_final.pth")
    ppo_trainer.save(final_model_path)
    stats.save_to_file(str(stats_dir / "training_stats_final_full.json"))
    stats.save_summary_only(str(stats_dir / "training_stats_final_summary.json"))
    logger.info(f"Final model: {final_model_path}")
    logger.info(f"Training log: {logger.get_log_file()}")
    logger.info(f"Run directory: {base_output_dir}")
    return {
        "run_dir": str(base_output_dir),
        "final_model_path": final_model_path,
        "final_stats_path": str(stats_dir / "training_stats_final_full.json"),
        "final_summary_path": str(stats_dir / "training_stats_final_summary.json"),
        "log_file": logger.get_log_file(),
    }


def _infer_episode_from_checkpoint(checkpoint_file: Path) -> Optional[int]:
    parts = checkpoint_file.stem.split("_")
    for idx, part in enumerate(parts):
        if part == "episode" and idx + 1 < len(parts):
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def _load_previous_stats(stats: TrainingStats, stats_dir: Path, resume_episode: int, logger: TrainingLogger) -> None:
    summary_path = stats_dir / f"training_stats_episode_{resume_episode}_summary.json"
    if not summary_path.exists():
        return
    try:
        with open(summary_path, "r", encoding="utf-8") as handle:
            old_stats = json.load(handle)
        stats.episode_rewards = old_stats.get("episode_rewards", [])[:resume_episode]
        stats.episode_avg_rewards = old_stats.get("episode_avg_rewards", [])[:resume_episode]
        stats.actor_losses = old_stats.get("actor_losses", [])[:resume_episode]
        stats.critic_losses = old_stats.get("critic_losses", [])[:resume_episode]
        stats.episode_times = old_stats.get("episode_times", [])[:resume_episode]
        stats.episode_steps = old_stats.get("episode_steps", [])[:resume_episode]
        stats.episode_avg_height_change = old_stats.get("episode_avg_height_change", [])[:resume_episode]
        stats.episode_avg_height_ratio = old_stats.get("episode_avg_height_ratio", [])[:resume_episode]
        stats.kl_divergences = old_stats.get("kl_divergences", [])[:resume_episode]
        stats.entropies = old_stats.get("entropies", [])[:resume_episode]
        stats.policy_ratios = old_stats.get("policy_ratios", [])[:resume_episode]
        stats.clip_fractions = old_stats.get("clip_fractions", [])[:resume_episode]
        stats.total_steps = sum(stats.episode_steps)

        full_stats_path = stats_dir / f"training_stats_episode_{resume_episode}_full.json"
        if full_stats_path.exists():
            with open(full_stats_path, "r", encoding="utf-8") as handle:
                full_stats = json.load(handle)
            stats.connection_records = [
                record for record in full_stats.get("connection_records", []) if record.get("episode", 0) <= resume_episode
            ]
        logger.info(f"Recovered statistics through episode {resume_episode}.")
    except Exception as exc:
        logger.warning(f"Failed to recover training statistics: {exc}")


def _save_checkpoint_and_stats(
    ppo_trainer: PPOTrainer,
    stats: TrainingStats,
    model_dir: Path,
    stats_dir: Path,
    episode: int,
    logger: TrainingLogger,
) -> None:
    model_path = str(model_dir / f"rl_deaco_tuner_episode_{episode}.pth")
    ppo_trainer.save(model_path)
    stats.save_to_file(str(stats_dir / f"training_stats_episode_{episode}_full.json"))
    stats.save_summary_only(str(stats_dir / f"training_stats_episode_{episode}_summary.json"))
    logger.info(f"Checkpoint saved: {model_path}")


def find_latest_checkpoint(training_base_dir: str) -> Optional[Tuple[str, int]]:
    """Find the newest episode checkpoint under a training output tree."""
    training_base = Path(training_base_dir)
    if not training_base.exists():
        return None

    run_dirs = sorted([path for path in training_base.iterdir() if path.is_dir() and path.name.startswith("run_")], reverse=True)
    for run_dir in run_dirs:
        models_dir = run_dir / "models"
        if not models_dir.exists():
            continue
        checkpoint_files = list(models_dir.glob("rl_deaco_tuner_episode_*.pth"))
        if not checkpoint_files:
            continue
        checkpoint_files.sort(key=lambda path: _infer_episode_from_checkpoint(path) or -1, reverse=True)
        latest = checkpoint_files[0]
        episode_num = _infer_episode_from_checkpoint(latest)
        if episode_num is not None and episode_num > 0:
            return str(latest), episode_num
    return None


def _print_status() -> int:
    script_dir = Path(__file__).parent
    training_base_dir = script_dir / "rl_training"
    print("\n" + "=" * 70)
    print("Training status")
    print("=" * 70)
    if not training_base_dir.exists():
        print(f"Training directory does not exist: {training_base_dir}")
        return 1
    run_dirs = sorted([path for path in training_base_dir.iterdir() if path.is_dir() and path.name.startswith("run_")], reverse=True)
    if not run_dirs:
        print("No training runs found.")
        return 1
    for run_dir in run_dirs[:5]:
        models_dir = run_dir / "models"
        stats_dir = run_dir / "stats"
        print(f"\n{run_dir.name}")
        print(f"   path: {run_dir}")
        if models_dir.exists():
            checkpoint_files = sorted(models_dir.glob("rl_deaco_tuner_episode_*.pth"))
            latest_episode = max([_infer_episode_from_checkpoint(path) or 0 for path in checkpoint_files], default=0)
            print(f"   checkpoints: {len(checkpoint_files)} (latest episode {latest_episode})")
            if (models_dir / "rl_deaco_tuner_final.pth").exists():
                print("   final model: yes")
        if stats_dir.exists():
            print(f"   summaries: {len(list(stats_dir.glob('training_stats_episode_*_summary.json')))}")
    latest = find_latest_checkpoint(str(training_base_dir))
    if latest:
        checkpoint_path, episode_num = latest
        print("\nLatest checkpoint:")
        print(f"   file: {checkpoint_path}")
        print(f"   episode: {episode_num}")
        print(f"   resume: python piping/train.py --checkpoint {checkpoint_path} --episodes {episode_num + 10}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the graph-aware PPO agent for DEACO-Green hyperparameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python piping/train.py --config configs/paper_reproduction_config.yaml
  python piping/train.py --episodes 200
  python piping/train.py --checkpoint /path/to/rl_deaco_tuner_episode_50.pth --episodes 200
  python piping/train.py --auto-resume --episodes 200
  python piping/train.py --status
        """,
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for resumed training")
    parser.add_argument("--resume-episode", type=int, default=None, help="Episode number to resume from")
    parser.add_argument("--episodes", type=int, default=100, help="Total number of training episodes")
    parser.add_argument("--auto-resume", action="store_true", help="Find the latest checkpoint and resume")
    parser.add_argument("--status", action="store_true", help="Print available training runs and checkpoints")
    parser.add_argument("--export-interval", type=int, default=0, help="Export routed layouts every N episodes")
    parser.add_argument("--debug-export-interval", type=int, default=0, help="Export debug visualizations every N episodes")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML/JSON reproduction config")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset root containing train/validation/test splits")
    parser.add_argument("--glb-directory", type=str, default=None, help="Directory containing the GLB equipment library")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for run_* training outputs")
    parser.add_argument("--seed", type=int, default=None, help="Random seed overriding the config value")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.status:
        raise SystemExit(_print_status())

    if args.auto_resume:
        training_base_dir = Path(__file__).parent / "rl_training"
        latest_checkpoint_info = find_latest_checkpoint(str(training_base_dir))
        if latest_checkpoint_info:
            args.checkpoint, args.resume_episode = latest_checkpoint_info
            print(f"Auto-resume checkpoint: {args.checkpoint} (episode {args.resume_episode})")
        else:
            print("No checkpoint found; starting from scratch.")

    if args.checkpoint and args.resume_episode is None:
        args.resume_episode = _infer_episode_from_checkpoint(Path(args.checkpoint))
        if args.resume_episode is not None:
            print(f"Inferred resume episode: {args.resume_episode}")

    main(
        checkpoint_path=args.checkpoint,
        resume_episode=args.resume_episode,
        num_episodes=args.episodes,
        export_interval=args.export_interval,
        debug_export_interval=args.debug_export_interval,
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        glb_directory=args.glb_directory,
        output_dir=args.output_dir,
        seed=args.seed,
    )
