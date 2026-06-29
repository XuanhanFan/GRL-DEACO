#!/usr/bin/env python3
"""DEACO-Green routing environment used by the PPO training loop."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

from ppo_trainer import DEACOActionSpace
from reward import RewardCalculator, RewardConfig, calculate_correction_penalty
from state import STATE_DIM, ConnectionState, StateExtractor


logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from deaco.aco import run_deaco
    from deaco.connections import (
        _build_port_lookup,
        _refresh_connections_from_ports,
        attach_connection_parameter_overrides,
        build_connections_from_config,
        build_tee_usage_map,
        filter_voxels_near_ports,
        infer_default_connections,
        insert_virtual_tees,
    )
    from deaco.grid import (
        build_sdbb_for_device,
        create_3d_grid_state_matrix,
        ensure_virtual_ports_clear_in_grid,
        update_state_matrix_with_path,
        voxelize_path,
    )
    from deaco.layout_io import create_placed_devices, load_layout_config, parse_scene_bounds
    from deaco.parameters import (
        DEACOParameters,
        initialize_scene_normalization_ranges,
    )
    from deaco.routing import try_multi_direction_extension
    from deaco.types import FitnessData, snap
    from deaco.visualization import visualize_state_matrix

    DEACO_AVAILABLE = True
    DEACO_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - optional geometry stack
    DEACO_AVAILABLE = False
    DEACO_IMPORT_ERROR = exc


class DEACOEnvironment:
    """Wrap one layout scenario as a sequential DEACO-Green routing environment."""

    def __init__(
        self,
        scenario_path: str,
        glb_directory: str,
        grid_res: float = 0.1,
        log_voxel_stats: bool = False,
        reward_config: Optional[RewardConfig | dict] = None,
        reward_calculator: Optional[RewardCalculator] = None,
        reproduction_config: Optional[dict] = None,
    ):
        if not DEACO_AVAILABLE:
            raise RuntimeError(
                "DEACO routing dependencies are unavailable. Install project requirements before "
                f"creating DEACOEnvironment. Original import error: {DEACO_IMPORT_ERROR}"
            )

        self.glb_directory = glb_directory
        self.grid_res = float(grid_res)
        self.log_voxel_stats = bool(log_voxel_stats)
        self.reproduction_config = reproduction_config or {}
        deaco_cfg = self.reproduction_config.get("deaco", {}) if isinstance(self.reproduction_config, dict) else {}
        self.deaco_geometry_config = deaco_cfg.get("geometry", {}) if isinstance(deaco_cfg, dict) else {}
        self.pipe_radius = float(self.deaco_geometry_config.get("pipe_radius_m", 0.05))
        self.pipe_safe_margin = float(self.deaco_geometry_config.get("pipe_safe_margin_m", 0.05))
        self.endpoint_extension = float(self.deaco_geometry_config.get("endpoint_extension_m", 0.25))
        self.endpoint_extension_step = float(self.deaco_geometry_config.get("endpoint_extension_step_m", 0.10))
        self.endpoint_extension_attempts = int(self.deaco_geometry_config.get("endpoint_extension_attempts", 10))
        self.fallback_extension_distances = list(
            self.deaco_geometry_config.get("fallback_extension_distances_m", [0.4, 0.6])
        )

        if os.path.isfile(scenario_path):
            layout_file = scenario_path
            self.scenario_dir = os.path.dirname(scenario_path)
        elif os.path.isdir(scenario_path):
            layout_file = os.path.join(scenario_path, "layout.json")
            self.scenario_dir = scenario_path
        else:
            raise ValueError(f"Scenario path does not exist: {scenario_path}")

        self.config = load_layout_config(layout_file)
        bounds_tuple = parse_scene_bounds(self.config["scene"])
        self.scene_bounds = bounds_tuple[:6]
        self.placed_devices = create_placed_devices(self.config, glb_directory)

        if self.config.get("connections"):
            base_connections = build_connections_from_config(self.config["connections"], self.placed_devices)
        else:
            base_connections = infer_default_connections(self.placed_devices)
        self.connections = attach_connection_parameter_overrides(base_connections, self.config)
        self.original_connections = list(self.connections)
        self.placed_devices, self.connections = insert_virtual_tees(
            self.placed_devices,
            self.connections,
            grid_res=self.grid_res,
            scene_bounds=self.scene_bounds,
        )

        self.state_matrix, self.grid_info = create_3d_grid_state_matrix(
            self.placed_devices,
            self.config["scene"],
            pitch=self.grid_res,
        )
        adjusted_ports = ensure_virtual_ports_clear_in_grid(
            self.placed_devices,
            self.state_matrix,
            self.grid_info,
            self.grid_res,
        )
        if adjusted_ports:
            self.connections = _refresh_connections_from_ports(self.connections, self.placed_devices)

        self.port_lookup = _build_port_lookup(self.placed_devices)
        self.tee_usage_total = build_tee_usage_map(self.connections)
        self.tee_usage_remaining = dict(self.tee_usage_total)

        scene_range_params = DEACOParameters.from_config(self.reproduction_config)
        initialize_scene_normalization_ranges(scene_range_params, self.grid_info, self.connections)
        self.scene_norm_ranges = getattr(scene_range_params, "scene_normalization_ranges", {}) or {}
        if isinstance(self.scene_norm_ranges, dict):
            self.grid_info["scene_normalization_ranges"] = self.scene_norm_ranges

        self.obstacle_cells = int(np.sum(self.state_matrix == 0))
        self.total_cells = int(self.state_matrix.size)
        self._clear_port_cells()

        self.obstacle_set = []
        for device in self.placed_devices:
            voxel_boxes = build_sdbb_for_device(device)
            if voxel_boxes:
                self.obstacle_set.append(voxel_boxes)

        self.grid_cell_size = (self.grid_res, self.grid_res, self.grid_res)
        self.current_connection_idx = 0
        self.episode_results = []

        if reward_calculator is not None:
            self.reward_config = reward_calculator.config
            self.reward_calculator = reward_calculator
        else:
            self.reward_config = reward_config if isinstance(reward_config, RewardConfig) else RewardConfig.from_dict(reward_config or {})
            self.reward_calculator = RewardCalculator(
                self.reward_config,
                scene_max_getter=self._get_scene_max,
                grid_res=self.grid_res,
                verbose=False,
            )

        logger.info(
            "DEACOEnvironment initialized: devices=%s, connections=%s, reward=configurable",
            len(self.placed_devices),
            len(self.connections),
        )

    def _clear_port_cells(self) -> None:
        """Ensure port cells and immediate neighbors are free before routing starts."""
        cleared = 0
        neighbors = 0
        shape = self.grid_info["shape"]
        bounds = self.grid_info["bounds"]
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        def clear_cell(point) -> int:
            i = int(round((point[0] - bounds["x"][0]) / self.grid_res))
            j = int(round((point[1] - bounds["y"][0]) / self.grid_res))
            k = int(round((point[2] - bounds["z"][0]) / self.grid_res))
            if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2] and self.state_matrix[i, j, k] == 0:
                self.state_matrix[i, j, k] = 1
                return 1
            return 0

        for device in self.placed_devices:
            for port in device["ports"]:
                port_pos = port["world_pos"]
                snapped = snap(port_pos, self.grid_res)
                cleared += clear_cell(snapped)

                base_i = int(round((snapped[0] - bounds["x"][0]) / self.grid_res))
                base_j = int(round((snapped[1] - bounds["y"][0]) / self.grid_res))
                base_k = int(round((snapped[2] - bounds["z"][0]) / self.grid_res))
                for di, dj, dk in directions:
                    point = (
                        bounds["x"][0] + (base_i + di) * self.grid_res,
                        bounds["y"][0] + (base_j + dj) * self.grid_res,
                        bounds["z"][0] + (base_k + dk) * self.grid_res,
                    )
                    neighbors += clear_cell(point)

                direction = np.asarray(port.get("direction", (0, 0, 0)), dtype=float)
                if np.linalg.norm(direction) <= 1e-8:
                    continue
                direction = direction / np.linalg.norm(direction)
                for ext_dist in (
                    self.endpoint_extension,
                    self.endpoint_extension + self.endpoint_extension_step,
                    self.endpoint_extension + 2 * self.endpoint_extension_step,
                    self.endpoint_extension + 3 * self.endpoint_extension_step,
                ):
                    ext_point = snap(tuple(np.asarray(port_pos) + direction * ext_dist), self.grid_res)
                    cleared += clear_cell(ext_point)

        self._update_obstacle_counter(-(cleared + neighbors))

    def _get_scene_max(self, key: str, fallback: float) -> float:
        ranges = getattr(self, "scene_norm_ranges", {})
        if isinstance(ranges, dict) and key in ranges:
            value = ranges[key]
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return float(value[1])
        return float(fallback)

    def reset(self, shuffle_connections: bool = False) -> Optional[ConnectionState]:
        """Reset connection order and return the first state."""
        self.current_connection_idx = 0
        self.episode_results = []
        self.tee_usage_remaining = dict(self.tee_usage_total)
        if shuffle_connections:
            import random

            self.connections = random.sample(self.original_connections, len(self.original_connections))
        else:
            self.connections = list(self.original_connections)
        self.reward_calculator.reset_episode()
        return self.get_current_state()

    def get_current_state(self) -> Optional[ConnectionState]:
        """Return the current connection state with dynamic obstacle density."""
        if self.current_connection_idx >= len(self.connections):
            return None
        return StateExtractor.build_state(
            self.config,
            self.placed_devices,
            self.connections[self.current_connection_idx],
            state_matrix=self.state_matrix,
            grid_info=self.grid_info,
            obstacle_cells=self.obstacle_cells,
            total_cells=self.total_cells,
            completed_connections=self.current_connection_idx,
            total_connections=len(self.connections),
        )

    def step(self, action_params: dict) -> Tuple[Optional[ConnectionState], float, bool, dict]:
        """Route the current connection with DEACO and return the RL transition."""
        if self.current_connection_idx >= len(self.connections):
            return None, 0.0, True, {}

        connection = self.connections[self.current_connection_idx]
        current_state = self.get_current_state()
        state_vector = current_state.to_vector() if current_state is not None else np.zeros(STATE_DIM, dtype=np.float32)
        task_complexity = float(current_state.task_complexity) if current_state is not None else 0.5

        original_action_params = action_params.copy()
        validated_action_params = DEACOActionSpace.validate_and_clamp_params(action_params)
        flow_rate_scale = float(validated_action_params.get("flow_rate_scale", 1.0))
        pipe_diameter_scale = float(validated_action_params.get("pipe_diameter_scale", 1.0))
        pipe_carbon_scale = float(validated_action_params.get("pipe_carbon_factor_scale", 1.0))
        param_kwargs = {
            key: value
            for key, value in validated_action_params.items()
            if key not in {"flow_rate_scale", "pipe_diameter_scale", "pipe_carbon_factor_scale"}
        }

        deaco_params = DEACOParameters.from_config(
            self.reproduction_config,
            early_stop_threshold=0.01,
            **param_kwargs,
        )
        deaco_params.flow_rate *= flow_rate_scale
        deaco_params.pipe_diameter *= pipe_diameter_scale
        deaco_params.pipe_carbon_factor *= pipe_carbon_scale
        validated_action_params.update(
            {
                "flow_rate_scale": flow_rate_scale,
                "pipe_diameter_scale": pipe_diameter_scale,
                "pipe_carbon_factor_scale": pipe_carbon_scale,
            }
        )

        start = snap(connection["from_pos"], self.grid_res)
        goal = snap(connection["to_pos"], self.grid_res)
        start, start_path = self._extend_endpoint(connection, endpoint="from", point=start)
        goal, goal_path = self._extend_endpoint(connection, endpoint="to", point=goal)
        goal_path.reverse()
        start_world = start
        goal_world = goal

        logger.info(
            "Running DEACO connection %s/%s: %s -> %s",
            self.current_connection_idx + 1,
            len(self.connections),
            connection["from"],
            connection["to"],
        )
        deaco_start_time = time.time()
        path, fitness_result = run_deaco(
            start_world=start_world,
            goal_world=goal_world,
            obstacle_set=self.obstacle_set,
            grid_info=self.grid_info,
            grid_cell_size=self.grid_cell_size,
            state_matrix=self.state_matrix,
            params=deaco_params,
            use_green=True,
        )
        elapsed_time = time.time() - deaco_start_time
        fitness_data = fitness_result if isinstance(fitness_result, FitnessData) else None
        fitness = float(fitness_data.J_total) if fitness_data is not None else float(fitness_result)
        stability_cv = 0.0 if path is not None and fitness != float("inf") else 1.0
        full_path = self._compose_full_path(path, start_path, goal_path)

        correction_penalty, correction_ratio, correction_details = calculate_correction_penalty(
            original_action_params,
            validated_action_params,
            coefficient=float(self.reward_config.get("correction_penalty_coefficient")),
        )
        reward_result = self.reward_calculator.calculate(
            full_path=full_path,
            fitness=fitness,
            fitness_data=fitness_data,
            deaco_params=deaco_params,
            validated_action_params=validated_action_params,
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
        reward = float(reward_result["reward"])
        success = bool(reward_result["success"])

        if full_path is not None and len(full_path) > 1:
            self._voxelize_successful_path(full_path, connection)

        connection_result = {
            "path": full_path,
            "fitness": fitness,
            "success": success,
            "params": validated_action_params,
            "reward_components": reward_result["reward_components"],
        }
        if fitness_data is not None:
            connection_result["fitness_data"] = {
                "J_total": float(fitness_data.J_total),
                "E_op": float(fitness_data.E_op),
                "CO2_op": float(fitness_data.CO2_op),
                "CO2_emb": float(fitness_data.CO2_emb),
                "L": int(fitness_data.L) if hasattr(fitness_data, "L") else 0,
                "N_bend": int(fitness_data.N_bend) if hasattr(fitness_data, "N_bend") else 0,
                "viol": float(fitness_data.viol),
                "alt": float(fitness_data.alt),
            }
            connection_result["green_bonus"] = float(reward_result["green_bonus"])
            if reward_result["green_metrics"] is not None:
                connection_result["green_metrics"] = self._convert_numpy_types(reward_result["green_metrics"])
        self.episode_results.append(connection_result)

        self.current_connection_idx += 1
        done = self.current_connection_idx >= len(self.connections)
        next_state = None if done else self.get_current_state()

        fitness_for_json = None if np.isinf(fitness) else float(fitness)
        info = {
            "connection_idx": int(self.current_connection_idx - 1),
            "fitness": fitness_for_json,
            "success": success,
            "path_length": int(reward_result["path_length"]),
            "total_height_change": float(reward_result["total_height_change"]),
            "height_ratio": float(reward_result["height_ratio"]),
            "correction_ratio": float(correction_ratio),
            "elapsed_time": float(elapsed_time),
            "time_penalty": float(reward_result["time_penalty"]),
            "stability_cv": float(stability_cv),
            "reward_components": self._convert_numpy_types(reward_result["reward_components"]),
        }
        if fitness_data is not None:
            info["N_bend"] = int(fitness_data.N_bend) if hasattr(fitness_data, "N_bend") else 0
            info["alt"] = float(fitness_data.alt) if hasattr(fitness_data, "alt") else 0.0
            info["viol"] = float(fitness_data.viol) if hasattr(fitness_data, "viol") else 0.0
            if hasattr(fitness_data, "L") and fitness_data.L > 0:
                info["path_length"] = int(fitness_data.L)
        if reward_result["green_metrics"] is not None:
            info["green_metrics"] = self._convert_numpy_types(reward_result["green_metrics"])
            info["green_bonus"] = float(reward_result["green_bonus"])

        return next_state, reward, done, info

    def _extend_endpoint(self, connection: dict, endpoint: str, point: tuple) -> Tuple[tuple, list]:
        """Apply the same endpoint extension policy used by DEACO layout routing."""
        path = [point]
        direction_key = "from_rota" if endpoint == "from" else "to_rota"
        name_key = "from" if endpoint == "from" else "to"
        device_name = connection[name_key].split(".")[0]
        if "_TEE_" in device_name:
            return point, path

        direction = np.asarray(connection[direction_key], dtype=float)
        if np.linalg.norm(direction) <= 1e-8:
            return point, path
        direction = direction / np.linalg.norm(direction)
        extension_distance = self.endpoint_extension
        shape = self.grid_info["shape"]
        bounds = self.grid_info["bounds"]

        for _ in range(self.endpoint_extension_attempts):
            candidate = snap(tuple(np.asarray(point) + direction * extension_distance), self.grid_res)
            i = int(round((candidate[0] - bounds["x"][0]) / self.grid_res))
            j = int(round((candidate[1] - bounds["y"][0]) / self.grid_res))
            k = int(round((candidate[2] - bounds["z"][0]) / self.grid_res))
            if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2] and self.state_matrix[i, j, k] == 1:
                path.append(candidate)
                return candidate, path
            extension_distance += self.endpoint_extension_step

        success, alt_point = try_multi_direction_extension(
            point,
            direction,
            self.grid_info,
            self.state_matrix,
            self.grid_res,
            extension_distances=self.fallback_extension_distances,
        )
        if success:
            path.append(alt_point)
            return alt_point, path
        return point, path

    def _compose_full_path(self, path_world, start_path: list, goal_path: list):
        if not path_world or len(path_world) <= 1:
            return path_world
        path_end = path_world[-1]
        goal_ext_start = goal_path[0] if goal_path else None
        middle_path = path_world[1:] if start_path else path_world
        tolerance = self.grid_res * 0.6

        if goal_ext_start is not None and np.allclose(path_end, goal_ext_start, atol=tolerance):
            end_path = goal_path[1:]
        elif goal_ext_start is not None:
            p1 = np.asarray(path_end)
            p2 = np.asarray(goal_ext_start)
            diff = p2 - p1
            axes_changed = [(idx, delta) for idx, delta in enumerate(diff) if abs(delta) > tolerance]
            bridge = []
            if len(axes_changed) == 2:
                intermediate = p1.copy()
                intermediate[2 if any(idx == 2 for idx, _ in axes_changed) else 1] = p2[
                    2 if any(idx == 2 for idx, _ in axes_changed) else 1
                ]
                bridge = [tuple(intermediate)]
            elif len(axes_changed) == 3:
                intermediate1 = p1.copy()
                intermediate1[2] = p2[2]
                intermediate2 = intermediate1.copy()
                intermediate2[1] = p2[1]
                bridge = [tuple(intermediate1), tuple(intermediate2)]
            end_path = bridge + goal_path
        else:
            end_path = goal_path

        return start_path + middle_path + end_path

    def _voxelize_successful_path(self, full_path: list, connection: dict) -> None:
        start_time = time.time()
        path_voxels = voxelize_path(full_path, self.pipe_radius, self.grid_res, safe_margin=self.pipe_safe_margin)
        if self.log_voxel_stats and time.time() - start_time > 1.0:
            logger.debug("Path voxelization took %.2fs", time.time() - start_time)
        if not path_voxels:
            return

        ports_to_keep_free = []
        for endpoint in ("from", "to"):
            port_name = connection.get(endpoint)
            if self.tee_usage_remaining.get(port_name, 0) > 1:
                ports_to_keep_free.append(port_name)
        if ports_to_keep_free:
            path_voxels = filter_voxels_near_ports(path_voxels, ports_to_keep_free, self.port_lookup, self.grid_res)

        added_count = update_state_matrix_with_path(self.state_matrix, self.grid_info, path_voxels)
        self._update_obstacle_counter(added_count)
        for endpoint in ("from", "to"):
            port_name = connection.get(endpoint)
            if port_name in self.tee_usage_remaining and self.tee_usage_remaining[port_name] > 0:
                self.tee_usage_remaining[port_name] -= 1

    def _convert_numpy_types(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        return obj

    def _update_obstacle_counter(self, delta: int) -> None:
        if delta == 0:
            return
        self.obstacle_cells = max(0, min(self.total_cells, self.obstacle_cells + int(delta)))

    def export_episode_results(self, output_dir: str, episode_name: str) -> None:
        """Export routed paths and reward components for one episode."""
        if not self.episode_results:
            logger.info("No episode results to export.")
            return
        os.makedirs(output_dir, exist_ok=True)
        export_data = {
            "scene": self.config.get("scene", {}),
            "devices": self.config.get("devices", []),
            "connections": [],
            "statistics": {
                "total_connections": len(self.episode_results),
                "successful_connections": sum(1 for result in self.episode_results if result["success"]),
                "total_fitness": sum(
                    result["fitness"] for result in self.episode_results if result["fitness"] != float("inf")
                ),
                "avg_fitness": None,
            },
        }

        for idx, (connection, result) in enumerate(zip(self.connections, self.episode_results)):
            export_data["connections"].append(
                {
                    "id": idx,
                    "from": connection["from"],
                    "to": connection["to"],
                    "from_pos": connection["from_pos"],
                    "to_pos": connection["to_pos"],
                    "success": result["success"],
                    "fitness": result["fitness"] if result["fitness"] != float("inf") else None,
                    "path_points": len(result["path"]) if result["path"] else 0,
                    "path": result["path"] if result["path"] else [],
                    "hyperparameters": result["params"],
                    "reward_components": result.get("reward_components", {}),
                }
            )

        valid_fitnesses = [
            result["fitness"]
            for result in self.episode_results
            if result["success"] and result["fitness"] != float("inf")
        ]
        if valid_fitnesses:
            export_data["statistics"]["avg_fitness"] = sum(valid_fitnesses) / len(valid_fitnesses)

        export_data = self._convert_numpy_types(export_data)
        info_file = os.path.join(output_dir, f"{episode_name}_info.json")
        with open(info_file, "w", encoding="utf-8") as handle:
            json.dump(export_data, handle, ensure_ascii=False, indent=2)
        logger.info("Episode results exported to: %s", info_file)

    def export_debug_visualization(self, output_dir: str, episode_name: str) -> None:
        """Export a lightweight obstacle-grid debug visualization when trimesh is installed."""
        os.makedirs(output_dir, exist_ok=True)
        try:
            obstacle_file = os.path.join(output_dir, f"{episode_name}_obstacles.glb")
            visualize_state_matrix(
                self.state_matrix,
                self.grid_info,
                output_file=obstacle_file,
                sample_rate=20,
                show_free_space=False,
            )
            logger.info("Debug obstacle visualization saved to: %s", obstacle_file)
        except Exception as exc:
            logger.warning("Debug visualization export failed: %s", exc)
