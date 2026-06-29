#!/usr/bin/env python3
"""State representation for graph-aware DEACO-Green policy training."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


STATE_DIM = 27
logger = logging.getLogger(__name__)


@dataclass
class ConnectionState:
    """Scene-level and active-connection features consumed by the PPO policy."""

    area_x: float
    area_y: float
    area_z: float
    area_volume: float
    num_devices: int
    obstacle_density: float
    avg_device_size: float
    scene_L_max: float
    scene_alt_max: float
    scene_clearance_max: float
    start_x_norm: float
    start_y_norm: float
    start_z_norm: float
    end_x_norm: float
    end_y_norm: float
    end_z_norm: float
    manhattan_distance: float
    euclidean_distance: float
    distance_ratio: float
    direction_x: float
    direction_y: float
    direction_z: float
    y_expected: float
    y_distance_ratio: float
    path_tortuosity: float
    vertical_complexity: float
    task_complexity: float

    def to_vector(self) -> np.ndarray:
        """Return the fixed 27-dimensional policy input vector."""
        vector = np.array(
            [
                self.area_x / 100.0,
                self.area_y / 100.0,
                self.area_z / 100.0,
                self.area_volume / 10000.0,
                self.num_devices / 10.0,
                self.obstacle_density,
                self.avg_device_size / 10.0,
                self.scene_L_max / 200.0,
                self.scene_alt_max / 50.0,
                self.scene_clearance_max / 30.0,
                self.start_x_norm,
                self.start_y_norm,
                self.start_z_norm,
                self.end_x_norm,
                self.end_y_norm,
                self.end_z_norm,
                self.manhattan_distance / 100.0,
                self.euclidean_distance / 100.0,
                self.distance_ratio / 2.0,
                self.direction_x,
                self.direction_y,
                self.direction_z,
                self.y_expected,
                self.y_distance_ratio,
                self.path_tortuosity,
                self.vertical_complexity,
                self.task_complexity,
            ],
            dtype=np.float32,
        )

        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            vector = np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)

        return vector

    @property
    def dim(self) -> int:
        """State dimension."""
        return STATE_DIM


class StateExtractor:
    """Build connection states from a layout scene and one routing task."""

    @staticmethod
    def extract_scene_features(
        config: dict,
        placed_devices: list,
        state_matrix: Optional[np.ndarray] = None,
        grid_info: Optional[dict] = None,
        obstacle_cells: Optional[int] = None,
        total_cells: Optional[int] = None,
    ) -> dict:
        """Extract global scene features, using cached obstacle counts when available."""
        area = config["scene"]["area"]
        area_x = float(area["x"])
        area_y = float(area["y"])
        area_z = float(area["z"])
        area_volume = area_x * area_y * area_z
        num_devices = len(placed_devices)

        if obstacle_cells is not None and total_cells is not None and total_cells > 0:
            obstacle_density = obstacle_cells / total_cells
        elif state_matrix is not None:
            total_cells = state_matrix.size
            start = time.time()
            obstacle_cells = int(np.sum(state_matrix == 0))
            elapsed = time.time() - start
            if elapsed > 0.1:
                logger.debug(
                    "Obstacle-density scan took %.2fs for %s cells; pass cached obstacle_cells "
                    "and total_cells during training.",
                    elapsed,
                    f"{total_cells:,}",
                )
            obstacle_density = obstacle_cells / total_cells if total_cells else 0.0
        else:
            total_device_volume = 0.0
            for device in placed_devices:
                bounds = device.get("size")
                if bounds is None:
                    continue
                total_device_volume += abs(
                    (bounds[1][0] - bounds[0][0])
                    * (bounds[1][1] - bounds[0][1])
                    * (bounds[1][2] - bounds[0][2])
                )
            obstacle_density = total_device_volume / area_volume if area_volume > 0 else 0.0

        device_sizes = []
        for device in placed_devices:
            bounds = device.get("size")
            if bounds is None:
                continue
            device_volume = abs(
                (bounds[1][0] - bounds[0][0])
                * (bounds[1][1] - bounds[0][1])
                * (bounds[1][2] - bounds[0][2])
            )
            device_sizes.append(device_volume ** (1.0 / 3.0))

        avg_device_size = float(np.mean(device_sizes)) if device_sizes else 0.0
        scene_L_max = area_x + area_y + area_z
        scene_alt_max = area_y
        scene_clearance_max = max(area_x, area_z)

        if grid_info is not None:
            scene_ranges = grid_info.get("scene_normalization_ranges")
            if scene_ranges:
                scene_L_max = scene_ranges.get("L", (0.0, scene_L_max))[1]
                scene_alt_max = scene_ranges.get("alt", (0.0, scene_alt_max))[1]
                scene_clearance_max = scene_ranges.get("f_Install", (0.0, scene_clearance_max))[1]

        return {
            "area_x": area_x,
            "area_y": area_y,
            "area_z": area_z,
            "area_volume": area_volume,
            "num_devices": num_devices,
            "obstacle_density": float(np.clip(obstacle_density, 0.0, 1.0)),
            "avg_device_size": avg_device_size,
            "scene_L_max": float(scene_L_max),
            "scene_alt_max": float(scene_alt_max),
            "scene_clearance_max": float(scene_clearance_max),
        }

    @staticmethod
    def extract_connection_features(connection: dict, scene_features: dict) -> dict:
        """Extract normalized geometric features for the active connection."""
        start_pos = np.asarray(connection["from_pos"], dtype=float)
        end_pos = np.asarray(connection["to_pos"], dtype=float)

        area_x = max(float(scene_features["area_x"]), 1e-6)
        area_y = max(float(scene_features["area_y"]), 1e-6)
        area_z = max(float(scene_features["area_z"]), 1e-6)

        start_x_norm = start_pos[0] / area_x
        start_y_norm = start_pos[1] / area_y
        start_z_norm = start_pos[2] / area_z
        end_x_norm = end_pos[0] / area_x
        end_y_norm = end_pos[1] / area_y
        end_z_norm = end_pos[2] / area_z

        delta = end_pos - start_pos
        manhattan_distance = float(np.sum(np.abs(delta)))
        euclidean_distance = float(np.linalg.norm(delta))
        distance_ratio = manhattan_distance / euclidean_distance if euclidean_distance > 1e-6 else 1.0

        direction = delta.copy()
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-6:
            direction = direction / direction_norm

        horizontal_dist = abs(delta[0]) + abs(delta[2])
        height_dist = abs(delta[1])
        total_distance = horizontal_dist + height_dist
        y_distance_ratio = height_dist / (total_distance + 1e-6)
        y_expected = end_y_norm - start_y_norm

        path_tortuosity = float(np.clip((distance_ratio - 1.0) / 0.73, 0.0, 1.0))
        vertical_complexity = float(np.clip(y_distance_ratio, 0.0, 1.0))
        distance_complexity = float(np.clip(manhattan_distance / 100.0, 0.0, 1.0))
        task_complexity = (
            0.35 * distance_complexity
            + 0.30 * vertical_complexity
            + 0.25 * path_tortuosity
            + 0.10 * float(scene_features.get("obstacle_density", 0.3))
        )

        return {
            "start_x_norm": float(start_x_norm),
            "start_y_norm": float(start_y_norm),
            "start_z_norm": float(start_z_norm),
            "end_x_norm": float(end_x_norm),
            "end_y_norm": float(end_y_norm),
            "end_z_norm": float(end_z_norm),
            "manhattan_distance": manhattan_distance,
            "euclidean_distance": euclidean_distance,
            "distance_ratio": float(distance_ratio),
            "direction_x": float(direction[0]),
            "direction_y": float(direction[1]),
            "direction_z": float(direction[2]),
            "y_expected": float(y_expected),
            "y_distance_ratio": float(y_distance_ratio),
            "path_tortuosity": path_tortuosity,
            "vertical_complexity": vertical_complexity,
            "task_complexity": float(np.clip(task_complexity, 0.0, 1.0)),
        }

    @staticmethod
    def build_state(
        config: dict,
        placed_devices: list,
        connection: dict,
        state_matrix: Optional[np.ndarray] = None,
        grid_info: Optional[dict] = None,
        obstacle_cells: Optional[int] = None,
        total_cells: Optional[int] = None,
        completed_connections: int = 0,
        total_connections: int = 1,
    ) -> ConnectionState:
        """Build a complete policy state.

        ``completed_connections`` and ``total_connections`` are retained for
        call-site compatibility but are intentionally not part of the state.
        The policy should not learn route-order artifacts.
        """
        _ = completed_connections, total_connections
        scene_features = StateExtractor.extract_scene_features(
            config,
            placed_devices,
            state_matrix,
            grid_info,
            obstacle_cells=obstacle_cells,
            total_cells=total_cells,
        )
        connection_features = StateExtractor.extract_connection_features(connection, scene_features)
        return ConnectionState(**scene_features, **connection_features)
