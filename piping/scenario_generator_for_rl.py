#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Procedural scenario generation for RL-based 3D pipe routing.

The generator samples process-equipment GLB assets, places them on a
multi-level industrial layout, rejects overlapping placements with an AABB
clearance test, and writes layout JSON files consumed by the DEACO routing
pipeline. It also supports reproducible train/validation/test splits and
K-fold splits for artifact evaluation.

Chinese equipment keywords are intentionally retained in a few classifier
tables because the bundled GLB assets are named in Chinese.
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


DEFAULT_PLATE_LEVELS = [0.0, 5.0, 9.0]


def _set_random_seed(seed: Optional[int]) -> None:
    """Seed all random generators used by this module."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    print(f"Using random seed: {seed}")


@dataclass
class DeviceInfo:
    """Static metadata for one equipment asset."""
    name: str
    glb_file: str
    height: float
    width: float  # Footprint width in the XZ plane.
    depth: float  # Footprint depth in the XZ plane.
    bounds: Optional[List[List[float]]]  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    num_ports: int = 0

    @property
    def footprint(self) -> float:
        """Return the horizontal footprint area."""
        return self.width * self.depth

    @property
    def volume(self) -> float:
        """Return the bounding-box volume."""
        return self.width * self.depth * self.height


@dataclass
class PlacedDevice:
    """Equipment asset after placement in a sampled scenario."""
    device_info: DeviceInfo
    position: Tuple[float, float, float]  # (x, y, z) center position.
    rotation_yaw: float  # Rotation around the Y axis in radians.
    layer_name: str
    row: int
    col: int

    def get_aabb(self) -> Tuple[List[float], List[float]]:
        """Return the axis-aligned bounding box after yaw rotation.

        Returns:
            tuple: ([min_x, min_y, min_z], [max_x, max_y, max_z])
        """
        cos_yaw = abs(math.cos(self.rotation_yaw))
        sin_yaw = abs(math.sin(self.rotation_yaw))

        rotated_width = self.device_info.width * cos_yaw + self.device_info.depth * sin_yaw
        rotated_depth = self.device_info.width * sin_yaw + self.device_info.depth * cos_yaw

        x, y, z = self.position
        w, h, d = rotated_width, self.device_info.height, rotated_depth

        min_point = [x - w/2, y, z - d/2]
        max_point = [x + w/2, y + h, z + d/2]

        return (min_point, max_point)


class ScenarioGenerator:
    """Generate reproducible equipment-layout scenarios for routing experiments."""

    def __init__(self,
                 glb_directory: str,
                 scene_bounds: Tuple[float, float, float] = (30.0, 20.0, 24.0),
                 plate_levels: Optional[List[float]] = None,
                 min_clearance: float = 0.5,
                 variable_scene_size: bool = False,
                 scene_size_ranges: Optional[Dict[str, Any]] = None):
        """Initialize the scenario generator.

        Args:
            glb_directory: Directory containing GLB equipment assets.
            scene_bounds: Fixed scene extent (x_max, y_max, z_max) used when
                variable_scene_size is disabled.
            plate_levels: Platform elevations used when variable_scene_size is
                disabled.
            min_clearance: Minimum AABB clearance in meters.
            variable_scene_size: Whether to sample scene dimensions per
                scenario.
            scene_size_ranges: Optional sampling ranges:
                {
                    'x': (min, max),
                    'y': (min, max),
                    'z': (min, max),
                    'L2_heights': [5.0, 6.0],
                    'L3_heights': [9.0, 12.0],
                }
        """
        self.glb_directory = Path(glb_directory)
        self.scene_bounds = scene_bounds
        self.plate_levels = list(plate_levels or DEFAULT_PLATE_LEVELS)
        self.min_clearance = min_clearance
        self.variable_scene_size = variable_scene_size

        # Default scene-size ranges used for domain-randomized scenarios.
        if scene_size_ranges is None:
            self.scene_size_ranges = {
                'x': (30.0, 60.0),
                'y': (20.0, 30.0),
                'z': (24.0, 48.0),
                'L2_heights': [5.0, 6.0],
                'L3_heights': [9.0, 12.0]
            }
        else:
            self.scene_size_ranges = scene_size_ranges

        self.device_library = self._load_device_library()
        if not self.device_library:
            raise ValueError(f"No usable equipment assets found in {self.glb_directory}")
        print(f"Loaded {len(self.device_library)} equipment assets.")

        # Keep this grid convention aligned with the downstream layout schema.
        self.grid_config = {
            'rows': 4,
            'cols': 6,
            'aisle_rows': [2],
            'aisle_cols': []
        }

        # Placement preferences inferred from the current asset naming scheme.
        self.layer_preferences = {
            'L1': [
                'reactor',
                'tower',
                'column',
                'drum',
                'vessel',
                'oxygenation',
                'hydration',
                'recovery',
                'deacetaldehyde',
            ],
            'L2': [
                'exchanger',
                'heat_exchanger',
                'cooler',
                'condenser',
                'heater',
                'reboiler',
            ],
            'L3': [
                'pump',
                'compressor',
                'fan',
                'blower',
                'filter',
                'valve',
            ]
        }

    def _get_preferred_layer(self, device_name: str) -> Optional[str]:
        """Return the preferred platform layer inferred from the asset name.

        Args:
            device_name: Equipment asset name.

        Returns:
            Preferred layer name (L1/L2/L3), or None if no rule matches.
        """
        normalized_name = device_name.lower()
        for layer, keywords in self.layer_preferences.items():
            for keyword in keywords:
                if keyword in normalized_name:
                    return layer
        return None

    def _get_device_type_category(self, device_name: str) -> str:
        """Classify equipment size/type for duplicate sampling.

        Args:
            device_name: Equipment asset name.

        Returns:
            Equipment class: ``small``, ``medium``, or ``large``.
        """
        normalized_name = device_name.lower()
        small_keywords = ['pump', 'fan', 'blower', 'filter', 'valve', 'compressor']
        medium_keywords = ['exchanger', 'cooler', 'condenser', 'heater', 'evaporator', 'reboiler']
        large_keywords = ['reactor', 'tower', 'column', 'tank', 'vessel', 'separator', 'drum']

        for keyword in small_keywords:
            if keyword in normalized_name:
                return 'small'

        for keyword in medium_keywords:
            if keyword in normalized_name:
                return 'medium'

        for keyword in large_keywords:
            if keyword in normalized_name:
                return 'large'

        return 'medium'

    def _select_devices_with_duplicates(self, num_devices: int) -> List[DeviceInfo]:
        """Select equipment while allowing realistic duplicate types.

        The proportions are a pragmatic prior for synthetic process layouts:
        large units are less duplicated, while pumps and auxiliary equipment
        are commonly repeated.

        Args:
            num_devices: Target number of equipment instances.

        Returns:
            Selected equipment list, possibly containing repeated assets.
        """
        selected_devices = []

        device_by_category = {
            'small': [],
            'medium': [],
            'large': []
        }

        for device in self.device_library:
            category = self._get_device_type_category(device.name)
            device_by_category[category].append(device)

        category_config = {
            'large': {
                'priority': 0.4,
                'count_range': (1, 2),
                'duplicate_prob': 0.3
            },
            'medium': {
                'priority': 0.35,
                'count_range': (1, 3),
                'duplicate_prob': 0.5
            },
            'small': {
                'priority': 0.25,
                'count_range': (1, 4),
                'duplicate_prob': 0.7
            }
        }

        target_counts = {
            'large': int(num_devices * category_config['large']['priority']),
            'medium': int(num_devices * category_config['medium']['priority']),
            'small': int(num_devices * category_config['small']['priority'])
        }

        total = sum(target_counts.values())
        if total < num_devices:
            target_counts['medium'] += (num_devices - total)

        for category in ['large', 'medium', 'small']:
            target = target_counts[category]
            available = device_by_category[category]

            if not available or target <= 0:
                continue

            config = category_config[category]

            num_types = max(1, min(len(available), target // 2))
            selected_types = random.sample(available, num_types)

            for device_type in selected_types:
                if target <= 0:
                    break

                if random.random() < config['duplicate_prob']:
                    count = random.randint(config['count_range'][0],
                                          min(config['count_range'][1], target))
                else:
                    count = 1

                for _ in range(count):
                    selected_devices.append(device_type)
                    target -= 1

        while len(selected_devices) < num_devices:
            device = random.choice(self.device_library)
            selected_devices.append(device)

        while len(selected_devices) > num_devices:
            selected_devices.pop(random.randint(0, len(selected_devices) - 1))

        random.shuffle(selected_devices)

        type_counts = {}
        for device in selected_devices:
            type_counts[device.name] = type_counts.get(device.name, 0) + 1

        print(f"\nEquipment composition ({len(selected_devices)} instances):")
        duplicates = []
        for device_name, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                duplicates.append(f"{device_name} x{count}")

        if duplicates:
            print(f"   Repeated assets: {', '.join(duplicates)}")
        print(f"   Unique asset types: {len(type_counts)}")

        return selected_devices

    def _load_device_library(self) -> List[DeviceInfo]:
        """Load asset metadata from the GLB directory."""
        model_size_file = self.glb_directory / "model_size_summary.json"

        if not model_size_file.exists():
            raise FileNotFoundError(f"Missing model size summary: {model_size_file}")

        with open(model_size_file, 'r', encoding='utf-8') as f:
            size_data = json.load(f)

        device_library = []

        for item in size_data:
            glb_file = item['glb_file']

            # Skip assets that are explicitly marked as missing ports.
            if 'missing' in glb_file.lower():
                continue

            info_file = self.glb_directory / glb_file.replace('.glb', '_info.json')
            num_ports = 0
            bounds = None

            if info_file.exists():
                try:
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info_data = json.load(f)
                        num_ports = len(info_data.get('ports', []))
                        bounds = info_data.get('geometry', {}).get('bounds', None)
                except Exception as e:
                    print(f"Warning: failed to read {info_file}: {e}")

            if num_ports > 0:
                device_info = DeviceInfo(
                    name=glb_file.replace('.glb', ''),
                    glb_file=glb_file,
                    height=item['height'],
                    width=item['xz_width'],
                    depth=item['xz_depth'],
                    bounds=bounds,
                    num_ports=num_ports
                )
                device_library.append(device_info)

        return device_library

    def _check_collision(self, device1: PlacedDevice, device2: PlacedDevice) -> bool:
        """Check whether two placed devices violate the clearance constraint.

        Args:
            device1: First placed device.
            device2: Second placed device.

        Returns:
            True if the AABBs overlap or are closer than ``min_clearance``.
        """
        aabb1_min, aabb1_max = device1.get_aabb()
        aabb2_min, aabb2_max = device2.get_aabb()

        margin = self.min_clearance

        for i in range(3):
            if aabb1_max[i] + margin <= aabb2_min[i]:
                return False
            if aabb2_max[i] + margin <= aabb1_min[i]:
                return False

        return True

    def _is_valid_position(self,
                          device: DeviceInfo,
                          position: Tuple[float, float, float],
                          rotation_yaw: float,
                          layer_name: str,
                          row: int,
                          col: int,
                          placed_devices: List[PlacedDevice],
                          scene_bounds: Optional[Tuple[float, float, float]] = None) -> bool:
        """Return whether a candidate placement is inside bounds and collision-free.

        Args:
            device: Equipment metadata.
            position: Candidate center position.
            rotation_yaw: Candidate yaw rotation in radians.
            layer_name: Platform layer name.
            row: Grid row.
            col: Grid column.
            placed_devices: Devices already accepted in the scenario.
            scene_bounds: Scene extent. Defaults to ``self.scene_bounds``.

        Returns:
            True if the placement is valid.
        """
        if scene_bounds is None:
            scene_bounds = self.scene_bounds

        temp_device = PlacedDevice(
            device_info=device,
            position=position,
            rotation_yaw=rotation_yaw,
            layer_name=layer_name,
            row=row,
            col=col
        )

        aabb_min, aabb_max = temp_device.get_aabb()
        if (aabb_min[0] < 0 or aabb_max[0] > scene_bounds[0] or
            aabb_min[2] < 0 or aabb_max[2] > scene_bounds[2]):
            return False

        for placed in placed_devices:
            if self._check_collision(temp_device, placed):
                return False

        return True

    def _get_cell_center(
        self,
        row: int,
        col: int,
        scene_bounds: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float]:
        """Return the center of a layout grid cell in the XZ plane.

        Args:
            row: Grid row.
            col: Grid column.
            scene_bounds: Scene extent. Defaults to ``self.scene_bounds``.

        Returns:
            ``(x, z)`` cell-center coordinates.
        """
        if scene_bounds is None:
            scene_bounds = self.scene_bounds

        cell_width = scene_bounds[0] / self.grid_config['cols']
        cell_depth = scene_bounds[2] / self.grid_config['rows']

        x = (col + 0.5) * cell_width
        z = (row + 0.5) * cell_depth

        return (x, z)

    def _generate_random_scene_config(self) -> Dict[str, Any]:
        """Sample scene dimensions and platform elevations.

        Returns:
            dict: {
                'scene_bounds': (x, y, z),
                'plate_levels': [L1, L2, L3]
            }
        """
        if not self.variable_scene_size:
            return {
                'scene_bounds': self.scene_bounds,
                'plate_levels': self.plate_levels
            }

        x_size = random.uniform(self.scene_size_ranges['x'][0], self.scene_size_ranges['x'][1])
        y_size = random.uniform(self.scene_size_ranges['y'][0], self.scene_size_ranges['y'][1])
        z_size = random.uniform(self.scene_size_ranges['z'][0], self.scene_size_ranges['z'][1])

        L2_height = random.choice(self.scene_size_ranges['L2_heights'])
        L3_height = random.choice(self.scene_size_ranges['L3_heights'])

        return {
            'scene_bounds': (round(x_size, 1), round(y_size, 1), round(z_size, 1)),
            'plate_levels': [0.0, L2_height, L3_height]
        }

    def generate_scenario(self,
                         num_devices: int = 10,
                         strategy: str = 'random',
                         allow_rotation: bool = True,
                         allow_duplicates: bool = True,
                         custom_scene_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate one scenario layout.

        Args:
            num_devices: Number of equipment instances. Values between 5 and
                15 usually produce practical layouts for the default scene.
            strategy: Placement ordering strategy: ``random``,
                ``layer_based``, or ``functional``.
            allow_rotation: Whether to sample 90-degree yaw rotations.
            allow_duplicates: Whether repeated equipment types are allowed.
            custom_scene_config: Optional fixed scene configuration. If None,
                the generator samples according to ``variable_scene_size``.

        Returns:
            Layout dictionary compatible with the downstream routing pipeline.
        """
        if custom_scene_config is None:
            scene_config = self._generate_random_scene_config()
        else:
            scene_config = custom_scene_config

        current_scene_bounds = scene_config['scene_bounds']
        current_plate_levels = scene_config['plate_levels']

        if self.variable_scene_size:
            print(
                f"\nScene size: X={current_scene_bounds[0]}m, "
                f"Y={current_scene_bounds[1]}m, Z={current_scene_bounds[2]}m"
            )
            print(
                "   Platform heights: "
                f"L1={current_plate_levels[0]}m, "
                f"L2={current_plate_levels[1]}m, "
                f"L3={current_plate_levels[2]}m"
            )

        if num_devices == -1:
            num_devices = random.randint(5, 12)

        num_devices = max(num_devices, 5)

        if allow_duplicates:
            selected_devices = self._select_devices_with_duplicates(num_devices)
        else:
            num_devices = min(num_devices, len(self.device_library))
            selected_devices = random.sample(self.device_library, num_devices)

        if strategy == 'layer_based':
            selected_devices.sort(key=lambda d: d.volume, reverse=True)
        elif strategy == 'functional':
            def device_type_priority(device):
                """Priority order for functional grouping by asset name."""
                name = device.name.lower()
                if 'tower' in name or 'column' in name:
                    return 0
                elif 'reactor' in name:
                    return 1
                elif 'tank' in name or 'vessel' in name or 'drum' in name:
                    return 2
                elif 'cooler' in name or 'condenser' in name or 'exchanger' in name:
                    return 3
                elif 'pump' in name:
                    return 4
                else:
                    return 5
            selected_devices.sort(key=device_type_priority)
        else:
            # Smaller assets are placed first to improve acceptance rate.
            selected_devices.sort(key=lambda d: d.volume)

        placed_devices = []
        device_configs = []

        device_id = 1
        for device in selected_devices:
            placed = False

            preferred_layer = self._get_preferred_layer(device.name)

            available_positions = []

            all_positions = []
            for row in range(self.grid_config['rows']):
                for col in range(self.grid_config['cols']):
                    if row in self.grid_config['aisle_rows'] or col in self.grid_config['aisle_cols']:
                        continue
                    for layer_idx, layer_name in enumerate(['L1', 'L2', 'L3']):
                        all_positions.append((layer_name, layer_idx, row, col))

            if preferred_layer:
                preferred_positions = [pos for pos in all_positions if pos[0] == preferred_layer]
                other_positions = [pos for pos in all_positions if pos[0] != preferred_layer]
                random.shuffle(preferred_positions)
                random.shuffle(other_positions)
                available_positions = preferred_positions + other_positions

                if preferred_positions:
                    print(f"   Preferred layer for '{device.name}': {preferred_layer}")
            else:
                available_positions = all_positions
                random.shuffle(available_positions)

            for layer_name, layer_idx, row, col in available_positions:
                x_cell, z_cell = self._get_cell_center(row, col, current_scene_bounds)
                y = current_plate_levels[layer_idx]

                cell_width = current_scene_bounds[0] / self.grid_config['cols']
                cell_depth = current_scene_bounds[2] / self.grid_config['rows']
                offset_range = 0.2

                x_offset = random.uniform(-cell_width * offset_range, cell_width * offset_range)
                z_offset = random.uniform(-cell_depth * offset_range, cell_depth * offset_range)

                position = (x_cell + x_offset, y, z_cell + z_offset)

                if allow_rotation:
                    rotation_yaw = random.choice([0, math.pi/2, math.pi, -math.pi/2])
                else:
                    rotation_yaw = 0

                if self._is_valid_position(device, position, rotation_yaw,
                                          layer_name, row, col, placed_devices,
                                          current_scene_bounds):
                    placed_device = PlacedDevice(
                        device_info=device,
                        position=position,
                        rotation_yaw=rotation_yaw,
                        layer_name=layer_name,
                        row=row,
                        col=col
                    )
                    placed_devices.append(placed_device)

                    device_config = {
                        'id': device_id,
                        'name': device.name,
                        'source_glb': device.glb_file,
                        'layer_name': layer_name,
                        'row': row,
                        'col': col,
                        'pose': {
                            'x': round(position[0], 2),
                            'y': round(position[1], 2),
                            'z': round(position[2], 2),
                            'yaw': round(rotation_yaw, 4),
                            'pitch': 0.0
                        }
                    }
                    device_configs.append(device_config)

                    device_id += 1
                    placed = True
                    break

            if not placed:
                preferred_info = f" (preferred layer: {preferred_layer})" if preferred_layer else ""
                print(f"Warning: failed to place {device.name}{preferred_info}; tried all candidate cells.")

        scenario = {
            'scene': {
                'area': {
                    'x': current_scene_bounds[0],
                    'y': current_scene_bounds[1],
                    'z': current_scene_bounds[2]
                },
                'level_ref': 'top',
                'piping': {
                    'pipe_radius': 0.05,
                    'collision_check_enabled': True,
                    'collision_buffer': 0.02
                },
                'plate_levels': current_plate_levels,
                'plate_thicknesses': [0.0, 0.2, 0.2],
                'layers': [
                    {
                        'name': f'L{i+1}',
                        'rows': self.grid_config['rows'],
                        'cols': self.grid_config['cols'],
                        'size_x': current_scene_bounds[0],
                        'size_z': current_scene_bounds[2],
                        'aisle_rows': self.grid_config['aisle_rows'],
                        'aisle_cols': self.grid_config['aisle_cols']
                    }
                    for i in range(3)
                ]
            },
            'devices': device_configs
        }

        layer_stats = {'L1': 0, 'L2': 0, 'L3': 0}
        for config in device_configs:
            layer_stats[config['layer_name']] += 1

        placed_count = len(device_configs)
        placement_rate = (placed_count / num_devices * 100) if num_devices > 0 else 0

        print(f"\nScenario generation completed:")
        print(f"   Placed devices: {placed_count}/{num_devices} ({placement_rate:.1f}%)")
        print(f"   Layer distribution: L1={layer_stats['L1']} | L2={layer_stats['L2']} | L3={layer_stats['L3']}")

        if placement_rate < 80:
            print(f"   Warning: low placement rate ({placement_rate:.1f}%).")
            print("      Consider fewer devices, larger scene bounds, or smaller min_clearance.")

        return scenario

    def generate_batch(self,
                      num_scenarios: int = 100,
                      output_dir: str = 'scenarios',
                      num_devices_range: Tuple[int, int] = (6, 12),
                      strategies: Optional[List[str]] = None) -> List[str]:
        """Generate a flat batch of scenario JSON files.

        Args:
            num_scenarios: Number of scenarios to generate.
            output_dir: Output directory.
            num_devices_range: Inclusive range for sampled device counts.
            strategies: Optional placement-ordering strategies.

        Returns:
            Paths to generated JSON files.
        """
        if strategies is None:
            strategies = ['random', 'layer_based', 'functional']

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated_files = []
        start_time = time.time()

        print("\nBatch scenario generation started.")
        print(f"   Target scenarios: {num_scenarios}")
        print(f"   Device count range: {num_devices_range[0]}-{num_devices_range[1]}")
        print(f"   Output directory: {output_path}")

        for i in range(num_scenarios):
            strategy = random.choice(strategies)
            num_devices = random.randint(num_devices_range[0], num_devices_range[1])

            try:
                scenario = self.generate_scenario(
                    num_devices=num_devices,
                    strategy=strategy,
                    allow_rotation=True
                )

                filename = f"layout_{i:04d}.json"
                filepath = output_path / filename

                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(scenario, f, indent=2, ensure_ascii=False)

                generated_files.append(str(filepath))

                if (i + 1) % 10 == 0 or i == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / (i + 1)
                    remaining = avg_time * (num_scenarios - i - 1)
                    print(f"   Progress: {i+1}/{num_scenarios} "
                          f"({(i+1)/num_scenarios*100:.1f}%) | "
                          f"avg: {avg_time:.2f}s/scenario | "
                          f"eta: {remaining/60:.1f}min")

            except Exception as e:
                print(f"Error: failed to generate scenario {i}: {e}")
                continue

        total_time = time.time() - start_time
        print("\nBatch scenario generation completed.")
        print(f"   Succeeded: {len(generated_files)}/{num_scenarios}")
        print(f"   Total time: {total_time/60:.1f}min")
        if generated_files:
            print(f"   Average: {total_time/len(generated_files):.2f}s/scenario")

        return generated_files

    def generate_rl_dataset_kfold(self,
                                 num_scenarios: int = 1000,
                                 output_dir: str = 'scenarios_rl_dataset_kfold',
                                 k_folds: int = 5,
                                 test_ratio: float = 0.2,
                                 random_seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate a K-fold scenario dataset with a held-out test split.

        Each fold can be used once as validation while the remaining folds are
        used for training. The held-out test split is intended for final
        evaluation only.

        Args:
            num_scenarios: Total number of scenarios.
            output_dir: Output root directory.
            k_folds: Number of folds.
            test_ratio: Held-out test split ratio.
            random_seed: Optional seed for reproducibility.

        Returns:
            Dictionary with this shape:
            {
                'folds': [fold1, fold2, ..., foldK],
                'test': {...},
                'fold_info': {...}
            }
        """
        if k_folds < 2:
            raise ValueError(f"k_folds must be >= 2, got {k_folds}")
        if not 0.0 <= test_ratio < 1.0:
            raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")

        _set_random_seed(random_seed)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_test = int(num_scenarios * test_ratio)
        num_kfold = num_scenarios - num_test
        base_fold_size = num_kfold // k_folds
        fold_remainder = num_kfold % k_folds
        fold_counts = [
            base_fold_size + (1 if fold_idx < fold_remainder else 0)
            for fold_idx in range(k_folds)
        ]

        print(f"\n{k_folds}-fold dataset split:")
        print(f"   Total scenarios: {num_scenarios}")
        print(f"   Test split: {num_test} ({test_ratio*100:.0f}%)")
        print(f"   K-fold pool: {num_kfold} ({(1-test_ratio)*100:.0f}%)")
        print(
            f"   Base fold size: {base_fold_size}"
            + (f" (+{fold_remainder} distributed remainder)" if fold_remainder else "")
        )
        print(f"   Output directory: {output_path}")

        result = {
            'folds': [],
            'test': None,
            'fold_info': {}
        }

        print("\n" + "="*70)
        print("Generating held-out test split")
        print("="*70)
        if random_seed is not None:
            _set_random_seed(random_seed + 9999)
        result['test'] = self.generate_diverse_batch(
            num_scenarios=num_test,
            output_dir=str(output_path / 'test')
        )

        for fold_idx in range(k_folds):
            print("\n" + "="*70)
            print(f"Generating fold {fold_idx + 1}/{k_folds}")
            print("="*70)

            if random_seed is not None:
                _set_random_seed(random_seed + fold_idx)

            fold_data = self.generate_diverse_batch(
                num_scenarios=fold_counts[fold_idx],
                output_dir=str(output_path / f'fold_{fold_idx + 1}')
            )

            result['folds'].append(fold_data)

        fold_info = {
            'k_folds': k_folds,
            'total_scenarios': num_scenarios,
            'test_set': {
                'count': num_test,
                'ratio': test_ratio,
                'usage': 'Final evaluation only, never use during training',
                'simple': len(result['test']['simple']),
                'medium': len(result['test']['medium']),
                'complex': len(result['test']['complex'])
            },
            'folds': []
        }

        for fold_idx in range(k_folds):
            fold_data = result['folds'][fold_idx]
            fold_info['folds'].append({
                'fold_id': fold_idx + 1,
                'count': fold_counts[fold_idx],
                'simple': len(fold_data['simple']),
                'medium': len(fold_data['medium']),
                'complex': len(fold_data['complex']),
                'usage': f'Can be used as validation set while other {k_folds-1} folds as training set'
            })

        result['fold_info'] = fold_info

        index_file = output_path / 'dataset_index_kfold.json'
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(fold_info, f, indent=2, ensure_ascii=False)

        training_guide = {
            'description': f'{k_folds}-Fold Cross Validation Training Guide',
            'total_training_rounds': k_folds,
            'training_rounds': []
        }

        for val_fold in range(k_folds):
            train_folds = [i+1 for i in range(k_folds) if i != val_fold]
            training_guide['training_rounds'].append({
                'round': val_fold + 1,
                'validation_fold': val_fold + 1,
                'training_folds': train_folds,
                'description': f'Use fold_{val_fold+1} as validation, folds {train_folds} as training'
            })

        training_guide['final_test'] = {
            'description': 'After all K rounds, use test set for final evaluation',
            'test_set_path': 'test/'
        }

        guide_file = output_path / 'training_guide_kfold.json'
        with open(guide_file, 'w', encoding='utf-8') as f:
            json.dump(training_guide, f, indent=2, ensure_ascii=False)

        print("\n" + "="*70)
        print(f"{k_folds}-fold dataset generation completed.")
        print("="*70)
        print("\nDataset structure:")
        print(f"   {output_path}/")
        print(f"   ├── fold_1/         # fold 1 ({fold_counts[0]} scenarios)")
        print(f"   │   ├── simple/")
        print(f"   │   ├── medium/")
        print(f"   │   └── complex/")
        for i in range(2, k_folds + 1):
            print(f"   ├── fold_{i}/{'         ' if i < 10 else '        '}# fold {i} ({fold_counts[i-1]} scenarios)")
            print(f"   │   ├── simple/")
            print(f"   │   ├── medium/")
            print(f"   │   └── complex/")
        print(f"   ├── test/           # held-out test split ({num_test} scenarios)")
        print(f"   │   ├── simple/")
        print(f"   │   ├── medium/")
        print(f"   │   └── complex/")
        print("   ├── dataset_index_kfold.json")
        print("   └── training_guide_kfold.json")

        print(f"\n{k_folds}-fold training protocol:")
        for i in range(k_folds):
            val_fold = i + 1
            train_folds = [j+1 for j in range(k_folds) if j != i]
            print(f"   Round {i+1}: validation=fold_{val_fold}, training=folds {train_folds}")
        print("   Final step: evaluate once on test/")

        print("\nRecommended reporting:")
        print(f"   1. Train {k_folds} rounds with a different validation fold each round.")
        print("   2. Report validation mean and standard deviation across folds.")
        print("   3. Use test/ only once for final evaluation.")

        print("\nMetadata files:")
        print(f"   Index: {index_file}")
        print(f"   Training guide: {guide_file}")

        return result

    def generate_rl_dataset(self,
                           num_scenarios: int = 1000,
                           output_dir: str = 'scenarios_rl_dataset',
                           train_ratio: float = 0.8,
                           val_ratio: float = 0.1,
                           test_ratio: float = 0.1,
                           random_seed: Optional[int] = None) -> Dict[str, Dict]:
        """Generate train/validation/test splits for RL experiments.

        Each split is internally stratified into simple, medium, and complex
        folders by device-count range.

        Args:
            num_scenarios: Total number of scenarios.
            output_dir: Output root directory.
            train_ratio: Training split ratio.
            val_ratio: Validation split ratio.
            test_ratio: Test split ratio.
            random_seed: Optional seed for reproducibility.

        Returns:
            ``{'train': {...}, 'val': {...}, 'test': {...}}``.
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        if min(train_ratio, val_ratio, test_ratio) < 0.0:
            raise ValueError("Split ratios must be non-negative.")

        _set_random_seed(random_seed)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_train = int(num_scenarios * train_ratio)
        num_val = int(num_scenarios * val_ratio)
        num_test = num_scenarios - num_train - num_val

        print("\nRL dataset split:")
        print(f"   Total scenarios: {num_scenarios}")
        print(f"   Train: {num_train} ({train_ratio*100:.0f}%)")
        print(f"   Validation: {num_val} ({val_ratio*100:.0f}%)")
        print(f"   Test: {num_test} ({test_ratio*100:.0f}%)")
        print(f"   Output directory: {output_path}")

        result = {}

        print("\n" + "="*70)
        print("Generating training split")
        print("="*70)
        result['train'] = self.generate_diverse_batch(
            num_scenarios=num_train,
            output_dir=str(output_path / 'train')
        )

        print("\n" + "="*70)
        print("Generating validation split")
        print("="*70)
        if random_seed is not None:
            _set_random_seed(random_seed + 1)
        result['val'] = self.generate_diverse_batch(
            num_scenarios=num_val,
            output_dir=str(output_path / 'val')
        )

        print("\n" + "="*70)
        print("Generating test split")
        print("="*70)
        if random_seed is not None:
            _set_random_seed(random_seed + 2)
        result['test'] = self.generate_diverse_batch(
            num_scenarios=num_test,
            output_dir=str(output_path / 'test')
        )

        dataset_index = {
            'total_scenarios': num_scenarios,
            'random_seed': random_seed,
            'split': {
                'train': {
                    'count': num_train,
                    'ratio': train_ratio,
                    'simple': len(result['train']['simple']),
                    'medium': len(result['train']['medium']),
                    'complex': len(result['train']['complex'])
                },
                'val': {
                    'count': num_val,
                    'ratio': val_ratio,
                    'simple': len(result['val']['simple']),
                    'medium': len(result['val']['medium']),
                    'complex': len(result['val']['complex'])
                },
                'test': {
                    'count': num_test,
                    'ratio': test_ratio,
                    'simple': len(result['test']['simple']),
                    'medium': len(result['test']['medium']),
                    'complex': len(result['test']['complex'])
                }
            },
            'usage_guidelines': {
                'train': 'Use for RL agent training (trial and error)',
                'val': 'Use periodically during training to monitor progress and prevent overfitting',
                'test': 'Use ONLY ONCE at the end for final evaluation'
            }
        }

        index_file = output_path / 'dataset_index.json'
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_index, f, indent=2, ensure_ascii=False)

        print("\n" + "="*70)
        print("RL dataset generation completed.")
        print("="*70)
        print("\nDataset structure:")
        print(f"   {output_path}/")
        print(f"   ├── train/          # training split ({num_train} scenarios)")
        print("   │   ├── simple/")
        print("   │   ├── medium/")
        print("   │   └── complex/")
        print(f"   ├── val/            # validation split ({num_val} scenarios)")
        print(f"   │   ├── simple/")
        print(f"   │   ├── medium/")
        print(f"   │   └── complex/")
        print(f"   ├── test/           # test split ({num_test} scenarios)")
        print(f"   │   ├── simple/")
        print(f"   │   ├── medium/")
        print(f"   │   └── complex/")
        print("   └── dataset_index.json")

        print("\nUsage protocol:")
        print("   Train on train/, tune or monitor with val/, and reserve test/ for final evaluation.")

        print(f"\nIndex file: {index_file}")

        return result

    def generate_diverse_batch(self,
                              num_scenarios: int = 1000,
                              output_dir: str = 'scenarios_diverse') -> Dict[str, List[str]]:
        """Generate simple, medium, and complex scenario folders.

        The split uses a fixed difficulty prior: 30% simple, 50% medium, and
        20% complex scenarios.

        Args:
            num_scenarios: Total number of scenarios.
            output_dir: Output directory.

        Returns:
            ``{'simple': [...], 'medium': [...], 'complex': [...]}``.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        num_simple = int(num_scenarios * 0.3)
        num_medium = int(num_scenarios * 0.5)
        num_complex = num_scenarios - num_simple - num_medium

        result = {
            'simple': [],
            'medium': [],
            'complex': []
        }

        print("\nGenerating difficulty-stratified scenario batch.")
        print(f"   Simple: {num_simple} scenarios (5-8 devices)")
        print(f"   Medium: {num_medium} scenarios (8-12 devices)")
        print(f"   Complex: {num_complex} scenarios (12-18 devices)")

        print("\nGenerating simple scenarios...")
        result['simple'] = self.generate_batch(
            num_scenarios=num_simple,
            output_dir=str(output_path / 'simple'),
            num_devices_range=(5, 8),
            strategies=['random']
        )

        print("\nGenerating medium scenarios...")
        result['medium'] = self.generate_batch(
            num_scenarios=num_medium,
            output_dir=str(output_path / 'medium'),
            num_devices_range=(8, 12),
            strategies=['random', 'layer_based']
        )

        print("\nGenerating complex scenarios...")
        result['complex'] = self.generate_batch(
            num_scenarios=num_complex,
            output_dir=str(output_path / 'complex'),
            num_devices_range=(12, 18),
            strategies=['random', 'layer_based', 'functional']
        )

        index = {
            'total_scenarios': num_scenarios,
            'simple': {
                'count': len(result['simple']),
                'device_range': [5, 8],
                'files': [str(Path(f).name) for f in result['simple']]
            },
            'medium': {
                'count': len(result['medium']),
                'device_range': [8, 12],
                'files': [str(Path(f).name) for f in result['medium']]
            },
            'complex': {
                'count': len(result['complex']),
                'device_range': [12, 18],
                'files': [str(Path(f).name) for f in result['complex']]
            }
        }

        index_file = output_path / 'scenario_index.json'
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        print("\nDifficulty-stratified scenario generation completed.")
        print(f"   Index file: {index_file}")

        return result

    def visualize_scenario(self, scenario: Dict, output_file: str = 'scenario_preview.glb') -> bool:
        """Export a GLB preview of the placed equipment only.

        Args:
            scenario: Layout dictionary.
            output_file: Output GLB file path.

        Returns:
            True if export succeeds.
        """
        try:
            import trimesh
        except ImportError as exc:
            raise ImportError(
                "GLB preview export requires the optional 'trimesh' dependency. "
                "Install the project requirements before using visualization."
            ) from exc

        print(f"\nGenerating scenario preview: {output_file}")

        try:
            scene = trimesh.Scene()

            scene_config = scenario['scene']
            devices = scenario['devices']

            loaded_count = 0
            failed_count = 0

            for device_config in devices:
                device_name = device_config['name']
                glb_file = device_config['source_glb']
                glb_path = self.glb_directory / glb_file

                try:
                    device_mesh = trimesh.load(str(glb_path), force='mesh')
                    if isinstance(device_mesh, trimesh.Scene):
                        meshes = []
                        for geometry in device_mesh.geometry.values():
                            if isinstance(geometry, trimesh.Trimesh):
                                meshes.append(geometry)
                        if meshes:
                            device_mesh = trimesh.util.concatenate(meshes)
                        else:
                            failed_count += 1
                            continue
                except Exception as e:
                    print(f"   Warning: failed to load {device_name}: {e}")
                    failed_count += 1
                    continue

                pose = device_config['pose']
                position = [pose['x'], pose['y'], pose['z']]
                yaw = pose.get('yaw', 0.0)
                pitch = pose.get('pitch', 0.0)
                roll = pose.get('roll', 0.0)

                transform_matrix = np.eye(4)

                transform_matrix[:3, 3] = position

                if yaw != 0.0 or pitch != 0.0 or roll != 0.0:
                    cos_yaw = np.cos(yaw)
                    sin_yaw = np.sin(yaw)
                    cos_pitch = np.cos(pitch)
                    sin_pitch = np.sin(pitch)
                    cos_roll = np.cos(roll)
                    sin_roll = np.sin(roll)

                    Ry = np.array([
                        [cos_yaw, 0, sin_yaw],
                        [0, 1, 0],
                        [-sin_yaw, 0, cos_yaw]
                    ])

                    Rx = np.array([
                        [1, 0, 0],
                        [0, cos_pitch, -sin_pitch],
                        [0, sin_pitch, cos_pitch]
                    ])

                    Rz = np.array([
                        [cos_roll, -sin_roll, 0],
                        [sin_roll, cos_roll, 0],
                        [0, 0, 1]
                    ])

                    rotation_matrix = Rz @ Rx @ Ry
                    transform_matrix[:3, :3] = rotation_matrix

                device_mesh_copy = device_mesh.copy()
                device_mesh_copy.apply_transform(transform_matrix)

                layer_name = device_config['layer_name']
                if layer_name == 'L1':
                    color = [100, 150, 255, 255]
                elif layer_name == 'L2':
                    color = [100, 255, 150, 255]
                else:  # L3
                    color = [255, 150, 100, 255]

                device_mesh_copy.visual.face_colors = color

                scene.add_geometry(device_mesh_copy, node_name=f"device_{device_config['id']}_{device_name}")
                loaded_count += 1

            area = scene_config['area']
            ground_size = [area['x'], 0.2, area['z']]
            ground_center = [area['x']/2, -0.1, area['z']/2]

            ground = trimesh.creation.box(
                extents=ground_size,
                transform=trimesh.transformations.translation_matrix(ground_center)
            )
            ground.visual.face_colors = [128, 128, 128, 255]
            scene.add_geometry(ground, node_name="ground")

            axes = trimesh.creation.axis(origin_size=0.3, axis_length=5.0)
            scene.add_geometry(axes, node_name="axes")

            scene.export(output_file)

            print("Visualization export completed.")
            print(f"   Loaded devices: {loaded_count}/{len(devices)}")
            print(f"   Failed devices: {failed_count}/{len(devices)}")
            print(f"   File: {output_file}")

            return True

        except Exception as e:
            print(f"Visualization export failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def _build_generator(args) -> ScenarioGenerator:
    """Create a scenario generator from parsed CLI arguments."""
    return ScenarioGenerator(
        glb_directory=str(args.glb_directory),
        scene_bounds=(args.scene_x, args.scene_y, args.scene_z),
        plate_levels=[args.l1_height, args.l2_height, args.l3_height],
        min_clearance=args.min_clearance,
        variable_scene_size=args.variable_scene_size,
        scene_size_ranges={
            'x': (args.scene_x_min, args.scene_x_max),
            'y': (args.scene_y_min, args.scene_y_max),
            'z': (args.scene_z_min, args.scene_z_max),
            'L2_heights': args.l2_height_choices,
            'L3_heights': args.l3_height_choices,
        },
    )


def _add_common_generator_args(parser: argparse.ArgumentParser) -> None:
    """Add shared generator arguments to a subcommand parser."""
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--glb-directory",
        type=Path,
        default=repo_root / "static" / "glb",
        help="GLB asset directory containing model_size_summary.json.",
    )
    parser.add_argument("--scene-x", type=float, default=30.0)
    parser.add_argument("--scene-y", type=float, default=20.0)
    parser.add_argument("--scene-z", type=float, default=24.0)
    parser.add_argument("--l1-height", type=float, default=0.0)
    parser.add_argument("--l2-height", type=float, default=5.0)
    parser.add_argument("--l3-height", type=float, default=9.0)
    parser.add_argument("--min-clearance", type=float, default=0.5)
    parser.add_argument(
        "--variable-scene-size",
        action="store_true",
        help="Sample scene size and platform levels from configured ranges.",
    )
    parser.add_argument("--scene-x-min", type=float, default=30.0)
    parser.add_argument("--scene-x-max", type=float, default=60.0)
    parser.add_argument("--scene-y-min", type=float, default=20.0)
    parser.add_argument("--scene-y-max", type=float, default=30.0)
    parser.add_argument("--scene-z-min", type=float, default=24.0)
    parser.add_argument("--scene-z-max", type=float, default=48.0)
    parser.add_argument(
        "--l2-height-choices",
        type=float,
        nargs="+",
        default=[5.0, 6.0],
        help="Candidate L2 platform heights when --variable-scene-size is used.",
    )
    parser.add_argument(
        "--l3-height-choices",
        type=float,
        nargs="+",
        default=[9.0, 12.0],
        help="Candidate L3 platform heights when --variable-scene-size is used.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for scenario generation."""
    parser = argparse.ArgumentParser(
        description="Generate procedural 3D pipe-routing scenarios for RL training."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Generate one scenario JSON.")
    _add_common_generator_args(single)
    single.add_argument("--num-devices", type=int, default=8)
    single.add_argument(
        "--strategy",
        choices=("random", "layer_based", "functional"),
        default="random",
    )
    single.add_argument("--output", type=Path, default=Path("scenarios/layout_test.json"))
    single.add_argument("--visualize", action="store_true")

    batch = subparsers.add_parser("batch", help="Generate a flat batch of scenarios.")
    _add_common_generator_args(batch)
    batch.add_argument("--num-scenarios", type=int, default=100)
    batch.add_argument("--output-dir", type=Path, default=Path("scenarios_batch"))
    batch.add_argument("--min-devices", type=int, default=6)
    batch.add_argument("--max-devices", type=int, default=12)
    batch.add_argument(
        "--strategies",
        nargs="+",
        default=["random", "layer_based", "functional"],
        choices=("random", "layer_based", "functional"),
    )

    diverse = subparsers.add_parser(
        "diverse", help="Generate simple/medium/complex scenario folders."
    )
    _add_common_generator_args(diverse)
    diverse.add_argument("--num-scenarios", type=int, default=1000)
    diverse.add_argument("--output-dir", type=Path, default=Path("scenarios_diverse"))

    dataset = subparsers.add_parser(
        "dataset", help="Generate train/val/test scenario splits."
    )
    _add_common_generator_args(dataset)
    dataset.add_argument("--num-scenarios", type=int, default=1000)
    dataset.add_argument("--output-dir", type=Path, default=Path("scenarios_rl_dataset"))
    dataset.add_argument("--train-ratio", type=float, default=0.8)
    dataset.add_argument("--val-ratio", type=float, default=0.1)
    dataset.add_argument("--test-ratio", type=float, default=0.1)

    kfold = subparsers.add_parser(
        "kfold", help="Generate K-fold scenario splits with a held-out test set."
    )
    _add_common_generator_args(kfold)
    kfold.add_argument("--num-scenarios", type=int, default=1000)
    kfold.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scenarios_rl_dataset_kfold"),
    )
    kfold.add_argument("--k-folds", type=int, default=5)
    kfold.add_argument("--test-ratio", type=float, default=0.2)

    visualize = subparsers.add_parser("visualize", help="Export GLB previews from JSON.")
    _add_common_generator_args(visualize)
    visualize.add_argument("--input", type=Path, required=True)
    visualize.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum files to process when --input is a directory.",
    )

    return parser


def main() -> None:
    """Command-line entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    _set_random_seed(args.seed)
    generator = _build_generator(args)

    if args.command == "single":
        scenario = generator.generate_scenario(
            num_devices=args.num_devices,
            strategy=args.strategy,
            allow_rotation=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as file:
            json.dump(scenario, file, indent=2, ensure_ascii=False)
        print(f"Scenario JSON written to: {args.output}")

        if args.visualize:
            generator.visualize_scenario(scenario, str(args.output.with_suffix(".glb")))
        return

    if args.command == "batch":
        generator.generate_batch(
            num_scenarios=args.num_scenarios,
            output_dir=str(args.output_dir),
            num_devices_range=(args.min_devices, args.max_devices),
            strategies=args.strategies,
        )
        return

    if args.command == "diverse":
        generator.generate_diverse_batch(
            num_scenarios=args.num_scenarios,
            output_dir=str(args.output_dir),
        )
        return

    if args.command == "dataset":
        generator.generate_rl_dataset(
            num_scenarios=args.num_scenarios,
            output_dir=str(args.output_dir),
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
        )
        return

    if args.command == "kfold":
        generator.generate_rl_dataset_kfold(
            num_scenarios=args.num_scenarios,
            output_dir=str(args.output_dir),
            k_folds=args.k_folds,
            test_ratio=args.test_ratio,
            random_seed=args.seed,
        )
        return

    if args.command == "visualize":
        target = args.input
        if target.is_file():
            with target.open("r", encoding="utf-8") as file:
                scenario = json.load(file)
            generator.visualize_scenario(scenario, str(target.with_suffix(".glb")))
            return

        if not target.is_dir():
            raise FileNotFoundError(f"Input path does not exist: {target}")

        json_files = sorted(target.glob("**/*.json"))
        if args.limit is not None:
            json_files = json_files[:args.limit]

        for json_file in json_files:
            with json_file.open("r", encoding="utf-8") as file:
                scenario = json.load(file)
            generator.visualize_scenario(scenario, str(json_file.with_suffix(".glb")))
        print(f"Processed {len(json_files)} scenario JSON files.")


if __name__ == "__main__":
    main()
