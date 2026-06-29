#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approximate voxel occupancy with compact axis-aligned bounding boxes.

The module converts a GLB mesh into filled voxel centers, groups connected
voxels, creates tight AABBs for each group, and optionally merges adjacent
boxes under size and fill-ratio constraints. The resulting boxes provide a
compact obstacle proxy for 3D pipe routing experiments.
"""

import argparse
import json
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import trimesh


Voxel = Tuple[float, float, float]


@dataclass
class TightBoundingBox:
    """Axis-aligned box tightly covering a connected voxel group."""

    min_point: np.ndarray
    max_point: np.ndarray
    voxel_count: int
    voxel_volume: float = 0.0

    @property
    def center(self) -> np.ndarray:
        """Return the box center."""
        return (self.min_point + self.max_point) / 2

    @property
    def size(self) -> np.ndarray:
        """Return the box side lengths."""
        return self.max_point - self.min_point

    @property
    def volume(self) -> float:
        """Return the box volume."""
        return float(np.prod(self.size))

    @property
    def fill_ratio(self) -> float:
        """Return the ratio between occupied voxel volume and box volume."""
        if self.volume < 1e-9:
            return 0.0
        return float(self.voxel_volume / self.volume)

    def to_dict(self) -> Dict:
        """Serialize the box for JSON export."""
        return {
            'min_point': self.min_point.tolist(),
            'max_point': self.max_point.tolist(),
            'center': self.center.tolist(),
            'size': self.size.tolist(),
            'volume': float(self.volume),
            'voxel_count': self.voxel_count,
            'voxel_volume': float(self.voxel_volume),
            'fill_ratio': float(self.fill_ratio)
        }


class VoxelBoxApproximator:
    """Build compact bounding-box approximations from voxelized meshes."""

    def __init__(
        self,
        pitch: float = 0.1,
        merge_threshold: float = 0.1,
        min_voxels_per_box: int = 5,
        max_box_size: float = 0.6,
        min_fill_ratio: float = 0.4,
    ):
        """Initialize the approximator.

        Args:
            pitch: Voxel size in meters.
            merge_threshold: Maximum separation for candidate box merging.
            min_voxels_per_box: Minimum occupied voxels required per box.
            max_box_size: Maximum side length allowed after merging.
            min_fill_ratio: Minimum occupied-volume ratio after merging.
        """
        if pitch <= 0:
            raise ValueError("pitch must be positive")
        if min_voxels_per_box < 1:
            raise ValueError("min_voxels_per_box must be at least 1")

        self.pitch = float(pitch)
        self.merge_threshold = float(merge_threshold)
        self.min_voxels_per_box = int(min_voxels_per_box)
        self.max_box_size = float(max_box_size)
        self.min_fill_ratio = float(min_fill_ratio)

        self.mesh: Optional[trimesh.Trimesh] = None
        self.voxels: Set[Voxel] = set()
        self.boxes: List[TightBoundingBox] = []
        self.stats: Dict[str, float] = {}
        self._reset_state()

    def _reset_state(self) -> None:
        """Clear loaded mesh data and derived approximation state."""
        self.mesh = None
        self.voxels = set()
        self.boxes = []
        self.stats = {
            'voxel_count': 0,
            'initial_boxes': 0,
            'merged_boxes': 0,
            'final_boxes': 0,
            'total_volume': 0.0,
            'average_voxels_per_box': 0.0
        }

    def load_glb(self, glb_path: str, device_transform: Optional[Dict] = None) -> bool:
        """Load a GLB file and voxelize it.

        Args:
            glb_path: Path to the GLB mesh file.
            device_transform: Optional transform dictionary with ``center`` and
                ``pose`` keys. Pose angles are interpreted as radians.

        Returns:
            True if the mesh was loaded and voxelized successfully.
        """
        self._reset_state()

        try:
            scene = trimesh.load(glb_path, force='scene')
            mesh = self._scene_to_mesh(scene)
            if mesh is None:
                print(f"No valid mesh geometry found in {glb_path}.")
                return False

            return self.load_mesh(mesh, device_transform=device_transform, reset=False)

        except Exception as e:
            print(f"Failed to load GLB file {glb_path}: {e}")
            return False

    def load_mesh(
        self,
        mesh: trimesh.Trimesh,
        device_transform: Optional[Dict] = None,
        reset: bool = True,
    ) -> bool:
        """Load an in-memory mesh and voxelize it.

        This helper avoids temporary GLB export when upstream code already owns
        a transformed ``trimesh.Trimesh`` instance.
        """
        if reset:
            self._reset_state()

        if not isinstance(mesh, trimesh.Trimesh):
            print("load_mesh expects a trimesh.Trimesh instance.")
            return False

        if mesh.is_empty or len(mesh.vertices) == 0:
            print("Cannot voxelize an empty mesh.")
            return False

        try:
            self.mesh = mesh.copy()
            if device_transform is not None:
                self._apply_transform(device_transform)

            return self._voxelize_mesh()

        except Exception as e:
            print(f"Failed to voxelize mesh: {e}")
            return False

    def _scene_to_mesh(self, scene) -> Optional[trimesh.Trimesh]:
        """Convert a loaded Trimesh object or scene into one mesh."""
        if isinstance(scene, trimesh.Trimesh):
            return scene

        if not isinstance(scene, trimesh.Scene):
            return None

        try:
            mesh = scene.dump(concatenate=True)
            if isinstance(mesh, trimesh.Trimesh) and not mesh.is_empty:
                return mesh
        except Exception:
            pass

        meshes = [
            geom
            for geom in scene.geometry.values()
            if isinstance(geom, trimesh.Trimesh) and not geom.is_empty
        ]
        if not meshes:
            return None
        return trimesh.util.concatenate(meshes)

    def _voxelize_mesh(self) -> bool:
        """Voxelize the loaded mesh and store snapped voxel centers."""
        if self.mesh is None:
            print("No mesh is loaded.")
            return False

        print("Loaded mesh; voxelizing occupancy...")
        voxels_obj = self.mesh.voxelized(pitch=self.pitch)
        filled_voxels_obj = voxels_obj.fill()

        self.voxels = {self._snap(point) for point in filled_voxels_obj.points}
        self.stats['voxel_count'] = len(self.voxels)

        if not self.voxels:
            print("Voxelization produced no occupied voxels.")
            return False

        print(f"   Occupied voxels: {self.stats['voxel_count']:,}")
        return True

    def _apply_transform(self, device_transform: Dict):
        """Apply translation and yaw/pitch/roll rotation to the current mesh."""
        if self.mesh is None:
            return

        center = device_transform.get('center', [0, 0, 0])
        pose = device_transform.get('pose', {})

        transform_matrix = np.eye(4)
        yaw = pose.get('yaw', 0.0)
        pitch_angle = pose.get('pitch', 0.0)
        roll = pose.get('roll', 0.0)

        if yaw != 0.0 or pitch_angle != 0.0 or roll != 0.0:
            Ry = np.array([
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)]
            ])
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(pitch_angle), -np.sin(pitch_angle)],
                [0, np.sin(pitch_angle), np.cos(pitch_angle)]
            ])
            Rz = np.array([
                [np.cos(roll), -np.sin(roll), 0],
                [np.sin(roll), np.cos(roll), 0],
                [0, 0, 1]
            ])
            transform_matrix[:3, :3] = Rz @ Rx @ Ry

        transform_matrix[:3, 3] = center
        self.mesh.apply_transform(transform_matrix)

    def _snap(self, point: np.ndarray) -> Voxel:
        """Snap a point to the voxel grid with stable floating-point keys."""
        return tuple(
            round(round(float(coord) / self.pitch) * self.pitch, 10)
            for coord in point
        )

    def _offset_voxel(self, voxel: Voxel, offset: Voxel) -> Voxel:
        """Return a neighboring voxel key with consistent rounding."""
        return tuple(round(float(voxel[i] + offset[i]), 10) for i in range(3))

    def generate_boxes(self, method: str = 'slice') -> List[TightBoundingBox]:
        """Generate tight bounding boxes from occupied voxels.

        Args:
            method: Box construction strategy: ``slice``, ``cluster``, or
                ``flood``. ``flood`` currently uses connected components with a
                reserved regularization hook.

        Returns:
            Generated bounding boxes.
        """
        if not self.voxels:
            print("No voxel data is available. Call load_glb() or load_mesh() first.")
            return []

        method = method.lower()
        print(f"\nGenerating tight bounding boxes with method='{method}'...")
        self.boxes = []
        self.stats.update({
            'initial_boxes': 0,
            'merged_boxes': 0,
            'final_boxes': 0,
            'total_volume': 0.0,
            'average_voxels_per_box': 0.0,
        })

        if method == 'slice':
            self.boxes = self._slice_method()
        elif method == 'cluster':
            self.boxes = self._cluster_method()
        elif method == 'flood':
            self.boxes = self._flood_fill_method()
        else:
            print(f"Unknown box generation method: {method}")
            return []

        self.stats['initial_boxes'] = len(self.boxes)
        print(f"   Initial boxes: {self.stats['initial_boxes']}")

        self.boxes = self._merge_adjacent_boxes(self.boxes)
        self.stats['merged_boxes'] = len(self.boxes)
        print(f"   Boxes after merging: {self.stats['merged_boxes']}")

        self.boxes = [
            box for box in self.boxes if box.voxel_count >= self.min_voxels_per_box
        ]
        self.stats['final_boxes'] = len(self.boxes)

        if self.boxes:
            self.stats['total_volume'] = float(sum(box.volume for box in self.boxes))
            self.stats['average_voxels_per_box'] = float(
                sum(box.voxel_count for box in self.boxes) / len(self.boxes)
            )

        print(f"   Final boxes: {self.stats['final_boxes']}")
        print(
            "   Average voxels per box: "
            f"{self.stats['average_voxels_per_box']:.1f}"
        )

        return self.boxes

    def _slice_method(self) -> List[TightBoundingBox]:
        """Create boxes by finding connected 2D regions on each Y slice."""
        boxes = []

        y_groups: Dict[float, List[Voxel]] = defaultdict(list)
        for voxel in self.voxels:
            y_coord = voxel[1]
            y_groups[y_coord].append(voxel)

        print(f"   Y-axis slices: {len(y_groups)}")

        for y_coord, layer_voxels in sorted(y_groups.items()):
            xz_boxes = self._find_connected_regions_2d(layer_voxels, y_coord)
            boxes.extend(xz_boxes)

        return boxes

    def _find_connected_regions_2d(
        self, voxels: List[Voxel], y_coord: float
    ) -> List[TightBoundingBox]:
        """Find connected voxel regions on one XZ slice."""
        if not voxels:
            return []

        voxel_set = set(voxels)
        visited: Set[Voxel] = set()
        boxes = []

        neighbors_2d = [
            (self.pitch, 0.0, 0.0), (-self.pitch, 0.0, 0.0),
            (0.0, 0.0, self.pitch), (0.0, 0.0, -self.pitch),
            (self.pitch, 0.0, self.pitch), (self.pitch, 0.0, -self.pitch),
            (-self.pitch, 0.0, self.pitch), (-self.pitch, 0.0, -self.pitch)
        ]

        for start_voxel in sorted(voxels):
            if start_voxel in visited:
                continue

            region = []
            queue = deque([start_voxel])
            visited.add(start_voxel)

            while queue:
                current = queue.popleft()
                region.append(current)

                for offset in neighbors_2d:
                    neighbor = self._offset_voxel(current, offset)
                    if neighbor in voxel_set and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(region) >= self.min_voxels_per_box:
                box = self._create_tight_box(region)
                boxes.append(box)

        return boxes

    def _cluster_method(self) -> List[TightBoundingBox]:
        """Create boxes from 3D 6-connected voxel components."""
        boxes = []
        visited: Set[Voxel] = set()

        neighbors_3d = [
            (self.pitch, 0.0, 0.0), (-self.pitch, 0.0, 0.0),
            (0.0, self.pitch, 0.0), (0.0, -self.pitch, 0.0),
            (0.0, 0.0, self.pitch), (0.0, 0.0, -self.pitch)
        ]

        print("   Running 3D connected-component analysis...")

        for start_voxel in sorted(self.voxels):
            if start_voxel in visited:
                continue

            region = []
            queue = deque([start_voxel])
            visited.add(start_voxel)

            while queue:
                current = queue.popleft()
                region.append(current)

                for offset in neighbors_3d:
                    neighbor = self._offset_voxel(current, offset)
                    if neighbor in self.voxels and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            if len(region) >= self.min_voxels_per_box:
                box = self._create_tight_box(region)
                boxes.append(box)

        return boxes

    def _flood_fill_method(self) -> List[TightBoundingBox]:
        """Create regularized boxes from connected components.

        The current implementation intentionally preserves the connected
        component boxes. The regularization hook is kept explicit so future
        experiments can add rectangle expansion without changing the public API.
        """
        regions_boxes = self._cluster_method()
        optimized_boxes = []

        for box in regions_boxes:
            expanded_box = self._expand_box_to_rectangle(box)
            optimized_boxes.append(expanded_box)

        return optimized_boxes

    def _create_tight_box(self, voxels: List[Voxel]) -> TightBoundingBox:
        """Create a tight AABB that covers complete voxel cells."""
        voxel_array = np.array(voxels)

        min_coords = voxel_array.min(axis=0)
        max_coords = voxel_array.max(axis=0)

        half_pitch = self.pitch / 2
        min_point = min_coords - half_pitch
        max_point = max_coords + half_pitch

        voxel_volume = len(voxels) * (self.pitch ** 3)

        return TightBoundingBox(
            min_point=min_point,
            max_point=max_point,
            voxel_count=len(voxels),
            voxel_volume=voxel_volume
        )

    def _expand_box_to_rectangle(self, box: TightBoundingBox) -> TightBoundingBox:
        """Return the current box; reserved for future box regularization."""
        return box

    def _merge_adjacent_boxes(self, boxes: List[TightBoundingBox]) -> List[TightBoundingBox]:
        """Merge neighboring boxes when the merged box remains compact."""
        if len(boxes) <= 1:
            return boxes

        print("   Merging adjacent boxes...")

        merged = True
        iteration = 0

        while merged and iteration < 10:
            merged = False
            iteration += 1
            new_boxes = []
            used = set()

            for i, box1 in enumerate(boxes):
                if i in used:
                    continue

                merged_with = None
                for j in range(i + 1, len(boxes)):
                    if j in used:
                        continue

                    box2 = boxes[j]

                    if self._can_merge(box1, box2):
                        merged_box = self._merge_two_boxes(box1, box2)
                        new_boxes.append(merged_box)
                        used.add(i)
                        used.add(j)
                        merged = True
                        merged_with = j
                        break

                if merged_with is None and i not in used:
                    new_boxes.append(box1)
                    used.add(i)

            boxes = new_boxes

            if merged:
                print(f"     Iteration {iteration}: {len(boxes)} boxes")

        return boxes

    def _can_merge(self, box1: TightBoundingBox, box2: TightBoundingBox) -> bool:
        """Return whether two boxes satisfy conservative merge constraints."""
        merged_box = self._merge_two_boxes(box1, box2)

        merged_size = merged_box.size
        if np.any(merged_size > self.max_box_size):
            return False

        if merged_box.fill_ratio < self.min_fill_ratio:
            return False

        for axis in range(3):
            gap = max(box1.min_point[axis], box2.min_point[axis]) - \
                  min(box1.max_point[axis], box2.max_point[axis])

            if gap <= self.merge_threshold:
                other_axes = [0, 1, 2]
                other_axes.remove(axis)

                overlaps = True
                for other_axis in other_axes:
                    overlap_min = max(box1.min_point[other_axis], box2.min_point[other_axis])
                    overlap_max = min(box1.max_point[other_axis], box2.max_point[other_axis])

                    if overlap_max <= overlap_min:
                        overlaps = False
                        break

                    overlap_length = overlap_max - overlap_min
                    box1_length = box1.max_point[other_axis] - box1.min_point[other_axis]
                    box2_length = box2.max_point[other_axis] - box2.min_point[other_axis]
                    min_length = min(box1_length, box2_length)

                    if overlap_length < min_length * 0.8:
                        overlaps = False
                        break

                if overlaps:
                    return True

        return False

    def _merge_two_boxes(self, box1: TightBoundingBox, box2: TightBoundingBox) -> TightBoundingBox:
        """Return one box covering both input boxes."""
        min_point = np.minimum(box1.min_point, box2.min_point)
        max_point = np.maximum(box1.max_point, box2.max_point)

        return TightBoundingBox(
            min_point=min_point,
            max_point=max_point,
            voxel_count=box1.voxel_count + box2.voxel_count,
            voxel_volume=box1.voxel_volume + box2.voxel_volume
        )

    def export_json(self, output_path: str) -> None:
        """Export box metadata and geometry to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'metadata': {
                'pitch': self.pitch,
                'merge_threshold': self.merge_threshold,
                'min_voxels_per_box': self.min_voxels_per_box,
                'max_box_size': self.max_box_size,
                'min_fill_ratio': self.min_fill_ratio
            },
            'statistics': self.stats,
            'boxes': [box.to_dict() for box in self.boxes]
        }

        with output_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nExported JSON: {output_path}")

    def export_visualization_glb(self, output_path: str, show_voxels: bool = False) -> None:
        """Export a GLB visualization of generated boxes and optional voxels."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        scene = trimesh.Scene()

        for i, box in enumerate(self.boxes):
            size = box.size
            center = box.center

            box_mesh = trimesh.creation.box(extents=size)
            box_mesh.apply_translation(center)

            ratio = box.voxel_count / max(self.stats['average_voxels_per_box'], 1)
            ratio = min(float(ratio), 1.0)
            color = [
                int(255 * ratio),
                100,
                int(255 * (1 - ratio)),
                120
            ]
            box_mesh.visual.face_colors = color

            scene.add_geometry(box_mesh, node_name=f"box_{i}")

        if show_voxels and self.voxels:
            voxel_points = np.array(sorted(self.voxels))
            point_cloud = trimesh.PointCloud(vertices=voxel_points)
            point_cloud.colors = [255, 50, 50, 255]
            scene.add_geometry(point_cloud, node_name="voxels")

        axes = trimesh.creation.axis(origin_size=0.05, axis_length=1.0)
        scene.add_geometry(axes, node_name="axes")

        scene.export(output_path)
        file_size_kb = output_path.stat().st_size / 1024
        print(f"\nExported visualization GLB: {output_path} ({file_size_kb:.1f} KB)")


def process_glb_file(
    glb_path: str,
    output_folder: str,
    pitch: float = 0.1,
    method: str = 'slice',
    merge_threshold: float = 0.1,
    min_voxels_per_box: int = 5,
    max_box_size: float = 0.6,
    min_fill_ratio: float = 0.4,
    show_voxels: bool = False,
) -> bool:
    """Process one GLB file and export JSON plus visualization outputs."""
    glb_path = Path(glb_path)
    output_folder = Path(output_folder)

    if not glb_path.exists():
        print(f"GLB file does not exist: {glb_path}")
        return False

    approximator = VoxelBoxApproximator(
        pitch=pitch,
        merge_threshold=merge_threshold,
        min_voxels_per_box=min_voxels_per_box,
        max_box_size=max_box_size,
        min_fill_ratio=min_fill_ratio
    )

    if not approximator.load_glb(str(glb_path)):
        return False

    boxes = approximator.generate_boxes(method=method)
    if not boxes:
        print(f"No boxes generated for {glb_path}.")
        return False

    output_folder.mkdir(parents=True, exist_ok=True)
    approximator.export_json(str(output_folder / f"{glb_path.stem}_boxes.json"))
    approximator.export_visualization_glb(
        str(output_folder / f"{glb_path.stem}_boxes.glb"),
        show_voxels=show_voxels
    )
    return True


def batch_process_glb_folder(
    glb_folder: str,
    output_folder: str,
    pitch: float = 0.1,
    method: str = 'slice',
    merge_threshold: float = 0.2,
    min_voxels_per_box: int = 5,
    max_box_size: float = 0.6,
    min_fill_ratio: float = 0.4,
    show_voxels: bool = False,
) -> None:
    """Process all GLB files in a folder."""
    glb_folder = Path(glb_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    glb_files = sorted(glb_folder.glob("*.glb"))

    if not glb_files:
        print(f"No GLB files found in {glb_folder}.")
        return

    print(f"\nFound {len(glb_files)} GLB files.")
    success_count = 0

    for i, glb_file in enumerate(glb_files, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(glb_files)}] Processing: {glb_file.name}")
        print(f"{'='*60}")

        ok = process_glb_file(
            str(glb_file),
            str(output_folder),
            pitch=pitch,
            method=method,
            merge_threshold=merge_threshold,
            min_voxels_per_box=min_voxels_per_box,
            max_box_size=max_box_size,
            min_fill_ratio=min_fill_ratio,
            show_voxels=show_voxels
        )
        success_count += int(ok)

    print(f"\nProcessed {success_count}/{len(glb_files)} GLB files successfully.")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for standalone preprocessing."""
    parser = argparse.ArgumentParser(
        description="Approximate GLB mesh occupancy with compact voxel boxes."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--glb", type=Path, help="Path to one GLB file.")
    input_group.add_argument(
        "--glb-folder", type=Path, help="Folder containing GLB files."
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("voxel_boxes_output"),
        help="Folder for exported JSON and visualization GLB files."
    )
    parser.add_argument("--pitch", type=float, default=0.1, help="Voxel size.")
    parser.add_argument(
        "--method",
        choices=("slice", "cluster", "flood"),
        default="slice",
        help="Box generation strategy."
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.1,
        help="Maximum separation for candidate box merging."
    )
    parser.add_argument(
        "--min-voxels-per-box",
        type=int,
        default=5,
        help="Minimum occupied voxels required per exported box."
    )
    parser.add_argument(
        "--max-box-size",
        type=float,
        default=0.6,
        help="Maximum side length allowed after merging."
    )
    parser.add_argument(
        "--min-fill-ratio",
        type=float,
        default=0.4,
        help="Minimum occupied-volume ratio allowed after merging."
    )
    parser.add_argument(
        "--show-voxels",
        action="store_true",
        help="Include occupied voxel centers in the visualization GLB."
    )
    return parser


def main() -> None:
    """Run the command-line preprocessing utility."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.glb is not None:
        process_glb_file(
            str(args.glb),
            str(args.output_folder),
            pitch=args.pitch,
            method=args.method,
            merge_threshold=args.merge_threshold,
            min_voxels_per_box=args.min_voxels_per_box,
            max_box_size=args.max_box_size,
            min_fill_ratio=args.min_fill_ratio,
            show_voxels=args.show_voxels
        )
        return

    batch_process_glb_folder(
        str(args.glb_folder),
        str(args.output_folder),
        pitch=args.pitch,
        method=args.method,
        merge_threshold=args.merge_threshold,
        min_voxels_per_box=args.min_voxels_per_box,
        max_box_size=args.max_box_size,
        min_fill_ratio=args.min_fill_ratio,
        show_voxels=args.show_voxels
    )


if __name__ == "__main__":
    main()
