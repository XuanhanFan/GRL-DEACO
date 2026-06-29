"""GLB device reader used by the DEACO routing input layer.

The reader extracts device mesh bounds, node transforms, and pipe port metadata
from GLB assets. Port nodes are interpreted as facing along their local positive
Y axis. Heavy geometry dependencies are imported only when a GLB file is read so
CLI help and configuration loading remain lightweight.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)

_EPSILON = 1e-8
_DEFAULT_DIRECTION = np.array([0.0, 1.0, 0.0], dtype=float)
_DEFAULT_PORT_KEYWORDS = ("\u63a5\u53e3", "gz", "port", "Port", "PORT")
_DEFAULT_EXCLUDE_KEYWORDS = ("ExportedModels", "\u8bbe\u5907", "\u7a7a\u7269\u4f53", "Empty", "Root", "Scene")


def _load_gltf2():
    try:
        from pygltflib import GLTF2
    except ImportError as exc:  # pragma: no cover - depends on optional geometry stack
        raise RuntimeError("pygltflib is required to read GLB device files.") from exc
    return GLTF2


def _load_trimesh():
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - depends on optional geometry stack
        raise RuntimeError("trimesh is required to read GLB mesh geometry.") from exc
    return trimesh


def snap_value(value: Any, precision: int = 1000) -> Any:
    """Round scalar or vector coordinates to a stable decimal grid."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return [round(float(item) * precision) / precision for item in value]
    return round(float(value) * precision) / precision


def _normalize_vector(vector: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON:
        return _DEFAULT_DIRECTION.copy() if fallback is None else fallback.copy()
    return vector / norm


def _quaternion_xyzw_to_rotation_matrix(quaternion: List[float] | np.ndarray) -> np.ndarray:
    """Convert a GLB quaternion in [x, y, z, w] order to a rotation matrix."""
    x, y, z, w = np.asarray(quaternion, dtype=float)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= _EPSILON:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=float,
    )


class GLBReader:
    """Read one GLB device asset and expose routing-ready device metadata."""

    def __init__(self) -> None:
        self.gltf = None
        self.file_path: Optional[str] = None
        self.mesh = None
        self.nodes_info: List[Dict[str, Any]] = []
        self.ports_info: List[Dict[str, Any]] = []
        self.device_bounds = None
        self.device_center = None
        self._node_parent_index: Dict[int, int] = {}
        self._world_transform_cache: Dict[int, np.ndarray] = {}

    def load_glb_file(self, file_path: str) -> bool:
        """Load a GLB file header and node tree."""
        path = Path(file_path)
        if not path.exists():
            logger.warning("GLB file does not exist: %s", path)
            return False

        try:
            GLTF2 = _load_gltf2()
            self.file_path = str(path)
            self.gltf = GLTF2().load_binary(str(path))
            self._world_transform_cache.clear()
            logger.debug("Loaded GLB file: %s", path.name)
            return True
        except Exception as exc:
            logger.warning("Failed to load GLB file %s: %s", path, exc)
            return False

    def _load_mesh_with_draco(self):
        """Load mesh geometry, decoding Draco-compressed primitives when needed."""
        trimesh = _load_trimesh()
        if "KHR_draco_mesh_compression" not in (self.gltf.extensionsUsed or []):
            return trimesh.load(self.file_path)

        try:
            from DracoPy import decode
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("DracoPy is required for Draco-compressed GLB files.") from exc

        primitive = self.gltf.meshes[0].primitives[0]
        draco_extension = primitive.extensions["KHR_draco_mesh_compression"]
        buffer_view_index = draco_extension["bufferView"]
        buffer_view = self.gltf.bufferViews[buffer_view_index]
        byte_offset = buffer_view.byteOffset or 0
        byte_length = buffer_view.byteLength
        binary_blob = self.gltf.binary_blob()
        draco_bytes = binary_blob[byte_offset : byte_offset + byte_length]
        draco_mesh = decode(draco_bytes)
        return trimesh.Trimesh(vertices=np.asarray(draco_mesh.points), faces=np.asarray(draco_mesh.faces))

    def extract_mesh_data(self) -> bool:
        """Extract mesh geometry, bounds, and the bottom-center device origin."""
        if self.gltf is None or self.file_path is None:
            logger.warning("Load a GLB file before extracting mesh data.")
            return False

        try:
            trimesh = _load_trimesh()
            mesh_or_scene = self._load_mesh_with_draco()
            if isinstance(mesh_or_scene, trimesh.Scene):
                self.mesh = mesh_or_scene.dump(concatenate=True)
            else:
                self.mesh = mesh_or_scene

            if self.mesh is None:
                logger.warning("No mesh geometry found in GLB file: %s", self.file_path)
                return False

            self.device_bounds = self.mesh.bounds
            min_point = self.device_bounds[0]
            max_point = self.device_bounds[1]
            # The project treats the device origin as the bottom center for
            # compatibility with existing layout placement data.
            self.device_center = np.array(
                [
                    (min_point[0] + max_point[0]) / 2,
                    min_point[1],
                    (min_point[2] + max_point[2]) / 2,
                ],
                dtype=float,
            )
            logger.debug(
                "Extracted GLB mesh: vertices=%s faces=%s",
                len(self.mesh.vertices),
                len(self.mesh.faces),
            )
            return True
        except Exception as exc:
            logger.warning("Failed to extract mesh data from %s: %s", self.file_path, exc)
            return False

    def extract_node_transform_matrix(self, node: Any) -> np.ndarray:
        """Return the local 4x4 transform matrix for a GLB node."""
        if node.matrix:
            return np.asarray(node.matrix, dtype=float).reshape(4, 4).T

        translation = np.asarray(node.translation if node.translation else [0.0, 0.0, 0.0], dtype=float)
        rotation = np.asarray(node.rotation if node.rotation else [0.0, 0.0, 0.0, 1.0], dtype=float)
        scale = np.asarray(node.scale if node.scale else [1.0, 1.0, 1.0], dtype=float)

        matrix = np.eye(4)
        matrix[:3, :3] = _quaternion_xyzw_to_rotation_matrix(rotation) @ np.diag(scale)
        matrix[:3, 3] = translation
        return matrix

    def _build_parent_index(self) -> None:
        self._node_parent_index = {}
        for parent_index, node in enumerate(self.gltf.nodes or []):
            for child_index in node.children or []:
                self._node_parent_index[int(child_index)] = int(parent_index)

    def _get_world_transform(self, node_index: int) -> np.ndarray:
        if node_index in self._world_transform_cache:
            return self._world_transform_cache[node_index]

        node = self.gltf.nodes[node_index]
        local_transform = self.extract_node_transform_matrix(node)
        parent_index = self._node_parent_index.get(node_index)
        if parent_index is None:
            world_transform = local_transform
        else:
            world_transform = self._get_world_transform(parent_index) @ local_transform
        self._world_transform_cache[node_index] = world_transform
        return world_transform

    def _get_directions_from_quaternion(self, quaternion: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotation_matrix = _quaternion_xyzw_to_rotation_matrix(quaternion)
        return rotation_matrix[:, 0], rotation_matrix[:, 1], rotation_matrix[:, 2]

    def extract_nodes_info(self) -> bool:
        """Extract node transforms and local coordinate axes."""
        if self.gltf is None:
            logger.warning("Load a GLB file before extracting node data.")
            return False

        try:
            self.nodes_info = []
            self._build_parent_index()
            self._world_transform_cache.clear()
            for node_index, node in enumerate(self.gltf.nodes or []):
                node_info: Dict[str, Any] = {
                    "index": node_index,
                    "name": node.name or f"node_{node_index}",
                    "translation": np.asarray(node.translation, dtype=float) if node.translation else None,
                    "rotation": node.rotation,
                    "scale": np.asarray(node.scale, dtype=float) if node.scale else None,
                    "matrix": np.asarray(node.matrix, dtype=float).reshape(4, 4) if node.matrix else None,
                    "mesh_index": node.mesh,
                    "children": node.children or [],
                }
                if node.rotation:
                    right, up, forward = self._get_directions_from_quaternion(node.rotation)
                    node_info["directions"] = {"right": right, "up": up, "forward": forward}
                self.nodes_info.append(node_info)
            logger.debug("Extracted %s GLB nodes.", len(self.nodes_info))
            return True
        except Exception as exc:
            logger.warning("Failed to extract node data from %s: %s", self.file_path, exc)
            return False

    def _get_port_direction_from_transform_matrix(
        self,
        relative_position: np.ndarray,
        rotation_matrix: np.ndarray,
        port_name: str = "",
    ) -> np.ndarray:
        """Return the normalized world direction of a port local +Y axis."""
        local_y_axis = rotation_matrix[:, 1]
        if np.linalg.norm(local_y_axis) > _EPSILON:
            return _normalize_vector(local_y_axis)
        if np.linalg.norm(relative_position) > _EPSILON:
            logger.debug("Port %s has a degenerate local Y axis; using relative position.", port_name)
            return _normalize_vector(relative_position)
        logger.debug("Port %s has degenerate direction and position; using default +Y.", port_name)
        return _DEFAULT_DIRECTION.copy()

    def extract_ports_info(self, port_keyword: str = "\u63a5\u53e3") -> bool:
        """Extract pipe port positions and directions from named GLB nodes."""
        if not self.nodes_info:
            logger.warning("Extract node data before extracting ports.")
            return False
        if self.device_center is None:
            logger.warning("Extract mesh data before extracting ports.")
            return False

        keywords = tuple(dict.fromkeys((port_keyword, *_DEFAULT_PORT_KEYWORDS)))
        try:
            self.ports_info = []
            for node_info in self.nodes_info:
                node_name = str(node_info["name"])
                is_port = any(keyword and keyword in node_name for keyword in keywords)
                is_excluded = any(keyword in node_name for keyword in _DEFAULT_EXCLUDE_KEYWORDS)
                if not is_port or is_excluded:
                    continue

                node_index = int(node_info["index"])
                local_transform = self.extract_node_transform_matrix(self.gltf.nodes[node_index])
                world_transform = self._get_world_transform(node_index)
                world_position = world_transform[:3, 3]
                local_position = local_transform[:3, 3]
                relative_position = world_position - np.asarray(self.device_center, dtype=float)
                direction = self._get_port_direction_from_transform_matrix(
                    relative_position,
                    world_transform[:3, :3],
                    node_name,
                )
                self.ports_info.append(
                    {
                        "name": node_name,
                        "node_index": node_index,
                        "world_position": snap_value(world_position.tolist()),
                        "local_position": snap_value(local_position.tolist()),
                        "relative_position": snap_value(relative_position.tolist()),
                        "direction": snap_value(direction.tolist()),
                        "rotation_quat": node_info.get("rotation"),
                        "transform_matrix": world_transform.tolist(),
                    }
                )

            logger.debug("Extracted %s GLB ports.", len(self.ports_info))
            return True
        except Exception as exc:
            logger.warning("Failed to extract port data from %s: %s", self.file_path, exc)
            return False

    def get_device_info(self) -> Dict[str, Any]:
        """Return routing-compatible device metadata."""
        return {
            "file_path": self.file_path,
            "file_name": Path(self.file_path).name if self.file_path else None,
            "mesh": self.mesh,
            "bounds": self.device_bounds,
            "center": self.device_center,
            "nodes": self.nodes_info,
            "ports": self.ports_info,
            "vertex_count": len(self.mesh.vertices) if self.mesh is not None else 0,
            "face_count": len(self.mesh.faces) if self.mesh is not None else 0,
        }

    def save_info_to_json(self, output_path: str) -> bool:
        """Save serializable device metadata for inspection."""
        try:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(_to_jsonable(self.get_device_info()), handle, indent=2, ensure_ascii=False)
            logger.info("Saved GLB device metadata to %s", output_path)
            return True
        except Exception as exc:
            logger.warning("Failed to save GLB device metadata to %s: %s", output_path, exc)
            return False


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items() if key != "mesh"}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "vertices") and hasattr(value, "faces"):
        return "<mesh omitted>"
    return value


def analyze_glb_directory(directory_path: str, port_keyword: str = "\u63a5\u53e3") -> List[Dict[str, Any]]:
    """Analyze all GLB files in a directory and return device metadata records."""
    directory = Path(directory_path)
    if not directory.exists():
        logger.warning("GLB directory does not exist: %s", directory)
        return []

    devices_info: List[Dict[str, Any]] = []
    for glb_file in sorted(directory.glob("*.glb")):
        reader = GLBReader()
        if (
            reader.load_glb_file(str(glb_file))
            and reader.extract_mesh_data()
            and reader.extract_nodes_info()
            and reader.extract_ports_info(port_keyword)
        ):
            devices_info.append(reader.get_device_info())
    return devices_info


__all__ = ["GLBReader", "analyze_glb_directory", "snap_value"]
