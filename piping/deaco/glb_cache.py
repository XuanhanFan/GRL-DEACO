"""In-process cache for GLB device metadata."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .glb_reader import GLBReader


logger = logging.getLogger(__name__)

_device_info_cache: Dict[str, Dict[str, Any]] = {}


def _cache_key(glb_path: str | Path) -> str:
    return str(Path(glb_path).expanduser().resolve())


def load_glb_device_info(glb_path: str) -> Optional[dict]:
    """Load and cache complete GLB device metadata."""
    key = _cache_key(glb_path)
    if key in _device_info_cache:
        return copy.deepcopy(_device_info_cache[key])

    path = Path(key)
    if not path.exists():
        logger.warning("GLB file does not exist: %s", path)
        return None

    reader = GLBReader()
    try:
        if not reader.load_glb_file(str(path)):
            return None
        if not reader.extract_mesh_data():
            return None
        if not reader.extract_nodes_info():
            return None
        if not reader.extract_ports_info():
            return None
    except Exception as exc:
        logger.warning("Failed to load GLB device metadata for %s: %s", path, exc)
        return None

    device_info = reader.get_device_info()
    _device_info_cache[key] = copy.deepcopy(device_info)
    return copy.deepcopy(device_info)


def preload_all_glb_devices(glb_directory: str) -> Dict[str, int]:
    """Preload all GLB files in a directory into the in-process cache."""
    directory = Path(glb_directory).expanduser()
    if not directory.is_absolute():
        directory = directory.resolve()

    if not directory.is_dir():
        logger.warning("GLB directory does not exist: %s", directory)
        return {"loaded": 0, "failed": 0, "total": 0}

    glb_files = sorted(directory.glob("*.glb"))
    if not glb_files:
        logger.warning("No GLB files found in directory: %s", directory)
        return {"loaded": 0, "failed": 0, "total": 0}

    loaded = 0
    failed = 0
    for glb_file in glb_files:
        if load_glb_device_info(str(glb_file)) is None:
            failed += 1
        else:
            loaded += 1

    logger.info("Preloaded GLB device cache: loaded=%s failed=%s total=%s", loaded, failed, len(glb_files))
    return {"loaded": loaded, "failed": failed, "total": len(glb_files)}


def clear_cache() -> None:
    """Clear cached GLB device metadata."""
    _device_info_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Return cache metadata for diagnostics and tests."""
    return {
        "device_count": len(_device_info_cache),
        "cache_keys": sorted(_device_info_cache.keys()),
    }


def get_cached_device_info(glb_path: str) -> Optional[dict]:
    """Return cached metadata without loading the GLB file."""
    key = _cache_key(glb_path)
    cached = _device_info_cache.get(key)
    return copy.deepcopy(cached) if cached is not None else None


__all__ = [
    "clear_cache",
    "get_cache_stats",
    "get_cached_device_info",
    "load_glb_device_info",
    "preload_all_glb_devices",
]
