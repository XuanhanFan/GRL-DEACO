from __future__ import annotations
import logging
import math
import os
import tempfile
from typing import Optional
import numpy as np
from .layout_io import parse_scene_bounds
from .types import GridPoint, snap
try:
    from voxel_box_approximation import VoxelBoxApproximator
except ImportError:
    VoxelBoxApproximator = None
logger = logging.getLogger(__name__)

def build_sdbb_for_device(device, expand_margin=0.0, max_boxes=5, slice_method='adaptive'):
    if device.get('mesh') is None:
        return []
    voxel_boxes = []
    try:
        dev_model = device['mesh'].copy()
        center = device['center']
        pose = device.get('pose', {})
        transform_matrix = np.eye(4)
        transform_matrix[:3, 3] = center
        yaw = pose.get('yaw', 0.0)
        pitch_rot = pose.get('pitch', 0.0)
        roll = pose.get('roll', 0.0)
        if yaw != 0.0 or pitch_rot != 0.0 or roll != 0.0:
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            cos_pitch = np.cos(pitch_rot)
            sin_pitch = np.sin(pitch_rot)
            cos_roll = np.cos(roll)
            sin_roll = np.sin(roll)
            Ry = np.array([[cos_yaw, 0, sin_yaw], [0, 1, 0], [-sin_yaw, 0, cos_yaw]])
            Rx = np.array([[1, 0, 0], [0, cos_pitch, -sin_pitch], [0, sin_pitch, cos_pitch]])
            Rz = np.array([[cos_roll, -sin_roll, 0], [sin_roll, cos_roll, 0], [0, 0, 1]])
            rotation_matrix = Rz @ Rx @ Ry
            transform_matrix[:3, :3] = rotation_matrix
        dev_model.apply_transform(transform_matrix)
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp_file:
            tmp_glb_path = tmp_file.name
        try:
            dev_model.export(tmp_glb_path)
            approximator = VoxelBoxApproximator(pitch=0.1, merge_threshold=0.1, min_voxels_per_box=5, max_box_size=0.6, min_fill_ratio=0.4)
            if approximator.load_glb(tmp_glb_path):
                method = 'slice' if slice_method in ['adaptive', 'height'] else slice_method
                if method not in ['slice', 'cluster', 'flood']:
                    method = 'slice'
                boxes = approximator.generate_boxes(method=method)
                for box in boxes:
                    voxel_boxes.append((box.min_point[0], box.min_point[1], box.min_point[2], box.max_point[0], box.max_point[1], box.max_point[2]))
        finally:
            if os.path.exists(tmp_glb_path):
                os.unlink(tmp_glb_path)
        if len(voxel_boxes) == 0:
            bounds = dev_model.bounds
            voxel_boxes.append((bounds[0][0], bounds[0][1], bounds[0][2], bounds[1][0], bounds[1][1], bounds[1][2]))
    except Exception as e:
        logger.debug("SDBB voxel approximation failed for %s; falling back to mesh bounds.", device.get('name'), exc_info=True)
        try:
            dev_model = device['mesh'].copy()
            center = device['center']
            pose = device.get('pose', {})
            transform_matrix = np.eye(4)
            transform_matrix[:3, 3] = center
            yaw = pose.get('yaw', 0.0)
            pitch_rot = pose.get('pitch', 0.0)
            roll = pose.get('roll', 0.0)
            if yaw != 0.0 or pitch_rot != 0.0 or roll != 0.0:
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                cos_pitch = np.cos(pitch_rot)
                sin_pitch = np.sin(pitch_rot)
                cos_roll = np.cos(roll)
                sin_roll = np.sin(roll)
                Ry = np.array([[cos_yaw, 0, sin_yaw], [0, 1, 0], [-sin_yaw, 0, cos_yaw]])
                Rx = np.array([[1, 0, 0], [0, cos_pitch, -sin_pitch], [0, sin_pitch, cos_pitch]])
                Rz = np.array([[cos_roll, -sin_roll, 0], [sin_roll, cos_roll, 0], [0, 0, 1]])
                rotation_matrix = Rz @ Rx @ Ry
                transform_matrix[:3, :3] = rotation_matrix
            dev_model.apply_transform(transform_matrix)
            bounds = dev_model.bounds
            voxel_boxes.append((bounds[0][0], bounds[0][1], bounds[0][2], bounds[1][0], bounds[1][1], bounds[1][2]))
        except:
            pass
    return voxel_boxes

def create_3d_grid_state_matrix(placed, scene_config, pitch=0.15):
    min_x, max_x, min_y, max_y, min_z, max_z, plate_levels = parse_scene_bounds(scene_config)
    M = int(np.ceil((max_x - min_x) / pitch))
    N = int(np.ceil((max_y - min_y) / pitch))
    L = int(np.ceil((max_z - min_z) / pitch))
    state_matrix = np.ones((M, N, L), dtype=np.int8)
    obstacle_points = set()
    processed_devices = 0
    for device in placed:
        if device.get('mesh') is None:
            continue
        try:
            dev_model = device['mesh'].copy()
            center = device['center']
            pose = device.get('pose', {})
            transform_matrix = np.eye(4)
            transform_matrix[:3, 3] = center
            yaw = pose.get('yaw', 0.0)
            pitch_rot = pose.get('pitch', 0.0)
            roll = pose.get('roll', 0.0)
            if yaw != 0.0 or pitch_rot != 0.0 or roll != 0.0:
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                cos_pitch = np.cos(pitch_rot)
                sin_pitch = np.sin(pitch_rot)
                cos_roll = np.cos(roll)
                sin_roll = np.sin(roll)
                Ry = np.array([[cos_yaw, 0, sin_yaw], [0, 1, 0], [-sin_yaw, 0, cos_yaw]])
                Rx = np.array([[1, 0, 0], [0, cos_pitch, -sin_pitch], [0, sin_pitch, cos_pitch]])
                Rz = np.array([[cos_roll, -sin_roll, 0], [sin_roll, cos_roll, 0], [0, 0, 1]])
                rotation_matrix = Rz @ Rx @ Ry
                transform_matrix[:3, :3] = rotation_matrix
            dev_model.apply_transform(transform_matrix)
            voxels = dev_model.voxelized(pitch=pitch)
            filled_voxels = voxels.fill()
            for pt in filled_voxels.points:
                obstacle_points.add(snap(pt, pitch))
            processed_devices += 1
        except Exception as e:
            continue
    obstacle_count = 0
    for obstacle_pt in obstacle_points:
        x_idx = int(round((obstacle_pt[0] - min_x) / pitch))
        y_idx = int(round((obstacle_pt[1] - min_y) / pitch))
        z_idx = int(round((obstacle_pt[2] - min_z) / pitch))
        if 0 <= x_idx < M and 0 <= y_idx < N and (0 <= z_idx < L):
            if state_matrix[x_idx, y_idx, z_idx] == 1:
                state_matrix[x_idx, y_idx, z_idx] = 0
                obstacle_count += 1
    total_cells = M * N * L
    obstacle_cells = np.sum(state_matrix == 0)
    free_cells = np.sum(state_matrix == 1)
    grid_info = {'shape': (M, N, L), 'pitch': pitch, 'bounds': {'x': (min_x, max_x), 'y': (min_y, max_y), 'z': (min_z, max_z)}, 'plate_levels': plate_levels}
    return (state_matrix, grid_info)

def build_obstacles(placed, pitch=0.1):
    obstacles = set()
    for device in placed:
        if device.get('mesh') is None:
            continue
        try:
            dev_model = device['mesh'].copy()
            center = device['center']
            pose = device.get('pose', {})
            transform_matrix = np.eye(4)
            transform_matrix[:3, 3] = center
            yaw = pose.get('yaw', 0.0)
            pitch_rot = pose.get('pitch', 0.0)
            roll = pose.get('roll', 0.0)
            if yaw != 0.0 or pitch_rot != 0.0 or roll != 0.0:
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                cos_pitch = np.cos(pitch_rot)
                sin_pitch = np.sin(pitch_rot)
                cos_roll = np.cos(roll)
                sin_roll = np.sin(roll)
                Ry = np.array([[cos_yaw, 0, sin_yaw], [0, 1, 0], [-sin_yaw, 0, cos_yaw]])
                Rx = np.array([[1, 0, 0], [0, cos_pitch, -sin_pitch], [0, sin_pitch, cos_pitch]])
                Rz = np.array([[cos_roll, -sin_roll, 0], [sin_roll, cos_roll, 0], [0, 0, 1]])
                rotation_matrix = Rz @ Rx @ Ry
                transform_matrix[:3, :3] = rotation_matrix
            dev_model.apply_transform(transform_matrix)
            voxels = dev_model.voxelized(pitch=pitch)
            filled_voxels = voxels.fill()
            for pt in filled_voxels.points:
                obstacles.add(snap(pt, pitch))
        except Exception as e:
            continue
    return obstacles

def remove_port_obstacles(placed, device_obstacles, platform_obstacles, res):
    for device in placed:
        for port in device['ports']:
            port_pos = snap(port['world_pos'], res)
            device_obstacles.discard(port_pos)
            platform_obstacles.discard(port_pos)

def check_point_in_obstacles(point, state_matrix, grid_info, device_obstacles=None, debug=False):
    result = {'in_sdbb': False, 'in_voxel': False, 'grid_index': None, 'grid_value': None}
    if state_matrix is not None and grid_info is not None:
        M, N, L = grid_info['shape']
        pitch = grid_info['pitch']
        bounds = grid_info['bounds']
        i = int(round((point[0] - bounds['x'][0]) / pitch))
        j = int(round((point[1] - bounds['y'][0]) / pitch))
        k = int(round((point[2] - bounds['z'][0]) / pitch))
        result['grid_index'] = (i, j, k)
        if 0 <= i < M and 0 <= j < N and (0 <= k < L):
            grid_value = state_matrix[i, j, k]
            result['grid_value'] = int(grid_value)
            result['in_sdbb'] = grid_value == 0
            if debug:
                status = 'blocked' if result['in_sdbb'] else 'free'
                logger.debug("Point %s maps to grid index %s and is %s.", point, (i, j, k), status)
        elif debug:
            logger.debug("Point %s maps outside the grid at index %s.", point, (i, j, k))
    if device_obstacles is not None:
        result['in_voxel'] = tuple(point) in device_obstacles
        if debug and result['in_voxel']:
            logger.debug("Point %s is present in the device obstacle set.", point)
    return result

def clear_state_matrix_region(state_matrix, grid_info, point, radius, value=1):
    if state_matrix is None or grid_info is None:
        return
    pitch = grid_info['pitch']
    bounds = grid_info['bounds']
    radius_steps = max(1, int(np.ceil(radius / pitch)))
    base_i = int(round((point[0] - bounds['x'][0]) / pitch))
    base_j = int(round((point[1] - bounds['y'][0]) / pitch))
    base_k = int(round((point[2] - bounds['z'][0]) / pitch))
    M, N, L = state_matrix.shape
    for di in range(-radius_steps, radius_steps + 1):
        for dj in range(-radius_steps, radius_steps + 1):
            for dk in range(-radius_steps, radius_steps + 1):
                dist = np.sqrt((di * pitch) ** 2 + (dj * pitch) ** 2 + (dk * pitch) ** 2)
                if dist > radius:
                    continue
                ii = base_i + di
                jj = base_j + dj
                kk = base_k + dk
                if 0 <= ii < M and 0 <= jj < N and (0 <= kk < L):
                    state_matrix[ii, jj, kk] = value

def ensure_virtual_ports_clear_in_grid(placed, state_matrix, grid_info, grid_res, max_push=10):
    if state_matrix is None or grid_info is None:
        return 0
    adjustments = 0
    for device in placed:
        if '_TEE_' not in device.get('name', ''):
            continue
        for port in device.get('ports', []):
            pos = np.array(port['world_pos'], dtype=float)
            direction = np.array(port.get('direction', [1.0, 0.0, 0.0]), dtype=float)
            if np.linalg.norm(direction) < 1e-08:
                direction = np.array([1.0, 0.0, 0.0])
            direction = direction / np.linalg.norm(direction)
            check = check_point_in_obstacles(tuple(pos), state_matrix, grid_info, device_obstacles=None, debug=False)
            if check['in_sdbb'] or check['in_voxel']:
                moved = False
                for sign in [1.0, -1.0]:
                    dir_try = direction * sign
                    new_pos = np.array(pos, dtype=float)
                    for _ in range(max_push):
                        new_pos = new_pos + dir_try * grid_res
                        check = check_point_in_obstacles(tuple(new_pos), state_matrix, grid_info, device_obstacles=None, debug=False)
                        if not (check['in_sdbb'] or check['in_voxel']):
                            port['world_pos'] = new_pos.tolist()
                            moved = True
                            adjustments += 1
                            break
                    if moved:
                        break
                if not moved:
                    logger.debug("Virtual tee port could not be moved out of an occupied cell: %s", port.get('name'))
            pos = np.array(port['world_pos'], dtype=float)
            clear_state_matrix_region(state_matrix, grid_info, pos, radius=max(grid_res * 1.5, 0.15), value=1)
    if adjustments:
        logger.debug("Adjusted %s virtual tee ports to clear occupied cells.", adjustments)
    return adjustments

def world_to_grid(world_pos, grid_info):
    pitch = grid_info['pitch']
    bounds = grid_info['bounds']
    i = int(round((world_pos[0] - bounds['x'][0]) / pitch))
    j = int(round((world_pos[1] - bounds['y'][0]) / pitch))
    k = int(round((world_pos[2] - bounds['z'][0]) / pitch))
    return GridPoint(i, j, k)

def grid_to_world(grid_point, grid_info):
    pitch = grid_info['pitch']
    bounds = grid_info['bounds']
    x = bounds['x'][0] + (grid_point.x + 0.5) * pitch
    y = bounds['y'][0] + (grid_point.y + 0.5) * pitch
    z = bounds['z'][0] + (grid_point.z + 0.5) * pitch
    return (x, y, z)

def get_valid_manhattan_neighbors(point, state_matrix, grid_info):
    M, N, L = grid_info['shape']
    neighbors = []
    manhattan_moves = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for dx, dy, dz in manhattan_moves:
        neighbor = GridPoint(point.x + dx, point.y + dy, point.z + dz)
        if 0 <= neighbor.x < M and 0 <= neighbor.y < N and (0 <= neighbor.z < L):
            if state_matrix[neighbor.x, neighbor.y, neighbor.z] == 1:
                neighbors.append(neighbor)
    return neighbors

def calculate_manhattan_distance(p1, p2):
    return abs(p1.x - p2.x) + abs(p1.y - p2.y) + abs(p1.z - p2.z)

def calculate_min_distance_segment_to_obstacles(seg_start, seg_end, grid_info, grid_cell_size, obstacle_set):
    mid_point = GridPoint((seg_start.x + seg_end.x) // 2, (seg_start.y + seg_end.y) // 2, (seg_start.z + seg_end.z) // 2)
    world_coord = grid_to_world(mid_point, grid_info)
    min_dist = float('inf')
    for device_sdbbs in obstacle_set:
        for sdbb_box in device_sdbbs:
            box_min = np.array([sdbb_box[0], sdbb_box[1], sdbb_box[2]])
            box_max = np.array([sdbb_box[3], sdbb_box[4], sdbb_box[5]])
            box_center = (box_min + box_max) / 2
            dist = np.linalg.norm(np.array(world_coord) - box_center)
            min_dist = min(min_dist, dist)
    return min_dist

def voxelize_path(path, pipe_radius, pitch, safe_margin=0.0):
    voxel_points = set()
    if path is None or len(path) < 2:
        return voxel_points
    effective_radius = pipe_radius + safe_margin
    for i in range(len(path) - 1):
        start = np.array(path[i])
        end = np.array(path[i + 1])
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-08:
            voxel_points.add(snap(tuple(start), pitch))
            continue
        direction_normalized = direction / length
        if abs(direction_normalized[0]) < 0.9:
            perpendicular1 = np.array([1, 0, 0])
        else:
            perpendicular1 = np.array([0, 1, 0])
        perpendicular1 = perpendicular1 - np.dot(perpendicular1, direction_normalized) * direction_normalized
        perpendicular1 = perpendicular1 / np.linalg.norm(perpendicular1)
        perpendicular2 = np.cross(direction_normalized, perpendicular1)
        num_samples_along = max(2, int(np.ceil(length / pitch)) + 1)
        for j in range(num_samples_along):
            t = j / max(1, num_samples_along - 1)
            center_point = start + t * direction
            num_radial_layers = max(1, int(np.ceil(effective_radius / pitch)))
            for r_layer in range(num_radial_layers + 1):
                r = r_layer * pitch
                if r > effective_radius:
                    continue
                if r < 1e-08:
                    voxel_points.add(snap(tuple(center_point), pitch))
                else:
                    num_circumference = max(4, int(np.ceil(2 * np.pi * r / pitch)))
                    for k in range(num_circumference):
                        angle = 2 * np.pi * k / num_circumference
                        offset = r * (np.cos(angle) * perpendicular1 + np.sin(angle) * perpendicular2)
                        sample_point = center_point + offset
                        voxel_points.add(snap(tuple(sample_point), pitch))
    return voxel_points

def update_state_matrix_with_path(state_matrix, grid_info, path_voxels):
    if not path_voxels:
        return 0
    coords = np.asarray(list(path_voxels), dtype=np.float64)
    if coords.size == 0:
        return 0
    M, N, L = grid_info['shape']
    pitch = grid_info['pitch']
    bounds = grid_info['bounds']
    ix = np.rint((coords[:, 0] - bounds['x'][0]) / pitch).astype(np.int64)
    iy = np.rint((coords[:, 1] - bounds['y'][0]) / pitch).astype(np.int64)
    iz = np.rint((coords[:, 2] - bounds['z'][0]) / pitch).astype(np.int64)
    valid_mask = (ix >= 0) & (ix < M) & (iy >= 0) & (iy < N) & (iz >= 0) & (iz < L)
    if not np.any(valid_mask):
        return 0
    ix = ix[valid_mask]
    iy = iy[valid_mask]
    iz = iz[valid_mask]
    flat_idx = np.ravel_multi_index((ix, iy, iz), (M, N, L))
    flat_idx = np.unique(flat_idx)
    state_flat = state_matrix.reshape(-1)
    free_mask = state_flat[flat_idx] == 1
    if not np.any(free_mask):
        return 0
    state_flat[flat_idx[free_mask]] = 0
    return int(np.sum(free_mask))
