from __future__ import annotations
import glob
import json
import logging
import os
from pathlib import Path
import numpy as np
from .grid import build_sdbb_for_device
from .layout_io import parse_scene_bounds
try:
    import trimesh
except ImportError:
    trimesh = None
logger = logging.getLogger(__name__)

def visualize_sdbb_boxes(placed, scene_config, expand_margin=0.0, max_boxes=5, slice_method='adaptive', output_file='voxel_boxes_visualization.glb'):
    scene = trimesh.Scene()
    min_x, max_x, min_y, max_y, min_z, max_z, plate_levels = parse_scene_bounds(scene_config)
    total_boxes = 0
    device_count = 0
    colors = [[255, 0, 0, 120], [0, 255, 0, 120], [0, 0, 255, 120], [255, 255, 0, 120], [255, 0, 255, 120], [0, 255, 255, 120], [255, 128, 0, 120], [128, 0, 255, 120]]
    for device_idx, device in enumerate(placed):
        if device.get('mesh') is None:
            continue
        device_count += 1
        color = colors[device_idx % len(colors)]
        voxel_boxes = build_sdbb_for_device(device, expand_margin=expand_margin, max_boxes=5, slice_method='adaptive')
        if not voxel_boxes:
            continue
        for box_idx, simple_box in enumerate(voxel_boxes):
            box_min_x, box_min_y, box_min_z, box_max_x, box_max_y, box_max_z = simple_box
            min_pt = np.array([box_min_x, box_min_y, box_min_z])
            max_pt = np.array([box_max_x, box_max_y, box_max_z])
            size = max_pt - min_pt
            center = (min_pt + max_pt) / 2
            box = trimesh.creation.box(extents=size)
            box.apply_translation(center)
            alpha = 100 + device_idx * 5 % 50
            box.visual.face_colors = [color[0], color[1], color[2], alpha]
            scene.add_geometry(box, node_name=f'device_{device_idx}_box_{box_idx}')
            total_boxes += 1
    ground_size = max(max_x - min_x, max_y - min_y)
    ground = trimesh.creation.box(extents=[ground_size, ground_size, 0.1])
    ground.apply_translation([ground_size / 2, ground_size / 2, -0.05])
    ground.visual.face_colors = [200, 200, 200, 255]
    scene.add_geometry(ground, node_name='ground')
    axes = trimesh.creation.axis(origin_size=0.1, axis_length=2.0)
    scene.add_geometry(axes, node_name='axes')
    scene.export(output_file)
    return True

def visualize_state_matrix(state_matrix, grid_info, output_file='state_matrix_visualization.glb', sample_rate=10, show_free_space=False):
    scene = trimesh.Scene()
    M, N, L = grid_info['shape']
    pitch = grid_info['pitch']
    bounds = grid_info['bounds']
    obstacle_points = []
    free_points = []
    count = 0
    for i in range(M):
        for j in range(N):
            for k in range(L):
                x = bounds['x'][0] + (i + 0.5) * pitch
                y = bounds['y'][0] + (j + 0.5) * pitch
                z = bounds['z'][0] + (k + 0.5) * pitch
                if state_matrix[i, j, k] == 0:
                    if count % sample_rate == 0:
                        obstacle_points.append([x, y, z])
                    count += 1
                elif show_free_space:
                    free_points.append([x, y, z])
    if show_free_space:
        logger.debug("Collected %s sampled free-space voxels for visualization.", len(free_points))
    if len(obstacle_points) > 0:
        points = np.array(obstacle_points)
        point_cloud = trimesh.PointCloud(vertices=points)
        point_cloud.colors = [255, 0, 0, 255]
        scene.add_geometry(point_cloud, node_name='obstacle_points')
    if show_free_space and len(free_points) > 0:
        free_points_array = np.array(free_points)
        free_point_cloud = trimesh.PointCloud(vertices=free_points_array)
        free_point_cloud.colors = [0, 255, 0, 128]
        scene.add_geometry(free_point_cloud, node_name='free_space_points')
    area_x = bounds['x'][1] - bounds['x'][0]
    area_y = bounds['y'][1] - bounds['y'][0]
    ground = trimesh.creation.box(extents=[area_x, 0.2, area_y])
    ground.apply_translation([area_x / 2, -0.1, area_y / 2])
    ground.visual.face_colors = [200, 200, 200, 255]
    scene.add_geometry(ground, node_name='ground')
    x_arrow = trimesh.creation.cylinder(radius=0.03, height=2)
    x_arrow.apply_translation([1, 1, 1])
    x_arrow.visual.face_colors = [255, 0, 0, 255]
    scene.add_geometry(x_arrow, node_name='x_axis')
    y_arrow = trimesh.creation.cylinder(radius=0.03, height=2)
    y_arrow.apply_translation([1, 2, 1])
    y_arrow.visual.face_colors = [0, 255, 0, 255]
    scene.add_geometry(y_arrow, node_name='y_axis')
    z_arrow = trimesh.creation.cylinder(radius=0.03, height=2)
    z_arrow.apply_translation([1, 1, 2])
    z_arrow.visual.face_colors = [0, 0, 255, 255]
    scene.add_geometry(z_arrow, node_name='z_axis')
    try:
        scene.export(output_file)
        return True
    except Exception:
        logger.exception("Failed to export state-matrix visualization to %s.", output_file)
        return False

def visualize_grid_obstacles(placed, obstacles, scene_config, output_file='grid_obstacles_visualization.glb'):
    scene = trimesh.Scene()
    min_x, max_x, min_y, max_y, min_z, max_z, plate_levels = parse_scene_bounds(scene_config)
    obstacle_points = np.array(list(obstacles))
    sample_points = np.array(list(obstacles)[::10])
    if len(sample_points) > 0:
        points_cloud = trimesh.PointCloud(vertices=sample_points)
        points_cloud.colors = [255, 0, 0, 255]
        scene.add_geometry(points_cloud, node_name='obstacle_points_cloud')
    for device in placed:
        center = device['center']
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.15)
        sphere.apply_translation(center)
        sphere.visual.face_colors = [0, 255, 0, 255]
        scene.add_geometry(sphere, node_name=f"device_center_{device['id']}")
    area_x = max_x - min_x
    area_z = max_z - min_z
    ground = trimesh.creation.box(extents=[area_x, 0.2, area_z])
    ground.apply_translation([min_x + area_x / 2, -0.1, min_z + area_z / 2])
    ground.visual.face_colors = [200, 200, 200, 255]
    scene.add_geometry(ground, node_name='ground')
    try:
        scene.export(output_file)
        return True
    except Exception as e:
        return False

def generate_layout_only_scene(placed, scene_config):
    scene = trimesh.Scene()
    for device in placed:
        if device.get('mesh') is not None:
            device_mesh = device['mesh'].copy()
            center = device['center']
            pose = device.get('pose', {})
            transform_matrix = np.eye(4)
            transform_matrix[:3, 3] = center
            yaw = pose.get('yaw', 0.0)
            pitch = pose.get('pitch', 0.0)
            roll = pose.get('roll', 0.0)
            if yaw != 0.0 or pitch != 0.0 or roll != 0.0:
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                cos_pitch = np.cos(pitch)
                sin_pitch = np.sin(pitch)
                cos_roll = np.cos(roll)
                sin_roll = np.sin(roll)
                Ry = np.array([[cos_yaw, 0, sin_yaw], [0, 1, 0], [-sin_yaw, 0, cos_yaw]])
                Rx = np.array([[1, 0, 0], [0, cos_pitch, -sin_pitch], [0, sin_pitch, cos_pitch]])
                Rz = np.array([[cos_roll, -sin_roll, 0], [sin_roll, cos_roll, 0], [0, 0, 1]])
                rotation_matrix = Rz @ Rx @ Ry
                transform_matrix[:3, :3] = rotation_matrix
            device_mesh.apply_transform(transform_matrix)
            scene.add_geometry(device_mesh, node_name=f"device_{device['id']}")
    area_size = scene_config.get('area_size', [30, 30])
    ground_size = [area_size[0], 0.2, area_size[1]]
    ground_center = [area_size[0] / 2, -0.1, area_size[1] / 2]
    ground = trimesh.creation.box(extents=ground_size, transform=trimesh.transformations.translation_matrix(ground_center))
    ground.visual.face_colors = [128, 128, 128, 255]
    scene.add_geometry(ground, node_name='ground')
    output_file = 'layout_only_scene_deaco.glb'
    scene.export(output_file)

def find_glb_file(device_name, glb_directory):
    for filename in os.listdir(glb_directory):
        if not filename.endswith('.glb'):
            continue
        base_name = filename.replace('.glb', '')
        if base_name in device_name:
            glb_file = os.path.join(glb_directory, filename)
            return glb_file
    return None

def load_glb_model(glb_path):
    try:
        mesh = trimesh.load(glb_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            meshes = []
            for geometry in mesh.geometry.values():
                if isinstance(geometry, trimesh.Trimesh):
                    meshes.append(geometry)
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
            else:
                return None
        return mesh
    except Exception as e:
        return None

def create_device_mesh_from_info(device_info, glb_directory):
    glb_file = find_glb_file(device_info['name'], glb_directory)
    if glb_file is None:
        return None
    mesh = load_glb_model(glb_file)
    if mesh is None:
        return None
    center = device_info['center']
    pose = device_info['pose']
    transform_matrix = np.eye(4)
    yaw = pose.get('yaw', 0.0)
    pitch = pose.get('pitch', 0.0)
    roll = pose.get('roll', 0.0)
    if yaw != 0.0 or pitch != 0.0 or roll != 0.0:
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)
        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        Ry = np.array([[cos_yaw, 0, sin_yaw], [0, 1, 0], [-sin_yaw, 0, cos_yaw]])
        Rx = np.array([[1, 0, 0], [0, cos_pitch, -sin_pitch], [0, sin_pitch, cos_pitch]])
        Rz = np.array([[cos_roll, -sin_roll, 0], [sin_roll, cos_roll, 0], [0, 0, 1]])
        rotation_matrix = Rz @ Rx @ Ry
        transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = center
    mesh_copy = mesh.copy()
    mesh_copy.apply_transform(transform_matrix)
    return mesh_copy

def create_pipe_segment(start, end, radius=0.05):
    direction = np.array(end) - np.array(start)
    length = np.linalg.norm(direction)
    if length < 1e-06:
        return None
    cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
    direction_normalized = direction / length
    z_axis = np.array([0, 0, 1])
    if np.allclose(direction_normalized, z_axis):
        pass
    elif np.allclose(direction_normalized, -z_axis):
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        cylinder.apply_transform(rotation_matrix)
    else:
        rotation_axis = np.cross(z_axis, direction_normalized)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        if rotation_axis_norm > 1e-08:
            rotation_axis = rotation_axis / rotation_axis_norm
            cos_angle = np.dot(z_axis, direction_normalized)
            rotation_angle = np.arccos(np.clip(cos_angle, -1, 1))
            rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
            cylinder.apply_transform(rotation_matrix)
    center = (np.array(start) + np.array(end)) / 2
    translation = trimesh.transformations.translation_matrix(center)
    cylinder.apply_transform(translation)
    return cylinder

def create_ground_plane(size=50.0, thickness=0.2, y_position=-0.1):
    ground = trimesh.creation.box(extents=[size, thickness, size])
    ground.apply_translation([size / 2, y_position, size / 2])
    ground.visual.face_colors = [128, 128, 128, 255]
    return ground

def visualize_path_obstacles(state_matrix, grid_info, path_obstacles_list, output_file='path_obstacles_evolution.glb', show_devices=True, placed_devices=None):
    scene = trimesh.Scene()
    colors = [[255, 100, 100, 180], [100, 255, 100, 180], [100, 100, 255, 180], [255, 255, 100, 180], [255, 100, 255, 180], [100, 255, 255, 180], [255, 150, 100, 180], [150, 100, 255, 180]]
    for path_idx, (path_id, voxel_set) in enumerate(path_obstacles_list):
        if len(voxel_set) == 0:
            continue
        sample_rate = max(1, len(voxel_set) // 1000)
        sampled_points = list(voxel_set)[::sample_rate]
        if len(sampled_points) > 0:
            points_array = np.array(sampled_points)
            point_cloud = trimesh.PointCloud(vertices=points_array)
            color = colors[path_idx % len(colors)]
            point_cloud.colors = color
            scene.add_geometry(point_cloud, node_name=f'path_{path_id}_obstacles')
    if show_devices and placed_devices is not None:
        for device in placed_devices[:5]:
            if device.get('mesh') is not None:
                device_mesh = device['mesh'].copy()
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
                device_mesh.apply_transform(transform_matrix)
                device_mesh.visual.face_colors = [150, 150, 150, 100]
                scene.add_geometry(device_mesh, node_name=f"device_{device['id']}")
    try:
        scene.export(output_file)
        return True
    except Exception as e:
        return False

def simplify_path(waypoints):
    if len(waypoints) <= 2:
        return waypoints
    simplified = [waypoints[0]]
    if len(waypoints) >= 2:
        simplified.append(waypoints[1])
    start_idx = 2
    end_idx = len(waypoints) - 2
    for i in range(start_idx, end_idx):
        p0 = np.array(waypoints[i - 1])
        p1 = np.array(waypoints[i])
        p2 = np.array(waypoints[i + 1])
        dir1 = p1 - p0
        dir2 = p2 - p1
        norm1 = np.linalg.norm(dir1)
        norm2 = np.linalg.norm(dir2)
        if norm1 < 1e-06 or norm2 < 1e-06:
            continue
        dir1_norm = dir1 / norm1
        dir2_norm = dir2 / norm2
        dot_product = np.dot(dir1_norm, dir2_norm)
        if abs(dot_product) < 0.99:
            simplified.append(waypoints[i])
    if len(waypoints) >= 2:
        simplified.append(waypoints[-2])
    simplified.append(waypoints[-1])
    result = [simplified[0]]
    for i in range(1, len(simplified)):
        if not np.allclose(np.array(simplified[i]), np.array(result[-1]), atol=1e-06):
            result.append(simplified[i])
    return result

def show_scene_with_pipes(placed, scene_config, connections, paths, pipe_radius=0.05, adjusted_extensions=None):
    if adjusted_extensions is None:
        adjusted_extensions = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    glb_directory = os.path.join(script_dir, '..', 'static', 'glb')
    scene = trimesh.Scene()
    device_count = 0
    segment_count = 0
    for device in placed:
        if device.get('mesh') is not None:
            device_mesh = device['mesh'].copy()
            center = device['center']
            pose = device.get('pose', {})
            transform_matrix = np.eye(4)
            transform_matrix[:3, 3] = center
            yaw = pose.get('yaw', 0.0)
            pitch = pose.get('pitch', 0.0)
            roll = pose.get('roll', 0.0)
            if yaw != 0.0 or pitch != 0.0 or roll != 0.0:
                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)
                cos_pitch = np.cos(pitch)
                sin_pitch = np.sin(pitch)
                cos_roll = np.cos(roll)
                sin_roll = np.sin(roll)
                Ry = np.array([[cos_yaw, 0, sin_yaw], [0, 1, 0], [-sin_yaw, 0, cos_yaw]])
                Rx = np.array([[1, 0, 0], [0, cos_pitch, -sin_pitch], [0, sin_pitch, cos_pitch]])
                Rz = np.array([[cos_roll, -sin_roll, 0], [sin_roll, cos_roll, 0], [0, 0, 1]])
                rotation_matrix = Rz @ Rx @ Ry
                transform_matrix[:3, :3] = rotation_matrix
            device_mesh.apply_transform(transform_matrix)
            device_mesh.visual.face_colors = [100, 150, 255, 255]
            scene.add_geometry(device_mesh, node_name=f"device_{device['id']}_{device['name']}")
            device_count += 1
        else:
            logger.debug("Skipping device without mesh in visualization: %s.", device.get('name', device.get('id')))
    pipe_count = 0
    total_original_segments = 0
    total_simplified_segments = 0
    for path in paths:
        if len(path) < 2:
            continue
        simplified_path = simplify_path(path)
        original_segments = len(path) - 1
        simplified_segments = len(simplified_path) - 1
        total_original_segments += original_segments
        total_simplified_segments += simplified_segments
        for i in range(len(simplified_path) - 1):
            start_point = simplified_path[i]
            end_point = simplified_path[i + 1]
            pipe_segment = create_pipe_segment(start_point, end_point, radius=pipe_radius)
            if pipe_segment is not None:
                pipe_segment.visual.face_colors = [255, 215, 0, 255]
                scene.add_geometry(pipe_segment, node_name=f'pipe_{pipe_count}_{segment_count}')
                segment_count += 1
        pipe_count += 1
    if total_original_segments > 0:
        reduction_pct = (1 - total_simplified_segments / total_original_segments) * 100
    port_markers = 0
    extension_markers = 0

    def get_extension_distance_for_display(device_name):
        if device_name in adjusted_extensions:
            return adjusted_extensions[device_name]
        return 0.25
    target_device = None
    target_found = False
    for device in placed:
        device_name = device['name']
        extension_distance = get_extension_distance_for_display(device_name)
        if device_name == target_device:
            target_found = True
        for port in device['ports']:
            port_id = f"{device_name}.{port['name']}"
            port_pos = port['world_pos']
            port_sphere = trimesh.primitives.Sphere(radius=0.08, center=port_pos)
            port_sphere.visual.face_colors = [255, 50, 50, 200]
            scene.add_geometry(port_sphere, node_name=f'port_{port_id}')
            port_markers += 1
            if device_name == target_device:
                logger.debug("Added port marker for target device %s at %s.", device_name, port_pos)
            port_direction = np.array(port['direction'])
            if np.linalg.norm(port_direction) > 1e-08:
                normalized_direction = port_direction / np.linalg.norm(port_direction)
                extension_pos = np.array(port_pos) + normalized_direction * extension_distance
                ext_sphere = trimesh.primitives.Sphere(radius=0.06, center=extension_pos)
                ext_sphere.visual.face_colors = [50, 150, 255, 200]
                scene.add_geometry(ext_sphere, node_name=f'ext_{port_id}')
                extension_markers += 1
                if device_name == target_device:
                    logger.debug("Added extension marker for target device %s at %s.", device_name, extension_pos)
    if not target_found and target_device is not None:
        logger.debug("Target device %s was not found while building visualization.", target_device)
    min_x, max_x, min_y, max_y, min_z, max_z, plate_levels = parse_scene_bounds(scene_config)
    ground = create_ground_plane(size=max(max_x - min_x, max_z - min_z), thickness=0.2, y_position=-0.1)
    scene.add_geometry(ground, node_name='ground')
    output_file = 'rendered_scene_deaco.glb'
    try:
        scene.export(output_file)
        return True
    except Exception:
        logger.exception("Failed to export DEACO scene visualization to %s.", output_file)
        return False

def export_layout_info(placed, connections, paths, output_file='layout_info_deaco.json'):

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return obj
    layout_info = {'devices': [], 'connections': [], 'paths': [], 'statistics': {'device_count': len(placed), 'connection_count': len(connections), 'path_count': len(paths), 'total_pipe_segments': sum((len(path) - 1 for path in paths if len(path) > 1))}}
    for device in placed:
        device_info = {'id': device['id'], 'name': device['name'], 'center': convert_to_serializable(device['center']), 'pose': convert_to_serializable(device['pose']), 'layer_name': device['layer_name'], 'size': convert_to_serializable(device['size']), 'ports': []}
        for port in device['ports']:
            port_info = {'name': port['name'], 'relative_position': convert_to_serializable(port['relative_position']), 'world_position': convert_to_serializable(port['world_pos']), 'direction': convert_to_serializable(port['direction'])}
            device_info['ports'].append(port_info)
        layout_info['devices'].append(device_info)
    for i, connection in enumerate(connections):
        conn_info = {'id': i, 'from': connection['from'], 'to': connection['to'], 'from_position': convert_to_serializable(connection['from_pos']), 'to_position': convert_to_serializable(connection['to_pos']), 'from_direction': convert_to_serializable(connection['from_rota']), 'to_direction': convert_to_serializable(connection['to_rota']), 'distance': float(connection['distance'])}
        layout_info['connections'].append(conn_info)
    for i, path in enumerate(paths):
        path_info = {'id': i, 'connection_id': i if i < len(connections) else None, 'waypoints': convert_to_serializable(path), 'segment_count': len(path) - 1 if len(path) > 1 else 0, 'total_length': 0.0}
        if len(path) > 1:
            total_length = 0.0
            for j in range(len(path) - 1):
                p1 = np.array(path[j])
                p2 = np.array(path[j + 1])
                total_length += np.linalg.norm(p2 - p1)
            path_info['total_length'] = float(total_length)
        layout_info['paths'].append(path_info)
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(layout_info, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        return False
