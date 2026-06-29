from __future__ import annotations
import json
import logging
import math
import os

from .glb_cache import load_glb_device_info
from .glb_reader import GLBReader

logger = logging.getLogger(__name__)

def load_layout_config(layout_file):
    try:
        with open(layout_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        return None

def load_device_from_glb(glb_path, device_config):
    device_info = load_glb_device_info(glb_path)
    if device_info is not None:
        device_info.update({'id': device_config['id'], 'name': device_config['name'], 'layer_name': device_config['layer_name'], 'row': device_config['row'], 'col': device_config['col'], 'pose': device_config['pose']})
        return device_info
    if not os.path.exists(glb_path):
        return None
    reader = GLBReader()
    if not reader.load_glb_file(glb_path):
        return None
    reader.extract_mesh_data()
    reader.extract_nodes_info()
    reader.extract_ports_info()
    device_info = reader.get_device_info()
    device_info.update({'id': device_config['id'], 'name': device_config['name'], 'layer_name': device_config['layer_name'], 'row': device_config['row'], 'col': device_config['col'], 'pose': device_config['pose']})
    return device_info

def create_placed_devices(config, glb_directory):
    placed_devices = []
    for device_config in config['devices']:
        glb_file = device_config['source_glb']
        glb_path = os.path.join(glb_directory, glb_file)
        device_info = load_device_from_glb(glb_path, device_config)
        if device_info is None:
            continue
        pose = device_config['pose']
        device_center = [pose['x'], pose['y'], pose['z']]
        world_ports = []
        for port in device_info['ports']:
            yaw = pose.get('yaw', 0.0)
            pitch = pose.get('pitch', 0.0)
            roll = pose.get('roll', 0.0)
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            cos_pitch = math.cos(pitch)
            sin_pitch = math.sin(pitch)
            cos_roll = math.cos(roll)
            sin_roll = math.sin(roll)
            rel_pos = port['relative_position']
            temp_x = rel_pos[0] * cos_yaw + rel_pos[2] * sin_yaw
            temp_y = rel_pos[1]
            temp_z = -rel_pos[0] * sin_yaw + rel_pos[2] * cos_yaw
            rotated_x = temp_x
            rotated_y = temp_y * cos_pitch - temp_z * sin_pitch
            rotated_z = temp_y * sin_pitch + temp_z * cos_pitch
            world_pos = [device_center[0] + rotated_x, device_center[1] + rotated_y, device_center[2] + rotated_z]
            direction = port['direction']
            temp_dir_x = direction[0] * cos_yaw + direction[2] * sin_yaw
            temp_dir_y = direction[1]
            temp_dir_z = -direction[0] * sin_yaw + direction[2] * cos_yaw
            rotated_dir_x = temp_dir_x
            rotated_dir_y = temp_dir_y * cos_pitch - temp_dir_z * sin_pitch
            rotated_dir_z = temp_dir_y * sin_pitch + temp_dir_z * cos_pitch
            world_direction = [rotated_dir_x, rotated_dir_y, rotated_dir_z]
            world_ports.append({'name': port['name'], 'relative_position': port['relative_position'], 'world_pos': world_pos, 'direction': world_direction})
        placed_device = {'id': device_config['id'], 'name': device_config['name'], 'center': device_center, 'size': device_info.get('bounds', [[0, 0, 0], [1, 1, 1]]), 'ports': world_ports, 'mesh': device_info.get('mesh'), 'layer_name': device_config['layer_name'], 'pose': pose}
        placed_devices.append(placed_device)
    return placed_devices

def parse_scene_bounds(scene_config):
    default_bounds = (0, 30, 0, 20, 0, 24)
    default_plate_levels = [0.0]
    try:
        if 'area' in scene_config:
            area = scene_config['area']
            min_x = scene_config.get('min_x', 0)
            max_x = min_x + area.get('x', 30)
            min_y = scene_config.get('min_y', 0)
            max_y = min_y + area.get('y', 30)
            min_z = scene_config.get('min_z', 0)
            max_z = min_z + area.get('z', 10)
            plate_levels = scene_config.get('plate_levels', default_plate_levels)
        elif 'scene_bounds' in scene_config:
            bounds = scene_config['scene_bounds']
            if isinstance(bounds, (list, tuple)) and len(bounds) == 3:
                min_x, max_x = (0, bounds[0])
                min_y, max_y = (0, bounds[1])
                min_z, max_z = (0, bounds[2])
            else:
                min_x, max_x, min_y, max_y, min_z, max_z = default_bounds
            plate_levels = scene_config.get('plate_levels', default_plate_levels)
        else:
            min_x, max_x, min_y, max_y, min_z, max_z = default_bounds
            plate_levels = default_plate_levels
        return (min_x, max_x, min_y, max_y, min_z, max_z, plate_levels)
    except Exception as e:
        return (*default_bounds, default_plate_levels)
