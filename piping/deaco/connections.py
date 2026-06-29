from __future__ import annotations
import copy
import logging
from collections import defaultdict
import numpy as np
from .parameters import create_preset_params
from .types import snap
logger = logging.getLogger(__name__)

def infer_default_connections(placed):
    connections = []
    connected_ports = set()
    excluded_devices = []
    for device1 in placed:
        if device1['name'] in excluded_devices:
            continue
        for port1 in device1['ports']:
            port1_id = f"{device1['name']}.{port1['name']}"
            if port1_id in connected_ports:
                continue
            best_match = None
            best_distance = float('inf')
            for device2 in placed:
                if device1['id'] == device2['id']:
                    continue
                if device2['name'] in excluded_devices:
                    continue
                for port2 in device2['ports']:
                    port2_id = f"{device2['name']}.{port2['name']}"
                    if port2_id in connected_ports:
                        continue
                    pos1 = np.array(port1['world_pos'])
                    pos2 = np.array(port2['world_pos'])
                    distance = np.linalg.norm(pos2 - pos1)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = {'device': device2, 'port': port2, 'port_id': port2_id, 'distance': distance}
            if best_match is not None:
                connection = {'from': port1_id, 'to': best_match['port_id'], 'from_pos': port1['world_pos'], 'to_pos': best_match['port']['world_pos'], 'from_rota': port1['direction'], 'to_rota': best_match['port']['direction'], 'distance': best_match['distance']}
                connections.append(connection)
                connected_ports.add(port1_id)
                connected_ports.add(best_match['port_id'])
    unconnected_port_count = 0
    for device1 in placed:
        if device1['name'] in excluded_devices:
            continue
        for port1 in device1['ports']:
            port1_id = f"{device1['name']}.{port1['name']}"
            if port1_id in connected_ports:
                continue
            unconnected_port_count += 1
            best_match = None
            best_distance = float('inf')
            for device2 in placed:
                if device1['id'] == device2['id']:
                    continue
                if device2['name'] in excluded_devices:
                    continue
                for port2 in device2['ports']:
                    port2_id = f"{device2['name']}.{port2['name']}"
                    if port2_id in connected_ports:
                        continue
                    pos1 = np.array(port1['world_pos'])
                    pos2 = np.array(port2['world_pos'])
                    distance = np.linalg.norm(pos2 - pos1)
                    if distance < best_distance:
                        best_distance = distance
                        best_match = {'device': device2, 'port': port2, 'port_id': port2_id, 'distance': distance}
            if best_match is not None:
                connection = {'from': port1_id, 'to': best_match['port_id'], 'from_pos': port1['world_pos'], 'to_pos': best_match['port']['world_pos'], 'from_rota': port1['direction'], 'to_rota': best_match['port']['direction'], 'distance': best_match['distance']}
                connections.append(connection)
                connected_ports.add(port1_id)
                connected_ports.add(best_match['port_id'])
    total_ports = sum((len(d['ports']) for d in placed if d['name'] not in excluded_devices))
    unconnected = []
    for device in placed:
        if device['name'] in excluded_devices:
            continue
        for port in device['ports']:
            port_id = f"{device['name']}.{port['name']}"
            if port_id not in connected_ports:
                unconnected.append(port_id)
    if unconnected:
        logger.debug("Unconnected inferred ports: %s", unconnected)
    devices_without_ports = [d['name'] for d in placed if len(d.get('ports', [])) == 0]
    if devices_without_ports:
        logger.debug("Devices without ports: %s", devices_without_ports)
    return connections

def _connection_key(from_port, to_port):
    return f'{from_port}->{to_port}'

def _normalize_connection_overrides(raw_overrides):
    normalized = {}
    if not raw_overrides:
        return normalized
    if isinstance(raw_overrides, dict):
        for key, value in raw_overrides.items():
            if isinstance(value, dict):
                normalized[key] = value
    elif isinstance(raw_overrides, list):
        for item in raw_overrides:
            if not isinstance(item, dict):
                continue
            key = item.get('name')
            if not key:
                from_endpoint = item.get('from')
                to_endpoint = item.get('to')
                if from_endpoint and to_endpoint:
                    key = _connection_key(from_endpoint, to_endpoint)
            if key and isinstance(item.get('params', item), dict):
                normalized[key] = item.get('params', item)
    return normalized

def _apply_medium_presets(payload):
    if payload is None:
        return {}
    medium_type = payload.get('medium_type')
    pipe_condition = payload.get('pipe_condition')
    operation_mode = payload.get('operation_mode')
    if not (medium_type or pipe_condition or operation_mode):
        return payload
    preset = create_preset_params(medium_type=medium_type or 'water', pipe_condition=pipe_condition or 'new_steel', operation_mode=operation_mode or 'continuous', use_hybrid_normalization=False, verbose=False)
    fields_to_copy = ['flow_rate', 'fluid_density', 'dynamic_viscosity', 'pipe_diameter', 'pipe_roughness', 'darcy_friction', 'pipe_carbon_factor', 'elbow_carbon_factor']
    for field in fields_to_copy:
        payload.setdefault(field, getattr(preset, field))
    payload.pop('medium_type', None)
    payload.pop('pipe_condition', None)
    payload.pop('operation_mode', None)
    return payload

def attach_connection_parameter_overrides(connections, config):
    raw_overrides = config.get('connection_parameters', {})
    device_defaults = config.get('device_pipe_defaults', {})
    normalized_overrides = _normalize_connection_overrides(raw_overrides)
    normalized_device_defaults = device_defaults if isinstance(device_defaults, dict) else {}
    if not normalized_overrides and (not normalized_device_defaults):
        return connections
    for conn in connections:
        override_payload = {}
        from_device = conn['from'].split('.')[0]
        to_device = conn['to'].split('.')[0]
        if from_device in normalized_device_defaults:
            override_payload.update(normalized_device_defaults[from_device])
        if to_device in normalized_device_defaults:
            for key, value in normalized_device_defaults[to_device].items():
                override_payload.setdefault(key, value)
        conn_key = _connection_key(conn['from'], conn['to'])
        if conn_key in normalized_overrides:
            override_payload.update(normalized_overrides[conn_key])
        override_payload = _apply_medium_presets(override_payload)
        if override_payload:
            conn['param_override'] = override_payload
    return connections

def _build_port_lookup(placed):
    lookup = {}
    for device in placed:
        for port in device.get('ports', []):
            port_id = f"{device['name']}.{port['name']}"
            world_pos = np.array(port.get('world_pos', [0.0, 0.0, 0.0]), dtype=float)
            direction = np.array(port.get('direction', [1.0, 0.0, 0.0]), dtype=float)
            norm = np.linalg.norm(direction)
            if norm < 1e-06:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / norm
            lookup[port_id] = {'device': device, 'port': port, 'world_pos': world_pos.tolist(), 'direction': direction.tolist()}
    return lookup

def _refresh_connections_from_ports(connections, placed):
    port_lookup = _build_port_lookup(placed)
    for conn in connections:
        from_info = port_lookup.get(conn['from'])
        to_info = port_lookup.get(conn['to'])
        if from_info:
            conn['from_pos'] = from_info['world_pos']
            conn['from_rota'] = from_info['direction']
        if to_info:
            conn['to_pos'] = to_info['world_pos']
            conn['to_rota'] = to_info['direction']
        if from_info and to_info:
            conn['distance'] = float(np.linalg.norm(np.array(conn['to_pos']) - np.array(conn['from_pos'])))
    return connections

def build_tee_usage_map(connections):
    usage = {}
    for conn in connections:
        for endpoint in ('from', 'to'):
            port_name = conn.get(endpoint)
            if not port_name:
                continue
            if '_TEE_' in port_name.split('.')[0]:
                usage[port_name] = usage.get(port_name, 0) + 1
    return usage

def filter_voxels_near_ports(path_voxels, port_names, port_lookup, grid_res, keep_radius_factor=2.5):
    if not path_voxels or not port_names:
        return path_voxels
    positions = []
    for name in port_names:
        info = port_lookup.get(name)
        if info:
            positions.append(np.array(info['world_pos'], dtype=float))
    if not positions:
        return path_voxels
    keep_radius = max(grid_res * keep_radius_factor, grid_res * 2.0)
    keep_radius_sq = keep_radius ** 2
    filtered = set()
    for voxel in path_voxels:
        point = np.array(voxel, dtype=float)
        if any((np.sum((point - pos) ** 2) <= keep_radius_sq for pos in positions)):
            continue
        filtered.add(voxel)
    return filtered

def _normalize_vector(vec, fallback=None):
    norm = np.linalg.norm(vec)
    if norm < 1e-06:
        if fallback is not None:
            return np.array(fallback, dtype=float)
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vec / norm

def _create_connection_dict(from_port, to_port, port_lookup):
    if from_port not in port_lookup or to_port not in port_lookup:
        return None
    from_info = port_lookup[from_port]
    to_info = port_lookup[to_port]
    pos_from = np.array(from_info['world_pos'])
    pos_to = np.array(to_info['world_pos'])
    return {'from': from_port, 'to': to_port, 'from_pos': from_info['world_pos'], 'to_pos': to_info['world_pos'], 'from_rota': from_info['direction'], 'to_rota': to_info['direction'], 'distance': float(np.linalg.norm(pos_to - pos_from))}

def _update_connection_endpoint(conn, endpoint, port_name, port_lookup):
    info = port_lookup.get(port_name)
    if not info:
        return conn
    if endpoint == 'from':
        conn['from'] = port_name
        conn['from_pos'] = info['world_pos']
        conn['from_rota'] = info['direction']
    else:
        conn['to'] = port_name
        conn['to_pos'] = info['world_pos']
        conn['to_rota'] = info['direction']
    pos_from = np.array(conn['from_pos'])
    pos_to = np.array(conn['to_pos'])
    conn['distance'] = float(np.linalg.norm(pos_to - pos_from))
    return conn

def insert_virtual_tees(placed, connections, grid_res=0.1, tee_offset=0.2, tee_radius=0.05, scene_bounds=None):
    if not connections:
        return (placed, connections)
    placed = copy.deepcopy(placed)
    connections = [copy.deepcopy(conn) for conn in connections]
    tee_counter = 0
    handled_ports = set()

    def rebuild_lookup():
        return _build_port_lookup(placed)
    port_lookup = rebuild_lookup()
    min_offset = max(tee_offset, grid_res * 2.0)
    min_radius = max(tee_radius, grid_res * 0.75)
    grid_step = grid_res * 3.0
    bounds_info = None
    if scene_bounds is not None and len(scene_bounds) >= 6:
        min_x, max_x, min_y, max_y, min_z, max_z = scene_bounds[:6]

        def axis_margin(min_v, max_v):
            span = max(max_v - min_v, grid_res * 4.0)
            preferred = max(grid_res * 1.5, span * 0.02)
            return min(preferred, span * 0.45)
        bounds_info = {'x': {'min': min_x, 'max': max_x, 'margin': axis_margin(min_x, max_x)}, 'y': {'min': min_y, 'max': max_y, 'margin': axis_margin(min_y, max_y)}, 'z': {'min': min_z, 'max': max_z, 'margin': axis_margin(min_z, max_z)}}

    def clip_to_bounds(point):
        if bounds_info is None:
            return np.array(point, dtype=float)
        clipped = np.array(point, dtype=float)
        for axis, idx in zip(['x', 'y', 'z'], range(3)):
            info = bounds_info[axis]
            span = info['max'] - info['min']
            if span <= 1e-06:
                clipped[idx] = info['min']
                continue
            margin = min(info['margin'], span / 2.0 - 1e-06) if span > 2e-06 else 0.0
            low = info['min'] + max(margin, 0.0)
            high = info['max'] - max(margin, 0.0)
            if low > high:
                mid = (info['min'] + info['max']) / 2.0
                clipped[idx] = mid
            else:
                clipped[idx] = np.clip(clipped[idx], low, high)
        return clipped

    def is_point_in_device(point):
        pt = np.array(point, dtype=float)
        for device in placed:
            bounds = device.get('size')
            if bounds is None or len(bounds) < 2:
                continue
            min_corner = np.array(bounds[0], dtype=float)
            max_corner = np.array(bounds[1], dtype=float)
            if np.all(pt >= min_corner) and np.all(pt <= max_corner):
                return True
        return False

    def relax_point_from_obstacle(candidate, anchor_points, fallback_dir):
        adjusted = np.array(candidate, dtype=float)
        anchors = [np.array(point, dtype=float) for point in anchor_points or [] if point is not None]
        for _ in range(5):
            if not is_point_in_device(adjusted):
                break
            if anchors:
                anchor_sum = np.sum(anchors, axis=0)
                adjusted = (anchor_sum + adjusted) / (len(anchors) + 1.0)
            else:
                break
        else:
            direction = fallback_dir
            if np.linalg.norm(direction) < 1e-08:
                direction = np.array([1.0, 0.0, 0.0])
            direction = direction / np.linalg.norm(direction)
            adjusted = anchors[0] + direction * min_offset if anchors else adjusted + direction * min_offset
        reference = anchors[0] if anchors else adjusted
        adjusted = ensure_min_displacement(reference, adjusted, fallback_dir)
        return clip_to_bounds(adjusted)

    def push_out_of_devices(point, direction, step=None, max_steps=6):
        step_size = step or grid_res
        direction = np.array(direction, dtype=float)
        if np.linalg.norm(direction) < 1e-08:
            direction = np.array([1.0, 0.0, 0.0])
        direction = direction / np.linalg.norm(direction)
        adjusted = np.array(point, dtype=float)
        for _ in range(max_steps):
            if not is_point_in_device(adjusted):
                break
            adjusted = adjusted + direction * step_size
        return adjusted

    def blend_with_anchor_average(initial_point, anchor_points):
        points = [np.array(initial_point, dtype=float)]
        for anchor in anchor_points:
            if anchor is not None:
                points.append(np.array(anchor, dtype=float))
        if len(points) == 0:
            return np.array(initial_point, dtype=float)
        return np.mean(points, axis=0)

    def create_tee_device(base_name, center):
        nonlocal tee_counter
        tee_counter += 1
        tee_name = f'{base_name}_TEE_{tee_counter:03d}'
        center = np.array(center, dtype=float)
        size = [(center - 0.1).tolist(), (center + 0.1).tolist()]
        device = {'id': f'virtual_tee_{tee_counter}', 'name': tee_name, 'center': center.tolist(), 'size': size, 'ports': [], 'mesh': None, 'layer_name': 'virtual', 'pose': {'x': float(center[0]), 'y': float(center[1]), 'z': float(center[2]), 'yaw': 0.0, 'pitch': 0.0}}
        return device

    def add_port(device, name, position, direction):
        port = {'name': name, 'relative_position': [0.0, 0.0, 0.0], 'world_pos': position, 'direction': direction}
        device['ports'].append(port)

    def compute_centroid(port_names, base_point=None, base_weight=1.0, peer_weight=1.0, peer_weight_map=None):
        points = []
        weights = []
        if base_point is not None:
            points.append(np.array(base_point, dtype=float))
            weights.append(base_weight)
        for name in port_names:
            info = port_lookup.get(name)
            if not info:
                continue
            weight = peer_weight_map.get(name, peer_weight) if peer_weight_map else peer_weight
            points.append(np.array(info['world_pos'], dtype=float))
            weights.append(weight)
        count = len(points)
        if count == 0:
            return None
        weights = np.array(weights, dtype=float)
        weights = np.where(weights <= 0, 1.0, weights)
        weighted_sum = np.zeros(3, dtype=float)
        for pt, w in zip(points, weights):
            weighted_sum += pt * w
        centroid = weighted_sum / max(count, 1)
        return centroid

    def ensure_min_displacement(base_point, candidate_point, fallback_dir, min_distance=None):
        base_point = np.array(base_point, dtype=float)
        candidate_point = np.array(candidate_point, dtype=float)
        vec = candidate_point - base_point
        dist = np.linalg.norm(vec)
        required = min_distance if min_distance is not None else min_offset
        required = max(required, grid_res * 3.0, min_radius)
        if dist < max(required, 1e-06):
            direction = fallback_dir
            if np.linalg.norm(direction) < 1e-08:
                direction = np.array([1.0, 0.0, 0.0])
            direction = direction / np.linalg.norm(direction)
            return base_point + direction * required
        return candidate_point
    out_usage = defaultdict(list)
    for idx, conn in enumerate(connections):
        out_usage[conn['from']].append(idx)
    skip_indices = set()
    new_connections = []
    for port_name, idx_list in out_usage.items():
        if len(idx_list) <= 1:
            continue
        port_info = port_lookup.get(port_name)
        if not port_info:
            continue
        handled_ports.add(port_name)
        device_name = port_name.split('.')[0]
        base_pos = np.array(port_info['world_pos'])
        direction = _normalize_vector(np.array(port_info['direction']))
        target_ports = [connections[idx]['to'] for idx in idx_list]
        centroid = compute_centroid(target_ports, base_pos, base_weight=1.0, peer_weight=1.0)
        if centroid is not None:
            tee_center = centroid
        else:
            tee_center = base_pos + direction * min_offset
        tee_center = ensure_min_displacement(base_pos, tee_center, direction)
        anchor_points = [base_pos] + [port_lookup[target]['world_pos'] for target in target_ports if target in port_lookup]
        tee_center = blend_with_anchor_average(tee_center, anchor_points)
        tee_center = relax_point_from_obstacle(tee_center, anchor_points, direction)
        tee_center = clip_to_bounds(tee_center)
        tee_center = push_out_of_devices(tee_center, direction)
        tee_device = create_tee_device(device_name, tee_center)
        add_port(tee_device, 'tee_in', tee_center.tolist(), direction.tolist())
        tee_out_names = []
        for idx, conn_idx in enumerate(idx_list, start=1):
            target_port = connections[conn_idx]['to']
            target_info = port_lookup.get(target_port)
            if target_info:
                target_pos = np.array(target_info['world_pos'])
                out_dir = _normalize_vector(target_pos - tee_center, direction)
            else:
                out_dir = direction
                target_pos = tee_center + out_dir * min_offset
            if np.linalg.norm(out_dir) < 1e-06:
                out_dir = direction
            port_label = f'tee_out_{idx}'
            add_port(tee_device, port_label, tee_center.tolist(), out_dir.tolist())
            tee_out_names.append(f"{tee_device['name']}.{port_label}")
        placed.append(tee_device)
        port_lookup = rebuild_lookup()
        tee_in_full = f"{tee_device['name']}.tee_in"
        tee_connection = _create_connection_dict(port_name, tee_in_full, port_lookup)
        if tee_connection:
            new_connections.append(tee_connection)
        for mapped_port, conn_idx in zip(tee_out_names, idx_list):
            skip_indices.add(conn_idx)
            updated_conn = copy.deepcopy(connections[conn_idx])
            updated_conn = _update_connection_endpoint(updated_conn, 'from', mapped_port, port_lookup)
            new_connections.append(updated_conn)
    updated_connections = [copy.deepcopy(conn) for idx, conn in enumerate(connections) if idx not in skip_indices] + new_connections
    connections = updated_connections
    port_lookup = rebuild_lookup()
    in_usage = defaultdict(list)
    for idx, conn in enumerate(connections):
        in_usage[conn['to']].append(idx)
    skip_indices = set()
    new_connections = []
    for port_name, idx_list in in_usage.items():
        if len(idx_list) <= 1 or port_name in handled_ports:
            continue
        port_info = port_lookup.get(port_name)
        if not port_info:
            continue
        handled_ports.add(port_name)
        device_name = port_name.split('.')[0]
        base_pos = np.array(port_info['world_pos'])
        direction = _normalize_vector(np.array(port_info['direction']))
        source_ports = [connections[idx]['from'] for idx in idx_list]
        centroid = compute_centroid(source_ports, base_pos, base_weight=1.0, peer_weight=1.0)
        if centroid is not None:
            tee_center = base_pos + 0.4 * (centroid - base_pos)
        else:
            tee_center = base_pos - direction * min_offset
        tee_center = ensure_min_displacement(base_pos, tee_center, -direction)
        anchor_points = [base_pos] + [connections[idx]['from_pos'] for idx in idx_list if 'from_pos' in connections[idx]]
        tee_center = blend_with_anchor_average(tee_center, anchor_points)
        tee_center = relax_point_from_obstacle(tee_center, anchor_points, -direction)
        tee_center = clip_to_bounds(tee_center)
        tee_center = push_out_of_devices(tee_center, -direction)
        tee_device = create_tee_device(device_name, tee_center)
        to_base_vec = _normalize_vector(base_pos - tee_center, direction)
        add_port(tee_device, 'tee_out', tee_center.tolist(), to_base_vec.tolist())
        tee_in_names = []
        for idx, conn_idx in enumerate(idx_list, start=1):
            source_port = connections[conn_idx]['from']
            source_info = port_lookup.get(source_port)
            if source_info:
                source_pos = np.array(source_info['world_pos'])
                in_dir = _normalize_vector(tee_center - source_pos, -direction)
            else:
                in_dir = -direction
                source_pos = tee_center - in_dir * min_offset
            port_label = f'tee_in_{idx}'
            add_port(tee_device, port_label, tee_center.tolist(), (-in_dir).tolist())
            tee_in_names.append(f"{tee_device['name']}.{port_label}")
        placed.append(tee_device)
        port_lookup = rebuild_lookup()
        tee_out_full = f"{tee_device['name']}.tee_out"
        tee_to_target = _create_connection_dict(tee_out_full, port_name, port_lookup)
        if tee_to_target:
            new_connections.append(tee_to_target)
        for mapped_port, conn_idx in zip(tee_in_names, idx_list):
            skip_indices.add(conn_idx)
            updated_conn = copy.deepcopy(connections[conn_idx])
            updated_conn = _update_connection_endpoint(updated_conn, 'to', mapped_port, port_lookup)
            new_connections.append(updated_conn)
    updated_connections = [copy.deepcopy(conn) for idx, conn in enumerate(connections) if idx not in skip_indices] + new_connections
    return (placed, updated_connections)

def build_connections_from_config(connection_specs, placed):
    if not connection_specs:
        return []
    port_lookup = {}
    for device in placed:
        for port in device.get('ports', []):
            port_id = f"{device['name']}.{port['name']}"
            port_lookup[port_id] = (device, port)
    device_lookup_by_id = {device.get('id', i): device for i, device in enumerate(placed)}
    device_lookup_by_name = {device['name']: device for device in placed}
    connections = []
    for spec in connection_specs:
        if 'from' in spec and 'to' in spec:
            from_id = spec.get('from')
            to_id = spec.get('to')
            if not from_id or not to_id:
                continue
            if from_id not in port_lookup or to_id not in port_lookup:
                continue
            from_device, from_port = port_lookup[from_id]
            to_device, to_port = port_lookup[to_id]
        elif 'from_device' in spec and 'to_device' in spec:
            from_device_name = spec.get('from_device')
            to_device_name = spec.get('to_device')
            from_device_id = spec.get('from_device_id')
            to_device_id = spec.get('to_device_id')
            if from_device_id is not None and from_device_id in device_lookup_by_id:
                from_device = device_lookup_by_id[from_device_id]
            elif from_device_name in device_lookup_by_name:
                from_device = device_lookup_by_name[from_device_name]
            else:
                continue
            if to_device_id is not None and to_device_id in device_lookup_by_id:
                to_device = device_lookup_by_id[to_device_id]
            elif to_device_name in device_lookup_by_name:
                to_device = device_lookup_by_name[to_device_name]
            else:
                continue
            from_ports = from_device.get('ports', [])
            to_ports = to_device.get('ports', [])
            if not from_ports or not to_ports:
                continue
            min_dist = float('inf')
            best_from_port = None
            best_to_port = None
            for fp in from_ports:
                for tp in to_ports:
                    fp_pos = np.array(fp['world_pos'])
                    tp_pos = np.array(tp['world_pos'])
                    dist = np.linalg.norm(tp_pos - fp_pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_from_port = fp
                        best_to_port = tp
            if not best_from_port or not best_to_port:
                continue
            from_port = best_from_port
            to_port = best_to_port
            from_id = f"{from_device_name}.{from_port['name']}"
            to_id = f"{to_device_name}.{to_port['name']}"
        else:
            continue
        from_pos = from_port['world_pos']
        to_pos = to_port['world_pos']
        distance = np.linalg.norm(np.array(to_pos) - np.array(from_pos))
        connection = {'from': from_id, 'to': to_id, 'from_pos': from_pos, 'to_pos': to_pos, 'from_rota': from_port['direction'], 'to_rota': to_port['direction'], 'distance': distance}
        manual_params = spec.get('param_override') or spec.get('params')
        if manual_params:
            connection['param_override'] = _apply_medium_presets(manual_params)
        connections.append(connection)
    return connections
