from __future__ import annotations
import copy
import logging
import time
from typing import Any, Optional
import numpy as np
from .aco import run_deaco
from .connections import _build_port_lookup, _refresh_connections_from_ports, attach_connection_parameter_overrides, build_connections_from_config, build_tee_usage_map, filter_voxels_near_ports, infer_default_connections, insert_virtual_tees
from .fitness import calculate_path_fitness_green
from .grid import build_obstacles, build_sdbb_for_device, create_3d_grid_state_matrix, ensure_virtual_ports_clear_in_grid, remove_port_obstacles, update_state_matrix_with_path, voxelize_path, world_to_grid
from .layout_io import create_placed_devices, load_layout_config, parse_scene_bounds
from .parameters import DEACOParameters, clone_params_with_override, initialize_scene_normalization_ranges, validate_parameters
from .types import RoutingResult, snap
logger = logging.getLogger(__name__)

def try_multi_direction_extension(port_pos, original_direction, grid_info, state_matrix, grid_res, extension_distances=[0.4, 0.6]):
    M, N, L = grid_info['shape']
    abs_dir = np.abs(original_direction)
    main_axis = np.argmax(abs_dir)
    directions = [np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
    candidate_directions = []
    for dir_vec in directions:
        dot_product = np.abs(np.dot(original_direction / (np.linalg.norm(original_direction) + 1e-09), dir_vec))
        if dot_product < 0.5:
            candidate_directions.append(dir_vec)
    if len(candidate_directions) == 0:
        candidate_directions = []
        for axis in [0, 1, 2]:
            if axis != main_axis:
                candidate_directions.append(np.array([1.0 if i == axis else 0.0 for i in range(3)]))
                candidate_directions.append(np.array([-1.0 if i == axis else 0.0 for i in range(3)]))
    for alt_dir in candidate_directions:
        for ext_dist in extension_distances:
            step_point = np.array(port_pos) + alt_dir * ext_dist
            snapped_ext_point = snap(tuple(step_point), grid_res)
            ext_i = int(round((snapped_ext_point[0] - grid_info['bounds']['x'][0]) / grid_res))
            ext_j = int(round((snapped_ext_point[1] - grid_info['bounds']['y'][0]) / grid_res))
            ext_k = int(round((snapped_ext_point[2] - grid_info['bounds']['z'][0]) / grid_res))
            if 0 <= ext_i < M and 0 <= ext_j < N and (0 <= ext_k < L) and (state_matrix[ext_i, ext_j, ext_k] == 1):
                return (True, snapped_ext_point)
    return (False, None)

def process_single_connection_deaco(args):
    if len(args) >= 10:
        conn_idx, conn, grid_res, grid_info, grid_cell_size, state_matrix, obstacle_set, deaco_params, adjusted_extensions, use_green = args
    else:
        conn_idx, conn, grid_res, grid_info, grid_cell_size, state_matrix, obstacle_set, deaco_params, adjusted_extensions = args
        use_green = False
    connection_name = f"{conn['from']} -> {conn['to']}"
    try:
        start = snap(conn['from_pos'], grid_res)
        goal = snap(conn['to_pos'], grid_res)
        start_path = []
        goal_path = []

        def get_extension_distance(connection_endpoint):
            device_name = connection_endpoint.split('.')[0]
            if device_name in adjusted_extensions:
                return adjusted_extensions[device_name]
            return 0.25

        def is_virtual_tee(endpoint_name):
            return '_TEE_' in endpoint_name.split('.')[0]

        def ensure_endpoint_clear(endpoint_name, world_pos):
            if is_virtual_tee(endpoint_name):
                clear_state_matrix_region(state_matrix, grid_info, world_pos, radius=max(grid_res * 1.5, 0.15), value=1)
        ensure_endpoint_clear(conn['from'], start)
        ensure_endpoint_clear(conn['to'], goal)
        start_direction = np.array(conn['from_rota'])
        skip_start_extension = is_virtual_tee(conn['from'])
        if skip_start_extension or np.linalg.norm(start_direction) <= 1e-08:
            start_path = [start]
        else:
            normalized_start_direction = start_direction / np.linalg.norm(start_direction)
            start_path = [start]
            base_extension = get_extension_distance(conn['from'])
            extension_distance = base_extension
            max_attempts = 10
            M, N, L = grid_info['shape']
            extension_success = False
            for attempt in range(max_attempts):
                step_point = np.array(start) + normalized_start_direction * extension_distance
                snapped_ext_point = snap(tuple(step_point), grid_res)
                ext_i = int(round((snapped_ext_point[0] - grid_info['bounds']['x'][0]) / grid_res))
                ext_j = int(round((snapped_ext_point[1] - grid_info['bounds']['y'][0]) / grid_res))
                ext_k = int(round((snapped_ext_point[2] - grid_info['bounds']['z'][0]) / grid_res))
                if 0 <= ext_i < M and 0 <= ext_j < N and (0 <= ext_k < L) and (state_matrix[ext_i, ext_j, ext_k] == 1):
                    start_path.append(snapped_ext_point)
                    extension_success = True
                    break
                else:
                    extension_distance += 0.1
            if not extension_success:
                alt_success, alt_ext_point = try_multi_direction_extension(start, normalized_start_direction, grid_info, state_matrix, grid_res, extension_distances=[0.4, 0.6])
                if alt_success:
                    start_path.append(alt_ext_point)
                else:
                    return (conn_idx, None, False, connection_name, False)
            start = start_path[-1]
        goal_direction = np.array(conn['to_rota'])
        skip_goal_extension = is_virtual_tee(conn['to'])
        if skip_goal_extension or np.linalg.norm(goal_direction) <= 1e-08:
            goal_path = [goal]
        else:
            normalized_goal_direction = goal_direction / np.linalg.norm(goal_direction)
            goal_path = [goal]
            base_extension = get_extension_distance(conn['to'])
            extension_distance = base_extension
            max_attempts = 10
            M, N, L = grid_info['shape']
            extension_success = False
            for attempt in range(max_attempts):
                step_point = np.array(goal) + normalized_goal_direction * extension_distance
                snapped_ext_point = snap(tuple(step_point), grid_res)
                ext_i = int(round((snapped_ext_point[0] - grid_info['bounds']['x'][0]) / grid_res))
                ext_j = int(round((snapped_ext_point[1] - grid_info['bounds']['y'][0]) / grid_res))
                ext_k = int(round((snapped_ext_point[2] - grid_info['bounds']['z'][0]) / grid_res))
                if 0 <= ext_i < M and 0 <= ext_j < N and (0 <= ext_k < L) and (state_matrix[ext_i, ext_j, ext_k] == 1):
                    goal_path.append(snapped_ext_point)
                    extension_success = True
                    break
                else:
                    extension_distance += 0.1
            if not extension_success:
                alt_success, alt_ext_point = try_multi_direction_extension(goal, normalized_goal_direction, grid_info, state_matrix, grid_res, extension_distances=[0.4, 0.6])
                if alt_success:
                    goal_path.append(alt_ext_point)
                else:
                    return (conn_idx, None, False, connection_name, False)
            goal = goal_path[-1]
        goal_path.reverse()
        effective_params = clone_params_with_override(deaco_params, conn.get('param_override'), context=connection_name)
        result = run_deaco(start_world=start, goal_world=goal, obstacle_set=obstacle_set, grid_info=grid_info, grid_cell_size=grid_cell_size, state_matrix=state_matrix, params=effective_params, use_green=use_green)
        path_world, fitness = result
        path = path_world if path_world is not None else None
        if use_green and isinstance(fitness, FitnessData):
            fitness_value = fitness.J_total
        else:
            fitness_value = fitness if isinstance(fitness, (int, float)) else float('inf')
        if path and len(path) > 1:
            path_end = path[-1]
            goal_ext_start = goal_path[0] if len(goal_path) > 0 else None
            middle_path = path[1:] if len(start_path) > 0 else path
            tolerance = grid_res * 0.6
            if goal_ext_start is not None and np.allclose(path_end, goal_ext_start, atol=tolerance):
                end_path = goal_path[1:]
            elif goal_ext_start is not None:
                p1 = np.array(path_end)
                p2 = np.array(goal_ext_start)
                diff = p2 - p1
                axes_changed = []
                axis_threshold = grid_res * 0.6
                for axis_idx, (axis_name, delta) in enumerate(zip(['X', 'Y', 'Z'], diff)):
                    if abs(delta) > axis_threshold:
                        axes_changed.append((axis_idx, axis_name, delta))
                middle_conn_points = []
                if len(axes_changed) == 2:
                    if any((axis_name == 'Z' for _, axis_name, _ in axes_changed)):
                        intermediate = p1.copy()
                        intermediate[2] = p2[2]
                        middle_conn_points = [tuple(intermediate)]
                    else:
                        intermediate = p1.copy()
                        intermediate[1] = p2[1]
                        middle_conn_points = [tuple(intermediate)]
                elif len(axes_changed) == 3:
                    intermediate1 = p1.copy()
                    intermediate1[2] = p2[2]
                    intermediate2 = intermediate1.copy()
                    intermediate2[1] = p2[1]
                    middle_conn_points = [tuple(intermediate1), tuple(intermediate2)]
                end_path = middle_conn_points + goal_path
            else:
                end_path = goal_path
            full_path = start_path + middle_path + end_path
            manhattan_tolerance = grid_res * 0.6
            full_manhattan_result = validate_manhattan_path(full_path, tolerance=manhattan_tolerance)
            is_manhattan = full_manhattan_result['is_valid']
            return (conn_idx, full_path, True, connection_name, is_manhattan)
        else:
            return (conn_idx, None, False, connection_name, False)
    except Exception as e:
        return (conn_idx, None, False, connection_name, False)

def validate_manhattan_path(path, tolerance=1e-06):
    if not path or len(path) < 2:
        return {'is_valid': True, 'total_segments': 0, 'invalid_segments': [], 'invalid_count': 0}
    invalid_segments = []
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        diff = p2 - p1
        changes = np.abs(diff) > tolerance
        num_changes = np.sum(changes)
        if num_changes != 1:
            invalid_segments.append({'segment_index': i, 'from': path[i], 'to': path[i + 1], 'diff': diff.tolist(), 'num_changes': int(num_changes), 'changed_axes': [axis_name for axis_name, changed in zip(['X', 'Y', 'Z'], changes) if changed]})
    return {'is_valid': len(invalid_segments) == 0, 'total_segments': len(path) - 1, 'invalid_segments': invalid_segments, 'invalid_count': len(invalid_segments)}

def route_connection(
    connection_idx: int,
    connection: dict[str, Any],
    grid_res: float,
    grid_info: dict[str, Any],
    grid_cell_size,
    state_matrix,
    obstacle_set,
    params: DEACOParameters,
    adjusted_extensions: dict[str, float] | None = None,
    use_green: bool = True,
) -> RoutingResult:
    start_time = time.time()
    result = process_single_connection_deaco((
        connection_idx,
        connection,
        grid_res,
        grid_info,
        grid_cell_size,
        state_matrix,
        obstacle_set,
        params,
        adjusted_extensions or {},
        use_green,
    ))
    idx, full_path, success, connection_name, is_manhattan = result
    fitness = float("inf")
    fitness_data = None
    if success and full_path is not None:
        path_grid = [world_to_grid(point, grid_info) for point in full_path]
        fitness_data = calculate_path_fitness_green(
            path_grid,
            grid_info,
            grid_cell_size,
            obstacle_set,
            params,
            connection["from_pos"],
            connection["to_pos"],
        )
        fitness = float(fitness_data.J_total)
    return RoutingResult(
        success=bool(success),
        path=full_path,
        fitness=fitness,
        fitness_data=fitness_data,
        elapsed_time=time.time() - start_time,
        diagnostics={
            "connection_idx": int(idx),
            "connection_name": connection_name,
            "is_manhattan": bool(is_manhattan),
        },
    )

def route_scene(scenario_path: str, glb_directory: str, deaco_params: DEACOParameters | None=None, grid_res: float=0.1, use_green: bool=True, save_layouts: bool=False, output_dir: str | None=None, scenario_name: str='') -> dict[str, Any]:
    start_time = time.time()
    config = load_layout_config(scenario_path)
    placed = create_placed_devices(config, glb_directory)
    if config.get('connections'):
        connections = build_connections_from_config(config['connections'], placed)
    else:
        connections = infer_default_connections(placed)
    connections = attach_connection_parameter_overrides(connections, config)
    scene_bounds_tuple = parse_scene_bounds(config['scene'])
    scene_bounds = scene_bounds_tuple[:6]
    placed, connections = insert_virtual_tees(placed, connections, grid_res=grid_res, scene_bounds=scene_bounds)
    state_matrix, grid_info = create_3d_grid_state_matrix(placed, config['scene'], pitch=grid_res)
    adjusted_ports = ensure_virtual_ports_clear_in_grid(placed, state_matrix, grid_info, grid_res)
    if adjusted_ports:
        connections = _refresh_connections_from_ports(connections, placed)
    port_lookup = _build_port_lookup(placed)
    tee_usage_remaining = dict(build_tee_usage_map(connections))
    obstacle_set = []
    for device in placed:
        voxel_boxes = build_sdbb_for_device(device, expand_margin=0.0, max_boxes=5, slice_method='adaptive')
        if voxel_boxes:
            obstacle_set.append(voxel_boxes)
    grid_cell_size = (grid_res, grid_res, grid_res)
    params = deaco_params or DEACOParameters()
    initialize_scene_normalization_ranges(params, grid_info, connections)
    success_count = 0
    connection_results = []
    paths = []
    metrics_ready_records = []
    adjusted_extensions: dict[str, float] = {}
    for conn_idx, conn in enumerate(connections):
        connection_name = f"{conn['from']} -> {conn['to']}"
        effective_params = clone_params_with_override(params, conn.get('param_override'), context=connection_name)
        route_start = time.time()
        result = process_single_connection_deaco((conn_idx, conn, grid_res, grid_info, grid_cell_size, state_matrix, obstacle_set, effective_params, adjusted_extensions, use_green))
        _idx, full_path, success, conn_name, is_manhattan = result
        elapsed = time.time() - route_start
        record = {'connection_idx': conn_idx, 'connection_name': conn_name, 'success': bool(success), 'path': full_path, 'elapsed_time': elapsed, 'is_manhattan': bool(is_manhattan)}
        if success and full_path is not None:
            success_count += 1
            paths.append(full_path)
            path_grid = [world_to_grid(point, grid_info) for point in full_path]
            fitness_data = calculate_path_fitness_green(path_grid, grid_info, grid_cell_size, obstacle_set, effective_params, conn['from_pos'], conn['to_pos'])
            record.update({'fitness': float(fitness_data.J_total), 'fitness_data': fitness_data, 'path_length': len(full_path), 'E_op': float(fitness_data.E_op), 'CO2_op': float(fitness_data.CO2_op), 'CO2_emb': float(fitness_data.CO2_emb), 'N_bend': int(fitness_data.N_bend), 'alt': float(fitness_data.alt), 'viol': float(fitness_data.viol)})
            metrics_ready_records.append(record)
            path_voxels = voxelize_path(full_path, 0.05, grid_res, safe_margin=0.05)
            ports_to_keep_free = []
            for endpoint in ('from', 'to'):
                port_name = conn.get(endpoint)
                if tee_usage_remaining.get(port_name, 0) > 1:
                    ports_to_keep_free.append(port_name)
            if ports_to_keep_free and path_voxels:
                path_voxels = filter_voxels_near_ports(path_voxels, ports_to_keep_free, port_lookup, grid_res)
            if path_voxels:
                update_state_matrix_with_path(state_matrix, grid_info, path_voxels)
            for endpoint in ('from', 'to'):
                port_name = conn.get(endpoint)
                if port_name in tee_usage_remaining and tee_usage_remaining[port_name] > 0:
                    tee_usage_remaining[port_name] -= 1
        connection_results.append(record)
    return {'success': True, 'scenario_name': scenario_name, 'connections': connections, 'connection_results': connection_results, 'success_count': success_count, 'failed_count': len(connections) - success_count, 'total_connections': len(connections), 'total_time': time.time() - start_time, 'paths': paths, 'metrics_ready_records': metrics_ready_records, 'placed_devices': placed, 'config': config}
