from __future__ import annotations
import logging
import math
import numpy as np
from .parameters import normalize_value
from .types import FitnessData, GridPoint
from .grid import calculate_manhattan_distance, grid_to_world, calculate_min_distance_segment_to_obstacles
logger = logging.getLogger(__name__)

def calculate_reynolds_number(flow_rate, pipe_diameter, fluid_density, dynamic_viscosity):
    if dynamic_viscosity < 1e-09:
        return float('inf')
    Re = 4 * fluid_density * flow_rate / (np.pi * pipe_diameter * dynamic_viscosity + 1e-09)
    return Re

def calculate_swamee_jain_friction(pipe_diameter, pipe_roughness, reynolds_number):
    if reynolds_number < 1e-06:
        return 0.064
    relative_roughness = pipe_roughness / (pipe_diameter + 1e-09)
    log_arg = relative_roughness / 3.7 + 5.74 / reynolds_number ** 0.9
    log_arg = max(log_arg, 1e-10)
    f = 0.25 / np.log10(log_arg) ** 2
    f = np.clip(f, 0.008, 0.1)
    return f

def calculate_darcy_friction_factor(params, flow_rate=None, pipe_diameter=None):
    if params.darcy_friction is not None:
        return params.darcy_friction
    Q = flow_rate if flow_rate is not None else params.flow_rate
    D = pipe_diameter if pipe_diameter is not None else params.pipe_diameter
    Re = calculate_reynolds_number(Q, D, params.fluid_density, params.dynamic_viscosity)
    f = calculate_swamee_jain_friction(D, params.pipe_roughness, Re)
    return f

def calculate_path_fitness(path_grid, grid_info, grid_cell_size, obstacle_set, params):
    if path_grid is None or len(path_grid) < 2:
        return float('inf')
    f_Length = len(path_grid) - 1
    f_Bends = 0
    control_points = [path_grid[0]]
    for i in range(1, len(path_grid) - 1):
        p_prev = path_grid[i - 1]
        p_curr = path_grid[i]
        p_next = path_grid[i + 1]
        vec1 = np.array([p_curr.x - p_prev.x, p_curr.y - p_prev.y, p_curr.z - p_prev.z])
        vec2 = np.array([p_next.x - p_curr.x, p_next.y - p_curr.y, p_next.z - p_curr.z])
        if np.dot(vec1, vec2) == 0:
            f_Bends += 1
            control_points.append(p_curr)
    control_points.append(path_grid[-1])
    f_Energy = 0
    M, N, L = grid_info['shape']
    dx, dy, dz = grid_cell_size
    world_max_x = M * dx
    world_max_y = N * dy
    world_max_z = L * dz
    for p in path_grid:
        world_coord = grid_to_world(p, grid_info)
        energy_p = min(min(world_coord[0], world_max_x - world_coord[0]), min(world_coord[1], world_max_y - world_coord[1]), min(world_coord[2], world_max_z - world_coord[2]))
        f_Energy += energy_p
    f_Install = 0
    for j in range(len(control_points) - 1):
        seg_start = control_points[j]
        seg_end = control_points[j + 1]
        min_dist = calculate_min_distance_segment_to_obstacles(seg_start, seg_end, grid_info, grid_cell_size, obstacle_set)
        f_Install += min_dist
    f_Height = 0
    total_height_change = 0
    total_height = 0
    for i in range(1, len(path_grid)):
        prev_p = path_grid[i - 1]
        curr_p = path_grid[i]
        total_height_change += abs(curr_p.y - prev_p.y)
        world_coord = grid_to_world(curr_p, grid_info)
        total_height += world_coord[1]
    avg_height = total_height / len(path_grid) if len(path_grid) > 0 else 0
    f_Height = total_height_change + avg_height * 0.5
    fitness = params.omega_Length * f_Length + params.omega_Bend * f_Bends + params.omega_Energy * f_Energy + params.omega_Install * f_Install + params.omega_height_penalty * f_Height
    return fitness

def calculate_clearance_violation(path_grid, grid_info, obstacle_set, min_clearance=0.2):
    if path_grid is None or len(path_grid) < 2:
        return 0.0
    violation_sum = 0.0
    pitch = grid_info['pitch']
    control_points = [path_grid[0]]
    for i in range(1, len(path_grid) - 1):
        p_prev = path_grid[i - 1]
        p_curr = path_grid[i]
        p_next = path_grid[i + 1]
        vec1 = np.array([p_curr.x - p_prev.x, p_curr.y - p_prev.y, p_curr.z - p_prev.z])
        vec2 = np.array([p_next.x - p_curr.x, p_next.y - p_curr.y, p_next.z - p_curr.z])
        if np.dot(vec1, vec2) == 0:
            control_points.append(p_curr)
    control_points.append(path_grid[-1])
    for cp in control_points:
        world_coord = grid_to_world(cp, grid_info)
        min_dist = float('inf')
        for device_sdbbs in obstacle_set:
            for sdbb_box in device_sdbbs:
                box_min = np.array([sdbb_box[0], sdbb_box[1], sdbb_box[2]])
                box_max = np.array([sdbb_box[3], sdbb_box[4], sdbb_box[5]])
                closest_point = np.clip(world_coord, box_min, box_max)
                dist = np.linalg.norm(np.array(world_coord) - closest_point)
                min_dist = min(min_dist, dist)
        if min_dist < min_clearance:
            violation_sum += (min_clearance - min_dist) ** 2
    return violation_sum

def calculate_path_fitness_green(path_grid, grid_info, grid_cell_size, obstacle_set, params, start_world, goal_world):
    if path_grid is None or len(path_grid) < 2:
        return FitnessData(J_total=float('inf'), E_op=float('inf'), CO2_op=float('inf'), CO2_emb=float('inf'), L=0, N_bend=0, viol=float('inf'), alt=0.0, f_Energy=float('inf'), f_Install=float('inf'), f_Height=float('inf'))
    unique_path_grid = [path_grid[0]]
    for i in range(1, len(path_grid)):
        curr_point = path_grid[i]
        prev_point = unique_path_grid[-1]
        if not (curr_point.x == prev_point.x and curr_point.y == prev_point.y and (curr_point.z == prev_point.z)):
            unique_path_grid.append(curr_point)
    L = len(unique_path_grid) - 1
    N_bend = 0
    control_points = [unique_path_grid[0]]
    for i in range(1, len(unique_path_grid) - 1):
        p_prev = unique_path_grid[i - 1]
        p_curr = unique_path_grid[i]
        p_next = unique_path_grid[i + 1]
        vec1 = np.array([p_curr.x - p_prev.x, p_curr.y - p_prev.y, p_curr.z - p_prev.z])
        vec2 = np.array([p_next.x - p_curr.x, p_next.y - p_curr.y, p_next.z - p_curr.z])
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        if vec1_norm > 1e-10 and vec2_norm > 1e-10:
            vec1_normalized = vec1 / vec1_norm
            vec2_normalized = vec2 / vec2_norm
            dot_product = np.dot(vec1_normalized, vec2_normalized)
            if abs(dot_product) < 1e-06:
                N_bend += 1
                control_points.append(p_curr)
    control_points.append(unique_path_grid[-1])
    pitch = grid_info['pitch']
    L_physical = L * pitch
    dy = grid_cell_size[1] if len(grid_cell_size) > 1 else pitch
    total_positive_lift = 0.0
    positive_height_sum = 0.0
    positive_height_samples = 0
    for i in range(1, len(path_grid)):
        prev_p = path_grid[i - 1]
        curr_p = path_grid[i]
        delta_cells = curr_p.y - prev_p.y
        if delta_cells > 0:
            delta_m = delta_cells * dy
            total_positive_lift += delta_m
            world_coord = grid_to_world(curr_p, grid_info)
            positive_height_sum += world_coord[1]
            positive_height_samples += 1
    pipe_area = 0.25 * np.pi * params.pipe_diameter ** 2
    V = params.flow_rate / (pipe_area + 1e-09)
    V_sq_2g = V ** 2 / (2 * params.gravity)
    f = calculate_darcy_friction_factor(params)
    h_friction = f * (L_physical / params.pipe_diameter) * V_sq_2g
    h_local = N_bend * params.bend_loss_K * V_sq_2g
    h_tot = h_friction + h_local + total_positive_lift
    delta_p = params.fluid_density * params.gravity * h_tot
    P_pump = delta_p * params.flow_rate / (params.pump_efficiency + 1e-09)
    E_op = P_pump * params.annual_hours / 1000.0
    CO2_op = E_op * params.grid_carbon_intensity
    reference_diameter = max(params.reference_pipe_diameter, params.epsilon)
    diameter_ratio = max(params.pipe_diameter / reference_diameter, 1e-06)
    carbon_scale = diameter_ratio ** params.pipe_carbon_scale_exponent
    effective_pipe_carbon = params.pipe_carbon_factor * carbon_scale
    CO2_emb = effective_pipe_carbon * L_physical + params.elbow_carbon_factor * N_bend
    alt = total_positive_lift
    viol = calculate_clearance_violation(path_grid, grid_info, obstacle_set, min_clearance=0.2)
    f_Energy = 0
    M, N, L_grid = grid_info['shape']
    dx, dy, dz = grid_cell_size
    world_max_x = M * dx
    world_max_y = N * dy
    world_max_z = L_grid * dz
    for p in path_grid:
        world_coord = grid_to_world(p, grid_info)
        energy_p = min(min(world_coord[0], world_max_x - world_coord[0]), min(world_coord[1], world_max_y - world_coord[1]), min(world_coord[2], world_max_z - world_coord[2]))
        f_Energy += energy_p
    f_Install = 0
    for j in range(len(control_points) - 1):
        seg_start = control_points[j]
        seg_end = control_points[j + 1]
        min_dist = calculate_min_distance_segment_to_obstacles(seg_start, seg_end, grid_info, grid_cell_size, obstacle_set)
        f_Install += min_dist
    avg_height = positive_height_sum / positive_height_samples if positive_height_samples > 0 else 0.0
    f_Height = total_positive_lift + avg_height * 0.5
    E_op_norm = normalize_value(E_op, 'E_op', 0, params=params)
    CO2_op_norm = normalize_value(CO2_op, 'CO2_op', 0, params=params)
    CO2_emb_norm = normalize_value(CO2_emb, 'CO2_emb', 0, params=params)
    L_norm = normalize_value(L, 'L', 0, params=params)
    N_bend_norm = normalize_value(N_bend, 'N_bend', 0, params=params)
    alt_norm = normalize_value(alt, 'alt', 0, params=params)
    viol_norm = normalize_value(viol, 'viol', 0, params=params)
    if params.use_hybrid_normalization:
        f_Energy_value = f_Energy
        f_Install_value = f_Install
        f_Height_value = f_Height
    else:
        f_Energy_value = normalize_value(f_Energy, 'f_Energy', 0, params=params)
        f_Install_value = normalize_value(f_Install, 'f_Install', 0, params=params)
        f_Height_value = normalize_value(f_Height, 'f_Height', 0, params=params)
    contribution_E_op = params.w_op * E_op_norm
    contribution_CO2_emb = params.w_emb * CO2_emb_norm
    contribution_L = params.w_L * L_norm
    contribution_N_bend = params.w_bend * N_bend_norm
    contribution_alt = params.w_alt * alt_norm
    contribution_viol = params.w_clear * viol_norm
    if params.use_hybrid_normalization:
        contribution_f_Energy = params.omega_Energy_raw * f_Energy_value
        contribution_f_Install = params.omega_Install_raw * f_Install_value
        contribution_f_Height = params.omega_height_penalty_raw * f_Height_value
    else:
        contribution_f_Energy = params.omega_Energy * f_Energy_value
        contribution_f_Install = params.omega_Install * f_Install_value
        contribution_f_Height = params.omega_height_penalty * f_Height_value
    J_total = contribution_E_op + contribution_CO2_emb + contribution_L + contribution_N_bend + contribution_alt + contribution_viol + contribution_f_Energy + contribution_f_Install + contribution_f_Height
    fitness_data = FitnessData(J_total=J_total, E_op=E_op, CO2_op=CO2_op, CO2_emb=CO2_emb, L=L, N_bend=N_bend, viol=viol, alt=alt, f_Energy=f_Energy, f_Install=f_Install, f_Height=f_Height)
    return fitness_data
