from __future__ import annotations
import logging
import random
import numpy as np
from .parameters import DEACOParameters, normalize_value
from .types import GridPoint
from .grid import calculate_manhattan_distance, get_valid_manhattan_neighbors, grid_to_world, world_to_grid
from .fitness import calculate_path_fitness, calculate_path_fitness_green
logger = logging.getLogger(__name__)

def calculate_spatial_heuristic(neighbors, current_point, previous_point, goal_point, params):
    eta_values = {}
    for n in neighbors:
        xz_dist = abs(n.x - goal_point.x) + abs(n.z - goal_point.z)
        y_dist = abs(n.y - goal_point.y)
        if getattr(params, 'use_anisotropic_height_weight', True):
            u = float(xz_dist)
            delta_xz = getattr(params, 'delta_xz', 10.0)
            s_sigmoid = getattr(params, 's_sigmoid', 3.0)
            kappa_y = getattr(params, 'kappa_y', 1.0)
            sigmoid_arg = (u - delta_xz) / (s_sigmoid + 1e-09)
            sigmoid_value = 1.0 / (1.0 + np.exp(-sigmoid_arg))
            alpha_y = 1.0 + kappa_y * sigmoid_value
            dis = xz_dist + alpha_y * y_dist
        elif xz_dist > 2:
            y_move = n.y - current_point.y
            y_towards_goal = y_move > 0 and n.y < goal_point.y or (y_move < 0 and n.y > goal_point.y)
            if y_move != 0:
                if y_towards_goal:
                    dis = xz_dist + y_dist * (1.5 + params.omega_height_penalty * 0.5)
                else:
                    dis = xz_dist + y_dist * (2.0 + params.omega_height_penalty * 1.5)
            else:
                dis = xz_dist + y_dist
        else:
            dis = xz_dist + y_dist
        vec_move = np.array([n.x - current_point.x, n.y - current_point.y, n.z - current_point.z])
        vec_goal = np.array([goal_point.x - current_point.x, goal_point.y - current_point.y, goal_point.z - current_point.z])
        dot_product = np.dot(vec_move, vec_goal)
        direction = 1 if dot_product > 0 else 0
        bend = 0
        if previous_point is not None:
            vec1 = np.array([current_point.x - previous_point.x, current_point.y - previous_point.y, current_point.z - previous_point.z])
            vec2 = np.array([n.x - current_point.x, n.y - current_point.y, n.z - current_point.z])
            if np.dot(vec1, vec2) == 0:
                bend = 1
        numerator = 1 + params.omega_direction_reward * direction
        denominator = (1 + dis) * (1 + params.omega_bend_penalty * bend)
        eta_values[n] = numerator / max(denominator, 1e-09)
    return eta_values

def calculate_green_heuristic(neighbors, current_point, previous_point, goal_point, params, grid_info):
    eta_values = {}
    pipe_area = 0.25 * np.pi * params.pipe_diameter ** 2
    V = params.flow_rate / (pipe_area + 1e-09)
    V_sq_2g = V ** 2 / (2 * params.gravity)
    pitch = grid_info['pitch']
    L_cn = pitch
    L_cn_over_D = L_cn / (params.pipe_diameter + 1e-09)
    f_local = calculate_darcy_friction_factor(params)
    for n in neighbors:
        bend = 0
        if previous_point is not None:
            vec1 = np.array([current_point.x - previous_point.x, current_point.y - previous_point.y, current_point.z - previous_point.z])
            vec2 = np.array([n.x - current_point.x, n.y - current_point.y, n.z - current_point.z])
            if np.dot(vec1, vec2) == 0:
                bend = 1
        vec_move = np.array([n.x - current_point.x, n.y - current_point.y, n.z - current_point.z])
        vec_goal = np.array([goal_point.x - current_point.x, goal_point.y - current_point.y, goal_point.z - current_point.z])
        dot_product = np.dot(vec_move, vec_goal)
        direction = 1 if dot_product > 0 else 0
        xz_dist = abs(n.x - goal_point.x) + abs(n.z - goal_point.z)
        y_dist = abs(n.y - goal_point.y)
        if getattr(params, 'use_anisotropic_height_weight', True):
            u = float(xz_dist)
            delta_xz = getattr(params, 'delta_xz', 10.0)
            s_sigmoid = getattr(params, 's_sigmoid', 3.0)
            kappa_y = getattr(params, 'kappa_y', 1.0)
            sigmoid_arg = (u - delta_xz) / (s_sigmoid + 1e-09)
            sigmoid_value = 1.0 / (1.0 + np.exp(-sigmoid_arg))
            alpha_y = 1.0 + kappa_y * sigmoid_value
            dis_to_goal = xz_dist + alpha_y * y_dist
        else:
            dis_to_goal = calculate_manhattan_distance(n, goal_point)
        e_loc_friction = params.lambda_1 * f_local * L_cn_over_D * V_sq_2g
        e_loc_bend = params.lambda_2 * bend * V_sq_2g
        current_world = grid_to_world(current_point, grid_info)
        n_world = grid_to_world(n, grid_info)
        delta_y = n_world[1] - current_world[1]
        e_loc_elevation = params.lambda_3 * max(0, delta_y)
        E_loc = e_loc_friction + e_loc_bend + e_loc_elevation
        numerator = 1 + params.omega_direction_reward * direction
        denominator = (1 + dis_to_goal) * (1 + params.omega_bend_penalty * bend) * (1 + params.lambda_loc * E_loc)
        eta_values[n] = numerator / max(denominator, 1e-09)
    return eta_values

def calculate_probabilities(neighbors, pheromone_matrix, eta_values, alpha, beta):
    probabilities = {}
    raw_scores = {}
    denominator = 0.0
    for n in neighbors:
        tau_n = max(pheromone_matrix[n.x, n.y, n.z], 1e-09)
        eta_n = max(eta_values.get(n, 1e-09), 1e-09)
        score = tau_n ** alpha * eta_n ** beta
        raw_scores[n] = score
        denominator += score
    if denominator < 1e-09:
        for n in neighbors:
            probabilities[n] = 1.0 / len(neighbors)
    else:
        for n in neighbors:
            probabilities[n] = raw_scores[n] / denominator
    return probabilities

def select_next_node(probabilities, current_q0):
    q = random.random()
    if q < current_q0:
        return max(probabilities.items(), key=lambda x: x[1])[0]
    else:
        nodes = list(probabilities.keys())
        probs = list(probabilities.values())
        total = sum(probs)
        if total > 0:
            probs = np.array([p / total for p in probs])
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(probs)) / len(probs)
        return np.random.choice(nodes, p=probs)

def construct_ant_path_manhattan(start_grid, goal_grid, pheromone_matrix, state_matrix, grid_info, params, current_q0, use_green=False):
    current_point = start_grid
    path = [current_point]
    previous_point = None
    step_count = 0
    while current_point != goal_grid and step_count < params.max_steps:
        neighbors = get_valid_manhattan_neighbors(current_point, state_matrix, grid_info)
        if previous_point is not None and len(neighbors) > 1:
            neighbors = [n for n in neighbors if n != previous_point]
        if len(neighbors) == 0:
            return None
        if use_green:
            eta_values = calculate_green_heuristic(neighbors, current_point, previous_point, goal_grid, params, grid_info)
        else:
            eta_values = calculate_spatial_heuristic(neighbors, current_point, previous_point, goal_grid, params)
        probabilities = calculate_probabilities(neighbors, pheromone_matrix, eta_values, params.alpha, params.beta)
        next_point = select_next_node(probabilities, current_q0)
        path.append(next_point)
        previous_point = current_point
        current_point = next_point
        step_count += 1
    if current_point == goal_grid:
        return path
    else:
        return None

def remove_concave_pockets_manhattan(path, state_matrix, grid_info):
    if path is None or len(path) < 2:
        return path
    optimized_path = path.copy()
    i = 0
    while i < len(optimized_path) - 1:
        if i > 0 and i < len(optimized_path) - 1:
            p_prev = optimized_path[i - 1]
            p_curr = optimized_path[i]
            p_next = optimized_path[i + 1]
            if p_prev == p_next:
                optimized_path.pop(i + 1)
                optimized_path.pop(i)
                i = max(0, i - 1)
                continue
        if i > 0 and i < len(optimized_path) - 2:
            p_A = optimized_path[i - 1]
            p_B = optimized_path[i]
            p_C = optimized_path[i + 1]
            p_D = optimized_path[i + 2]
            vec_AB = np.array([p_B.x - p_A.x, p_B.y - p_A.y, p_B.z - p_A.z])
            vec_BC = np.array([p_C.x - p_B.x, p_C.y - p_B.y, p_C.z - p_B.z])
            vec_CD = np.array([p_D.x - p_C.x, p_D.y - p_C.y, p_D.z - p_C.z])
            is_u_shape = np.dot(vec_AB, vec_BC) == 0 and np.dot(vec_BC, vec_CD) == 0 and (np.dot(vec_AB, vec_CD) == -1)
            if is_u_shape:
                shortcut_safe = check_manhattan_shortcut_safe(p_A, p_D, state_matrix, grid_info)
                if shortcut_safe:
                    optimized_path.pop(i + 1)
                    optimized_path.pop(i)
                    i = max(0, i - 1)
                    continue
        i += 1
    return optimized_path

def check_manhattan_shortcut_safe(start, end, state_matrix, grid_info):
    points = []
    current = GridPoint(start.x, start.y, start.z)
    while current.x != end.x:
        current = GridPoint(current.x + (1 if current.x < end.x else -1), current.y, current.z)
        points.append(current)
    while current.y != end.y:
        current = GridPoint(current.x, current.y + (1 if current.y < end.y else -1), current.z)
        points.append(current)
    while current.z != end.z:
        current = GridPoint(current.x, current.y, current.z + (1 if current.z < end.z else -1))
        points.append(current)
    M, N, L = grid_info['shape']
    for p in points:
        if not (0 <= p.x < M and 0 <= p.y < N and (0 <= p.z < L)):
            return False
        if state_matrix[p.x, p.y, p.z] == 0:
            return False
    return True

def update_pheromones(pheromone_matrix, paths_data, params, current_tau_min, current_tau_max, grid_info):
    M, N, L = grid_info['shape']
    pheromone_matrix *= 1 - params.rho
    for path, fitness in paths_data:
        if path is None or len(path) < 2:
            continue
        adjusted_fitness = max(fitness, 1e-09)
        exp_arg = min(adjusted_fitness / 100.0, 700.0)
        delta_tau = params.Q / (np.exp(exp_arg) - 1 + 1e-09)
        delta_tau = max(0, delta_tau)
        S = len(path) - 1
        for step, p in enumerate(path):
            if 0 <= p.x < M and 0 <= p.y < N and (0 <= p.z < L):
                gamma = params.delta_gamma ** ((S - step) / max(S, 1))
                pheromone_matrix[p.x, p.y, p.z] += gamma * delta_tau
    pheromone_matrix[:] = np.clip(pheromone_matrix, current_tau_min, current_tau_max)

def update_pheromones_green(pheromone_matrix, paths_data, start_grid, goal_grid, params, current_tau_min, current_tau_max, grid_info, iteration_best_energy=None, global_best_energy=None):
    M, N, L = grid_info['shape']
    pheromone_matrix *= 1 - params.rho
    total_manhattan_dist = calculate_manhattan_distance(start_grid, goal_grid)
    benchmark_candidates = [val for val in (iteration_best_energy, global_best_energy, params.reference_energy) if val is not None and np.isfinite(val) and (val > params.epsilon)]
    benchmark_energy = min(benchmark_candidates) if benchmark_candidates else max(params.reference_energy, 1.0)
    for path, fitness_data in paths_data:
        if path is None or len(path) < 2:
            continue
        E_op = fitness_data.E_op
        if not np.isfinite(E_op):
            continue
        relative_gain = max(0.0, (benchmark_energy - E_op) / (benchmark_energy + params.epsilon))
        energy_ratio = E_op / max(benchmark_energy, params.epsilon)
        energy_savings = np.exp(-energy_ratio) * (1.0 + relative_gain)
        S = len(path) - 1
        for step, p in enumerate(path):
            if 0 <= p.x < M and 0 <= p.y < N and (0 <= p.z < L):
                dist_to_goal = calculate_manhattan_distance(p, goal_grid)
                gamma_gain = max(0, 1.0 - dist_to_goal / max(total_manhattan_dist, 1))
                delta_tau_step = gamma_gain * energy_savings
                pheromone_matrix[p.x, p.y, p.z] += delta_tau_step
    pheromone_matrix[:] = np.clip(pheromone_matrix, current_tau_min, current_tau_max)

def run_deaco(start_world, goal_world, obstacle_set, grid_info, grid_cell_size, state_matrix, params=None, use_green=False):
    if params is None:
        params = DEACOParameters()
    start_grid = world_to_grid(start_world, grid_info)
    goal_grid = world_to_grid(goal_world, grid_info)
    algorithm_name = 'DEACO-Green' if use_green else 'DEACO'
    M, N, L = grid_info['shape']
    if 0 <= start_grid.x < M and 0 <= start_grid.y < N and (0 <= start_grid.z < L):
        start_status = 'blocked' if state_matrix[start_grid.x, start_grid.y, start_grid.z] == 0 else 'free'
    else:
        start_status = 'out_of_bounds'
    if 0 <= goal_grid.x < M and 0 <= goal_grid.y < N and (0 <= goal_grid.z < L):
        goal_status = 'blocked' if state_matrix[goal_grid.x, goal_grid.y, goal_grid.z] == 0 else 'free'
    else:
        goal_status = 'out_of_bounds'
    logger.debug(
        "%s route: start=%s (%s), goal=%s (%s), grid_shape=%s",
        algorithm_name,
        start_grid,
        start_status,
        goal_grid,
        goal_status,
        grid_info['shape'],
    )
    pheromone_matrix = np.full((M, N, L), params.tau_0, dtype=np.float32)
    global_best_path = None
    global_best_fitness = float('inf')
    global_best_fitness_data = None
    global_best_energy = float('inf')
    no_improvement_count = 0
    last_best_fitness = float('inf')
    for k in range(1, params.K_iterations + 1):
        current_tau_min = params.tau_min0 * 0.9 ** (k // 10)
        current_tau_max = params.tau_max0 * 1.1 ** (k // 10)
        current_q0 = params.A_q0 / (1 + np.exp(-10 / params.K_iterations * (k - 0.5 * params.K_iterations))) + params.B_q0
        current_iteration_paths_data = []
        for m in range(params.M_ants):
            path = construct_ant_path_manhattan(start_grid, goal_grid, pheromone_matrix, state_matrix, grid_info, params, current_q0, use_green=use_green)
            if path is not None:
                optimized_path = remove_concave_pockets_manhattan(path, state_matrix, grid_info)
                if use_green:
                    fitness_data = calculate_path_fitness_green(optimized_path, grid_info, grid_cell_size, obstacle_set, params, start_world, goal_world)
                    fitness = fitness_data.J_total
                    current_iteration_paths_data.append((optimized_path, fitness_data))
                else:
                    fitness = calculate_path_fitness(optimized_path, grid_info, grid_cell_size, obstacle_set, params)
                    current_iteration_paths_data.append((optimized_path, fitness))
                if fitness < global_best_fitness:
                    global_best_fitness = fitness
                    global_best_path = optimized_path.copy()
                    if use_green:
                        global_best_fitness_data = fitness_data
                        if np.isfinite(fitness_data.E_op):
                            global_best_energy = min(global_best_energy, fitness_data.E_op)
        iteration_best_energy = None
        if use_green and current_iteration_paths_data:
            finite_energies = [fd.E_op for _, fd in current_iteration_paths_data if np.isfinite(fd.E_op)]
            if finite_energies:
                iteration_best_energy = min(finite_energies)
        if use_green:
            update_pheromones_green(pheromone_matrix, current_iteration_paths_data, start_grid, goal_grid, params, current_tau_min, current_tau_max, grid_info, iteration_best_energy=iteration_best_energy, global_best_energy=global_best_energy)
        else:
            update_pheromones(pheromone_matrix, current_iteration_paths_data, params, current_tau_min, current_tau_max, grid_info)
        improvement = last_best_fitness - global_best_fitness
        if improvement < params.early_stop_threshold:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        last_best_fitness = global_best_fitness
        if logger.isEnabledFor(logging.DEBUG) and (k % 10 == 0 or k == 1):
            success_rate = len(current_iteration_paths_data) / params.M_ants * 100
            if use_green and global_best_fitness_data:
                logger.debug(
                    "DEACO iteration %s/%s: success_rate=%.1f%%, best=%.6g, E_op=%.6g, CO2_emb=%.6g",
                    k,
                    params.K_iterations,
                    success_rate,
                    global_best_fitness,
                    global_best_fitness_data.E_op,
                    global_best_fitness_data.CO2_emb,
                )
            else:
                logger.debug(
                    "DEACO iteration %s/%s: success_rate=%.1f%%, best=%.6g",
                    k,
                    params.K_iterations,
                    success_rate,
                    global_best_fitness,
                )
        if no_improvement_count >= params.early_stop_patience:
            logger.debug("Early stopping after %s iterations without sufficient improvement.", no_improvement_count)
            break
    if global_best_path is not None:
        best_path_world = [grid_to_world(p, grid_info) for p in global_best_path]
        logger.debug("%s succeeded with %s path points and fitness %.6g.", algorithm_name, len(best_path_world), global_best_fitness)
        if use_green and global_best_fitness_data:
            return (best_path_world, global_best_fitness_data)
        else:
            return (best_path_world, global_best_fitness)
    else:
        logger.debug("%s failed to find a path.", algorithm_name)
        if use_green:
            return (None, FitnessData(J_total=float('inf'), E_op=float('inf'), CO2_op=float('inf'), CO2_emb=float('inf'), L=0, N_bend=0, viol=float('inf'), alt=0.0, f_Energy=float('inf'), f_Install=float('inf'), f_Height=float('inf')))
        else:
            return (None, float('inf'))
