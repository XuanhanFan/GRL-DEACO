from __future__ import annotations
import heapq
import logging
import numpy as np
from .grid import calculate_manhattan_distance, get_valid_manhattan_neighbors, grid_to_world, world_to_grid
from .types import GridPoint
logger = logging.getLogger(__name__)

def a_star_3d_directional(start, goal, device_obstacles, platform_obstacles, bounds, res=0.12, max_iterations=50000):

    def neighbors(n, prev_dir=None):
        base_dirs = [(res, 0, 0), (-res, 0, 0), (0, res, 0), (0, -res, 0), (0, 0, res), (0, 0, -res)]
        xy_dist_to_goal = abs(n[0] - goal[0]) + abs(n[1] - goal[1])
        z_dist_to_goal = abs(n[2] - goal[2])
        if prev_dir is not None:
            dirs = [prev_dir]
            other_dirs = [d for d in base_dirs if d != prev_dir]
            xy_dirs = [d for d in other_dirs if d[2] == 0]
            z_dirs = [d for d in other_dirs if d[2] != 0]
            if xy_dist_to_goal > res * 3:
                dirs.extend(xy_dirs)
                dirs.extend(z_dirs)
            else:
                dirs.extend(xy_dirs)
                dirs.extend(z_dirs)
        else:
            xy_dirs = [(res, 0, 0), (-res, 0, 0), (0, res, 0), (0, -res, 0)]
            z_dirs = [(0, 0, res), (0, 0, -res)]
            if xy_dist_to_goal > res * 3:
                dirs = xy_dirs + z_dirs
            else:
                dirs = xy_dirs + z_dirs
        for dx, dy, dz in dirs:
            nn = snap((n[0] + dx, n[1] + dy, n[2] + dz), res)
            if not (bounds[0] <= nn[0] <= bounds[1] and bounds[2] <= nn[1] <= bounds[3] and (bounds[4] <= nn[2] <= bounds[5])):
                continue
            is_z_move = dz != 0 and dx == dy == 0
            blocked = nn in device_obstacles or (nn in platform_obstacles and (not is_z_move))
            if not blocked:
                yield (nn, (dx, dy, dz))

    def enhanced_heuristic(a, b):
        manhattan_dist = sum((abs(a[i] - b[i]) for i in range(3)))
        xy_dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
        z_dist = abs(a[2] - b[2])
        if xy_dist > res * 3 and z_dist > 0:
            manhattan_dist += z_dist * 0.8
        if xy_dist > z_dist * 2:
            manhattan_dist *= 0.95
        elif z_dist > xy_dist * 2:
            manhattan_dist *= 1.05
        return manhattan_dist

    def direction_cost(curr_dir, prev_dir, curr_pos):
        if prev_dir is None:
            return 0
        xy_dist_remaining = abs(curr_pos[0] - goal[0]) + abs(curr_pos[1] - goal[1])
        z_dist_remaining = abs(curr_pos[2] - goal[2])
        if curr_dir == prev_dir:
            return -0.2
        if tuple((-x for x in curr_dir)) == prev_dir:
            return 1.0
        is_z_move = curr_dir[2] != 0
        is_xy_move = curr_dir[0] != 0 or curr_dir[1] != 0
        if is_z_move:
            if xy_dist_remaining > res * 3:
                return 1.2
            elif z_dist_remaining < res * 2:
                return 0.6
            else:
                return 0.3
        elif is_xy_move:
            if z_dist_remaining > res * 3 and xy_dist_remaining < res * 2:
                return 0.4
            else:
                return 0.1
        return 0.3
    openset = [(enhanced_heuristic(start, goal), 0, start, [start], None)]
    closed = set()
    g_scores = {start: 0}
    iterations = 0
    while openset:
        iterations += 1
        if iterations > max_iterations:
            return None
        f, g, curr, path, prev_dir = heapq.heappop(openset)
        if curr == goal:
            return path
        if curr in closed:
            continue
        closed.add(curr)
        for nei, curr_dir in neighbors(curr, prev_dir):
            if nei in closed:
                continue
            tentative_g = g + 1 + direction_cost(curr_dir, prev_dir, curr)
            if nei not in g_scores or tentative_g < g_scores[nei]:
                g_scores[nei] = tentative_g
                f_score = tentative_g + enhanced_heuristic(nei, goal)
                new_path = path + [nei]
                heapq.heappush(openset, (f_score, tentative_g, nei, new_path, curr_dir))
    return None

def dynamic_manhattan_path(start, goal, device_obstacles, platform_obstacles, bounds, res=0.12, max_jump=50, max_iterations=10000):
    path = [start]
    curr = list(start)
    last_direction = None
    iterations = 0
    while any((abs(curr[i] - goal[i]) > 0.0001 for i in range(3))):
        iterations += 1
        if iterations > max_iterations:
            return None
        found = False
        xy_dist = abs(curr[0] - goal[0]) + abs(curr[1] - goal[1])
        z_dist = abs(curr[2] - goal[2])
        if last_direction is not None:
            axis = last_direction[0]
            if abs(curr[axis] - goal[axis]) > 0.0001:
                step = res if curr[axis] < goal[axis] else -res
                next_p = list(curr)
                next_p[axis] += step
                next_p = snap(tuple(next_p), res)
                changes = sum((1 for i in range(3) if abs(next_p[i] - curr[i]) > 1e-06))
                if changes == 1:
                    is_z_move = axis == 2
                    blocked = next_p in device_obstacles or (next_p in platform_obstacles and (not is_z_move))
                    if not blocked:
                        path.append(next_p)
                        curr = list(next_p)
                        found = True
        if not found:
            if xy_dist > res * 2:
                axis_priority = [0, 1, 2]
            else:
                axis_priority = [0, 1, 2]
            if last_direction is not None:
                last_axis = last_direction[0]
                axis_priority = [a for a in axis_priority if a != last_axis] + [last_axis]
            for axis in axis_priority:
                if abs(curr[axis] - goal[axis]) > 0.0001:
                    step = res if curr[axis] < goal[axis] else -res
                    next_p = list(curr)
                    next_p[axis] += step
                    next_p = snap(tuple(next_p), res)
                    changes = sum((1 for i in range(3) if abs(next_p[i] - curr[i]) > 1e-06))
                    if changes == 1:
                        is_z_move = axis == 2
                        blocked = next_p in device_obstacles or (next_p in platform_obstacles and (not is_z_move))
                        if not blocked:
                            path.append(next_p)
                            curr = list(next_p)
                            last_direction = (axis, step)
                            found = True
                            break
        if found:
            continue
        xy_axes = [0, 1]
        z_axis = [2]
        axes_to_try = xy_axes if xy_dist > res * 2 else [0, 1, 2]
        axes_sorted = sorted(axes_to_try, key=lambda i: -abs(curr[i] - goal[i]))
        jumped = False
        for axis in axes_sorted:
            jump_dist = abs(goal[axis] - curr[axis])
            max_step = int(jump_dist / abs(res))
            for d in range(3, min(max_jump, max_step) + 1):
                step = res if curr[axis] < goal[axis] else -res
                probe = list(curr)
                probe[axis] += step * d
                probe = snap(tuple(probe), res)
                is_z_move = axis == 2
                blocked = probe in device_obstacles or (probe in platform_obstacles and (not is_z_move))
                if not blocked and all((bounds[2 * i] <= probe[i] <= bounds[2 * i + 1] for i in range(3))):
                    local_path = a_star_3d_directional(tuple(curr), probe, device_obstacles, platform_obstacles, bounds, res=res, max_iterations=5000)
                    if local_path and len(local_path) > 1:
                        valid_path = True
                        for j in range(len(local_path) - 1):
                            p1, p2 = (local_path[j], local_path[j + 1])
                            changes = sum((1 for k in range(3) if abs(p1[k] - p2[k]) > 1e-06))
                            if changes != 1:
                                valid_path = False
                                break
                        if valid_path:
                            path += local_path[1:]
                            curr = list(local_path[-1])
                            jumped = True
                            break
            if jumped:
                break
        if not found and (not jumped):
            local_path = a_star_3d_directional(tuple(curr), goal, device_obstacles, platform_obstacles, bounds, res=res, max_iterations=50000)
            if local_path and len(local_path) > 1:
                valid_path = True
                for j in range(len(local_path) - 1):
                    p1, p2 = (local_path[j], local_path[j + 1])
                    changes = sum((1 for k in range(3) if abs(p1[k] - p2[k]) > 1e-06))
                    if changes != 1:
                        valid_path = False
                        break
                if valid_path:
                    path += local_path[1:]
                    curr = list(local_path[-1])
                    continue
            break
    return path
