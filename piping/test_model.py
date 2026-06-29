#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained GRL-DEACO policy on validation/test scenario splits.

The evaluator records per-connection path quality, green metrics, success
flags, processing time, selected DEACO hyperparameters, and optional layout
exports. It is designed for user-provided datasets and model checkpoints; the
public artifact does not ship private trained weights.

Example:
    python3 evaluate_trained_policy.py \
        --model-path /path/to/your/model_file \
        --dataset-dir /path/to/your/dataset \
        --glb-directory /path/to/your/glb_library \
        --use-gnn

    python3 evaluate_trained_policy.py \
        --model-path /path/to/your/model_file \
        --dataset-dir /path/to/your/dataset \
        --glb-directory /path/to/your/glb_library \
        --n-workers <num_workers>
"""

from __future__ import annotations

import os
import json
import time
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from functools import partial

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - keeps CLI --help usable in minimal environments
    torch = None
    TORCH_AVAILABLE = False

from state import ConnectionState
from ppo_trainer import DEACOActionSpace as CurrentDEACOActionSpace, PPOTrainer
from environment import DEACOEnvironment
from metrics_collector import MetricsCollector


def _require_torch() -> None:
    """Raise a clear error only when model evaluation actually needs PyTorch."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for trained-policy evaluation. Install project requirements first.")


class LegacyDEACOActionSpace(CurrentDEACOActionSpace):
    """Action space for older 32-dimensional checkpoints.

    Older internal checkpoints included three physical scaling factors in the
    actor output. During public evaluation those factors are still decoded for
    checkpoint compatibility, then fixed to 1.0 before calling the environment.
    """

    def __post_init__(self):
        super().__post_init__()
        self.param_names = self.param_names + [
            'flow_rate_scale',
            'pipe_diameter_scale',
            'pipe_carbon_factor_scale'
        ]
        self.param_ranges.update({
            'flow_rate_scale': (0.7, 1.3),
            'pipe_diameter_scale': (0.8, 1.2),
            'pipe_carbon_factor_scale': (0.8, 1.2)
        })


def get_action_space_for_model(model_path: str):
    """Select the action-space decoder from the checkpoint output dimension."""
    _require_torch()
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    if 'actor' in checkpoint:
        mean_layer_weight = checkpoint['actor'].get('mean_layer.weight', None)
        if mean_layer_weight is not None:
            action_dim = mean_layer_weight.shape[0]
            print(f"Detected model action dimension: {action_dim}")

            if action_dim == 32:
                print("   Using legacy 32-D action space with scale terms.")
                print("   Scale terms are fixed to 1.0 during evaluation.")
                return LegacyDEACOActionSpace()
            else:
                print(f"   Using current {action_dim}-D action space.")
                return CurrentDEACOActionSpace()

    print("   Could not detect action dimension; using current action space.")
    return CurrentDEACOActionSpace()

try:
    from deaco.visualization import (
        show_scene_with_pipes,
        export_layout_info,
    )
    LAYOUT_EXPORT_AVAILABLE = True
    LAYOUT_EXPORT_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - keeps CLI --help usable without trimesh
    show_scene_with_pipes = None
    export_layout_info = None
    LAYOUT_EXPORT_AVAILABLE = False
    LAYOUT_EXPORT_IMPORT_ERROR = exc

try:
    from graph_neural_networks import build_scene_graph
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False


def save_scenario_layout_files(env, scenario_info, scenario_idx, output_base_dir=None):
    """Save layout JSON and visualization files for one evaluated scenario.

    Args:
        env: Evaluated DEACO environment.
        scenario_info: Scenario metadata dictionary.
        scenario_idx: Scenario index in the evaluation split.
        output_base_dir: Base output directory.
    """
    if not LAYOUT_EXPORT_AVAILABLE:
        print(f"    Layout export unavailable; skipping. Import error: {LAYOUT_EXPORT_IMPORT_ERROR}")
        return

    try:
        scenario_name = f"{scenario_info['difficulty']}_{Path(scenario_info['file']).stem}"
        if output_base_dir is None:
            output_base_dir = Path("evaluation_results")
        else:
            output_base_dir = Path(output_base_dir)
        scenario_output_dir = output_base_dir / "scenario_layouts" / f"scenario_{scenario_idx:03d}_{scenario_name}"
        scenario_output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        if hasattr(env, "episode_results"):
            paths = [res["path"] for res in env.episode_results if res.get("path")]

        if not paths:
            print(f"    No valid paths for {scenario_name}; skipping layout export.")
            return

        print(f"    Saving layout files to: {scenario_output_dir}")

        current_dir = os.getcwd()

        try:
            os.chdir(scenario_output_dir)

            print("    Generating GLB visualization...")
            show_scene_with_pipes(
                placed=env.placed_devices,
                scene_config=env.config.get("scene", {}),
                connections=env.connections,
                paths=paths,
            )

            print("    Exporting layout metadata...")
            export_layout_info(
                placed=env.placed_devices,
                connections=env.connections,
                paths=paths,
                output_file=f"{scenario_name}_layout_info.json",
            )

            scenario_summary = {
                "scenario_info": {
                    "file": scenario_info['file'],
                    "difficulty": scenario_info['difficulty'],
                    "fold": scenario_info['fold'],
                    "path": scenario_info['path']
                },
                "scene_config": env.config.get("scene", {}),
                "connections_summary": {
                    "total_connections": len(env.connections),
                    "successful_paths": len(paths),
                    "success_rate": len(paths) / len(env.connections) if env.connections else 0.0
                },
                "device_summary": {
                    "total_devices": len(env.placed_devices),
                    "device_names": [device.get('name', 'Unknown') for device in env.placed_devices]
                }
            }

            if hasattr(env, "episode_results"):
                scenario_summary["connection_results"] = []
                for i, result in enumerate(env.episode_results):
                    conn_summary = {
                        "connection_idx": i,
                        "success": result.get("success", False),
                        "path_length": len(result.get("path", [])),
                        "fitness": result.get("fitness"),
                        "fitness_data": result.get("fitness_data")
                    }
                    if i < len(env.connections):
                        conn = env.connections[i]
                        conn_summary["from_device"] = conn.get("from", "Unknown")
                        conn_summary["to_device"] = conn.get("to", "Unknown")

                    scenario_summary["connection_results"].append(conn_summary)

            summary_file = f"{scenario_name}_scenario_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(scenario_summary, f, indent=2, ensure_ascii=False, default=str)

            print("    Layout files saved:")
            print("       - GLB visualization: rendered_scene_deaco_model.glb")
            print(f"       - Layout metadata: {scenario_name}_layout_info.json")
            print(f"       - Scenario summary: {summary_file}")

        finally:
            os.chdir(current_dir)

    except Exception as e:
        print(f"    Failed to save layout files: {str(e)}")
        import traceback
        traceback.print_exc()


def _to_device_tensor(data, device):
    """Convert array-like data into a tensor on the target device."""
    _require_torch()
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return torch.tensor(data, dtype=torch.float32, device=device)


def _compute_fitness_breakdown(fitness_data: Optional[Dict], action_params: Optional[Dict]) -> Optional[Dict]:
    """Build a post-hoc fitness contribution summary for exported details.

    The environment owns the authoritative DEACO fitness computation. This
    helper only reports approximate normalized contributions so users can
    inspect which terms the policy emphasized in each connection.
    """
    if fitness_data is None or action_params is None:
        return None

    try:
        w_L = action_params.get('w_L', 1.0)
        w_bend = action_params.get('w_bend', 1.0)
        w_op = action_params.get('w_op', 1.0)
        w_emb = action_params.get('w_emb', 1.0)
        w_alt = action_params.get('w_alt', 1.0)
        w_viol = action_params.get('w_viol', 1.0)

        E_op = fitness_data.get('E_op', 0.0)
        CO2_emb = fitness_data.get('CO2_emb', 0.0)
        L = fitness_data.get('L', 0)
        N_bend = fitness_data.get('N_bend', 0)
        alt = fitness_data.get('alt', 0.0)
        viol = fitness_data.get('viol', 0.0)

        # Reference scales are used for inspection only; they are not the
        # authoritative normalization used by DEACOEnvironment.
        ref_energy = 50000.0  # kWh/year
        ref_carbon = 1000.0   # kgCO2
        ref_length = 1000     # grid steps
        ref_bend = 100        # bends
        ref_alt = 50.0        # m

        E_op_norm = min(E_op / ref_energy, 1.0) if ref_energy > 0 else 0.0
        CO2_emb_norm = min(CO2_emb / ref_carbon, 1.0) if ref_carbon > 0 else 0.0
        L_norm = min(L / ref_length, 1.0) if ref_length > 0 else 0.0
        N_bend_norm = min(N_bend / ref_bend, 1.0) if ref_bend > 0 else 0.0
        alt_norm = min(alt / ref_alt, 1.0) if ref_alt > 0 else 0.0
        viol_norm = viol

        contrib_E_op = w_op * E_op_norm
        contrib_CO2_emb = w_emb * CO2_emb_norm
        contrib_L = w_L * L_norm
        contrib_N_bend = w_bend * N_bend_norm
        contrib_alt = w_alt * alt_norm
        contrib_viol = w_viol * viol_norm

        breakdown = {
            'green_items': {
                'operational_energy': {
                    'contribution': float(contrib_E_op),
                    'weight': float(w_op),
                    'normalized_value': float(E_op_norm),
                    'raw_value': float(E_op),
                    'unit': 'kWh/year'
                },
                'embedded_carbon': {
                    'contribution': float(contrib_CO2_emb),
                    'weight': float(w_emb),
                    'normalized_value': float(CO2_emb_norm),
                    'raw_value': float(CO2_emb),
                    'unit': 'kgCO2'
                },
                'path_length': {
                    'contribution': float(contrib_L),
                    'weight': float(w_L),
                    'normalized_value': float(L_norm),
                    'raw_value': int(L),
                    'unit': 'grid_steps'
                },
                'bend_count': {
                    'contribution': float(contrib_N_bend),
                    'weight': float(w_bend),
                    'normalized_value': float(N_bend_norm),
                    'raw_value': int(N_bend),
                    'unit': 'count'
                },
                'altitude_change': {
                    'contribution': float(contrib_alt),
                    'weight': float(w_alt),
                    'normalized_value': float(alt_norm),
                    'raw_value': float(alt),
                    'unit': 'm'
                },
                'clearance_violation': {
                    'contribution': float(contrib_viol),
                    'weight': float(w_viol),
                    'normalized_value': float(viol_norm),
                    'raw_value': float(viol),
                    'unit': 'dimensionless'
                }
            },
            'total_fitness': fitness_data.get('J_total', float('inf'))
        }

        return breakdown

    except Exception as e:
        print(f"    Failed to compute fitness breakdown: {e}")
        return None


def _deterministic_policy_action(ppo_trainer: PPOTrainer,
                                state_vector: np.ndarray,
                                node_features: Optional[torch.Tensor],
                                adj_matrix: Optional[torch.Tensor]) -> np.ndarray:
    """Evaluate with the policy mean action to avoid sampling noise."""
    state_tensor = torch.as_tensor(state_vector, dtype=torch.float32, device=ppo_trainer.device).unsqueeze(0)

    with torch.no_grad():
        if ppo_trainer.use_gnn and node_features is not None and adj_matrix is not None:
            node_tensor = _to_device_tensor(node_features, ppo_trainer.device)
            adj_tensor = _to_device_tensor(adj_matrix, ppo_trainer.device)

            if node_tensor.dim() == 2:
                node_tensor = node_tensor.unsqueeze(0)
            if adj_tensor.dim() == 2:
                adj_tensor = adj_tensor.unsqueeze(0)

            mean, _, _ = ppo_trainer.actor(state_tensor, node_tensor, adj_tensor)
        else:
            mean, _ = ppo_trainer.actor(state_tensor)

    return mean.squeeze(0).cpu().numpy()


def _fix_public_physical_scales(action_params: Dict) -> Dict:
    """Keep physical operating parameters fixed for fair policy comparison."""
    action_params['flow_rate_scale'] = 1.0
    action_params['pipe_diameter_scale'] = 1.0
    action_params['pipe_carbon_factor_scale'] = 1.0
    return action_params


def _fitness_data_to_dict(fitness_data) -> Optional[Dict]:
    """Convert a FitnessData-like object into a serializable dictionary."""
    if fitness_data is None:
        return None
    if isinstance(fitness_data, dict):
        return fitness_data

    try:
        return {
            'J_total': float(fitness_data.J_total) if hasattr(fitness_data, 'J_total') else 0.0,
            'E_op': float(fitness_data.E_op) if hasattr(fitness_data, 'E_op') else 0.0,
            'CO2_op': float(fitness_data.CO2_op) if hasattr(fitness_data, 'CO2_op') else 0.0,
            'CO2_emb': float(fitness_data.CO2_emb) if hasattr(fitness_data, 'CO2_emb') else 0.0,
            'L': int(fitness_data.L) if hasattr(fitness_data, 'L') else 0,
            'N_bend': int(fitness_data.N_bend) if hasattr(fitness_data, 'N_bend') else 0,
            'viol': float(fitness_data.viol) if hasattr(fitness_data, 'viol') else 0.0,
            'alt': float(fitness_data.alt) if hasattr(fitness_data, 'alt') else 0.0,
            'f_Energy': float(fitness_data.f_Energy) if hasattr(fitness_data, 'f_Energy') else 0.0,
            'f_Install': float(fitness_data.f_Install) if hasattr(fitness_data, 'f_Install') else 0.0,
            'f_Height': float(fitness_data.f_Height) if hasattr(fitness_data, 'f_Height') else 0.0,
        }
    except Exception:
        return None


def _extract_fitness_data(env, info: Dict, connection_idx: int) -> Dict:
    """Extract connection metrics from environment results or fallback info."""
    fitness_data = None

    if hasattr(env, 'episode_results') and len(env.episode_results) > connection_idx:
        result = env.episode_results[connection_idx]
        fitness_data = _fitness_data_to_dict(result.get('fitness_data'))

    if fitness_data is not None:
        return fitness_data

    green_metrics = info.get('green_metrics', {})
    return {
        'E_op': green_metrics.get('E_op', 0.0),
        'CO2_emb': green_metrics.get('CO2_emb', 0.0),
        'CO2_op': green_metrics.get('CO2_op', 0.0),
        'N_bend': info.get('N_bend', 0),
        'alt': info.get('alt', 0.0),
        'viol': info.get('viol', 0.0),
        'L': info.get('path_length', 0),
    }


def _connection_info(
    scenario_info: Dict,
    connection_idx: int,
    total_connections: int,
    inference_time: float,
    deaco_time: float,
    connection_time: float,
) -> Dict:
    """Build metadata stored with one evaluated connection."""
    return {
        'connection_idx': connection_idx,
        'total_connections': total_connections,
        'difficulty': scenario_info['difficulty'],
        'scenario_file': scenario_info['file'],
        'inference_time': inference_time,
        'deaco_time': deaco_time,
        'total_time': connection_time,
    }


def _connection_state_dim() -> int:
    """Return the neural state dimension expected by ConnectionState."""
    return ConnectionState(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0,
        0, 0, 0,
    ).dim


def evaluate_single_scenario_worker(
    scenario_info: Dict,
    scenario_idx: int,
    model_path: str,
    glb_directory: str,
    use_gnn: bool,
    device: str,
    save_layouts: bool = False,
    output_dir: Optional[Path] = None
) -> Tuple[MetricsCollector, Dict, str]:
    """Evaluate one scenario inside a worker process."""
    scenario_file = scenario_info['path']
    scenario_name = f"fold_{scenario_info['fold']}/{scenario_info['difficulty']}/{scenario_info['file']}"

    state_dim = _connection_state_dim()
    action_space = get_action_space_for_model(model_path)
    action_dim = action_space.dim

    ppo_trainer = PPOTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        node_feature_dim=24,
        use_gnn=use_gnn and GNN_AVAILABLE,
        device=device
    )
    ppo_trainer.load(model_path)

    collector = MetricsCollector()

    try:
        scenario_start_time = time.time()

        env = DEACOEnvironment(scenario_file, glb_directory)
        state = env.reset()
        if state is None:
            return collector, {
                'scenario': scenario_name,
                'status': 'failed',
                'reason': 'initialization_failed'
            }, scenario_name

        if ppo_trainer.use_gnn:
            node_features, adj_matrix = build_scene_graph(
                env.placed_devices,
                env.connections,
                completed_connections=0
            )
        else:
            node_features = None
            adj_matrix = None

        connection_idx = 0

        while True:
            connection_start_time = time.time()
            state_vector = state.to_vector()

            inference_start = time.time()
            action_raw = _deterministic_policy_action(
                ppo_trainer, state_vector, node_features, adj_matrix
            )
            action_normalized = action_space.normalize_action(action_raw)
            action_params = action_space.action_to_params(action_normalized)
            action_params = _fix_public_physical_scales(action_params)
            inference_time = time.time() - inference_start

            deaco_start = time.time()
            next_state, reward, done, info = env.step(action_params)
            deaco_time = time.time() - deaco_start
            connection_time = time.time() - connection_start_time

            if 'deaco_time' not in info:
                info['deaco_time'] = deaco_time
            if 'inference_time' not in info:
                info['inference_time'] = inference_time

            fitness_data = _extract_fitness_data(env, info, connection_idx)
            success = info.get('success', False)
            fitness_breakdown = _compute_fitness_breakdown(fitness_data, action_params)

            collector.add_connection_result(
                connection_idx=connection_idx,
                success=success,
                fitness_data=fitness_data,
                path_length=int(fitness_data.get('L', 0)),
                processing_time=connection_time,
                reward=reward,
                fitness=info.get('fitness'),
                scenario_name=scenario_name,
                connection_info=_connection_info(
                    scenario_info,
                    connection_idx,
                    len(env.connections),
                    inference_time,
                    info.get('deaco_time', deaco_time),
                    connection_time,
                ),
                action_params=action_params,
                fitness_breakdown=fitness_breakdown,
            )

            if done:
                break
            state = next_state
            connection_idx += 1

            if ppo_trainer.use_gnn:
                node_features, adj_matrix = build_scene_graph(
                    env.placed_devices,
                    env.connections,
                    completed_connections=env.current_connection_idx
                )

        scenario_time = time.time() - scenario_start_time
        summary = {
            'scenario': scenario_name,
            'status': 'success',
            'n_connections': len(env.connections),
            'time': scenario_time
        }
        print(f"  [{scenario_idx+1}] {scenario_name} ({scenario_time:.1f}s)")

        if save_layouts and output_dir is not None:
            save_scenario_layout_files(env, scenario_info, scenario_idx, output_dir)

        return collector, summary, scenario_name

    except Exception as e:
        print(f"  [{scenario_idx+1}] {scenario_name}: {e}")
        return collector, {
            'scenario': scenario_name,
            'status': 'error',
            'error': str(e)
        }, scenario_name


def evaluate_model_on_scenarios_parallel(
    model_path: str,
    scenarios: List[Dict],
    glb_directory: str,
    use_gnn: bool = True,
    device: str = 'cpu',
    save_layouts: bool = False,
    output_dir: Optional[Path] = None,
    n_workers: int = -1
) -> Tuple[MetricsCollector, Dict]:
    """Evaluate scenarios with one or more worker processes.

    Args:
        n_workers: Number of workers. Use -1 for auto, 1 for serial.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating model: {model_path}")
    print(f"{'='*70}")
    print(f"Scenario count: {len(scenarios)}")

    if n_workers == -1:
        n_workers = max(1, mp.cpu_count() - 1)
    n_workers = max(1, min(n_workers, len(scenarios)))
    print(f"Worker processes: {n_workers} (CPU cores: {mp.cpu_count()})\n")

    start_time = time.time()

    if n_workers == 1:
        print("Serial evaluation mode\n")
        global_collector = MetricsCollector()
        scenario_summaries = []
        for idx, scenario_info in enumerate(scenarios):
            collector, summary, _ = evaluate_single_scenario_worker(
                scenario_info, idx, model_path, glb_directory,
                use_gnn, device, save_layouts, output_dir
            )
            global_collector.merge(collector)
            scenario_summaries.append(summary)

    else:
        print(f"Parallel evaluation mode ({n_workers} workers)\n")
        eval_func = partial(
            evaluate_single_scenario_worker,
            model_path=model_path,
            glb_directory=glb_directory,
            use_gnn=use_gnn,
            device=device,
            save_layouts=save_layouts,
            output_dir=output_dir
        )

        tasks = [(scenario_info, idx) for idx, scenario_info in enumerate(scenarios)]
        global_collector = MetricsCollector()
        scenario_summaries = []

        with mp.Pool(processes=n_workers) as pool:
            results = pool.starmap(eval_func, tasks)
            for collector, summary, _ in results:
                global_collector.merge(collector)
                scenario_summaries.append(summary)

    total_time = time.time() - start_time
    stats = global_collector.compute_statistics()

    summary_stats = {
        'model': model_path,
        'total_scenarios': len(scenarios),
        'total_time': total_time,
        'n_workers': n_workers,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stats': stats,
        'scenario_summaries': scenario_summaries
    }

    print(f"\n{'='*70}")
    scenarios_per_minute = len(scenarios) / total_time * 60 if total_time > 0 else 0.0
    print(f"Completed in {total_time:.2f}s ({scenarios_per_minute:.1f} scenarios/min)")
    print(f"Success rate: {stats['algorithm_performance']['success_rate']:.2%}")
    print("")
    return global_collector, summary_stats


def evaluate_model_on_scenarios(
    model_path: str,
    scenarios: List[Dict],
    glb_directory: str,
    use_gnn: bool = True,
    device: Optional[str] = None,
    save_layouts: bool = False,
    output_dir: Optional[Path] = None
) -> Tuple[MetricsCollector, Dict]:
    """Evaluate a trained policy on a list of scenarios serially.

    Args:
        model_path: Checkpoint path.
        scenarios: Scenario dictionaries with path, fold, difficulty, and file.
        glb_directory: GLB asset library directory.
        use_gnn: Whether to use the graph-aware policy path.
        device: Compute device.
        save_layouts: Whether to export per-scenario layout files.
        output_dir: Base output directory for layout files.

    Returns:
        (metrics_collector, summary_stats)
    """
    _require_torch()
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n{'='*70}")
    print(f"Evaluating model: {model_path}")
    print(f"{'='*70}")
    print(f"Scenario count: {len(scenarios)}")
    print(f"Use GNN: {use_gnn}")
    print(f"Device: {device}\n")

    state_dim = _connection_state_dim()
    action_space = get_action_space_for_model(model_path)
    action_dim = action_space.dim

    ppo_trainer = PPOTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        node_feature_dim=24,
        use_gnn=use_gnn and GNN_AVAILABLE,
        device=device
    )

    print(f"Loading model: {model_path}")
    ppo_trainer.load(model_path)
    print("Model loaded.\n")

    collector = MetricsCollector()

    for scenario_idx, scenario_info in enumerate(scenarios):
        scenario_file = scenario_info['path']
        scenario_name = f"fold_{scenario_info['fold']}/{scenario_info['difficulty']}/{scenario_info['file']}"

        print(f"[{scenario_idx + 1}/{len(scenarios)}] Scenario: {scenario_name}")

        try:
            scenario_start_time = time.time()

            env = DEACOEnvironment(scenario_file, glb_directory)
            state = env.reset()
            if state is None:
                print(f"  Scenario {scenario_name} failed to initialize; skipping.")
                continue

            if ppo_trainer.use_gnn:
                node_features, adj_matrix = build_scene_graph(
                    env.placed_devices,
                    env.connections,
                    completed_connections=0
                )
            else:
                node_features = None
                adj_matrix = None

            connection_idx = 0

            while True:
                connection_start_time = time.time()

                state_vector = state.to_vector()

                inference_start = time.time()
                action_raw = _deterministic_policy_action(
                    ppo_trainer,
                    state_vector,
                    node_features,
                    adj_matrix
                )

                action_normalized = action_space.normalize_action(action_raw)
                action_params = action_space.action_to_params(action_normalized)
                action_params = _fix_public_physical_scales(action_params)
                inference_time = time.time() - inference_start

                deaco_start = time.time()
                next_state, reward, done, info = env.step(action_params)
                deaco_time = time.time() - deaco_start
                connection_time = time.time() - connection_start_time

                if 'deaco_time' not in info:
                    info['deaco_time'] = deaco_time
                if 'inference_time' not in info:
                    info['inference_time'] = inference_time

                fitness_data = _extract_fitness_data(env, info, connection_idx)
                fitness_breakdown = _compute_fitness_breakdown(fitness_data, action_params)

                collector.add_connection_result(
                    connection_idx=connection_idx,
                    success=info.get('success', False),
                    fitness_data=fitness_data,
                    path_length=info.get('path_length', 0),
                    processing_time=connection_time,
                    reward=reward,
                    fitness=info.get('fitness'),
                    scenario_name=scenario_name,
                    connection_info=_connection_info(
                        scenario_info,
                        connection_idx,
                        len(env.connections),
                        inference_time,
                        info.get('deaco_time', deaco_time),
                        connection_time,
                    ),
                    action_params=action_params,
                    fitness_breakdown=fitness_breakdown,
                )

                connection_idx += 1

                if done:
                    break

                state = next_state

                if ppo_trainer.use_gnn:
                    node_features, adj_matrix = build_scene_graph(
                        env.placed_devices,
                        env.connections,
                        completed_connections=env.current_connection_idx
                    )

            scenario_time = time.time() - scenario_start_time
            print(f"  Completed {connection_idx} connections in {scenario_time:.2f}s")

            if save_layouts:
                save_scenario_layout_files(env, scenario_info, scenario_idx, output_dir)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Evaluation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    summary_stats = collector.compute_statistics()

    return collector, summary_stats


def _connection_details_payload(connection_details: List[Dict]) -> Dict:
    """Build the detailed connection-result payload written to JSON."""
    return {
        'description': (
            'Per-connection evaluation details, including decoded RL '
            'hyperparameters and post-hoc fitness contribution summaries.'
        ),
        'fields': {
            'action_params': 'Decoded DEACO hyperparameters selected by the policy.',
            'fitness_breakdown': (
                'Approximate contribution summary for green and routing terms.'
            ),
            'fitness_data': (
                'Raw fitness data such as J_total, E_op, CO2_emb, L, '
                'N_bend, alt, and viol.'
            ),
            'connection_info': 'Scenario difficulty, source file, and timing metadata.',
        },
        'connections': connection_details,
    }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate a trained GRL-DEACO policy')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to a trained .pth checkpoint')
    parser.add_argument('--dataset-dir', type=str,
                       default='scenarios_rl_dataset_kfold',
                       help='Dataset root directory')
    parser.add_argument('--glb-directory', type=str,
                        default='static/glb',
                       help='GLB equipment library directory')
    parser.add_argument('--fold-idx', type=int, default=1,
                       help='Fold index for k-fold datasets')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory; generated automatically when omitted')
    parser.add_argument('--use-gnn', action='store_true', default=False,
                       help='Use the graph-aware policy architecture')
    parser.add_argument('--force-gnn', action='store_true',
                       help='Deprecated alias; use --use-gnn')
    parser.add_argument('--device', type=str, default='auto',
                       help='Compute device: auto, cuda, or cpu')
    parser.add_argument('--no-save-layouts', dest='save_layouts', action='store_false', default=True,
                       help='Disable per-scenario layout exports')
    parser.add_argument('--test-subdir', type=str, default='test',
                       help='Test split subdirectory name')
    parser.add_argument('--skip-validation', action='store_true', default=True,
                       help='Skip validation split evaluation')
    parser.add_argument('--eval-validation', action='store_true', default=False,
                       help='Evaluate validation split in addition to test')
    parser.add_argument('--n-workers', type=int, default=1,
                       help='Number of worker processes (-1=auto, 1=serial, >1=parallel)')

    args = parser.parse_args()

    _require_torch()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    piping_dir = Path(__file__).resolve().parent
    repo_root = piping_dir.parent

    dataset_arg = Path(args.dataset_dir)
    dataset_dir = dataset_arg if dataset_arg.is_absolute() else (piping_dir / dataset_arg).resolve()

    glb_arg = Path(args.glb_directory)
    glb_dir = glb_arg if glb_arg.is_absolute() else (repo_root / glb_arg).resolve()
    if not glb_dir.exists():
        raise FileNotFoundError(
            f"GLB directory does not exist. Use --glb-directory to set it: {glb_dir}"
        )

    model_arg = Path(args.model_path)
    model_path = model_arg if model_arg.is_absolute() else (piping_dir / model_arg).resolve()

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = piping_dir / "evaluation_results" / f"eval_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_validation:
        args.skip_validation = False

    print(f"\n{'='*70}")
    print("Evaluation configuration")
    print(f"{'='*70}")
    print(f"Model path: {model_path}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"GLB directory: {glb_dir}")
    print(f"Fold index: {args.fold_idx}")
    print(f"Test split subdirectory: {args.test_subdir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Evaluate validation split: {not args.skip_validation}")
    print(f"Save layout files: {args.save_layouts}")
    print(f"Worker processes: {args.n_workers if args.n_workers > 0 else 'auto'}")
    print(f"Use GNN: {args.use_gnn}")
    print(f"{'='*70}\n")

    print(f"{'='*70}")
    print("Training metadata")
    print(f"{'='*70}")
    print("The public evaluator does not read or print private training metadata.")
    print(f"{'='*70}\n")

    print("Loading dataset splits...")

    val_scenarios = []
    if not args.skip_validation:
        val_fold_dir = dataset_dir / f"fold_{args.fold_idx}"
        for difficulty in ['simple', 'medium', 'complex']:
            difficulty_dir = val_fold_dir / difficulty
            if difficulty_dir.exists():
                for json_file in sorted(difficulty_dir.glob('*.json')):
                    val_scenarios.append({
                        'path': str(json_file),
                        'fold': args.fold_idx,
                        'difficulty': difficulty,
                        'file': json_file.name
                    })

    test_scenarios = []
    test_dir = dataset_dir / args.test_subdir
    if test_dir.exists():
        for difficulty in ['simple', 'medium', 'complex']:
            difficulty_dir = test_dir / difficulty
            if difficulty_dir.exists():
                for json_file in sorted(difficulty_dir.glob('*.json')):
                    test_scenarios.append({
                        'path': str(json_file),
                        'fold': 0,
                        'difficulty': difficulty,
                        'file': json_file.name
                    })

    print(f"Validation scenarios: {len(val_scenarios)}")
    print(f"Test scenarios: {len(test_scenarios)}\n")

    val_results = {}
    if val_scenarios:
        print(f"\n{'='*70}")
        print("Evaluating validation split")
        print(f"{'='*70}\n")
        if args.n_workers != 1:
            val_collector, val_stats = evaluate_model_on_scenarios_parallel(
                str(model_path),
                val_scenarios,
                str(glb_dir),
                use_gnn=args.use_gnn,
                device=device if device == 'cpu' else 'cpu',
                save_layouts=args.save_layouts,
                output_dir=output_dir,
                n_workers=args.n_workers
            )
        else:
            val_collector, val_stats = evaluate_model_on_scenarios(
                str(model_path),
                val_scenarios,
                str(glb_dir),
                use_gnn=args.use_gnn,
                device=device,
                save_layouts=args.save_layouts,
                output_dir=output_dir
            )
        val_results = {
            'collector': val_collector,
            'stats': val_stats,
            'scenarios': val_scenarios
        }

    test_results = {}
    if test_scenarios:
        print(f"\n{'='*70}")
        print("Evaluating test split")
        print(f"{'='*70}\n")
        if args.n_workers != 1:
            parallel_device = 'cpu'
            if device != 'cpu':
                print("Parallel mode uses CPU to avoid GPU memory contention.")
            test_collector, test_stats = evaluate_model_on_scenarios_parallel(
                str(model_path),
                test_scenarios,
                str(glb_dir),
                use_gnn=args.use_gnn,
                device=parallel_device,
                save_layouts=args.save_layouts,
                output_dir=output_dir,
                n_workers=args.n_workers
            )
        else:
            print(f"Serial mode, device: {device}")
            test_collector, test_stats = evaluate_model_on_scenarios(
                str(model_path),
                test_scenarios,
                str(glb_dir),
                use_gnn=args.use_gnn,
                device=device,
                save_layouts=args.save_layouts,
                output_dir=output_dir
            )
        test_results = {
            'collector': test_collector,
            'stats': test_stats,
            'scenarios': test_scenarios
        }

    print(f"\n{'='*70}")
    print("Saving evaluation results")
    print(f"{'='*70}\n")

    if val_results:
        val_output = {
            'dataset_type': 'validation',
            'fold_idx': args.fold_idx,
            'model_path': str(model_path),
            'statistics': val_results['stats'],
            'scenario_count': len(val_scenarios)
        }

        val_json_path = output_dir / "validation_metrics.json"
        with open(val_json_path, 'w', encoding='utf-8') as f:
            json.dump(val_output, f, indent=2, ensure_ascii=False)
        print(f"Validation metrics saved: {val_json_path}")

        val_details_path = output_dir / "validation_details.json"
        details_data = _connection_details_payload(
            val_results['collector'].connection_details
        )
        with open(val_details_path, 'w', encoding='utf-8') as f:
            json.dump(details_data, f, indent=2, ensure_ascii=False)
        print(f"Validation details saved: {val_details_path}")
        print(f"   Connections exported: {len(val_results['collector'].connection_details)}")

    if test_results:
        test_output = {
            'dataset_type': 'test',
            'model_path': str(model_path),
            'statistics': test_results['stats'],
            'scenario_count': len(test_scenarios)
        }

        test_json_path = output_dir / "test_metrics.json"
        with open(test_json_path, 'w', encoding='utf-8') as f:
            json.dump(test_output, f, indent=2, ensure_ascii=False)
        print(f"Test metrics saved: {test_json_path}")

        test_details_path = output_dir / "test_details.json"
        details_data = _connection_details_payload(
            test_results['collector'].connection_details
        )
        with open(test_details_path, 'w', encoding='utf-8') as f:
            json.dump(details_data, f, indent=2, ensure_ascii=False)
        print(f"Test details saved: {test_details_path}")
        print(f"   Connections exported: {len(test_results['collector'].connection_details)}")

    print(f"\n{'='*70}")
    print("Evaluation summary")
    print(f"{'='*70}\n")

    if val_results:
        print("Validation metrics:")
        val_stats_data = val_results['stats'].get('stats', val_results['stats'])
        print(f"  Success rate: {val_stats_data['algorithm_performance']['success_rate']:.2%}")
        print(f"  Mean path length L: {val_stats_data['path_quality'].get('L_mean', 0):.2f}")
        print(f"  Mean bends N_bend: {val_stats_data['path_quality'].get('N_bend_mean', 0):.2f}")
        print(f"  Mean elevation change H_alt: {val_stats_data['path_quality'].get('H_alt_mean', 0):.4f} m")
        print(f"  Clearance violation ratio: {val_stats_data['path_quality'].get('N_viol_ratio', 0):.2%}")
        print(f"  Annual operational energy E_op: {val_stats_data['green_metrics'].get('E_op_mean', 0):.2f} kWh/year")
        print(f"  Embodied carbon CO2_emb: {val_stats_data['green_metrics'].get('CO2_emb_mean', 0):.2f} kgCO2")
        print(f"  Mean processing time: {val_stats_data['algorithm_performance'].get('avg_time_per_connection', 0):.3f} s/connection")
        print()

    if test_results:
        print("Test metrics:")
        stats_data = test_results['stats'].get('stats', test_results['stats'])
        print(f"  Success rate: {stats_data['algorithm_performance']['success_rate']:.2%}")
        print(f"  Mean path length L: {stats_data['path_quality'].get('L_mean', 0):.2f}")
        print(f"  Mean bends N_bend: {stats_data['path_quality'].get('N_bend_mean', 0):.2f}")
        print(f"  Mean elevation change H_alt: {stats_data['path_quality'].get('H_alt_mean', 0):.4f} m")
        print(f"  Clearance violation ratio: {stats_data['path_quality'].get('N_viol_ratio', 0):.2%}")
        print(f"  Annual operational energy E_op: {stats_data['green_metrics'].get('E_op_mean', 0):.2f} kWh/year")
        print(f"  Embodied carbon CO2_emb: {stats_data['green_metrics'].get('CO2_emb_mean', 0):.2f} kgCO2")
        print(f"  Mean processing time: {stats_data['algorithm_performance'].get('avg_time_per_connection', 0):.3f} s/connection")
        print()

    print(f"All results saved to: {output_dir}\n")


if __name__ == '__main__':
    main()
