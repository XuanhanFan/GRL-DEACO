#!/usr/bin/env python3
"""Evaluate fixed DEACO-Green parameters on scenario splits.

The script is intentionally thin: it loads fixed DEACO parameters, routes each
scenario through ``deaco.routing.route_scene``, records metrics, and writes JSON
summaries. PPO policy evaluation lives in ``test_model.py`` and
``evaluate_trained_policy.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - handled at config load time
    yaml = None

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from deaco.parameters import DEACOParameters
from deaco.routing import route_scene
from deaco.visualization import export_layout_info, show_scene_with_pipes
from metrics_collector import MetricsCollector


DIFFICULTIES = ("simple", "medium", "complex")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_project_path(path_value: Optional[str], fallback: str) -> Path:
    path = Path(path_value or fallback)
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _json_default(obj: Any) -> Any:
    """Convert dataclasses and NumPy scalars into JSON-compatible values."""
    if hasattr(obj, "__dataclass_fields__"):
        return {key: _json_default(getattr(obj, key)) for key in obj.__dataclass_fields__}
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {key: _json_default(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_default(value) for value in obj]
    return obj


def load_reproduction_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load the YAML/JSON reproduction config used for fixed-parameter evaluation."""
    path = _resolve_project_path(config_path, "configs/paper_reproduction_config.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Reproduction config not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs. Install project requirements first.")
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def load_parameter_payload(params: str) -> Dict[str, Any]:
    """Load a parameter dictionary from a JSON file path or inline JSON."""
    if os.path.exists(params):
        with open(params, "r", encoding="utf-8") as handle:
            return json.load(handle)
    try:
        payload = json.loads(params)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse parameter payload: {params}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Parameter payload must be a JSON object.")
    return payload


def load_custom_params(params_dict: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> DEACOParameters:
    """Create DEACO parameters from config defaults plus fixed-parameter overrides."""
    params = DEACOParameters.from_config(config or {})
    payload = dict(params_dict)

    flow_rate_scale = float(payload.pop("flow_rate_scale", 1.0) or 1.0)
    pipe_diameter_scale = float(payload.pop("pipe_diameter_scale", 1.0) or 1.0)
    pipe_carbon_factor_scale = float(payload.pop("pipe_carbon_factor_scale", 1.0) or 1.0)

    for key, value in payload.items():
        if hasattr(params, key):
            setattr(params, key, value)

    params.flow_rate *= flow_rate_scale
    params.pipe_diameter *= pipe_diameter_scale
    params.pipe_carbon_factor *= pipe_carbon_factor_scale
    return params


def resolve_test_dir(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    """Resolve either a dataset root or an explicit test split directory."""
    if args.dataset_dir:
        base = _resolve_project_path(args.dataset_dir, args.dataset_dir)
        candidate = base / "test"
        return candidate if candidate.exists() else base

    if args.test_dir:
        return _resolve_project_path(args.test_dir, args.test_dir)

    dataset_dir = (config.get("paths") or {}).get("dataset_dir")
    if dataset_dir:
        base = _resolve_project_path(dataset_dir, dataset_dir)
        candidate = base / "test"
        return candidate if candidate.exists() else base

    return _resolve_project_path("piping/scenarios_rl_dataset_kfold/test", "piping/scenarios_rl_dataset_kfold/test")


def resolve_glb_directory(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    """Resolve the GLB library from CLI override, config, or public default."""
    configured = (config.get("paths") or {}).get("glb_directory")
    return _resolve_project_path(args.glb_directory or configured, "static/glb")


def resolve_output_dir(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    """Resolve the fixed-parameter evaluation output directory."""
    if args.output_dir:
        return _resolve_project_path(args.output_dir, args.output_dir)
    configured = (config.get("paths") or {}).get("output_dir")
    if configured:
        return _resolve_project_path(str(Path(configured) / "fixed_deaco_evaluation"), "evaluation_results")
    return _resolve_project_path("evaluation_results", "evaluation_results")


def collect_test_scenarios(test_dir: Path, limit: int | None = None) -> List[Dict[str, str]]:
    """Collect layout JSON files from simple, medium, and complex splits."""
    scenarios: List[Dict[str, str]] = []
    for difficulty in DIFFICULTIES:
        difficulty_dir = test_dir / difficulty
        if not difficulty_dir.exists():
            continue
        for layout_file in sorted(difficulty_dir.glob("*.json")):
            scenarios.append({
                "path": str(layout_file),
                "difficulty": difficulty,
                "file": layout_file.name,
                "full_name": f"{test_dir.name}/{difficulty}/{layout_file.name}",
            })
            if limit is not None and len(scenarios) >= limit:
                return scenarios
    return scenarios


def add_route_result_to_metrics(collector: MetricsCollector, scenario_name: str, route_result: Dict[str, Any]) -> None:
    """Record per-connection routing metrics in the shared collector."""
    for record in route_result.get("connection_results", []):
        success = bool(record.get("success"))
        fitness_data = record.get("fitness_data")
        collector.add_connection_result(
            connection_idx=int(record.get("connection_idx", 0)),
            success=success,
            fitness_data={
                "E_op": getattr(fitness_data, "E_op", 0.0),
                "CO2_op": getattr(fitness_data, "CO2_op", 0.0),
                "CO2_emb": getattr(fitness_data, "CO2_emb", 0.0),
                "L": getattr(fitness_data, "L", 0),
                "N_bend": getattr(fitness_data, "N_bend", 0),
                "viol": getattr(fitness_data, "viol", 0.0),
                "alt": getattr(fitness_data, "alt", 0.0),
                "J_total": getattr(fitness_data, "J_total", record.get("fitness", 0.0)),
            } if success and fitness_data is not None else None,
            path_length=int(record.get("path_length", 0) or 0),
            processing_time=float(record.get("elapsed_time", 0.0) or 0.0),
            reward=0.0,
            fitness=record.get("fitness"),
            scenario_name=scenario_name,
            connection_info={"connection_name": record.get("connection_name", "")},
        )


def save_layout_export(route_result: Dict[str, Any], scenario: Dict[str, str], scenario_idx: int, output_dir: Path) -> None:
    """Write optional GLB and layout metadata exports for one routed scenario."""
    paths = route_result.get("paths", [])
    if not paths:
        return
    scenario_name = f"scenario_{scenario_idx:03d}_{scenario['difficulty']}_{Path(scenario['file']).stem}"
    scenario_output_dir = output_dir / "scenario_layouts" / scenario_name
    scenario_output_dir.mkdir(parents=True, exist_ok=True)
    current_dir = os.getcwd()
    try:
        os.chdir(scenario_output_dir)
        show_scene_with_pipes(
            placed=route_result["placed_devices"],
            scene_config=route_result["config"].get("scene", {}),
            connections=route_result["connections"],
            paths=paths,
        )
        export_layout_info(
            placed=route_result["placed_devices"],
            connections=route_result["connections"],
            paths=paths,
            output_file="layout_info_deaco.json",
        )
    finally:
        os.chdir(current_dir)


def evaluate_scenarios(
    scenarios: Iterable[Dict[str, str]],
    deaco_params: DEACOParameters,
    glb_directory: str,
    output_dir: Path,
    save_layouts: bool = False,
) -> tuple[MetricsCollector, List[Dict[str, Any]]]:
    """Route all scenarios and return the populated collector plus results."""
    collector = MetricsCollector()
    scenario_results: List[Dict[str, Any]] = []
    for idx, scenario in enumerate(scenarios):
        try:
            route_result = route_scene(
                scenario_path=scenario["path"],
                glb_directory=glb_directory,
                deaco_params=deaco_params,
                scenario_name=scenario["full_name"],
            )
            add_route_result_to_metrics(collector, scenario["full_name"], route_result)
            if save_layouts:
                save_layout_export(route_result, scenario, idx, output_dir)
            scenario_results.append({
                "scenario": scenario,
                "success": True,
                "connections": route_result.get("total_connections", 0),
                "success_count": route_result.get("success_count", 0),
                "failed_count": route_result.get("failed_count", 0),
                "total_time": route_result.get("total_time", 0.0),
                "connection_results": route_result.get("connection_results", []),
            })
        except Exception as exc:
            scenario_results.append({
                "scenario": scenario,
                "success": False,
                "error": str(exc),
                "connections": 0,
                "success_count": 0,
                "failed_count": 0,
            })
    return collector, scenario_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate fixed DEACO-Green parameters")
    parser.add_argument("--params", type=str, required=True, help="Path to a parameter JSON file, or an inline JSON object")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON reproduction config. Defaults to configs/paper_reproduction_config.yaml")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Dataset root containing a test split")
    parser.add_argument("--test-dir", type=str, default=None, help="Explicit test split directory")
    parser.add_argument("--glb-directory", "--glb", dest="glb_directory", type=str, default=None, help="GLB equipment library directory")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--save-layouts", action="store_true", help="Save per-scenario layout JSON and GLB visualizations")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many scenarios")
    return parser


def main() -> dict[str, Any]:
    args = build_parser().parse_args()
    config = load_reproduction_config(args.config)
    params_dict = load_parameter_payload(args.params)
    deaco_params = load_custom_params(params_dict, config)

    test_dir = resolve_test_dir(args, config)
    glb_directory = resolve_glb_directory(args, config)

    scenarios = collect_test_scenarios(test_dir, limit=args.limit)
    if not scenarios:
        raise FileNotFoundError(f"No scenario JSON files found under {test_dir}")

    output_dir = resolve_output_dir(args, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    collector, scenario_results = evaluate_scenarios(
        scenarios=scenarios,
        deaco_params=deaco_params,
        glb_directory=str(glb_directory),
        output_dir=output_dir,
        save_layouts=args.save_layouts,
    )
    statistics = collector.compute_statistics()
    output_file = output_dir / f"deaco_evaluation_{timestamp}.json"
    payload = {
        "timestamp": timestamp,
        "config": str(_resolve_project_path(args.config, "configs/paper_reproduction_config.yaml")),
        "params": params_dict,
        "effective_deaco_params": deaco_params,
        "test_dir": str(test_dir),
        "glb_directory": str(glb_directory),
        "scenarios": scenario_results,
        "statistics": statistics,
    }
    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(_json_default(payload), handle, indent=2, ensure_ascii=False)

    perf = statistics.get("algorithm_performance", {})
    print(f"Evaluated scenarios: {len(scenarios)}")
    print(f"Total connections: {perf.get('total_connections', 0)}")
    print(f"Success rate: {perf.get('success_rate', 0.0) * 100:.2f}%")
    print(f"Results: {output_file}")
    return {"output_file": str(output_file), "statistics": statistics}


if __name__ == "__main__":
    main()
