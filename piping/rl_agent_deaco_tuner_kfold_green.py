#!/usr/bin/env python3
"""Thin K-fold orchestration for GRL-DEACO public reproduction.

This script intentionally does not implement PPO, reward calculation, graph
construction, or environment stepping. Each fold is trained through
``piping/train.py`` and evaluated through the existing trained-policy evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DIFFICULTIES = ("simple", "medium", "complex")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sanitize_json(value):
    if isinstance(value, dict):
        return {key: _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_sanitize_json(payload), handle, indent=2, ensure_ascii=False, default=_json_default, allow_nan=False)


def _read_reproduction_config(config_path: Optional[Path]) -> dict:
    if config_path is None:
        config_path = _repo_root() / "configs" / "paper_reproduction_config.yaml"
    if not config_path.exists():
        return {}
    if config_path.suffix.lower() == ".json":
        with open(config_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    try:
        import yaml
    except ImportError:
        return {}
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(path: Optional[str], default: Path, base: Optional[Path] = None) -> Path:
    if path is None:
        return default.resolve()
    value = Path(path)
    if value.is_absolute():
        return value
    return ((base or Path.cwd()) / value).resolve()


def _collect_split_scenarios(split_dir: Path, fold: Optional[int | str] = None) -> List[Dict]:
    scenarios = []
    for difficulty in DIFFICULTIES:
        difficulty_dir = split_dir / difficulty
        if not difficulty_dir.exists():
            continue
        for layout_file in sorted(difficulty_dir.glob("layout_*.json")):
            scenario = {
                "difficulty": difficulty,
                "file": layout_file.name,
                "path": str(layout_file),
            }
            if fold is not None:
                scenario["fold"] = fold
            scenarios.append(scenario)
    return scenarios


def _count_by_difficulty(scenarios: Iterable[Dict]) -> Dict[str, int]:
    counts = {difficulty: 0 for difficulty in DIFFICULTIES}
    for scenario in scenarios:
        counts[scenario["difficulty"]] += 1
    counts["total"] = sum(counts.values())
    return counts


def validate_kfold_dataset(dataset_dir: Path, k_folds: int, require_test: bool = True) -> Dict:
    """Validate the expected K-fold scenario layout and return split counts."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"K-fold dataset directory does not exist: {dataset_dir}")
    if k_folds < 2:
        raise ValueError(f"k_folds must be at least 2, got {k_folds}")

    fold_counts = {}
    for fold_id in range(1, k_folds + 1):
        fold_dir = dataset_dir / f"fold_{fold_id}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Missing fold directory: {fold_dir}")
        for difficulty in DIFFICULTIES:
            difficulty_dir = fold_dir / difficulty
            if not difficulty_dir.exists():
                raise FileNotFoundError(f"Missing difficulty directory: {difficulty_dir}")
        fold_counts[f"fold_{fold_id}"] = _count_by_difficulty(_collect_split_scenarios(fold_dir, fold=fold_id))

    test_counts = None
    test_dir = dataset_dir / "test"
    if require_test:
        if not test_dir.exists():
            raise FileNotFoundError(f"Missing held-out test directory: {test_dir}")
        for difficulty in DIFFICULTIES:
            difficulty_dir = test_dir / difficulty
            if not difficulty_dir.exists():
                raise FileNotFoundError(f"Missing test difficulty directory: {difficulty_dir}")
        test_counts = _count_by_difficulty(_collect_split_scenarios(test_dir, fold="test"))

    return {"folds": fold_counts, "test": test_counts}


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source.resolve())
        return "symlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def materialize_fold_train_view(
    *,
    dataset_dir: Path,
    output_dir: Path,
    validation_fold: int,
    k_folds: int,
    config_path: Optional[Path],
    seed: Optional[int],
) -> Tuple[Path, dict]:
    """Create a standard train/ view for train.main from K-1 source folds."""
    fold_output_dir = output_dir / f"fold_{validation_fold}"
    view_dir = fold_output_dir / "dataset_view"
    train_view = view_dir / "train"
    if view_dir.exists():
        shutil.rmtree(view_dir)
    for difficulty in DIFFICULTIES:
        (train_view / difficulty).mkdir(parents=True, exist_ok=True)

    train_folds = [fold_id for fold_id in range(1, k_folds + 1) if fold_id != validation_fold]
    linked_files = []
    train_counts = {difficulty: 0 for difficulty in DIFFICULTIES}
    link_modes = {"symlink": 0, "copy": 0}

    for source_fold in train_folds:
        source_fold_dir = dataset_dir / f"fold_{source_fold}"
        for difficulty in DIFFICULTIES:
            for source_file in sorted((source_fold_dir / difficulty).glob("layout_*.json")):
                target_name = f"fold{source_fold}_{source_file.name}"
                destination = train_view / difficulty / target_name
                mode = _link_or_copy(source_file, destination)
                link_modes[mode] += 1
                train_counts[difficulty] += 1
                linked_files.append(
                    {
                        "source_fold": source_fold,
                        "difficulty": difficulty,
                        "source": str(source_file),
                        "target": str(destination),
                        "mode": mode,
                    }
                )

    train_counts["total"] = sum(train_counts.values())
    validation_scenarios = _collect_split_scenarios(dataset_dir / f"fold_{validation_fold}", fold=validation_fold)
    test_scenarios = _collect_split_scenarios(dataset_dir / "test", fold="test")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "dataset_view_dir": str(view_dir),
        "train_view_dir": str(train_view),
        "validation_fold": validation_fold,
        "train_folds": train_folds,
        "test_split": str(dataset_dir / "test"),
        "config_path": str(config_path) if config_path is not None else None,
        "seed": seed,
        "file_counts": {
            "train": train_counts,
            "validation": _count_by_difficulty(validation_scenarios),
            "test": _count_by_difficulty(test_scenarios),
        },
        "link_modes": link_modes,
        "linked_files": linked_files,
    }
    _write_json(fold_output_dir / "split_manifest.json", manifest)
    return view_dir, manifest


def _evaluate_split(
    *,
    model_path: str,
    scenarios: List[Dict],
    glb_directory: Path,
    output_dir: Path,
    use_gnn: bool,
    split_name: str,
    fold_id: int,
) -> dict:
    from test_model import evaluate_model_on_scenarios

    collector, stats = evaluate_model_on_scenarios(
        model_path=model_path,
        scenarios=scenarios,
        glb_directory=str(glb_directory),
        use_gnn=use_gnn,
        save_layouts=False,
        output_dir=output_dir,
    )
    payload = {
        "fold": fold_id,
        "split": split_name,
        "model_path": model_path,
        "total_scenarios": len(scenarios),
        "stats": stats,
        "connection_details": collector.connection_details,
    }
    _write_json(output_dir / f"{split_name}_results.json", payload)
    return payload


def _compact_metric_record(fold_result: dict, split_name: str) -> dict:
    result = fold_result.get(split_name)
    if not result:
        return {
            "fold": fold_result["fold"],
            "model_path": fold_result.get("model_path"),
            "total_scenarios": 0,
            "total_connections": 0,
            "success_rate": 0.0,
            "avg_reward": 0.0,
            "avg_fitness": None,
        }
    performance = result["stats"].get("algorithm_performance", {})
    avg_fitness = performance.get("avg_fitness")
    if isinstance(avg_fitness, float) and not math.isfinite(avg_fitness):
        avg_fitness = None
    return {
        "fold": fold_result["fold"],
        "model_path": fold_result.get("model_path"),
        "total_scenarios": int(result.get("total_scenarios", 0)),
        "total_connections": int(performance.get("total_connections", 0)),
        "success_rate": float(performance.get("success_rate", 0.0)),
        "avg_reward": float(performance.get("avg_reward", 0.0)),
        "avg_fitness": avg_fitness,
    }


def _mean_std(values: List[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def summarize_split(fold_results: List[dict], split_name: str) -> dict:
    records = [_compact_metric_record(result, split_name) for result in fold_results if result.get(split_name)]
    finite_fitness = [record["avg_fitness"] for record in records if record["avg_fitness"] is not None]
    return {
        "split": split_name,
        "evaluated_folds": len(records),
        "total_scenarios": int(sum(record["total_scenarios"] for record in records)),
        "total_connections": int(sum(record["total_connections"] for record in records)),
        "folds": records,
        "statistics": {
            "success_rate": _mean_std([record["success_rate"] for record in records]),
            "avg_reward": _mean_std([record["avg_reward"] for record in records]),
            "avg_fitness": _mean_std([float(value) for value in finite_fitness]),
            "valid_fitness_count": len(finite_fitness),
        },
    }


def run_fold(
    *,
    dataset_dir: Path,
    output_dir: Path,
    validation_fold: int,
    k_folds: int,
    config_path: Optional[Path],
    glb_directory: Path,
    episodes: int,
    seed: Optional[int],
    skip_test: bool,
    export_interval: int,
    debug_export_interval: int,
    use_gnn_for_eval: bool,
) -> dict:
    from train import main as train_main

    fold_output_dir = output_dir / f"fold_{validation_fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    fold_seed = seed + validation_fold - 1 if seed is not None else None
    dataset_view_dir, manifest = materialize_fold_train_view(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        validation_fold=validation_fold,
        k_folds=k_folds,
        config_path=config_path,
        seed=fold_seed,
    )

    train_metadata = train_main(
        num_episodes=episodes,
        export_interval=export_interval,
        debug_export_interval=debug_export_interval,
        config_path=str(config_path) if config_path is not None else None,
        dataset_dir=str(dataset_view_dir),
        glb_directory=str(glb_directory),
        output_dir=str(fold_output_dir / "training"),
        seed=fold_seed,
    )
    if not train_metadata or not train_metadata.get("final_model_path"):
        raise RuntimeError(f"Fold {validation_fold} training did not produce a final model.")

    train_run = {
        "fold": validation_fold,
        "train_folds": manifest["train_folds"],
        "validation_fold": validation_fold,
        "seed": fold_seed,
        "train_metadata": train_metadata,
    }
    _write_json(fold_output_dir / "train_run.json", train_run)

    model_path = train_metadata["final_model_path"]
    validation_scenarios = _collect_split_scenarios(dataset_dir / f"fold_{validation_fold}", fold=validation_fold)
    validation_result = _evaluate_split(
        model_path=model_path,
        scenarios=validation_scenarios,
        glb_directory=glb_directory,
        output_dir=fold_output_dir,
        use_gnn=use_gnn_for_eval,
        split_name="validation",
        fold_id=validation_fold,
    )

    test_result = None
    if not skip_test:
        test_result = _evaluate_split(
            model_path=model_path,
            scenarios=_collect_split_scenarios(dataset_dir / "test", fold="test"),
            glb_directory=glb_directory,
            output_dir=fold_output_dir,
            use_gnn=use_gnn_for_eval,
            split_name="test",
            fold_id=validation_fold,
        )

    return {
        "fold": validation_fold,
        "train_folds": manifest["train_folds"],
        "seed": fold_seed,
        "model_path": model_path,
        "train_run": train_run,
        "validation": validation_result,
        "test": test_result,
    }


def run_kfold(args: argparse.Namespace) -> dict:
    repo_root = _repo_root()
    dataset_dir = _resolve_path(args.dataset_dir, Path("piping/scenarios_rl_dataset_kfold"), base=repo_root)
    config_path = _resolve_path(args.config, repo_root / "configs" / "paper_reproduction_config.yaml", base=repo_root)
    glb_directory = _resolve_path(args.glb_directory, repo_root / "static" / "glb", base=repo_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _resolve_path(args.output_dir, dataset_dir.parent / "rl_training_kfold" / f"kfold_{timestamp}")

    validate_kfold_dataset(dataset_dir, args.k_folds, require_test=not args.skip_test)
    config = _read_reproduction_config(config_path)
    use_gnn_for_eval = bool(config.get("ppo", {}).get("use_gnn", True))

    if args.fold is not None:
        if args.fold < 1 or args.fold > args.k_folds:
            raise ValueError(f"--fold must be in [1, {args.k_folds}], got {args.fold}")
        folds = [args.fold]
    else:
        folds = list(range(1, args.k_folds + 1))

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    for fold_id in folds:
        print(f"\n=== K-fold round: validation fold {fold_id}/{args.k_folds} ===")
        fold_results.append(
            run_fold(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                validation_fold=fold_id,
                k_folds=args.k_folds,
                config_path=config_path,
                glb_directory=glb_directory,
                episodes=args.episodes,
                seed=args.seed,
                skip_test=args.skip_test,
                export_interval=args.export_interval,
                debug_export_interval=args.debug_export_interval,
                use_gnn_for_eval=use_gnn_for_eval,
            )
        )

    kfold_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "config_path": str(config_path),
        "glb_directory": str(glb_directory),
        "output_dir": str(output_dir),
        "k_folds": args.k_folds,
        "episodes": args.episodes,
        "seed": args.seed,
        "folds_requested": folds,
        "validation": summarize_split(fold_results, "validation"),
    }
    _write_json(output_dir / "kfold_summary.json", kfold_summary)

    test_summary = None
    if not args.skip_test:
        test_summary = summarize_split(fold_results, "test")
        _write_json(output_dir / "test_summary.json", test_summary)

    print(f"\nK-fold orchestration complete: {output_dir}")
    return {"output_dir": str(output_dir), "kfold_summary": kfold_summary, "test_summary": test_summary}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run thin K-fold orchestration for GRL-DEACO training and evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=str, default="piping/scenarios_rl_dataset_kfold",
                        help="K-fold dataset root containing fold_i/ and test/ splits")
    parser.add_argument("--config", type=str, default="configs/paper_reproduction_config.yaml",
                        help="YAML/JSON reproduction config passed to piping/train.py")
    parser.add_argument("--glb-directory", type=str, default="static/glb",
                        help="Directory containing the GLB equipment library")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Root directory for fold outputs")
    parser.add_argument("--k-folds", type=int, default=5,
                        help="Number of fold_i directories to use")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Training episodes per fold")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional base seed; fold N uses seed+N-1")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only one validation fold")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip held-out test evaluation")
    parser.add_argument("--export-interval", type=int, default=0,
                        help="Forwarded to train.main episode export interval")
    parser.add_argument("--debug-export-interval", type=int, default=0,
                        help="Forwarded to train.main debug export interval")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_kfold(args)


if __name__ == "__main__":
    main()
