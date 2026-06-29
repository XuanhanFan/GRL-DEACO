# Paper Traceability

This document maps the WWW 2026 paper artifact to the public code paths used by this repository:

**Green Industrial Engineering on the Web: Agent-Driven Ant Colony Optimization Tuning for Energy-Efficient 3D Pipe Routing**.

The goal is to make each algorithmic component, formula family, metric, and configurable constant traceable from the paper narrative to code and configuration. The repository exposes the method and experiment pipeline; table-level numerical comparison still requires compatible scenario splits and GLB assets.

## Main References

- Public configuration: `configs/paper_reproduction_config.yaml`
- Minimal runnable configuration: `configs/public_example_config.yaml`
- Original paper figures: `docs/FRAMEWORK_OVERVIEW.md`
- Training entry point: `piping/train.py`
- Fixed-parameter DEACO-Green evaluation: `piping/evaluate_deaco_with_params.py`
- Trained-policy evaluation: `piping/evaluate_trained_policy.py` and `piping/test_model.py`
- Core routing package: `piping/deaco/`

## Algorithm Traceability

| Paper component | Code location | Config location | Reproduction output |
|---|---|---|---|
| Industrial scene and equipment layout ingestion | `piping/deaco/layout_io.py`, `piping/deaco/glb_reader.py`, `piping/deaco/glb_cache.py` | `paths.dataset_dir`, `paths.glb_directory`, `physical_parameters` | Loaded scene config, placed devices, port metadata, cached GLB device info |
| 3D occupancy grid and obstacle representation | `piping/deaco/grid.py`, `piping/voxel_box_approximation.py` | `deaco.geometry.grid_spacing_m`, `pipe_radius_m`, `pipe_safe_margin_m`, `sdbb_*` | `state_matrix`, obstacle counts, path voxel updates |
| Explicit connections, inferred connections, and virtual tees | `piping/deaco/connections.py`, `piping/deaco/routing.py` | Layout JSON `connections`, optional connection overrides | Ordered connection list, tee usage map, per-connection routing records |
| DEACO ant path construction | `piping/deaco/aco.py::run_deaco()` | `deaco.aco.M_ants`, `K_iterations`, `max_steps`, `early_stop_*` | Per-connection path, fitness, elapsed time, diagnostics |
| Pheromone and heuristic transition policy | `piping/deaco/aco.py` transition and selection helpers | `deaco.aco.alpha`, `beta`, `rho`, `Q`, `tau_0`, `tau_max0`, `tau_min0` | Ant transition probabilities and pheromone updates |
| Adaptive exploitation threshold `q0` | `piping/deaco/aco.py::run_deaco()` | `deaco.aco.A_q0`, `B_q0`, `delta_gamma` | Iteration-level exploitation/exploration behavior |
| Spatial routing heuristic | `piping/deaco/aco.py::calculate_spatial_heuristic()` | `deaco.fitness_weights.w_L`, `w_bend`, `w_alt`, `w_clear`, `delta_xz`, `kappa_y`, `s_sigmoid` | Direction, length, bend, altitude, and clearance guidance |
| Green routing heuristic | `piping/deaco/aco.py::calculate_green_heuristic()` | `deaco.green_coefficients`, `physical_parameters` | Energy/carbon-aware transition shaping |
| Path fitness and green objective terms | `piping/deaco/fitness.py` | `deaco.fitness_weights`, `deaco.green_coefficients`, `physical_parameters` | `FitnessData`: length, bends, altitude, clearance violation, energy, carbon |
| Sequential multi-connection routing | `piping/deaco/routing.py::route_connection()`, `route_scene()` | Routing and geometry config under `deaco` | Scene-level `connection_results`, `paths`, `success_count`, `total_time` |
| Classical routing baselines | `piping/deaco/baselines.py` | Baseline-specific CLI/config settings where used | A*/Manhattan-style baseline paths and metrics |
| RL state vector | `piping/state.py` | Implied by `STATE_DIM = 27` and scene/routing inputs | 27-D connection state used by PPO |
| Scene graph encoder | `piping/graph_neural_networks.py` | `ppo.use_gnn`, `ppo.use_attention` | Node features `[N, 24]`, graph-aware actor/critic inputs |
| 29-D DEACO action vector | `piping/ppo_trainer.py::DEACOActionSpace` | `deaco_action_ranges` | Mapped DEACO parameter override for each connection |
| PPO actor/critic training | `piping/ppo_trainer.py`, `piping/train.py` | `ppo.*`, `training.*` | Checkpoints, stats JSON, policy logs |
| Reward formulation and shaping | `piping/reward.py` | `reward.normalizer`, `reward.component_weights`, `reward.component_scales`, `reward.component_clips`, `reward.failure`, `reward.success_scaling`, `reward.adaptive_shaping` | Scalar reward plus structured `reward_components` |
| K-fold experiment orchestration | `piping/rl_agent_deaco_tuner_kfold_green.py` | Same config consumed by `piping/train.py` | Per-fold train run metadata, validation/test summaries |

## Formula and Parameter Ownership

| Paper formula family | Implemented in | Main configurable constants |
|---|---|---|
| Hydraulic operating energy | `piping/deaco/fitness.py` | `physical_parameters.flow_rate_m3_s`, `pipe_diameter_m`, `fluid_density_kg_m3`, `gravity_m_s2`, `pump_efficiency`, `annual_hours`, `darcy_friction_factor` |
| Operating carbon | `piping/deaco/fitness.py` | `physical_parameters.grid_carbon_intensity_kgco2_kwh` |
| Embodied carbon | `piping/deaco/fitness.py` | `physical_parameters.pipe_carbon_factor_kgco2_m`, `elbow_carbon_factor_kgco2_each`, `reference_pipe_diameter_m`, `pipe_carbon_scale_exponent` |
| Green fitness combination | `piping/deaco/fitness.py` | `deaco.fitness_weights.w_op`, `w_emb`, `w_L`, `w_bend`, `w_alt`, `w_clear`; `deaco.green_coefficients.lambda_*` |
| Transition probability | `piping/deaco/aco.py` | `deaco.aco.alpha`, `beta`, `A_q0`, `B_q0`, `delta_gamma` |
| Pheromone evaporation and deposition | `piping/deaco/aco.py` | `deaco.aco.rho`, `Q`, `tau_0`, `tau_max0`, `tau_min0` |
| PPO clipped objective inputs | `piping/ppo_trainer.py` | `ppo.gamma`, `lambda_gae`, `epsilon_clip`, `k_epochs`, `entropy_coef`, learning rates |
| Reward component weighting | `piping/reward.py` | `reward.component_weights`, `component_scales`, `component_clips` |
| Failure and partial-route reward | `piping/reward.py` | `reward.failure.partial_base`, `partial_progress_coefficient`, `complete`, `lower_bound`, correction penalty settings |

## Metric Traceability

| Reported metric | Source object/function | Evaluation output |
|---|---|---|
| Success rate | `MetricsCollector.compute_statistics()` | `algorithm_performance.success_rate` |
| Total scenarios and connections | `piping/evaluate_deaco_with_params.py`, `piping/test_model.py` | `scenario_results`, `algorithm_performance.total_connections` |
| Path length `L` | `FitnessData.L` from `piping/deaco/fitness.py` | Per-connection records and aggregate metric summaries |
| Bend count `N_bend` | `FitnessData.N_bend` | Per-connection records and aggregate metric summaries |
| Altitude change `H_alt` | `FitnessData.alt` | Per-connection records and aggregate metric summaries |
| Clearance violation | `FitnessData.viol` | Per-connection records, green fitness diagnostics |
| Operating energy | `FitnessData.E_op` | Per-connection records, aggregate energy statistics |
| Embodied carbon | `FitnessData.CO2_emb` | Per-connection records, aggregate carbon statistics |
| Operating carbon | `FitnessData.CO2_op` | Per-connection records, aggregate carbon statistics |
| Reward decomposition | `RewardCalculator.calculate()` | `reward_components.raw`, `scaled`, `weighted`, and final reward |
| Training stability and checkpoint status | `piping/train.py` | Run directory logs, stats, and model checkpoints |

## Evaluation Commands

Fixed DEACO-Green uses the same physical and DEACO defaults as the training stack:

```bash
python piping/evaluate_deaco_with_params.py \
  --config configs/paper_reproduction_config.yaml \
  --params configs/example_deaco_params.json \
  --dataset-dir /path/to/scenarios_rl_dataset \
  --glb-directory /path/to/glb_library \
  --output-dir evaluation_results/deaco_green
```

GRL-DEACO policy evaluation:

```bash
python piping/evaluate_trained_policy.py \
  --model-path /path/to/rl_deaco_tuner_final.pth \
  --dataset-dir /path/to/scenarios_rl_dataset \
  --glb-directory /path/to/glb_library \
  --output-dir evaluation_results/grl_deaco
```

Training:

```bash
python piping/train.py \
  --config configs/paper_reproduction_config.yaml \
  --dataset-dir /path/to/scenarios_rl_dataset \
  --glb-directory /path/to/glb_library \
  --episodes 100 \
  --seed 2026
```

## Artifact Boundaries

- The public code does not redistribute private GLB equipment assets, proprietary industrial layouts, trained checkpoints, or experiment logs.
- The core DEACO-Green implementation lives in `piping/deaco/`; removed monolithic scripts are not part of the reproduction path.
- The exploratory adaptive-q0 script is intentionally not exposed as a core module. The reproducible q0 behavior is the `A_q0/B_q0/delta_gamma` policy in `piping/deaco/aco.py`.
- CLI scripts may print final summaries for users. Library modules use `logging` so training, evaluation, and JSON outputs stay clean.
- Table-level comparison requires equivalent scenario splits and GLB assets. With user-provided compatible assets, the repository supports method-level runs, ablation, and regression testing.
