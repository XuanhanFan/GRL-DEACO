# GRL-DEACO: Green 3D Pipe Routing with RL-Tuned Ant Colony Optimization

Public code for:

**Green Industrial Engineering on the Web: Agent-Driven Ant Colony Optimization Tuning for Energy-Efficient 3D Pipe Routing**
WWW 2026, DOI: `10.1145/3774904.3793013`

GRL-DEACO combines a DEACO-Green 3D pipe router with a graph-aware PPO policy. For each routing connection, the policy observes scene and connection state, emits a 29-dimensional DEACO parameter action, and routes the pipe with an energy- and carbon-aware ant-colony optimizer.

## Reproduction Levels

| Level | Supported by this repository | Required inputs |
|---|---|---|
| Static checks | Import checks, CLI help, config loading, and paper-to-code traceability | No private assets |
| Method-level runs | Run DEACO-Green, train/evaluate GRL-DEACO, and collect reported metrics | Compatible layout JSON files and GLB equipment models |
| Paper-scale reruns | Run the same evaluation pipeline at the scale described in the paper | Equivalent industrial GLB assets and scenario splits |

The original experiments used private industrial assets that cannot be redistributed. This repository provides the public implementation and configuration needed to run the method, while leaving private GLB files, proprietary layouts, checkpoints, logs, and generated outputs outside the artifact.

## Framework Figure

![GRL-DEACO actor-critic architecture with graph-aware perception.](docs/assets/paper_figures/paper_figure_3_actor_critic_architecture.png)

Additional original paper figures referenced by this repository are available in `docs/FRAMEWORK_OVERVIEW.md`.

## Main Components

- Modular DEACO-Green routing core under `piping/deaco/`
- Graph-aware PPO training stack: `piping/state.py`, `piping/reward.py`, `piping/environment.py`, `piping/ppo_trainer.py`, `piping/train.py`
- Scene graph encoder and graph-aware Actor/Critic networks in `piping/graph_neural_networks.py`
- Fixed-parameter and trained-policy evaluators
- Public reproduction configs and traceability documentation
- Scenario generation utilities for user-provided GLB libraries

## Repository Map

```text
configs/
  paper_reproduction_config.yaml     # Main public reproduction config
  public_example_config.yaml         # Smaller example config
  example_deaco_params.json          # Fixed DEACO-Green parameter example

docs/
  DATA_FORMAT.md                     # Layout JSON and GLB expectations
  FRAMEWORK_OVERVIEW.md              # Original paper figures used by the public artifact
  REPRODUCIBILITY.md                 # End-to-end reproduction guide
  PAPER_TRACEABILITY.md              # Paper algorithm/formula/metric to code mapping

piping/
  deaco/                             # Routing core: parameters, IO, grid, fitness, ACO, routing
  train.py                           # Main PPO training entry point
  evaluate_trained_policy.py         # Trained GRL-DEACO evaluation
  evaluate_deaco_with_params.py      # Fixed-parameter DEACO-Green evaluation
  rl_agent_deaco_tuner_kfold_green.py # Thin k-fold orchestrator
  graph_neural_networks.py           # Scene graph encoder and graph-aware policies
  scenario_generator_for_rl.py       # Procedural scenario generation
```

For the original paper figures and paper-to-code traceability, start with `docs/FRAMEWORK_OVERVIEW.md` and `docs/PAPER_TRACEABILITY.md`.

## Environment

Commands below assume they are run from the repository root. A clean conda environment is recommended:

```bash
conda create -n grl-deaco python=3.10
conda activate grl-deaco
pip install -r requirements.txt
```

Alternatively, use any Python 3.10+ environment with the dependencies in `requirements.txt` installed. For GPU training, install a PyTorch build matching your CUDA version before long PPO runs. CPU execution is sufficient for syntax checks, CLI help, small routing smoke tests, and fixed-parameter debugging.

## Data Layout

The expected scenario split is scenario-level, not connection-level:

```text
piping/scenarios_rl_dataset/
  train/
    simple/*.json
    medium/*.json
    complex/*.json
  validation/
    simple/*.json
    medium/*.json
    complex/*.json
  test/
    simple/*.json
    medium/*.json
    complex/*.json
```

GLB assets are expected under `static/glb/` by default. Both paths can be overridden from the CLI or in `configs/paper_reproduction_config.yaml`.

Each layout JSON should define scene bounds, device placements, port metadata, and either explicit process connections or enough port information for connection inference. See `docs/DATA_FORMAT.md` for details.

## Configuration

`configs/paper_reproduction_config.yaml` is the main public configuration file for:

- Paper metadata and expected split description
- Physical constants used by green fitness calculations
- DEACO geometry, ACO, pheromone, heuristic, and green coefficients
- PPO hyperparameters and scheduler controls
- 29-dimensional action ranges
- Reward component weights, clips, scales, failure penalties, and shaping terms
- Metric names used for reporting

Fixed-parameter evaluation also reads this config. Parameter JSON files only override selected DEACO values on top of the config defaults.

## Quick Checks

Run these before training:

```bash
python -m py_compile \
  piping/deaco/*.py \
  piping/state.py \
  piping/ppo_trainer.py \
  piping/reward.py \
  piping/environment.py \
  piping/train.py \
  piping/evaluate_deaco_with_params.py \
  piping/evaluate_trained_policy.py

python piping/train.py --help
python piping/evaluate_deaco_with_params.py --help
python piping/evaluate_trained_policy.py --help
```

Check that the fixed-parameter evaluator reads the YAML config:

```bash
python -c "import sys; sys.path.append('piping'); from evaluate_deaco_with_params import load_reproduction_config, load_parameter_payload, load_custom_params; cfg=load_reproduction_config('configs/paper_reproduction_config.yaml'); p=load_custom_params(load_parameter_payload('configs/example_deaco_params.json'), cfg); print(p.pipe_diameter, p.flow_rate, p.M_ants)"
```

Expected output:

```text
0.154 0.0278 30
```

## Training

Train GRL-DEACO on a compatible dataset:

```bash
python piping/train.py \
  --config configs/paper_reproduction_config.yaml \
  --dataset-dir piping/scenarios_rl_dataset \
  --glb-directory static/glb \
  --output-dir piping/rl_training \
  --episodes 100 \
  --seed 2026
```

Training outputs are written under `run_*` directories:

- `logs/`: text logs
- `models/`: checkpoints and final model
- `stats/`: training statistics
- `debug/`: optional routed layout exports

## Evaluation

Evaluate a trained GRL-DEACO policy:

```bash
python piping/evaluate_trained_policy.py \
  --model-path piping/rl_training/run_YYYYMMDD_HHMMSS/models/rl_deaco_tuner_final.pth \
  --dataset-dir piping/scenarios_rl_dataset \
  --glb-directory static/glb \
  --output-dir evaluation_results/grl_deaco
```

Evaluate fixed DEACO-Green parameters:

```bash
python piping/evaluate_deaco_with_params.py \
  --config configs/paper_reproduction_config.yaml \
  --params configs/example_deaco_params.json \
  --dataset-dir piping/scenarios_rl_dataset \
  --glb-directory static/glb \
  --output-dir evaluation_results/deaco_green
```

The fixed-parameter evaluator constructs `DEACOParameters` from the reproduction config first, then applies JSON overrides. This keeps physical constants, green coefficients, and normalization behavior aligned with training.

Fixed-parameter evaluation writes JSON summaries containing:

- effective DEACO parameters
- scenario-level routing status
- per-connection fitness and physical metrics
- aggregate success, energy, carbon, and path-quality statistics

## K-Fold Runs

The k-fold script is a thin orchestrator. It builds fold-specific train views, calls `piping/train.py`, evaluates each fold model, and aggregates metrics:

```bash
python piping/rl_agent_deaco_tuner_kfold_green.py \
  --dataset-dir piping/scenarios_rl_dataset_kfold \
  --config configs/paper_reproduction_config.yaml \
  --glb-directory static/glb \
  --output-dir evaluation_results/kfold \
  --episodes 100
```

Use `--fold N` for a single fold and `--skip-test` to skip held-out test evaluation.

K-fold outputs include per-fold train metadata, validation/test results, split manifests, and aggregate summary JSON files.

## Reported Metrics

The paper protocol reports:

- Success rate
- Path length `L`
- Bend count `N_bend`
- Altitude change `H_alt`
- Clearance violation
- Operating energy `E_op`
- Embodied carbon `CO2_emb`
- Operational carbon `CO2_op`
- Policy inference overhead

Metrics are collected through `piping/metrics_collector.py` and per-connection `FitnessData` records from `piping/deaco/fitness.py`.

## Paper Traceability

`docs/PAPER_TRACEABILITY.md` maps paper-level concepts to code and configuration, including:

- scene and GLB ingestion
- occupancy-grid construction
- DEACO transition rules and pheromone updates
- green fitness and carbon/energy terms
- PPO state, action, reward, and scheduler components
- fixed-parameter and trained-policy evaluation metrics

`docs/FRAMEWORK_OVERVIEW.md` references the original paper figures for the problem motivation, web platform workflow, GRL-DEACO actor-critic framework, attention visualization, and evaluation behavior.

## Documentation

- `docs/REPRODUCIBILITY.md`: reproduction workflow and acceptance checks
- `docs/DATA_FORMAT.md`: layout JSON and GLB conventions
- `docs/FRAMEWORK_OVERVIEW.md`: original paper figures and framework overview
- `docs/PAPER_TRACEABILITY.md`: mapping from paper components to code, configs, and outputs

## Citation

```bibtex
@inproceedings{fan2026grldeaco,
  title = {Green Industrial Engineering on the Web: Agent-Driven Ant Colony Optimization Tuning for Energy-Efficient 3D Pipe Routing},
  author = {Fan, Xuanhan and Ding, Rui and Liu, Haojie and Zhou, Jibin and Liu, Han and Li, Yuanman and Wang, Wei and Ye, Mao},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  year = {2026},
  doi = {10.1145/3774904.3793013}
}
```
