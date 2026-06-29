# Reproducibility Guide

This guide documents how to reproduce the public GRL-DEACO artifact for the WWW 2026 paper.

## 1. Environment

Install dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build for full training runs. CPU execution is suitable for syntax checks, small routing smoke tests, and fixed-parameter DEACO debugging.

## 2. Inputs

The full paper results require a compatible GLB equipment library and scenario splits. The private industrial assets used for the paper are not redistributed.

Expected split shape:

```text
piping/scenarios_rl_dataset/
  train/{simple,medium,complex}/*.json
  validation/{simple,medium,complex}/*.json
  test/{simple,medium,complex}/*.json
```

The paper protocol uses 100 scenarios: 64 train, 16 validation, and 20 test. The test set contains 6 simple scenes with 38 connections, 10 medium scenes with 95 connections, and 4 complex scenes with 51 connections.

## 3. Configuration

Use `configs/paper_reproduction_config.yaml` as the public source of truth for:

- paper metadata and DOI
- public physical constants from the appendix
- DEACO-Green geometry, ACO, fitness, normalization, and green-coefficient defaults
- PPO defaults (`gamma=0.99`, `lambda_gae=0.95`, gradient clipping documented as 0.5/0.3)
- action ranges for the 29-dimensional DEACO hyperparameter vector
- expected metrics and baselines

See `docs/PAPER_TRACEABILITY.md` for the paper-to-code mapping of algorithms, formula families, metrics, and configuration keys.

Local paths can be overridden from the CLI:

```bash
python piping/train.py \
  --config configs/paper_reproduction_config.yaml \
  --dataset-dir /path/to/scenarios_rl_dataset \
  --glb-directory /path/to/glb_library \
  --output-dir /path/to/rl_training \
  --episodes 100 \
  --seed 2026
```

## 4. Training GRL-DEACO

Each training episode routes every connection in one sampled scene. The PPO agent observes the scene graph and current connection state, emits a 29-dimensional continuous action, and the action is mapped to DEACO-Green parameters with the paper's `tanh` squashing rule.

The training script writes:

- logs under `run_*/logs/`
- checkpoints under `run_*/models/`
- training statistics under `run_*/stats/`
- optional debug exports under `run_*/debug/`

These generated artifacts are ignored by git.

## 5. Evaluation

Evaluate a trained policy:

```bash
python piping/evaluate_trained_policy.py \
  --model-path /path/to/rl_deaco_tuner_final.pth \
  --dataset-dir /path/to/scenarios_rl_dataset \
  --glb-directory /path/to/glb_library \
  --output-dir evaluation_results/grl_deaco
```

Evaluate fixed DEACO-Green:

```bash
python piping/evaluate_deaco_with_params.py \
  --params configs/example_deaco_params.json \
  --dataset-dir /path/to/scenarios_rl_dataset \
  --glb-directory /path/to/glb_library \
  --output-dir evaluation_results/deaco_green
```

Report mean and standard deviation per connection for:

- path length `L`
- bend count `N_bend`
- altitude change `H_alt`
- operating energy `E_op`
- embodied carbon `CO2_emb`
- operational carbon `CO2_op`
- success rate
- policy inference overhead

## 6. Acceptance Checks

A public artifact run is considered healthy when:

- all Python entry points pass `py_compile`
- `--help` works for training and evaluation scripts
- fixed DEACO-Green can route at least one compatible public or user-provided scenario
- the trained-policy evaluator writes `test_metrics.json` and detailed connection records
- no private GLB assets, layouts, checkpoints, logs, or generated outputs are tracked

Exact numerical reproduction of the paper tables requires equivalent GLB assets and generated layouts. Without those assets, the code supports method reproduction and regression testing on user-provided scenes.
