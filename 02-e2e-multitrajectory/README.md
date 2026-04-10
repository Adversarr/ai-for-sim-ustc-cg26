# Multi-Trajectory End-to-End Solver

This stage extends [02-e2e-solver](../02-e2e-solver) from a single reference rollout to a trajectory bank.

The goal is to show that the learned transition rule is not only memorizing one path through state space.

## Dataset Setup

We generate `500` double-pendulum trajectories by varying the initial angles and angular velocities around the default demo parameters.

- `400` trajectories are used for training
- `100` trajectories are used for validation
- the final recursive rollout test is still performed on the default trajectory from `demo_params()`

Every training script in this folder uses mini-batch optimization with:

- batch size `64`

## What Changes Compared With Stage 02

The model architecture is the same transition MLP from [src/ai_for_sim/e2e_solver.py](../src/ai_for_sim/e2e_solver.py).

The difference is in how training data is built:

1. Simulate many trajectories instead of one.
2. Convert every rollout into one-step transition pairs.
3. Concatenate all transition pairs from the training split.
4. Train with shuffled mini-batches.
5. Track validation loss on held-out trajectories.
6. Test generalization by rolling out on the default trajectory.

This better illustrates whether the learned solver has captured a reusable update rule rather than a single-trajectory shortcut.

## Commands

Run the supervised variant:

```bash
uv run python 02-e2e-multitrajectory/train_supervised.py
uv run python 02-e2e-multitrajectory/export_supervised.py
```

Run the hybrid variant:

```bash
uv run python 02-e2e-multitrajectory/train_hybrid.py
uv run python 02-e2e-multitrajectory/export_hybrid.py
```

Run both pipelines:

```bash
uv run python 02-e2e-multitrajectory/simulate.py
```

Run the tests covering the shared solver code:

```bash
uv run pytest tests/test_e2e_solver.py
```
