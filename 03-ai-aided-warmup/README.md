# AI-Aided Implicit Solver for the Double Pendulum

This stage now learns a direct warm start for each implicit Euler step. It does
not predict an LLT factor, SPD matrix, or Gauss-Newton preconditioner anymore.
The neural network predicts a state-space warmup delta, then the classical
Newton solver finishes the implicit solve from that learned warm start.

For implicit Euler, each time step solves

$$
z - y_n - \Delta t f(z) = 0.
$$

The explicit Euler guess is

$$
z_{\mathrm{explicit}} = y_n + \Delta t f(y_n).
$$

The learned model takes `(z_explicit, dt)` and predicts

$$
\Delta z_\theta \in \mathbb{R}^4,
$$

which defines the warm start

$$
z_{\mathrm{warm}} = z_{\mathrm{explicit}} + \Delta z_\theta.
$$

Newton corrections then run from `z_warm` until the implicit residual reaches
the configured tolerance. The final converged state is still produced by the
classical solver.

## Implementation

The main implementation lives in
[src/ai_for_sim/aided_solver.py](../src/ai_for_sim/aided_solver.py).

Training data is generated from exact implicit Euler solves over a trajectory
bank. For each step, the dataset stores:

- the explicit Euler iterate and `dt` as input features,
- the target warmup delta `z_exact - z_explicit`,
- the exact converged implicit state.

The training loss is direct regression on the warmup delta. Exported artifacts
still compare:

- explicit residual norm before the learned warmup,
- residual norm after the learned warmup,
- Newton correction counts after warm starting,
- final aided trajectory versus the exact implicit rollout.

## Artifacts

Training saves a checkpoint under
[03-ai-aided-warmup/artifacts](./artifacts):

- `direct_warmup/direct_warmup_checkpoint.pt`

Export generates:

- `direct_warmup_prediction.npz`
- `direct_warmup_summary.png`
- `direct_warmup_loss.png`
- `direct_warmup_animation.gif`
- `direct_warmup_comparison.gif`

The `.npz` file stores the exact implicit rollout, aided rollout, residual
histories, per-step iteration counts, warm-start residual ratios, and time
grid.

## Commands

Train the aided solver:

```bash
uv run python 03-ai-aided-warmup/train.py
```

Export artifacts:

```bash
uv run python 03-ai-aided-warmup/export.py
```

Run the full stage:

```bash
uv run python 03-ai-aided-warmup/simulate.py
```

Run the tests:

```bash
uv run pytest tests/test_aided_solver.py
```
