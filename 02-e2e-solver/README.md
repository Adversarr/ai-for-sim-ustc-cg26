# End-to-End Solver for the Double Pendulum

This stage learns a discrete transition rule

$$
y(t) = [\theta_1(t), \theta_2(t), \omega_1(t), \omega_2(t)]
\quad \longmapsto \quad
y(t+\Delta t).
$$

Instead of integrating the ODE directly as in [00-discrete-solver](00-discrete-solver), and instead of fitting a continuous function of time as in [01-implicit-neural-representation](01-implicit-neural-representation), this model learns a one-step state update.

The main implementation lives in [src/ai_for_sim/e2e_solver.py](src/ai_for_sim/e2e_solver.py).

## Main Idea

The numerical solver gives a trajectory

$$
y_0, y_1, y_2, \dots, y_N
$$

on a uniform time grid with spacing $\Delta t$.

From that rollout we build training pairs:

- input: the current state $y_t$
- target: the state increment

$$
\Delta y_t = y_{t+\Delta t} - y_t.
$$

The network predicts this increment, not the full next state:

$$
\widehat{\Delta y_t} = f_\theta(y_t),
\qquad
\hat{y}_{t+\Delta t} = y_t + \widehat{\Delta y_t}.
$$

This delta formulation is helpful for teaching and for optimization:

- consecutive states are usually close, so the network learns a smaller quantity,
- the update rule is easy to compare with classical numerical stepping,
- the meaning of prediction error is easier to interpret physically.

## How It Differs From the Earlier Stages

### Stage 00: numerical solver

In [00-discrete-solver](00-discrete-solver), the next state is produced by a hand-designed numerical integration algorithm applied to the ODE.

### Stage 01: implicit neural representation

In [01-implicit-neural-representation](01-implicit-neural-representation), the model takes time `t` as input and directly predicts the full state at that time.

### Stage 02: end-to-end transition model

Here the model takes the current state as input and predicts one discrete update. To generate a full trajectory, we apply the learned map recursively:

$$
\hat{y}_{k+1} = \hat{y}_k + f_\theta(\hat{y}_k).
$$

This is called a recursive rollout or free-running rollout.

## Why Recursive Rollout Matters

During training, the model usually sees true states from the reference trajectory. This is often called teacher forcing.

During evaluation, the model must consume its own previous prediction after the first step. That is harder, because a small local error at one step becomes part of the input for the next step.

This is why stage 02 is a useful teaching example:

- one-step loss may look small,
- but long-horizon rollout can still drift,
- so students can clearly see error accumulation in learned simulators.

## Variants

The folder contains two variants.

### Supervised

The supervised model minimizes the one-step delta loss

$$
\mathcal{L}_{\text{data}}
=
\frac{1}{N}\sum_{t}
\left\|
f_\theta(y_t) - (y_{t+\Delta t} - y_t)
\right\|_2^2.
$$

This variant is purely data-driven.

### Hybrid

The hybrid model keeps the supervised loss and adds a local physics prior.

From the ODE

$$
\frac{dy}{dt} = g(y),
$$

a first-order Euler estimate of the state change is

$$
\Delta y_t \approx \Delta t \, g(y_t).
$$

So the hybrid variant adds the loss

$$
\mathcal{L}_{\text{physics}}
=
\frac{1}{N}\sum_t
\left\|
f_\theta(y_t) - \Delta t \, g(y_t)
\right\|_2^2.
$$

The final objective is

$$
\mathcal{L}
=
\lambda_{\text{data}} \mathcal{L}_{\text{data}}
+
\lambda_{\text{physics}} \mathcal{L}_{\text{physics}}.
$$

This does not replace supervision. Instead, it tells the network that a good learned update should stay close to what the governing equations predict locally.

## Tuned Demo Defaults

For the teaching demo in this repository, the scripts use tuned defaults selected by recursive-rollout quality on the `6 s`, `20 fps` horizon:

- supervised:
  - model width `128`
  - hidden layers `4`
  - epochs `1200`
  - learning rate `8e-4`
- hybrid:
  - model width `128`
  - hidden layers `3`
  - epochs `900`
  - learning rate `8e-4`
  - physics weight `0.15`

These settings were chosen using rollout quality rather than only one-step loss, because long-horizon stability is the main teaching point of this stage.

## Artifacts

Training saves checkpoints under [02-e2e-solver/artifacts](02-e2e-solver/artifacts):

- `supervised/supervised_checkpoint.pt`
- `hybrid/hybrid_checkpoint.pt`

Export scripts generate:

- `*_prediction.npz`
- `*_summary.png`
- `*_loss.png`
- `*_animation.gif`
- `*_comparison.gif`

The summary figure is designed around rollout quality, so students can compare:

- reference vs. predicted angles,
- reference vs. predicted angular velocities,
- Cartesian end-mass path,
- rollout error and energy drift.

## Commands

Run the supervised variant:

```bash
uv run python 02-e2e-solver/train_supervised.py
uv run python 02-e2e-solver/export_supervised.py
```

Run the hybrid variant:

```bash
uv run python 02-e2e-solver/train_hybrid.py
uv run python 02-e2e-solver/export_hybrid.py
```

Run both pipelines:

```bash
uv run python 02-e2e-solver/simulate.py
```

Run the tests for this stage:

```bash
uv run pytest tests/test_e2e_solver.py
```
