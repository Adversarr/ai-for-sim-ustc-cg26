# AI-Aided Preconditioner for the Double Pendulum

This stage is about the large-scale use case where the nonlinear implicit step
is handled by a classical Gauss-Newton or Newton method, and each linearized
subproblem is solved with CG or Newton-CG. In that setting, the learned object
is the preconditioner used inside CG: an operator that approximately applies
\(M^{-1}\) so the Krylov solve converges in fewer iterations.

The current double-pendulum implementation is only a tiny 4x4 educational
analogue of that workflow. It is not the intended deployment regime, but it
now follows the correct pipeline: outer Gauss-Newton iterations with an inner
Conjugate Gradient solve, optionally preconditioned by the learned SPD model.
The dense SPD prediction is still only a small-scale stand-in for what would
be a large sparse preconditioner in a real simulation codebase.

For implicit Euler, each step solves

$$
z - y_n - \Delta t f(z) = 0.
$$

For a large problem, the outer nonlinear iteration linearizes this system and
solves a Gauss-Newton or Newton subproblem. The inner solve is then performed
by CG or Newton-CG, where a good preconditioner is valuable because we do not
want to form or invert a dense Hessian-like matrix directly.

In the toy analogue used here, the local system matrix is

$$
A(z) = I - \Delta t J_f(z),
$$

and the small-system proxy target is

$$
M^\star(z) = (A(z)^\top A(z) + \lambda I)^{-1}.
$$

The network predicts a lower-triangular factor and constructs an SPD matrix

$$
M_\theta = L L^\top + \varepsilon I,
$$

which is used as the preconditioner inside the toy Conjugate Gradient solve. In
the true large-scale setting, the role of the model is to improve the linear
solve by approximately applying a useful preconditioning operator, not to
replace the classical nonlinear solver.

The corresponding preconditioned action in this toy setting is

$$
d = - M_\theta A^\top r,
$$

which is exactly the operator application used by the toy preconditioned CG
solve on the Gauss-Newton system.

The implementation lives in
[src/ai_for_sim/aided_preconditioner.py](../src/ai_for_sim/aided_preconditioner.py).

In this repo, each nonlinear step is solved by Gauss-Newton with an inner CG
solve. The learned model acts inside that CG loop as the preconditioner. The
classical outer solve remains in control of the final converged answer. This
keeps the example easy to inspect while still illustrating the intended role of
learned preconditioning in a classical solver stack.

This pipeline is educational. The success criterion is to show that a learned
preconditioner can reduce inner CG work inside the classical nonlinear solve
while the classical method retains robustness and accuracy. The goal is not to
beat a dense direct solve on a 4x4 system.

Current tuned defaults for this stage:

- model width `64`
- hidden layers `3`
- epochs `240`
- learning rate `1.5e-3`
- matrix loss weight `0.15`
- action loss weight `6.0`
- projection damping `1e-2`

Observed tuned demo behavior:

- accepted preconditioner rate: about `100%`
- mean classical inner CG iterations: problem-dependent
- mean preconditioned inner CG iterations: problem-dependent
- trajectory MSE to exact implicit solve: essentially `0`

This is the version that should go into the slide deck: for realistic large
problems, AI is used to learn the preconditioner inside a Gauss-Newton /
Newton-CG pipeline. The double-pendulum code here is a tiny analogue that makes
that idea concrete without claiming to be the real large-scale solver.

The summary figure for this stage is preconditioner-specific. It shows:

- preconditioned residual vs explicit-Euler residual in the toy analogue,
- the distribution of residual-ratio improvements after the first
  preconditioned Gauss-Newton step,
- the distribution of inner CG iteration counts,
- and the relationship between preconditioner quality and downstream CG work.

The goal is to explain local preconditioner quality and its effect on the
classical solve, not to present the learned module as a standalone nonlinear
solver.

## Artifacts

Training writes a checkpoint under
[03-ai-aided-preconditioner/artifacts](./artifacts):

- `gauss_newton_preconditioner/gauss_newton_preconditioner_checkpoint.pt`

Export generates:

- `gauss_newton_preconditioner_prediction.npz`
- `gauss_newton_preconditioner_summary.png`
- `gauss_newton_preconditioner_loss.png`
- `gauss_newton_preconditioner_animation.gif`
- `gauss_newton_preconditioner_comparison.gif`

These artifacts are generated from the toy 4x4 analogue, but the stage now does
run an explicit Conjugate Gradient inner solve. The limitation is scale, not
the solver structure.

## Commands

```bash
uv run python 03-ai-aided-preconditioner/train.py
uv run python 03-ai-aided-preconditioner/export.py
uv run python 03-ai-aided-preconditioner/simulate.py
uv run pytest tests/test_aided_preconditioner.py
```
