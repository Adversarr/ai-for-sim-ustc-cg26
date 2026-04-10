from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ai_for_sim.aided_preconditioner import (
    AidedModelConfig,
    AidedTrainingConfig,
    ImplicitSolverConfig,
    export_preconditioner_artifacts,
    load_checkpoint,
    save_checkpoint_payload,
    train_gauss_newton_preconditioner,
)
from ai_for_sim.aided_solver import (
    conjugate_gradient_direction,
    demo_params,
    exact_gauss_newton_direction,
    explicit_euler_guess,
    lower_triangular_spd_from_raw,
    projected_hessian_inverse,
    residual_energy,
    residual_energy_gradient,
    system_matrix,
)
from ai_for_sim.e2e_solver import TrajectorySplitConfig


def test_spd_construction_and_projected_target_are_positive_definite():
    raw_factor = torch.linspace(-0.4, 0.4, 10, dtype=torch.float32).unsqueeze(0)
    spd_matrix = lower_triangular_spd_from_raw(raw_factor, diagonal_epsilon=1e-3).squeeze(0).numpy()

    assert np.allclose(spd_matrix, spd_matrix.T, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(spd_matrix) > 0.0)

    params = demo_params()
    current_state = params.initial_state.astype(float)
    dt = 1.0 / params.fps
    iterate = explicit_euler_guess(current_state, dt, params)
    matrix = system_matrix(iterate, dt, params)
    target = projected_hessian_inverse(matrix, damping=1e-3)
    assert np.allclose(target, target.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(target) > 0.0)


def test_oracle_projected_hessian_beats_identity_on_residual_energy():
    params = demo_params()
    solver_config = ImplicitSolverConfig(nonlinear_tol=1e-8, max_iterations=12)
    current_state = params.initial_state.astype(float)
    dt = 1.0 / params.fps
    iterate = explicit_euler_guess(current_state, dt, params)

    residual, matrix, gradient = residual_energy_gradient(
        iterate,
        current_state,
        dt,
        params,
        jacobian_epsilon=solver_config.jacobian_epsilon,
    )
    del residual
    target = projected_hessian_inverse(matrix, damping=1e-3)

    identity_candidate = iterate - gradient
    oracle_candidate = iterate - target @ gradient

    identity_energy = residual_energy(identity_candidate, current_state, dt, params)
    oracle_energy = residual_energy(oracle_candidate, current_state, dt, params)
    start_energy = residual_energy(iterate, current_state, dt, params)

    assert oracle_energy < start_energy
    assert oracle_energy < identity_energy


def test_conjugate_gradient_matches_exact_gauss_newton_direction():
    params = demo_params()
    current_state = params.initial_state.astype(float)
    dt = 1.0 / params.fps
    iterate = explicit_euler_guess(current_state, dt, params)
    _, matrix, gradient = residual_energy_gradient(iterate, current_state, dt, params)

    exact_direction = exact_gauss_newton_direction(gradient, matrix, damping=1e-3)
    cg_direction, cg_iterations, converged = conjugate_gradient_direction(
        gradient,
        matrix,
        damping=1e-3,
        tol=1e-12,
        max_iterations=8,
    )

    assert converged
    assert 1 <= cg_iterations <= 4
    assert np.allclose(cg_direction, exact_direction, atol=1e-8, rtol=1e-6)


def test_preconditioner_pipeline_checkpoint_and_exports(tmp_path: Path):
    model, result = train_gauss_newton_preconditioner(
        model_config=AidedModelConfig(hidden_width=16, hidden_layers=1),
        training_config=AidedTrainingConfig(
            epochs=10,
            learning_rate=2e-3,
            min_learning_rate=1e-4,
            batch_size=32,
            log_interval=1000,
            matrix_loss_weight=0.2,
            action_loss_weight=2.0,
            device="cpu",
        ),
        solver_config=ImplicitSolverConfig(max_iterations=10, nonlinear_tol=1e-8),
        split_config=TrajectorySplitConfig(
            total_trajectories=6,
            train_trajectories=4,
            validation_trajectories=2,
            seed=5,
        ),
    )

    assert result.variant == "gauss_newton_preconditioner"
    assert result.history["loss"][-1] < result.history["loss"][0]
    assert np.isfinite(result.aided_rollout.result.state).all()
    assert np.all(result.exact_rollout.linear_iteration_counts > 0)
    assert np.all(result.aided_rollout.linear_iteration_counts > 0)

    checkpoint_path = tmp_path / "gauss_newton_preconditioner_checkpoint.pt"
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    _, loaded = load_checkpoint(checkpoint_path)
    outputs = export_preconditioner_artifacts(loaded, tmp_path / "artifacts")

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0

    data = np.load(outputs["predictions"])
    assert data["exact_state"].shape == loaded.exact_rollout.result.state.shape
    assert data["aided_state"].shape == loaded.aided_rollout.result.state.shape
    assert data["exact_linear_iteration_counts"].shape == loaded.exact_rollout.linear_iteration_counts.shape
    assert data["aided_linear_iteration_counts"].shape == loaded.aided_rollout.linear_iteration_counts.shape
