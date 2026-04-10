from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_for_sim.aided_solver import (
    AidedModelConfig,
    AidedTrainingConfig,
    DIRECT_WARMUP_VARIANT,
    ImplicitSolverConfig,
    _build_warmup_training_dataset,
    build_model,
    demo_params,
    explicit_euler_guess,
    export_aided_solver_artifacts,
    get_device,
    implicit_residual,
    load_checkpoint,
    save_checkpoint_payload,
    simulate_ai_aided_implicit_euler,
    simulate_implicit_euler,
    system_matrix,
    train_direct_warmup_solver,
)
from ai_for_sim.e2e_solver import TrajectorySplitConfig


def test_system_matrix_matches_finite_difference_residual_jacobian():
    params = demo_params()
    current_state = params.initial_state.astype(float)
    dt = 1.0 / params.fps
    iterate = explicit_euler_guess(current_state, dt, params)

    matrix = system_matrix(iterate, dt, params, jacobian_epsilon=1e-7)
    finite_difference = np.zeros_like(matrix)
    eps = 1e-7
    for column in range(4):
        perturbation = np.zeros(4, dtype=float)
        perturbation[column] = eps
        residual_plus = implicit_residual(iterate + perturbation, current_state, dt, params)
        residual_minus = implicit_residual(iterate - perturbation, current_state, dt, params)
        finite_difference[:, column] = (residual_plus - residual_minus) / (2.0 * eps)

    assert np.allclose(matrix, finite_difference, atol=1e-5, rtol=1e-4)


def test_warmup_dataset_targets_delta_from_explicit_to_exact():
    params = demo_params()
    split_config = TrajectorySplitConfig(
        total_trajectories=4,
        train_trajectories=2,
        validation_trajectories=2,
        seed=11,
    )
    solver_config = ImplicitSolverConfig(nonlinear_tol=1e-8, max_iterations=12)

    train_dataset, _, dt = _build_warmup_training_dataset(
        params=params,
        split_config=split_config,
        solver_config=solver_config,
    )

    explicit_iterate = train_dataset["explicit_iterate"][0]
    target_delta = train_dataset["target_delta"][0]
    target_state = train_dataset["target_state"][0]

    assert train_dataset["features"].shape[1] == 5
    assert np.isclose(train_dataset["features"][0, -1], dt)
    assert np.allclose(train_dataset["features"][0, :4], explicit_iterate)
    assert np.allclose(explicit_iterate + target_delta, target_state)


def test_exact_implicit_euler_returns_finite_converged_rollout():
    params = demo_params()
    solver_config = ImplicitSolverConfig(nonlinear_tol=1e-8, max_iterations=12)
    rollout = simulate_implicit_euler(params=params, solver_config=solver_config)

    assert rollout.result.state.shape == (len(params.time_grid), 4)
    assert np.isfinite(rollout.result.state).all()
    assert np.all(rollout.converged)
    assert np.all(rollout.iteration_counts >= 1)


def test_training_pipeline_checkpoint_and_exports(tmp_path: Path):
    params = demo_params()
    split_config = TrajectorySplitConfig(
        total_trajectories=6,
        train_trajectories=4,
        validation_trajectories=2,
        seed=3,
    )
    training_config = AidedTrainingConfig(
        epochs=20,
        learning_rate=2e-3,
        min_learning_rate=1e-4,
        batch_size=32,
        log_interval=1000,
        device="cpu",
    )
    solver_config = ImplicitSolverConfig(
        nonlinear_tol=1e-8,
        max_iterations=10,
        min_step_size=1e-4,
    )

    model, result = train_direct_warmup_solver(
        model_config=AidedModelConfig(hidden_width=32, hidden_layers=2),
        training_config=training_config,
        params=params,
        solver_config=solver_config,
        split_config=split_config,
    )

    assert result.variant == DIRECT_WARMUP_VARIANT
    assert result.history["loss"][-1] < result.history["loss"][0]
    assert result.history["val_loss"][-1] < result.history["val_loss"][0]
    assert np.isfinite(result.aided_rollout.result.state).all()
    assert np.all(result.aided_rollout.fallback_steps == 0)
    assert result.metadata["train_trajectories"] == 4
    assert result.metadata["validation_trajectories"] == 2

    checkpoint_path = tmp_path / "direct_warmup_checkpoint.pt"
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    _, loaded = load_checkpoint(checkpoint_path)
    outputs = export_aided_solver_artifacts(loaded, tmp_path / "artifacts")

    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0

    data = np.load(outputs["predictions"])
    assert data["exact_state"].shape == result.exact_rollout.result.state.shape
    assert data["aided_state"].shape == result.aided_rollout.result.state.shape
    assert np.allclose(data["aided_fallback_steps"], 0.0)


def test_ai_aided_rollout_converges_with_direct_warmup_model():
    model = build_model(AidedModelConfig(hidden_width=16, hidden_layers=1), get_device("cpu"))
    params = demo_params()
    solver_config = ImplicitSolverConfig(
        nonlinear_tol=1e-8,
        max_iterations=10,
        fallback_to_newton=True,
    )

    rollout = simulate_ai_aided_implicit_euler(model, params=params, solver_config=solver_config)

    assert rollout.result.state.shape == (len(params.time_grid), 4)
    assert np.isfinite(rollout.result.state).all()
    assert np.all(rollout.converged)
    assert np.all(rollout.iteration_counts >= 1)
