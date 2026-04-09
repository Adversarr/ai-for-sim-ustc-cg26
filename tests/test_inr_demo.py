from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_for_sim.double_pendulum import DoublePendulumParams, state_derivative
from ai_for_sim.inr import (
    INRTrainingConfig,
    export_inr_artifacts,
    load_checkpoint,
    save_checkpoint_payload,
    sparse_reference_samples,
    train_hybrid_inr,
    train_physics_inr,
    train_supervised_inr,
)


def test_supervised_pipeline_exports_and_reaches_reasonable_error(tmp_path: Path):
    params = DoublePendulumParams(duration=2.0, fps=10)
    training_config = INRTrainingConfig(epochs=120, learning_rate=2e-3, device="cpu")
    model, result = train_supervised_inr(training_config=training_config, params=params)

    assert result.history["loss"][-1] < result.history["loss"][0]
    assert np.isfinite(result.prediction).all()
    mse = np.mean((result.prediction - result.reference.state) ** 2)
    assert mse < 1.0

    checkpoint_path = tmp_path / "supervised_checkpoint.pt"
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    _, loaded = load_checkpoint(checkpoint_path)

    outputs = export_inr_artifacts(loaded, tmp_path / "supervised")
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0

    data = np.load(outputs["predictions"])
    assert data["prediction"].shape == result.prediction.shape
    assert "reference" in data
    assert outputs["animation"].suffix == ".gif"
    assert outputs["comparison_animation"].suffix == ".gif"


def test_physics_pipeline_exports_and_respects_initial_condition(tmp_path: Path):
    params = DoublePendulumParams(duration=1.5, fps=8)
    training_config = INRTrainingConfig(
        epochs=25,
        learning_rate=0.8,
        collocation_points=81,
        ic_weight=25.0,
        lbfgs_max_iter=12,
        lbfgs_history_size=15,
        device="cpu",
    )
    model, result = train_physics_inr(training_config=training_config, params=params)

    assert result.history["loss"][-1] < result.history["loss"][0]
    assert result.history["initial_condition_loss"][-1] < result.history["initial_condition_loss"][0]
    assert np.isfinite(result.prediction).all()
    assert np.max(np.abs(result.prediction[0] - params.initial_state)) < 0.8

    finite_difference = np.gradient(result.prediction, result.time, axis=0)
    rhs = np.array([state_derivative(t, state, params) for t, state in zip(result.time, result.prediction, strict=True)])
    residual_mse = np.mean((finite_difference - rhs) ** 2)
    assert residual_mse < 12.0

    checkpoint_path = tmp_path / "physics_checkpoint.pt"
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    _, loaded = load_checkpoint(checkpoint_path)

    outputs = export_inr_artifacts(loaded, tmp_path / "physics")
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0

    data = np.load(outputs["predictions"])
    assert "reference" in data
    assert data["reference"].shape == result.prediction.shape
    assert outputs["animation"].suffix == ".gif"
    assert outputs["comparison_animation"].suffix == ".gif"


def test_hybrid_pipeline_uses_sparse_points_and_exports(tmp_path: Path):
    params = DoublePendulumParams(duration=1.5, fps=8)
    training_config = INRTrainingConfig(
        epochs=20,
        learning_rate=0.8,
        collocation_points=81,
        ic_weight=10.0,
        residual_weight=1.0,
        sparse_data_weight=20.0,
        sparse_points=9,
        lbfgs_max_iter=10,
        lbfgs_history_size=15,
        device="cpu",
    )
    model, result = train_hybrid_inr(training_config=training_config, params=params)

    assert result.history["loss"][-1] < result.history["loss"][0]
    assert result.history["sparse_data_loss"][-1] < result.history["sparse_data_loss"][0]
    assert np.isfinite(result.prediction).all()

    sparse_time, sparse_state = sparse_reference_samples(result.reference, training_config.sparse_points)
    sparse_indices = np.array([np.argmin(np.abs(result.time - value)) for value in sparse_time])
    sparse_mse = np.mean((result.prediction[sparse_indices] - sparse_state) ** 2)
    assert sparse_mse < 0.5

    checkpoint_path = tmp_path / "hybrid_checkpoint.pt"
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    _, loaded = load_checkpoint(checkpoint_path)

    outputs = export_inr_artifacts(loaded, tmp_path / "hybrid")
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0

    data = np.load(outputs["predictions"])
    assert "reference" in data
    assert data["reference"].shape == result.prediction.shape
    assert outputs["animation"].suffix == ".gif"
    assert outputs["comparison_animation"].suffix == ".gif"
