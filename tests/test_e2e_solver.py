from __future__ import annotations

from pathlib import Path

import numpy as np

from ai_for_sim.double_pendulum import DoublePendulumParams
from ai_for_sim.e2e_solver import (
    E2ETrainingConfig,
    TrajectorySplitConfig,
    build_transition_dataset,
    build_transition_dataset_from_references,
    build_multitrajectory_references,
    evaluate_one_step,
    export_e2e_artifacts,
    load_checkpoint,
    save_checkpoint_payload,
    train_hybrid_e2e,
    train_hybrid_multitrajectory_e2e,
    train_supervised_e2e,
    train_supervised_multitrajectory_e2e,
)


def test_supervised_pipeline_exports_and_rolls_out_reasonably(tmp_path: Path):
    params = DoublePendulumParams(duration=2.0, fps=10)
    training_config = E2ETrainingConfig(epochs=180, learning_rate=2e-3, log_interval=1000, device="cpu")
    model, result = train_supervised_e2e(training_config=training_config, params=params)

    assert result.history["loss"][-1] < result.history["loss"][0]
    assert np.isfinite(result.rollout_prediction).all()
    assert result.rollout_prediction.shape == result.reference.state.shape
    assert np.allclose(result.time, result.reference.time)
    assert np.allclose(result.rollout_prediction[0], params.initial_state)

    input_state, target_delta, _ = build_transition_dataset(result.reference)
    predicted_delta = evaluate_one_step(model, input_state, device_name := "cpu")
    del device_name
    one_step_mse = np.mean((predicted_delta - target_delta) ** 2)
    rollout_mse = np.mean((result.rollout_prediction - result.reference.state) ** 2)
    assert one_step_mse < 0.1
    assert rollout_mse < 8.0

    checkpoint_path = tmp_path / "supervised_checkpoint.pt"
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    _, loaded = load_checkpoint(checkpoint_path)

    outputs = export_e2e_artifacts(loaded, tmp_path / "supervised")
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0

    data = np.load(outputs["predictions"])
    assert data["prediction"].shape == result.rollout_prediction.shape
    assert data["reference"].shape == result.reference.state.shape
    assert outputs["animation"].suffix == ".gif"
    assert outputs["comparison_animation"].suffix == ".gif"


def test_hybrid_pipeline_tracks_data_and_physics_losses(tmp_path: Path):
    params = DoublePendulumParams(duration=1.5, fps=8)
    training_config = E2ETrainingConfig(
        epochs=160,
        learning_rate=2e-3,
        log_interval=1000,
        supervised_weight=1.0,
        physics_weight=0.5,
        device="cpu",
    )
    model, result = train_hybrid_e2e(training_config=training_config, params=params)

    assert result.history["loss"][-1] < result.history["loss"][0]
    assert result.history["data_loss"][-1] < result.history["data_loss"][0]
    assert result.history["physics_loss"][-1] < result.history["physics_loss"][0]
    assert np.isfinite(result.rollout_prediction).all()
    assert result.rollout_prediction.shape == result.reference.state.shape
    assert np.allclose(result.rollout_prediction[0], params.initial_state)

    input_state, target_delta, _ = build_transition_dataset(result.reference)
    predicted_delta = evaluate_one_step(model, input_state, "cpu")
    one_step_mse = np.mean((predicted_delta - target_delta) ** 2)
    rollout_mse = np.mean((result.rollout_prediction - result.reference.state) ** 2)
    assert one_step_mse < 0.15
    assert rollout_mse < 8.0

    checkpoint_path = tmp_path / "hybrid_checkpoint.pt"
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    _, loaded = load_checkpoint(checkpoint_path)

    outputs = export_e2e_artifacts(loaded, tmp_path / "hybrid")
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0

    data = np.load(outputs["predictions"])
    assert "reference" in data
    assert float(data["dt"]) > 0.0


def test_multitrajectory_supervised_pipeline_tracks_validation_and_default_test_rollout(tmp_path: Path):
    params = DoublePendulumParams(duration=1.5, fps=8)
    split_config = TrajectorySplitConfig(
        total_trajectories=12,
        train_trajectories=8,
        validation_trajectories=4,
        seed=5,
    )
    training_config = E2ETrainingConfig(
        epochs=60,
        learning_rate=2e-3,
        log_interval=1000,
        batch_size=16,
        device="cpu",
    )

    train_references, validation_references = build_multitrajectory_references(params, split_config)
    assert len(train_references) == 8
    assert len(validation_references) == 4

    train_input_state, train_target_delta, train_dt = build_transition_dataset_from_references(train_references)
    validation_input_state, validation_target_delta, validation_dt = build_transition_dataset_from_references(
        validation_references
    )
    assert train_input_state.shape[0] > validation_input_state.shape[0]
    assert train_target_delta.shape == train_input_state.shape
    assert validation_target_delta.shape == validation_input_state.shape
    assert np.isclose(train_dt, validation_dt)

    model, result = train_supervised_multitrajectory_e2e(
        training_config=training_config,
        params=params,
        split_config=split_config,
    )

    assert result.history["loss"][-1] < result.history["loss"][0]
    assert result.history["val_loss"][-1] < result.history["val_loss"][0]
    assert np.isfinite(result.rollout_prediction).all()
    assert result.rollout_prediction.shape == result.reference.state.shape
    assert np.allclose(result.rollout_prediction[0], params.initial_state)
    assert result.metadata["train_trajectories"] == 8
    assert result.metadata["validation_trajectories"] == 4
    assert result.training_config.batch_size == 16

    default_input_state, default_target_delta, _ = build_transition_dataset(result.reference)
    predicted_delta = evaluate_one_step(model, default_input_state, "cpu")
    default_one_step_mse = np.mean((predicted_delta - default_target_delta) ** 2)
    assert default_one_step_mse < 0.4

    checkpoint_path = tmp_path / "multitrajectory_supervised_checkpoint.pt"
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    _, loaded = load_checkpoint(checkpoint_path)
    assert loaded.metadata["total_trajectories"] == 12

    outputs = export_e2e_artifacts(loaded, tmp_path / "multitrajectory_supervised")
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_multitrajectory_hybrid_pipeline_tracks_validation_and_physics(tmp_path: Path):
    params = DoublePendulumParams(duration=1.5, fps=8)
    split_config = TrajectorySplitConfig(
        total_trajectories=10,
        train_trajectories=6,
        validation_trajectories=4,
        seed=11,
    )
    training_config = E2ETrainingConfig(
        epochs=60,
        learning_rate=2e-3,
        log_interval=1000,
        batch_size=12,
        supervised_weight=1.0,
        physics_weight=0.1,
        device="cpu",
    )

    _, result = train_hybrid_multitrajectory_e2e(
        training_config=training_config,
        params=params,
        split_config=split_config,
    )

    assert result.history["loss"][-1] < result.history["loss"][0]
    assert result.history["data_loss"][-1] < result.history["data_loss"][0]
    assert result.history["physics_loss"][-1] < result.history["physics_loss"][0]
    assert result.history["val_loss"][-1] < result.history["val_loss"][0]
    assert result.history["val_physics_loss"][-1] < result.history["val_physics_loss"][0]
    assert np.isfinite(result.rollout_prediction).all()
    assert result.rollout_prediction.shape == result.reference.state.shape
    assert result.metadata["train_trajectories"] == 6
    assert result.metadata["validation_trajectories"] == 4
