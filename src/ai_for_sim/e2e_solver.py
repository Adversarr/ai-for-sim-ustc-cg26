from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .double_pendulum import (
    DoublePendulumParams,
    SimulationResult,
    cartesian_positions,
    simulate_double_pendulum,
    total_energy,
)


@dataclass(frozen=True)
class E2EModelConfig:
    hidden_width: int = 64
    hidden_layers: int = 3


@dataclass(frozen=True)
class E2ETrainingConfig:
    epochs: int = 500
    optimizer: Literal["adam"] = "adam"
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    seed: int = 7
    log_interval: int = 50
    device: str = "cpu"
    input_noise_std: float = 1e-2
    supervised_weight: float = 1.0
    physics_weight: float = 0.05
    batch_size: int | None = None


@dataclass(frozen=True)
class TrajectorySplitConfig:
    total_trajectories: int = 500
    train_trajectories: int = 400
    validation_trajectories: int = 100
    seed: int = 17
    theta1_offset_range: tuple[float, float] = (-0.8, 0.8)
    theta2_offset_range: tuple[float, float] = (-0.8, 0.8)
    omega1_range: tuple[float, float] = (-1.5, 1.5)
    omega2_range: tuple[float, float] = (-1.5, 1.5)


@dataclass(frozen=True)
class TrainedE2EResult:
    variant: str
    model_config: E2EModelConfig
    training_config: E2ETrainingConfig
    params: DoublePendulumParams
    dt: float
    rollout_prediction: np.ndarray
    history: dict[str, list[float]]
    reference: SimulationResult
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time(self) -> np.ndarray:
        return self.reference.time

    @property
    def prediction(self) -> np.ndarray:
        return self.rollout_prediction

    @property
    def prediction_result(self) -> SimulationResult:
        return SimulationResult(time=self.time, state=self.rollout_prediction, params=self.params)


class TransitionMLP(nn.Module):
    """Map one state y(t) to a delta that approximates y(t + dt) - y(t)."""

    def __init__(self, config: E2EModelConfig) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = 4
        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_width))
            layers.append(nn.SiLU())
            input_dim = config.hidden_width
        layers.append(nn.Linear(input_dim, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


def demo_params() -> DoublePendulumParams:
    return DoublePendulumParams(duration=6.0, fps=20)


def tuned_supervised_model_config() -> E2EModelConfig:
    """Configuration chosen from rollout-based tuning on the demo horizon."""

    return E2EModelConfig(hidden_width=64, hidden_layers=4)


def tuned_hybrid_model_config() -> E2EModelConfig:
    """A slightly smaller depth works better once the physics prior is added."""

    return E2EModelConfig(hidden_width=96, hidden_layers=4)


def quick_supervised_config() -> E2ETrainingConfig:
    return E2ETrainingConfig(
        epochs=5000,
        learning_rate=1.5e-3,
        log_interval=100,
    )


def quick_hybrid_config() -> E2ETrainingConfig:
    return E2ETrainingConfig(
        epochs=5000,
        learning_rate=1.5e-3,
        log_interval=100,
        input_noise_std=2e-2,
        supervised_weight=1.0,
        physics_weight=1e-4,
    )


def quick_multitrajectory_supervised_config() -> E2ETrainingConfig:
    return E2ETrainingConfig(
        epochs=50,
        learning_rate=1.5e-3,
        log_interval=10,
        batch_size=64,
    )


def quick_multitrajectory_hybrid_config() -> E2ETrainingConfig:
    return E2ETrainingConfig(
        epochs=50,
        learning_rate=1.5e-3,
        log_interval=10,
        input_noise_std=2e-2,
        supervised_weight=1.0,
        physics_weight=1e-4,
        batch_size=64,
    )


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_model(config: E2EModelConfig, device: torch.device) -> TransitionMLP:
    return TransitionMLP(config).to(device)


def build_optimizer(model: nn.Module, training_config: E2ETrainingConfig) -> torch.optim.Optimizer:
    if training_config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
        )
    raise ValueError(f"Unsupported optimizer: {training_config.optimizer}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: E2ETrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Use one cosine decay over the full run with no warm restart."""

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.epochs,
        eta_min=training_config.min_learning_rate,
    )


def build_transition_dataset(reference: SimulationResult) -> tuple[np.ndarray, np.ndarray, float]:
    """Build one-step samples from the numerical trajectory.

    The solver gives a full sequence y_0, y_1, ..., y_N. For end-to-end training we
    reinterpret that rollout as many local examples:

    - input: current state y_t
    - target: state increment y_{t+dt} - y_t

    This is the core difference from the INR demo, where the model takes time as input.
    """

    if len(reference.time) < 2:
        raise ValueError("Need at least two time samples to build one-step transitions.")

    dt = float(reference.time[1] - reference.time[0])
    input_state = reference.state[:-1]
    target_delta = reference.state[1:] - reference.state[:-1]
    return input_state, target_delta, dt


def build_transition_dataset_from_references(
    references: list[SimulationResult],
) -> tuple[np.ndarray, np.ndarray, float]:
    if not references:
        raise ValueError("Need at least one trajectory to build a transition dataset.")

    input_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    dt_reference: float | None = None
    for reference in references:
        input_state, target_delta, dt = build_transition_dataset(reference)
        if dt_reference is None:
            dt_reference = dt
        elif not np.isclose(dt_reference, dt):
            raise ValueError("All trajectories must share the same time step.")
        input_batches.append(input_state)
        target_batches.append(target_delta)

    return np.concatenate(input_batches, axis=0), np.concatenate(target_batches, axis=0), float(dt_reference)


def default_trajectory_split_config() -> TrajectorySplitConfig:
    return TrajectorySplitConfig()


def generate_trajectory_bank(
    base_params: DoublePendulumParams,
    split_config: TrajectorySplitConfig,
) -> list[DoublePendulumParams]:
    if split_config.total_trajectories <= 0:
        raise ValueError("total_trajectories must be positive.")
    if split_config.train_trajectories < 0 or split_config.validation_trajectories < 0:
        raise ValueError("Trajectory split sizes must be non-negative.")
    if split_config.train_trajectories + split_config.validation_trajectories != split_config.total_trajectories:
        raise ValueError("train_trajectories + validation_trajectories must equal total_trajectories.")

    rng = np.random.default_rng(split_config.seed)
    params_list: list[DoublePendulumParams] = []
    for _ in range(split_config.total_trajectories):
        params_list.append(
            replace(
                base_params,
                theta1_0=base_params.theta1_0
                + float(rng.uniform(*split_config.theta1_offset_range)),
                theta2_0=base_params.theta2_0
                + float(rng.uniform(*split_config.theta2_offset_range)),
                omega1_0=float(rng.uniform(*split_config.omega1_range)),
                omega2_0=float(rng.uniform(*split_config.omega2_range)),
            )
        )
    return params_list


def build_multitrajectory_references(
    params: DoublePendulumParams,
    split_config: TrajectorySplitConfig | None = None,
) -> tuple[list[SimulationResult], list[SimulationResult]]:
    split_config = split_config or default_trajectory_split_config()
    trajectory_params = generate_trajectory_bank(params, split_config)
    references = [simulate_double_pendulum(trajectory_param) for trajectory_param in trajectory_params]
    train_count = split_config.train_trajectories
    return references[:train_count], references[train_count:]


def add_input_noise(input_state: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Perturb only the model input so training sees a small neighborhood of states."""

    if noise_std <= 0.0:
        return input_state
    return input_state + noise_std * torch.randn_like(input_state)


def torch_rhs(state: torch.Tensor, params: DoublePendulumParams) -> torch.Tensor:
    theta1 = state[:, 0]
    theta2 = state[:, 1]
    omega1 = state[:, 2]
    omega2 = state[:, 3]
    delta = theta1 - theta2

    m1 = params.mass1
    m2 = params.mass2
    l1 = params.length1
    l2 = params.length2
    g = params.gravity

    sin_delta = torch.sin(delta)
    cos_delta = torch.cos(delta)
    denominator = m1 + m2 * sin_delta.square()

    theta1_dot = omega1
    theta2_dot = omega2
    omega1_dot = (
        m2 * g * torch.sin(theta2) * cos_delta
        - m2 * sin_delta * (l1 * omega1.square() * cos_delta + l2 * omega2.square())
        - (m1 + m2) * g * torch.sin(theta1)
    ) / (l1 * denominator)
    omega2_dot = (
        (m1 + m2)
        * (
            l1 * omega1.square() * sin_delta
            - g * torch.sin(theta2)
            + g * torch.sin(theta1) * cos_delta
        )
        + m2 * l2 * omega2.square() * sin_delta * cos_delta
    ) / (l2 * denominator)
    return torch.stack([theta1_dot, theta2_dot, omega1_dot, omega2_dot], dim=1)


def rollout_model(model: nn.Module, initial_state: np.ndarray, steps: int, device: torch.device) -> np.ndarray:
    """Free-run the learned transition model over the whole horizon.

    Training uses teacher forcing because each sample is paired with the true current
    state. Evaluation is harder: after the first step, the model must consume its own
    previous prediction. This recursive rollout is where error accumulation becomes
    visible, so we export it as the main teaching artifact.
    """

    model.eval()
    rollout = np.zeros((steps, 4), dtype=np.float32)
    rollout[0] = initial_state.astype(np.float32)

    with torch.no_grad():
        current_state = torch.tensor(initial_state[None, :], dtype=torch.float32, device=device)
        for step in range(1, steps):
            predicted_delta = model(current_state)
            next_state = current_state + predicted_delta
            rollout[step] = next_state.squeeze(0).cpu().numpy()
            current_state = next_state

    return rollout


def evaluate_one_step(model: nn.Module, input_state: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_state, dtype=torch.float32, device=device)
        predicted_delta = model(input_tensor)
    return predicted_delta.cpu().numpy()


def _batch_size_for_dataset(dataset_size: int, training_config: E2ETrainingConfig) -> int:
    if training_config.batch_size is None:
        return dataset_size
    return max(1, min(training_config.batch_size, dataset_size))


def _evaluate_supervised_dataset(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_delta_tensor: torch.Tensor,
    training_config: E2ETrainingConfig,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        noisy_input = add_input_noise(input_tensor, training_config.input_noise_std)
        predicted_delta = model(noisy_input)
        data_loss = F.mse_loss(predicted_delta, target_delta_tensor)
        loss = training_config.supervised_weight * data_loss
    return float(loss.detach().cpu()), float(data_loss.detach().cpu())


def _evaluate_hybrid_dataset(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_delta_tensor: torch.Tensor,
    dt: float,
    params: DoublePendulumParams,
    training_config: E2ETrainingConfig,
) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        noisy_input = add_input_noise(input_tensor, training_config.input_noise_std)
        predicted_delta = model(noisy_input)
        data_loss = F.mse_loss(predicted_delta, target_delta_tensor)
        physics_target = dt * torch_rhs(noisy_input, params)
        physics_loss = F.mse_loss(predicted_delta, physics_target)
        loss = (
            training_config.supervised_weight * data_loss
            + training_config.physics_weight * physics_loss
        )
    return (
        float(loss.detach().cpu()),
        float(data_loss.detach().cpu()),
        float(physics_loss.detach().cpu()),
    )


def train_supervised_e2e(
    model_config: E2EModelConfig | None = None,
    training_config: E2ETrainingConfig | None = None,
    params: DoublePendulumParams | None = None,
) -> tuple[nn.Module, TrainedE2EResult]:
    model_config = model_config or E2EModelConfig()
    training_config = training_config or quick_supervised_config()
    params = params or demo_params()

    set_seed(training_config.seed)
    reference = simulate_double_pendulum(params)
    input_state, target_delta, dt = build_transition_dataset(reference)

    device = get_device(training_config.device)
    model = build_model(model_config, device)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    input_tensor = torch.tensor(input_state, dtype=torch.float32, device=device)
    target_delta_tensor = torch.tensor(target_delta, dtype=torch.float32, device=device)

    history = {"loss": [], "data_loss": []}
    for epoch in range(training_config.epochs):
        optimizer.zero_grad()

        noisy_input = add_input_noise(input_tensor, training_config.input_noise_std)

        # Predicting a delta is easier to learn than predicting the full next state,
        # because most consecutive frames differ by a comparatively small increment.
        predicted_delta = model(noisy_input)
        # We keep MSE because large one-step misses tend to poison all later steps in
        # recursive rollout. Squaring the error makes those dangerous local mistakes
        # more expensive during training.
        data_loss = F.mse_loss(predicted_delta, target_delta_tensor)
        loss = training_config.supervised_weight * data_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        history["loss"].append(float(loss.detach().cpu()))
        history["data_loss"].append(float(data_loss.detach().cpu()))
        if epoch % training_config.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}: total_loss={history['loss'][-1]:.6f}, "
                f"data_loss={history['data_loss'][-1]:.6f}, lr={current_lr:.6e}"
            )

    rollout_prediction = rollout_model(model, params.initial_state, len(reference.time), device)
    return model, TrainedE2EResult(
        variant="supervised",
        model_config=model_config,
        training_config=training_config,
        params=params,
        dt=dt,
        rollout_prediction=rollout_prediction,
        history=history,
        reference=reference,
    )


def train_hybrid_e2e(
    model_config: E2EModelConfig | None = None,
    training_config: E2ETrainingConfig | None = None,
    params: DoublePendulumParams | None = None,
) -> tuple[nn.Module, TrainedE2EResult]:
    model_config = model_config or E2EModelConfig()
    training_config = training_config or quick_hybrid_config()
    params = params or demo_params()

    set_seed(training_config.seed)
    reference = simulate_double_pendulum(params)
    input_state, target_delta, dt = build_transition_dataset(reference)

    device = get_device(training_config.device)
    model = build_model(model_config, device)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    input_tensor = torch.tensor(input_state, dtype=torch.float32, device=device)
    target_delta_tensor = torch.tensor(target_delta, dtype=torch.float32, device=device)

    history = {"loss": [], "data_loss": [], "physics_loss": []}
    for epoch in range(training_config.epochs):
        optimizer.zero_grad()

        noisy_input = add_input_noise(input_tensor, training_config.input_noise_std)
        predicted_delta = model(noisy_input)
        data_loss = F.mse_loss(predicted_delta, target_delta_tensor)

        # The hybrid loss tells students how local physics enters a transition model:
        # over one short step, the Euler increment dt * f(y_t) is a first-order guess
        # for the true state change. The neural model can improve on that guess while
        # still being nudged toward physically plausible updates. The target must be
        # built from the same noisy state seen by the model; otherwise we regularize
        # the network toward physics from a different input than the one it consumed.
        physics_target = dt * torch_rhs(noisy_input, params)
        physics_loss = F.mse_loss(predicted_delta, physics_target)
        loss = (
            training_config.supervised_weight * data_loss
            + training_config.physics_weight * physics_loss
        )

        loss.backward()
        optimizer.step()
        scheduler.step()

        history["loss"].append(float(loss.detach().cpu()))
        history["data_loss"].append(float(data_loss.detach().cpu()))
        history["physics_loss"].append(float(physics_loss.detach().cpu()))
        if epoch % training_config.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}: total_loss={history['loss'][-1]:.6f}, "
                f"data_loss={history['data_loss'][-1]:.6f}, physics_loss={history['physics_loss'][-1]:.6f}, "
                f"lr={current_lr:.6e}"
            )

    rollout_prediction = rollout_model(model, params.initial_state, len(reference.time), device)
    return model, TrainedE2EResult(
        variant="hybrid",
        model_config=model_config,
        training_config=training_config,
        params=params,
        dt=dt,
        rollout_prediction=rollout_prediction,
        history=history,
        reference=reference,
    )


def train_supervised_multitrajectory_e2e(
    model_config: E2EModelConfig | None = None,
    training_config: E2ETrainingConfig | None = None,
    params: DoublePendulumParams | None = None,
    split_config: TrajectorySplitConfig | None = None,
) -> tuple[nn.Module, TrainedE2EResult]:
    model_config = model_config or tuned_supervised_model_config()
    training_config = training_config or quick_multitrajectory_supervised_config()
    params = params or demo_params()
    split_config = split_config or default_trajectory_split_config()

    set_seed(training_config.seed)
    train_references, validation_references = build_multitrajectory_references(params, split_config)
    train_input_state, train_target_delta, dt = build_transition_dataset_from_references(train_references)
    validation_input_state, validation_target_delta, validation_dt = build_transition_dataset_from_references(
        validation_references
    )
    if not np.isclose(dt, validation_dt):
        raise ValueError("Train and validation trajectories must share the same time step.")

    device = get_device(training_config.device)
    model = build_model(model_config, device)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    train_input_tensor = torch.tensor(train_input_state, dtype=torch.float32, device=device)
    train_target_delta_tensor = torch.tensor(train_target_delta, dtype=torch.float32, device=device)
    validation_input_tensor = torch.tensor(validation_input_state, dtype=torch.float32, device=device)
    validation_target_delta_tensor = torch.tensor(validation_target_delta, dtype=torch.float32, device=device)

    batch_size = _batch_size_for_dataset(len(train_input_tensor), training_config)
    history = {"loss": [], "data_loss": [], "val_loss": [], "val_data_loss": []}
    for epoch in range(training_config.epochs):
        model.train()
        permutation = torch.randperm(len(train_input_tensor), device=device)
        epoch_loss = 0.0
        epoch_data_loss = 0.0

        for start in range(0, len(train_input_tensor), batch_size):
            indices = permutation[start : start + batch_size]
            input_batch = train_input_tensor[indices]
            target_batch = train_target_delta_tensor[indices]

            optimizer.zero_grad()
            noisy_input = add_input_noise(input_batch, training_config.input_noise_std)
            predicted_delta = model(noisy_input)
            data_loss = F.mse_loss(predicted_delta, target_batch)
            loss = training_config.supervised_weight * data_loss
            loss.backward()
            optimizer.step()

            batch_items = len(indices)
            epoch_loss += float(loss.detach().cpu()) * batch_items
            epoch_data_loss += float(data_loss.detach().cpu()) * batch_items

        scheduler.step()

        history["loss"].append(epoch_loss / len(train_input_tensor))
        history["data_loss"].append(epoch_data_loss / len(train_input_tensor))
        val_loss, val_data_loss = _evaluate_supervised_dataset(
            model,
            validation_input_tensor,
            validation_target_delta_tensor,
            training_config,
        )
        history["val_loss"].append(val_loss)
        history["val_data_loss"].append(val_data_loss)
        if epoch % training_config.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}: total_loss={history['loss'][-1]:.6f}, "
                f"data_loss={history['data_loss'][-1]:.6f}, "
                f"val_loss={history['val_loss'][-1]:.6f}, "
                f"val_data_loss={history['val_data_loss'][-1]:.6f}, "
                f"lr={current_lr:.6e}"
            )

    reference = simulate_double_pendulum(params)
    rollout_prediction = rollout_model(model, params.initial_state, len(reference.time), device)
    return model, TrainedE2EResult(
        variant="multitrajectory_supervised",
        model_config=model_config,
        training_config=training_config,
        params=params,
        dt=dt,
        rollout_prediction=rollout_prediction,
        history=history,
        reference=reference,
        metadata={
            "train_trajectories": split_config.train_trajectories,
            "validation_trajectories": split_config.validation_trajectories,
            "total_trajectories": split_config.total_trajectories,
            "train_samples": int(len(train_input_state)),
            "validation_samples": int(len(validation_input_state)),
        },
    )


def train_hybrid_multitrajectory_e2e(
    model_config: E2EModelConfig | None = None,
    training_config: E2ETrainingConfig | None = None,
    params: DoublePendulumParams | None = None,
    split_config: TrajectorySplitConfig | None = None,
) -> tuple[nn.Module, TrainedE2EResult]:
    model_config = model_config or tuned_hybrid_model_config()
    training_config = training_config or quick_multitrajectory_hybrid_config()
    params = params or demo_params()
    split_config = split_config or default_trajectory_split_config()

    set_seed(training_config.seed)
    train_references, validation_references = build_multitrajectory_references(params, split_config)
    train_input_state, train_target_delta, dt = build_transition_dataset_from_references(train_references)
    validation_input_state, validation_target_delta, validation_dt = build_transition_dataset_from_references(
        validation_references
    )
    if not np.isclose(dt, validation_dt):
        raise ValueError("Train and validation trajectories must share the same time step.")

    device = get_device(training_config.device)
    model = build_model(model_config, device)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    train_input_tensor = torch.tensor(train_input_state, dtype=torch.float32, device=device)
    train_target_delta_tensor = torch.tensor(train_target_delta, dtype=torch.float32, device=device)
    validation_input_tensor = torch.tensor(validation_input_state, dtype=torch.float32, device=device)
    validation_target_delta_tensor = torch.tensor(validation_target_delta, dtype=torch.float32, device=device)

    batch_size = _batch_size_for_dataset(len(train_input_tensor), training_config)
    history = {
        "loss": [],
        "data_loss": [],
        "physics_loss": [],
        "val_loss": [],
        "val_data_loss": [],
        "val_physics_loss": [],
    }
    for epoch in range(training_config.epochs):
        model.train()
        permutation = torch.randperm(len(train_input_tensor), device=device)
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0

        for start in range(0, len(train_input_tensor), batch_size):
            indices = permutation[start : start + batch_size]
            input_batch = train_input_tensor[indices]
            target_batch = train_target_delta_tensor[indices]

            optimizer.zero_grad()
            noisy_input = add_input_noise(input_batch, training_config.input_noise_std)
            predicted_delta = model(noisy_input)
            data_loss = F.mse_loss(predicted_delta, target_batch)
            physics_target = dt * torch_rhs(noisy_input, params)
            physics_loss = F.mse_loss(predicted_delta, physics_target)
            loss = (
                training_config.supervised_weight * data_loss
                + training_config.physics_weight * physics_loss
            )
            loss.backward()
            optimizer.step()

            batch_items = len(indices)
            epoch_loss += float(loss.detach().cpu()) * batch_items
            epoch_data_loss += float(data_loss.detach().cpu()) * batch_items
            epoch_physics_loss += float(physics_loss.detach().cpu()) * batch_items

        scheduler.step()

        history["loss"].append(epoch_loss / len(train_input_tensor))
        history["data_loss"].append(epoch_data_loss / len(train_input_tensor))
        history["physics_loss"].append(epoch_physics_loss / len(train_input_tensor))
        val_loss, val_data_loss, val_physics_loss = _evaluate_hybrid_dataset(
            model,
            validation_input_tensor,
            validation_target_delta_tensor,
            dt,
            params,
            training_config,
        )
        history["val_loss"].append(val_loss)
        history["val_data_loss"].append(val_data_loss)
        history["val_physics_loss"].append(val_physics_loss)
        if epoch % training_config.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}: total_loss={history['loss'][-1]:.6f}, "
                f"data_loss={history['data_loss'][-1]:.6f}, "
                f"physics_loss={history['physics_loss'][-1]:.6f}, "
                f"val_loss={history['val_loss'][-1]:.6f}, "
                f"val_data_loss={history['val_data_loss'][-1]:.6f}, "
                f"val_physics_loss={history['val_physics_loss'][-1]:.6f}, "
                f"lr={current_lr:.6e}"
            )

    reference = simulate_double_pendulum(params)
    rollout_prediction = rollout_model(model, params.initial_state, len(reference.time), device)
    return model, TrainedE2EResult(
        variant="multitrajectory_hybrid",
        model_config=model_config,
        training_config=training_config,
        params=params,
        dt=dt,
        rollout_prediction=rollout_prediction,
        history=history,
        reference=reference,
        metadata={
            "train_trajectories": split_config.train_trajectories,
            "validation_trajectories": split_config.validation_trajectories,
            "total_trajectories": split_config.total_trajectories,
            "train_samples": int(len(train_input_state)),
            "validation_samples": int(len(validation_input_state)),
        },
    )


def save_checkpoint_payload(*, model: nn.Module, result: TrainedE2EResult, checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": result.variant,
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "params": asdict(result.params),
        "dt": result.dt,
        "rollout_prediction": result.rollout_prediction,
        "history": result.history,
        "reference_time": result.reference.time,
        "reference_state": result.reference.state,
        "metadata": result.metadata,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path, device_name: str = "cpu") -> tuple[nn.Module, TrainedE2EResult]:
    payload = torch.load(checkpoint_path, map_location=get_device(device_name), weights_only=False)
    model_config = E2EModelConfig(**payload["model_config"])
    training_config = E2ETrainingConfig(**payload["training_config"])
    params = DoublePendulumParams(**payload["params"])
    device = get_device(device_name)
    model = build_model(model_config, device)
    model.load_state_dict(payload["model_state_dict"])

    reference = SimulationResult(
        time=np.asarray(payload["reference_time"]),
        state=np.asarray(payload["reference_state"]),
        params=params,
    )
    result = TrainedE2EResult(
        variant=payload["variant"],
        model_config=model_config,
        training_config=training_config,
        params=params,
        dt=float(payload["dt"]),
        rollout_prediction=np.asarray(payload["rollout_prediction"]),
        history={key: list(value) for key, value in payload["history"].items()},
        reference=reference,
        metadata=dict(payload.get("metadata", {})),
    )
    return model, result


def save_prediction_arrays(result: TrainedE2EResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        time=result.time,
        prediction=result.rollout_prediction,
        reference=result.reference.state,
        dt=result.dt,
    )
    return output_path


def plot_prediction_summary(result: TrainedE2EResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted = result.prediction_result
    reference = result.reference
    _, _, x2_pred, y2_pred = cartesian_positions(predicted)
    _, _, x2_ref, y2_ref = cartesian_positions(reference)

    energy_shift = total_energy(predicted) - total_energy(predicted)[0]
    rollout_error = np.linalg.norm(result.rollout_prediction - reference.state, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    axes[0].plot(result.time, result.rollout_prediction[:, 0], label=r"pred $\theta_1$", color="#2563eb")
    axes[0].plot(result.time, result.rollout_prediction[:, 1], label=r"pred $\theta_2$", color="#0f766e")
    axes[0].plot(result.time, reference.state[:, 0], "--", label=r"ref $\theta_1$", color="#93c5fd")
    axes[0].plot(result.time, reference.state[:, 1], "--", label=r"ref $\theta_2$", color="#99f6e4")
    axes[0].set_title("Recursive rollout angles")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Angle [rad]")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(result.time, result.rollout_prediction[:, 2], label=r"pred $\omega_1$", color="#dc2626")
    axes[1].plot(result.time, result.rollout_prediction[:, 3], label=r"pred $\omega_2$", color="#7c3aed")
    axes[1].plot(result.time, reference.state[:, 2], "--", label=r"ref $\omega_1$", color="#fca5a5")
    axes[1].plot(result.time, reference.state[:, 3], "--", label=r"ref $\omega_2$", color="#c4b5fd")
    axes[1].set_title("Recursive rollout angular velocities")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Angular velocity [rad/s]")
    axes[1].legend(frameon=False, ncol=2)

    axes[2].plot(x2_pred, y2_pred, color="#111827", linewidth=1.6, label="prediction")
    axes[2].plot(x2_ref, y2_ref, "--", color="#f59e0b", linewidth=1.2, label="reference")
    axes[2].set_title("End-mass trajectory")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("y [m]")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend(frameon=False)

    axes[3].plot(result.time, rollout_error, color="#059669", label="rollout state error")
    axes[3].plot(result.time, energy_shift, color="#7c3aed", label="energy drift")
    axes[3].set_title("Rollout degradation indicators")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(frameon=False)

    fig.suptitle(f"Double pendulum end-to-end solver ({result.variant})", fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_loss_history(result: TrainedE2EResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, values in result.history.items():
        ax.plot(values, label=name.replace("_", " "))
    ax.set_title(f"Training curves ({result.variant})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_e2e_animation(result: TrainedE2EResult, output_path: Path) -> Path:
    from .inr import save_inr_animation

    return save_inr_animation(
        TrainedE2EResultAdapter(result),
        output_path,
    )


def save_comparison_animation(result: TrainedE2EResult, output_path: Path) -> Path:
    from .inr import save_comparison_animation as save_inr_comparison_animation

    return save_inr_comparison_animation(
        TrainedE2EResultAdapter(result),
        output_path,
    )


class TrainedE2EResultAdapter:
    """Small adapter so we can reuse the existing animation utilities."""

    def __init__(self, result: TrainedE2EResult) -> None:
        self.variant = result.variant
        self.params = result.params
        self.reference = result.reference
        self.prediction = result.rollout_prediction

    @property
    def prediction_result(self) -> SimulationResult:
        return SimulationResult(
            time=self.reference.time,
            state=self.prediction,
            params=self.params,
        )


def export_e2e_artifacts(result: TrainedE2EResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictions": save_prediction_arrays(result, output_dir / f"{result.variant}_prediction.npz"),
        "summary": plot_prediction_summary(result, output_dir / f"{result.variant}_summary.png"),
        "loss": plot_loss_history(result, output_dir / f"{result.variant}_loss.png"),
        "animation": save_e2e_animation(result, output_dir / f"{result.variant}_animation.gif"),
        "comparison_animation": save_comparison_animation(result, output_dir / f"{result.variant}_comparison.gif"),
    }
    return outputs
