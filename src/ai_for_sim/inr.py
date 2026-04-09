from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from torch import nn
import torch.nn.functional as F

from .double_pendulum import (
    DoublePendulumParams,
    SimulationResult,
    cartesian_positions,
    simulate_double_pendulum,
    total_energy,
)


@dataclass(frozen=True)
class INRModelConfig:
    hidden_width: int = 32
    hidden_layers: int = 3


@dataclass(frozen=True)
class INRTrainingConfig:
    epochs: int = 600
    optimizer: Literal["adam", "lbfgs"] = "adam"
    learning_rate: float = 1e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    seed: int = 7
    collocation_points: int = 181
    log_interval: int = 100
    ic_weight: float = 3.0
    residual_weight: float = 1.0
    sparse_data_weight: float = 1.0
    sparse_points: int = 17
    lbfgs_history_size: int = 20
    lbfgs_max_iter: int = 20
    device: str = "cpu"


@dataclass(frozen=True)
class TrainedINRResult:
    variant: str
    model_config: INRModelConfig
    training_config: INRTrainingConfig
    params: DoublePendulumParams
    prediction: np.ndarray
    history: dict[str, list[float]]
    reference: SimulationResult | None = None

    @property
    def time(self) -> np.ndarray:
        if self.reference is not None:
            return self.reference.time
        return self.params.time_grid

    @property
    def prediction_result(self) -> SimulationResult:
        return SimulationResult(time=self.time, state=self.prediction, params=self.params)


class TimeMLP(nn.Module):
    def __init__(self, config: INRModelConfig) -> None:
        super().__init__()

        input_dim = 2 * 16  # Use sine and cosine encoding of time as input
        self.input_layer = nn.Linear(input_dim, config.hidden_width)
        self.activation = nn.SiLU()
        self.residual_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_width, 2 * config.hidden_width),
                    nn.SiLU(),
                    nn.Linear(2 * config.hidden_width, config.hidden_width),
                )
                for _ in range(config.hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(config.hidden_width, 4)

    def forward(self, time_input: torch.Tensor) -> torch.Tensor:
        T_max = 8
        encoded_time = torch.cat([torch.sin(2 * np.pi * time_input * i / T_max) for i in range(1, 17)] +
                                 [torch.cos(2 * np.pi * time_input * i / T_max) for i in range(1, 17)], dim=-1)

        hidden = self.activation(self.input_layer(encoded_time))
        for residual_layer in self.residual_layers:
            hidden = hidden + residual_layer(F.normalize(hidden, dim=-1))

        return self.output_layer(F.normalize(hidden, dim=-1))


def demo_params() -> DoublePendulumParams:
    return DoublePendulumParams(duration=6.0, fps=20)


def quick_supervised_config() -> INRTrainingConfig:
    return INRTrainingConfig(
        epochs=2000,
        learning_rate=1e-3,
        log_interval=30,
    )


def quick_physics_config() -> INRTrainingConfig:
    return INRTrainingConfig(
        epochs=500,
        optimizer="lbfgs",
        learning_rate=0.5,
        collocation_points=241,
        log_interval=40,
        ic_weight=3.0,
        lbfgs_history_size=10,
        lbfgs_max_iter=20,
    )


def quick_hybrid_config() -> INRTrainingConfig:
    return INRTrainingConfig(
        epochs=500,
        optimizer="lbfgs",
        learning_rate=0.5,
        collocation_points=241,
        log_interval=40,
        ic_weight=3.0,
        residual_weight=1.0,
        sparse_data_weight=1.0,
        sparse_points=13, # 0.5s period
        lbfgs_history_size=10,
        lbfgs_max_iter=20,
    )


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_model(config: INRModelConfig, device: torch.device) -> TimeMLP:
    model = TimeMLP(config)
    return model.to(device)


def build_optimizer(
    model: nn.Module,
    training_config: INRTrainingConfig,
) -> torch.optim.Optimizer:
    if training_config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=training_config.learning_rate,
            betas=(training_config.adam_beta1, training_config.adam_beta2),
        )
    if training_config.optimizer == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(),
            lr=training_config.learning_rate,
            max_iter=training_config.lbfgs_max_iter,
            history_size=training_config.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
    raise ValueError(f"Unsupported optimizer: {training_config.optimizer}")


def normalize_time(time_values: torch.Tensor, duration: float) -> torch.Tensor:
    return 2.0 * (time_values / duration) - 1.0


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


def time_derivative(model: nn.Module, collocation_time: torch.Tensor, duration: float) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_time = normalize_time(collocation_time, duration)
    prediction = model(normalized_time)

    gradients = []
    for component in range(prediction.shape[1]):
        derivative = torch.autograd.grad(
            prediction[:, component].sum(),
            collocation_time,
            create_graph=True,
        )[0]
        gradients.append(derivative)
    return prediction, torch.cat(gradients, dim=1)


def evaluate_model(model: nn.Module, time_grid: np.ndarray, params: DoublePendulumParams) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        time_tensor = torch.tensor(time_grid[:, None], dtype=torch.float32, device=device)
        prediction = model(normalize_time(time_tensor, params.duration))
    return prediction.cpu().numpy()


def sparse_reference_samples(
    reference: SimulationResult,
    sparse_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    sparse_points = max(2, min(sparse_points, len(reference.time)))
    sparse_indices = np.linspace(0, len(reference.time) - 1, sparse_points, dtype=int)
    sparse_indices = np.unique(sparse_indices)
    return reference.time[sparse_indices], reference.state[sparse_indices]


def sample_collocation_time(
    params: DoublePendulumParams,
    collocation_points: int,
    device: torch.device,
) -> torch.Tensor:
    # collocation_time = params.duration * torch.rand(
    #     (collocation_points, 1),
    #     device=device,
    #     dtype=torch.float32,
    # )
    # collocation_time, _ = torch.sort(collocation_time, dim=0)
    collocation_time = torch.linspace(0.0, params.duration, collocation_points, device=device).unsqueeze(1)
    collocation_time.requires_grad_(True)
    return collocation_time


def train_supervised_inr(
    model_config: INRModelConfig | None = None,
    training_config: INRTrainingConfig | None = None,
    params: DoublePendulumParams | None = None,
) -> tuple[nn.Module, TrainedINRResult]:
    model_config = model_config or INRModelConfig()
    training_config = training_config or INRTrainingConfig()
    params = params or demo_params()

    set_seed(training_config.seed)
    reference = simulate_double_pendulum(params)
    device = get_device(training_config.device)
    model = build_model(model_config, device)
    optimizer = build_optimizer(model, training_config)

    time_tensor = torch.tensor(reference.time[:, None], dtype=torch.float32, device=device)
    normalized_time = normalize_time(time_tensor, params.duration)
    target_state = torch.tensor(reference.state, dtype=torch.float32, device=device)

    history = {"loss": []}
    for _ in range(training_config.epochs):
        if training_config.optimizer == "lbfgs":
            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                prediction = model(normalized_time)
                loss = torch.mean((prediction - target_state) ** 2)
                loss.backward()
                return loss

            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            prediction = model(normalized_time)
            loss = torch.mean((prediction - target_state) ** 2)
            loss.backward()
            optimizer.step()
        history["loss"].append(float(loss.detach().cpu()))

    prediction = evaluate_model(model, reference.time, params)
    return model, TrainedINRResult(
        variant="supervised",
        model_config=model_config,
        training_config=training_config,
        params=params,
        prediction=prediction,
        history=history,
        reference=reference,
    )


def train_physics_inr(
    model_config: INRModelConfig | None = None,
    training_config: INRTrainingConfig | None = None,
    params: DoublePendulumParams | None = None,
) -> tuple[nn.Module, TrainedINRResult]:
    model_config = model_config or INRModelConfig()
    training_config = training_config or INRTrainingConfig(epochs=100, ic_weight=3.0)
    params = params or demo_params()

    set_seed(training_config.seed)
    reference = simulate_double_pendulum(params)
    device = get_device(training_config.device)
    model = build_model(model_config, device)
    optimizer = build_optimizer(model, training_config)

    initial_state = torch.tensor(params.initial_state[None, :], dtype=torch.float32, device=device)
    history = {"loss": [], "residual_loss": [], "initial_condition_loss": []}

    def compute_losses(collocation_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction, derivative = time_derivative(model, collocation_time, params.duration)
        rhs = torch_rhs(prediction, params)
        # residual_loss = torch.mean((derivative - rhs) ** 2)
        residual_loss = F.huber_loss(derivative, rhs, delta=1.0)

        initial_time = torch.zeros((1, 1), dtype=torch.float32, device=device)
        initial_prediction = model(normalize_time(initial_time, params.duration))
        # initial_condition_loss = torch.mean((initial_prediction - initial_state) ** 2)
        initial_condition_loss = F.huber_loss(initial_prediction, initial_state, delta=1.0)

        loss = (
            training_config.residual_weight * residual_loss
            + training_config.ic_weight * initial_condition_loss
        )
        return loss, residual_loss, initial_condition_loss

    for i in range(training_config.epochs):
        collocation_time = sample_collocation_time(params, training_config.collocation_points, device)
        if training_config.optimizer == "lbfgs":
            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss, _, _ = compute_losses(collocation_time)
                loss.backward()
                return loss

            optimizer.step(closure)
            loss, residual_loss, initial_condition_loss = compute_losses(collocation_time)
        else:
            optimizer.zero_grad()
            loss, residual_loss, initial_condition_loss = compute_losses(collocation_time)
            loss.backward()
            optimizer.step()

        history["loss"].append(float(loss.detach().cpu()))
        history["residual_loss"].append(float(residual_loss.detach().cpu()))
        history["initial_condition_loss"].append(float(initial_condition_loss.detach().cpu()))
        if i % training_config.log_interval == 0:
            print(
                f"Epoch {i}: total_loss={history['loss'][-1]:.6f}, residual_loss={history['residual_loss'][-1]:.6f}, "
                f"initial_condition_loss={history['initial_condition_loss'][-1]:.6f}"
            )

    prediction = evaluate_model(model, params.time_grid, params)
    return model, TrainedINRResult(
        variant="physics",
        model_config=model_config,
        training_config=training_config,
        params=params,
        prediction=prediction,
        history=history,
        reference=reference,
    )


def train_hybrid_inr(
    model_config: INRModelConfig | None = None,
    training_config: INRTrainingConfig | None = None,
    params: DoublePendulumParams | None = None,
) -> tuple[nn.Module, TrainedINRResult]:
    model_config = model_config or INRModelConfig()
    training_config = training_config or INRTrainingConfig(epochs=900, ic_weight=3.0, sparse_data_weight=20.0)
    params = params or demo_params()

    set_seed(training_config.seed)
    reference = simulate_double_pendulum(params)
    sparse_time, sparse_state = sparse_reference_samples(reference, training_config.sparse_points)

    device = get_device(training_config.device)
    model = build_model(model_config, device)
    optimizer = build_optimizer(model, training_config)

    initial_state = torch.tensor(params.initial_state[None, :], dtype=torch.float32, device=device)
    sparse_time_tensor = torch.tensor(sparse_time[:, None], dtype=torch.float32, device=device)
    sparse_target_state = torch.tensor(sparse_state, dtype=torch.float32, device=device)

    history = {"loss": [], "residual_loss": [], "initial_condition_loss": [], "sparse_data_loss": []}

    def compute_losses(collocation_time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction, derivative = time_derivative(model, collocation_time, params.duration)
        rhs = torch_rhs(prediction, params)
        residual_loss = F.huber_loss(derivative, rhs, delta=1.0)

        initial_time = torch.zeros((1, 1), dtype=torch.float32, device=device)
        initial_prediction = model(normalize_time(initial_time, params.duration))
        initial_condition_loss = F.huber_loss(initial_prediction, initial_state, delta=1.0)

        sparse_prediction = model(normalize_time(sparse_time_tensor, params.duration))
        sparse_data_loss = F.huber_loss(sparse_prediction, sparse_target_state, delta=1.0)

        loss = (
            training_config.residual_weight * residual_loss
            + training_config.ic_weight * initial_condition_loss
            + training_config.sparse_data_weight * sparse_data_loss
        )
        return loss, residual_loss, initial_condition_loss, sparse_data_loss

    for i in range(training_config.epochs):
        collocation_time = sample_collocation_time(params, training_config.collocation_points, device)
        if training_config.optimizer == "lbfgs":
            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss, _, _, _ = compute_losses(collocation_time)
                loss.backward()
                return loss

            optimizer.step(closure)
            loss, residual_loss, initial_condition_loss, sparse_data_loss = compute_losses(collocation_time)
        else:
            optimizer.zero_grad()
            loss, residual_loss, initial_condition_loss, sparse_data_loss = compute_losses(collocation_time)
            loss.backward()
            optimizer.step()

        history["loss"].append(float(loss.detach().cpu()))
        history["residual_loss"].append(float(residual_loss.detach().cpu()))
        history["initial_condition_loss"].append(float(initial_condition_loss.detach().cpu()))
        history["sparse_data_loss"].append(float(sparse_data_loss.detach().cpu()))
        if i % training_config.log_interval == 0:
            print(
                f"Epoch {i}: total_loss={history['loss'][-1]:.6f}, residual_loss={history['residual_loss'][-1]:.6f}, "
                f"initial_condition_loss={history['initial_condition_loss'][-1]:.6f}, "
                f"sparse_data_loss={history['sparse_data_loss'][-1]:.6f}"
            )

    prediction = evaluate_model(model, params.time_grid, params)
    return model, TrainedINRResult(
        variant="hybrid",
        model_config=model_config,
        training_config=training_config,
        params=params,
        prediction=prediction,
        history=history,
        reference=reference,
    )


def save_checkpoint_payload(
    *,
    model: nn.Module,
    result: TrainedINRResult,
    checkpoint_path: Path,
) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": result.variant,
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "params": asdict(result.params),
        "history": result.history,
        "prediction": result.prediction,
        "reference_time": None if result.reference is None else result.reference.time,
        "reference_state": None if result.reference is None else result.reference.state,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path: Path, device_name: str = "cpu") -> tuple[nn.Module, TrainedINRResult]:
    payload = torch.load(checkpoint_path, map_location=get_device(device_name), weights_only=False)
    model_config = INRModelConfig(**payload["model_config"])
    training_config = INRTrainingConfig(**payload["training_config"])
    params = DoublePendulumParams(**payload["params"])
    model = build_model(model_config, get_device(device_name))
    model.load_state_dict(payload["model_state_dict"])

    reference = None
    if payload["reference_time"] is not None and payload["reference_state"] is not None:
        reference = SimulationResult(
            time=np.asarray(payload["reference_time"]),
            state=np.asarray(payload["reference_state"]),
            params=params,
        )

    result = TrainedINRResult(
        variant=payload["variant"],
        model_config=model_config,
        training_config=training_config,
        params=params,
        prediction=np.asarray(payload["prediction"]),
        history={key: list(value) for key, value in payload["history"].items()},
        reference=reference,
    )
    return model, result


def plot_prediction_summary(result: TrainedINRResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted = result.prediction_result
    _, _, x2_pred, y2_pred = cartesian_positions(predicted)
    predicted_energy = total_energy(predicted)
    energy_shift = predicted_energy - predicted_energy[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    sparse_time = None
    sparse_state = None
    if result.variant == "hybrid" and result.reference is not None:
        sparse_time, sparse_state = sparse_reference_samples(
            result.reference,
            result.training_config.sparse_points,
        )

    axes[0].plot(predicted.time, predicted.state[:, 0], label=r"pred $\theta_1$", color="#2563eb")
    axes[0].plot(predicted.time, predicted.state[:, 1], label=r"pred $\theta_2$", color="#0f766e")
    if result.reference is not None:
        axes[0].plot(result.reference.time, result.reference.state[:, 0], "--", label=r"ref $\theta_1$", color="#93c5fd")
        axes[0].plot(result.reference.time, result.reference.state[:, 1], "--", label=r"ref $\theta_2$", color="#99f6e4")
    if sparse_time is not None and sparse_state is not None:
        axes[0].scatter(sparse_time, sparse_state[:, 0], s=24, color="#1d4ed8", marker="o", label=r"sparse $\theta_1$")
        axes[0].scatter(sparse_time, sparse_state[:, 1], s=24, color="#0d9488", marker="x", label=r"sparse $\theta_2$")
    axes[0].set_title("Angles over time")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Angle [rad]")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(predicted.time, predicted.state[:, 2], label=r"pred $\omega_1$", color="#dc2626")
    axes[1].plot(predicted.time, predicted.state[:, 3], label=r"pred $\omega_2$", color="#7c3aed")
    if result.reference is not None:
        axes[1].plot(result.reference.time, result.reference.state[:, 2], "--", label=r"ref $\omega_1$", color="#fca5a5")
        axes[1].plot(result.reference.time, result.reference.state[:, 3], "--", label=r"ref $\omega_2$", color="#c4b5fd")
    if sparse_time is not None and sparse_state is not None:
        axes[1].scatter(sparse_time, sparse_state[:, 2], s=24, color="#b91c1c", marker="o", label=r"sparse $\omega_1$")
        axes[1].scatter(sparse_time, sparse_state[:, 3], s=24, color="#6d28d9", marker="x", label=r"sparse $\omega_2$")
    axes[1].set_title("Angular velocities")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Angular velocity [rad/s]")
    axes[1].legend(frameon=False, ncol=2)

    axes[2].plot(x2_pred, y2_pred, color="#111827", linewidth=1.6, label="prediction")
    if result.reference is not None:
        _, _, x2_ref, y2_ref = cartesian_positions(result.reference)
        axes[2].plot(x2_ref, y2_ref, "--", color="#f59e0b", linewidth=1.2, label="reference")
    if sparse_time is not None and sparse_state is not None:
        sparse_result = SimulationResult(time=sparse_time, state=sparse_state, params=result.params)
        _, _, x2_sparse, y2_sparse = cartesian_positions(sparse_result)
        axes[2].scatter(x2_sparse, y2_sparse, s=28, color="#ea580c", marker="o", label="sparse points")
    axes[2].set_title("End-mass trajectory")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("y [m]")
    axes[2].set_aspect("equal", adjustable="box")
    axes[2].legend(frameon=False)

    axes[3].plot(predicted.time, energy_shift, color="#059669")
    axes[3].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[3].set_title("Predicted energy drift")
    axes[3].set_xlabel("Time [s]")
    axes[3].set_ylabel(r"$E(t)-E(0)$")

    fig.suptitle(f"Double pendulum INR demo ({result.variant})", fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_loss_history(result: TrainedINRResult, output_path: Path) -> Path:
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


def save_prediction_arrays(result: TrainedINRResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "time": result.time,
        "prediction": result.prediction,
    }
    if result.reference is not None:
        payload["reference"] = result.reference.state
    np.savez(output_path, **payload)
    return output_path


def save_inr_animation(result: TrainedINRResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted = result.prediction_result
    x1_pred, y1_pred, x2_pred, y2_pred = cartesian_positions(predicted)
    reach = predicted.params.length1 + predicted.params.length2

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(-reach - 0.2, reach + 0.2)
    ax.set_ylim(-reach - 0.2, reach + 0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Double pendulum INR animation ({result.variant})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.25)

    rod, = ax.plot([], [], color="#111827", linewidth=2.0)
    trail, = ax.plot([], [], color="#0f766e", linewidth=1.2, alpha=0.8)
    masses = ax.scatter([0.0, 0.0], [0.0, 0.0], s=80, c=["#2563eb", "#dc2626"], zorder=3)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def update(frame: int):
        rod.set_data([0.0, x1_pred[frame], x2_pred[frame]], [0.0, y1_pred[frame], y2_pred[frame]])
        trail_start = max(0, frame - 80)
        trail.set_data(x2_pred[trail_start : frame + 1], y2_pred[trail_start : frame + 1])
        masses.set_offsets(np.array([[x1_pred[frame], y1_pred[frame]], [x2_pred[frame], y2_pred[frame]]]))
        time_text.set_text(f"prediction, t = {predicted.time[frame]:.2f} s")
        return rod, trail, masses, time_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(predicted.time),
        interval=1000 / predicted.params.fps,
        blit=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=predicted.params.fps), dpi=150)
    plt.close(fig)
    return output_path


def save_comparison_animation(result: TrainedINRResult, output_path: Path) -> Path:
    if result.reference is None:
        raise ValueError("Comparison animation requires a reference trajectory.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    predicted = result.prediction_result
    reference = result.reference
    x1_pred, y1_pred, x2_pred, y2_pred = cartesian_positions(predicted)
    x1_ref, y1_ref, x2_ref, y2_ref = cartesian_positions(reference)
    reach = predicted.params.length1 + predicted.params.length2

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 10.0), constrained_layout=True)
    titles = ["Prediction", "Ground truth"]
    series = [
        (x1_pred, y1_pred, x2_pred, y2_pred, "#0f766e"),
        (x1_ref, y1_ref, x2_ref, y2_ref, "#f59e0b"),
    ]

    rods = []
    trails = []
    masses = []
    for ax, title, (_, _, _, _, color) in zip(axes, titles, series, strict=True):
        ax.set_xlim(-reach - 0.2, reach + 0.2)
        ax.set_ylim(-reach - 0.2, reach + 0.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(alpha=0.25)
        rod, = ax.plot([], [], color="#111827", linewidth=2.0)
        trail, = ax.plot([], [], color=color, linewidth=1.2, alpha=0.85)
        mass = ax.scatter([0.0, 0.0], [0.0, 0.0], s=80, c=["#2563eb", "#dc2626"], zorder=3)
        rods.append(rod)
        trails.append(trail)
        masses.append(mass)

    time_text = fig.text(0.5, 0.98, "", ha="center", va="top")

    def update(frame: int):
        artists = [time_text]
        for rod, trail, mass, (x1, y1, x2, y2, _) in zip(rods, trails, masses, series, strict=True):
            rod.set_data([0.0, x1[frame], x2[frame]], [0.0, y1[frame], y2[frame]])
            trail_start = max(0, frame - 80)
            trail.set_data(x2[trail_start : frame + 1], y2[trail_start : frame + 1])
            mass.set_offsets(np.array([[x1[frame], y1[frame]], [x2[frame], y2[frame]]]))
            artists.extend([rod, trail, mass])
        time_text.set_text(f"{result.variant} vs. numerical GT, t = {predicted.time[frame]:.2f} s")
        return tuple(artists)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(predicted.time),
        interval=1000 / predicted.params.fps,
        blit=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=predicted.params.fps), dpi=150)
    plt.close(fig)
    return output_path


def export_inr_artifacts(result: TrainedINRResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "predictions": save_prediction_arrays(result, output_dir / f"{result.variant}_prediction.npz"),
        "summary": plot_prediction_summary(result, output_dir / f"{result.variant}_summary.png"),
        "loss": plot_loss_history(result, output_dir / f"{result.variant}_loss.png"),
        "animation": save_inr_animation(result, output_dir / f"{result.variant}_animation.gif"),
    }
    if result.reference is not None:
        outputs["comparison_animation"] = save_comparison_animation(
            result,
            output_dir / f"{result.variant}_comparison.gif",
        )
    return outputs
