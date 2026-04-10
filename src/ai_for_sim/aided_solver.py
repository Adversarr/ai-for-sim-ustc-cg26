from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

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
    state_derivative,
)
from .e2e_solver import TrajectorySplitConfig, generate_trajectory_bank


DIRECT_WARMUP_VARIANT = "direct_warmup"
PROJECTED_HESSIAN_VARIANT = "projected_hessian"


@dataclass(frozen=True)
class AidedModelConfig:
    hidden_width: int = 64
    hidden_layers: int = 3
    diagonal_epsilon: float = 1e-3


@dataclass(frozen=True)
class AidedTrainingConfig:
    epochs: int = 400
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-5
    batch_size: int | None = 128
    seed: int = 7
    log_interval: int = 50
    device: str = "cpu"
    matrix_loss_weight: float = 1.0
    action_loss_weight: float = 5.0
    projection_damping: float = 1e-3


@dataclass(frozen=True)
class ImplicitSolverConfig:
    nonlinear_tol: float = 1e-8
    max_iterations: int = 12
    line_search_shrink: float = 0.5
    min_step_size: float = 1e-4
    jacobian_epsilon: float = 1e-6
    fallback_to_newton: bool = True
    linear_cg_tol: float = 1e-10
    linear_cg_max_iterations: int = 12
    gauss_newton_damping: float = 1e-3


@dataclass(frozen=True)
class SolverRollout:
    result: SimulationResult
    iteration_counts: np.ndarray
    linear_iteration_counts: np.ndarray
    residual_history: np.ndarray
    explicit_guess_residual_norms: np.ndarray
    warm_start_residual_norms: np.ndarray
    line_search_steps: np.ndarray
    fallback_steps: np.ndarray
    converged: np.ndarray


@dataclass(frozen=True)
class TrainedAidedSolverResult:
    variant: str
    model_config: AidedModelConfig
    training_config: AidedTrainingConfig
    solver_config: ImplicitSolverConfig
    params: DoublePendulumParams
    dt: float
    exact_rollout: SolverRollout
    aided_rollout: SolverRollout
    history: dict[str, list[float]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def time(self) -> np.ndarray:
        return self.exact_rollout.result.time

    @property
    def exact_reference(self) -> SimulationResult:
        return self.exact_rollout.result

    @property
    def aided_result(self) -> SimulationResult:
        return self.aided_rollout.result


class DirectWarmupMLP(nn.Module):
    def __init__(self, config: AidedModelConfig) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = 5
        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_width))
            layers.append(nn.SiLU())
            input_dim = config.hidden_width
        layers.append(nn.Linear(input_dim, 4))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class ProjectedHessianMLP(nn.Module):
    def __init__(self, config: AidedModelConfig) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = 5
        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_width))
            layers.append(nn.SiLU())
            input_dim = config.hidden_width
        layers.append(nn.Linear(input_dim, 10))
        self.network = nn.Sequential(*layers)
        self.diagonal_epsilon = config.diagonal_epsilon

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw_factor = self.network(features)
        return lower_triangular_spd_from_raw(raw_factor, self.diagonal_epsilon)


def demo_params() -> DoublePendulumParams:
    return DoublePendulumParams(duration=6.0, fps=20)


def quick_model_config() -> AidedModelConfig:
    return AidedModelConfig(hidden_width=32, hidden_layers=2)


def quick_training_config() -> AidedTrainingConfig:
    return AidedTrainingConfig(
        epochs=60,
        learning_rate=2e-3,
        min_learning_rate=1e-4,
        batch_size=64,
        log_interval=10,
        device="cpu",
    )


def quick_solver_config() -> ImplicitSolverConfig:
    return ImplicitSolverConfig(
        nonlinear_tol=1e-8,
        max_iterations=12,
        line_search_shrink=0.5,
        min_step_size=1e-4,
        jacobian_epsilon=1e-6,
        fallback_to_newton=True,
    )


def default_split_config() -> TrajectorySplitConfig:
    return TrajectorySplitConfig(
        total_trajectories=24,
        train_trajectories=16,
        validation_trajectories=8,
        seed=17,
    )


def get_device(device_name: str) -> torch.device:
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_warmup_model(config: AidedModelConfig, device: torch.device) -> DirectWarmupMLP:
    return DirectWarmupMLP(config).to(device)


def build_preconditioner_model(config: AidedModelConfig, device: torch.device) -> ProjectedHessianMLP:
    return ProjectedHessianMLP(config).to(device)


def build_model(config: AidedModelConfig, device: torch.device) -> DirectWarmupMLP:
    return build_warmup_model(config, device)


def build_optimizer(
    model: nn.Module,
    training_config: AidedTrainingConfig,
) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=training_config.learning_rate)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: AidedTrainingConfig,
) -> torch.optim.lr_scheduler.LRScheduler:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.epochs,
        eta_min=training_config.min_learning_rate,
    )


def lower_triangular_spd_from_raw(raw_factor: torch.Tensor, diagonal_epsilon: float) -> torch.Tensor:
    batch_shape = raw_factor.shape[:-1]
    lower = raw_factor.new_zeros(*batch_shape, 4, 4)
    tril_indices = torch.tril_indices(row=4, col=4, offset=0, device=raw_factor.device)
    lower[..., tril_indices[0], tril_indices[1]] = raw_factor
    diagonal_raw = torch.diagonal(lower, dim1=-2, dim2=-1)
    strict_lower = torch.tril(lower, diagonal=-1)
    diagonal = torch.diag_embed(F.softplus(diagonal_raw) + diagonal_epsilon)
    matrix = strict_lower + diagonal
    eye = torch.eye(4, dtype=raw_factor.dtype, device=raw_factor.device)
    return matrix @ matrix.transpose(-1, -2) + diagonal_epsilon * eye


def rhs_numpy(state: np.ndarray, params: DoublePendulumParams) -> np.ndarray:
    return state_derivative(0.0, state, params)


def implicit_residual(
    next_state: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    params: DoublePendulumParams,
) -> np.ndarray:
    return next_state - current_state - dt * rhs_numpy(next_state, params)


def residual_energy(
    next_state: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    params: DoublePendulumParams,
) -> float:
    residual = implicit_residual(next_state, current_state, dt, params)
    return 0.5 * float(residual @ residual)


def system_matrix(
    next_state: np.ndarray,
    dt: float,
    params: DoublePendulumParams,
    jacobian_epsilon: float = 1e-6,
) -> np.ndarray:
    jacobian = np.zeros((4, 4), dtype=float)
    for column in range(4):
        perturbation = np.zeros(4, dtype=float)
        perturbation[column] = jacobian_epsilon
        rhs_plus = rhs_numpy(next_state + perturbation, params)
        rhs_minus = rhs_numpy(next_state - perturbation, params)
        jacobian[:, column] = (rhs_plus - rhs_minus) / (2.0 * jacobian_epsilon)
    return np.eye(4, dtype=float) - dt * jacobian


def residual_energy_gradient(
    next_state: np.ndarray,
    current_state: np.ndarray,
    dt: float,
    params: DoublePendulumParams,
    jacobian_epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = implicit_residual(next_state, current_state, dt, params)
    matrix = system_matrix(next_state, dt, params, jacobian_epsilon=jacobian_epsilon)
    gradient = matrix.T @ residual
    return residual, matrix, gradient


def explicit_euler_guess(current_state: np.ndarray, dt: float, params: DoublePendulumParams) -> np.ndarray:
    return current_state + dt * rhs_numpy(current_state, params)


def exact_newton_direction(residual: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.linalg.solve(matrix, -residual)


def projected_hessian_inverse(matrix: np.ndarray, damping: float) -> np.ndarray:
    normal_matrix = matrix.T @ matrix + damping * np.eye(matrix.shape[1], dtype=float)
    return np.linalg.inv(normal_matrix)


def exact_gauss_newton_direction(gradient: np.ndarray, matrix: np.ndarray, damping: float) -> np.ndarray:
    normal_matrix = matrix.T @ matrix + damping * np.eye(matrix.shape[1], dtype=float)
    return np.linalg.solve(normal_matrix, -gradient)


def conjugate_gradient_direction(
    gradient: np.ndarray,
    matrix: np.ndarray,
    damping: float,
    tol: float,
    max_iterations: int,
    preconditioner_action: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, int, bool]:
    normal_matrix = matrix.T @ matrix + damping * np.eye(matrix.shape[1], dtype=float)
    rhs = -gradient
    direction = np.zeros_like(rhs)
    residual = rhs.copy()
    if float(np.linalg.norm(residual)) <= tol:
        return direction, 0, True

    def apply_preconditioner(vector: np.ndarray) -> np.ndarray:
        if preconditioner_action is None:
            return vector.copy()
        return preconditioner_action(vector)

    preconditioned_residual = apply_preconditioner(residual)
    residual_dot = float(residual @ preconditioned_residual)
    if residual_dot <= 0.0:
        return direction, 0, False

    search = preconditioned_residual.copy()
    for iteration in range(1, max_iterations + 1):
        matrix_search = normal_matrix @ search
        denominator = float(search @ matrix_search)
        if denominator <= 0.0:
            return direction, iteration - 1, False

        alpha = residual_dot / denominator
        direction = direction + alpha * search
        residual = residual - alpha * matrix_search
        if float(np.linalg.norm(residual)) <= tol:
            return direction, iteration, True

        next_preconditioned_residual = apply_preconditioner(residual)
        next_residual_dot = float(residual @ next_preconditioned_residual)
        if next_residual_dot <= 0.0:
            return direction, iteration, False

        beta = next_residual_dot / residual_dot
        search = next_preconditioned_residual + beta * search
        preconditioned_residual = next_preconditioned_residual
        residual_dot = next_residual_dot

    return direction, max_iterations, False


def _line_search_on_energy(
    current_state: np.ndarray,
    current_iterate: np.ndarray,
    direction: np.ndarray,
    dt: float,
    params: DoublePendulumParams,
    solver_config: ImplicitSolverConfig,
) -> tuple[np.ndarray, float, bool]:
    current_energy = residual_energy(current_iterate, current_state, dt, params)
    step_size = 1.0
    while step_size >= solver_config.min_step_size:
        candidate = current_iterate + step_size * direction
        candidate_energy = residual_energy(candidate, current_state, dt, params)
        if candidate_energy < current_energy:
            return candidate, step_size, True
        step_size *= solver_config.line_search_shrink
    return current_iterate.copy(), 0.0, False


def _solve_one_implicit_step(
    current_state: np.ndarray,
    dt: float,
    params: DoublePendulumParams,
    solver_config: ImplicitSolverConfig,
    preconditioner: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    collect_training_samples: bool = False,
    projection_damping: float = 1e-3,
    initial_iterate: np.ndarray | None = None,
    use_gauss_newton_cg: bool = False,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, np.ndarray]]]:
    iterate = explicit_euler_guess(current_state, dt, params) if initial_iterate is None else initial_iterate.copy()
    residual_norms: list[float] = []
    line_search_steps: list[float] = []
    fallback_used = False
    training_samples: list[dict[str, np.ndarray]] = []
    linear_iterations = 0
    first_step_residual_norm: float | None = None

    for outer_iteration in range(solver_config.max_iterations):
        residual, matrix, gradient = residual_energy_gradient(
            iterate,
            current_state,
            dt,
            params,
            jacobian_epsilon=solver_config.jacobian_epsilon,
        )
        residual_norm = float(np.linalg.norm(residual))
        residual_norms.append(residual_norm)
        if residual_norm <= solver_config.nonlinear_tol:
            return iterate, {
                "residual_norms": residual_norms,
                "line_search_steps": line_search_steps,
                "used_fallback": fallback_used,
                "linear_iterations": linear_iterations,
                "first_step_residual_norm": residual_norms[0] if first_step_residual_norm is None else first_step_residual_norm,
                "converged": True,
            }, training_samples

        target_matrix = projected_hessian_inverse(matrix, projection_damping)
        if collect_training_samples:
            training_samples.append(
                {
                    "iterate": iterate.copy(),
                    "gradient": gradient.copy(),
                    "target_matrix": target_matrix.copy(),
                }
            )

        use_fallback = False
        if use_gauss_newton_cg:
            learned_matrix = preconditioner(iterate, gradient) if preconditioner is not None else None
            preconditioner_action = None if learned_matrix is None else lambda vector: learned_matrix @ vector
            direction, cg_iterations, cg_converged = conjugate_gradient_direction(
                gradient,
                matrix,
                damping=projection_damping,
                tol=solver_config.linear_cg_tol,
                max_iterations=solver_config.linear_cg_max_iterations,
                preconditioner_action=preconditioner_action,
            )
            linear_iterations += cg_iterations
            if not cg_converged or float(direction @ gradient) >= 0.0:
                use_fallback = True
        else:
            if preconditioner is None:
                direction = exact_newton_direction(residual, matrix)
            else:
                direction = preconditioner(iterate, gradient)
                if float(direction @ gradient) >= 0.0:
                    use_fallback = True
                else:
                    candidate, step_size, accepted = _line_search_on_energy(
                        current_state,
                        iterate,
                        direction,
                        dt,
                        params,
                        solver_config,
                    )
                    if accepted:
                        candidate_residual_norm = float(
                            np.linalg.norm(implicit_residual(candidate, current_state, dt, params))
                        )
                        if candidate_residual_norm <= 0.5 * residual_norm:
                            iterate = candidate
                            line_search_steps.append(step_size)
                            if outer_iteration == 0:
                                first_step_residual_norm = candidate_residual_norm
                            continue
                    use_fallback = True

        if use_fallback:
            if not solver_config.fallback_to_newton:
                break
            fallback_used = True
            if use_gauss_newton_cg:
                direction = exact_gauss_newton_direction(gradient, matrix, projection_damping)
            else:
                direction = exact_newton_direction(residual, matrix)

        candidate, step_size, accepted = _line_search_on_energy(
            current_state,
            iterate,
            direction,
            dt,
            params,
            solver_config,
        )
        line_search_steps.append(step_size)
        if not accepted:
            break
        if outer_iteration == 0:
            first_step_residual_norm = float(np.linalg.norm(implicit_residual(candidate, current_state, dt, params)))
        iterate = candidate

    return iterate, {
        "residual_norms": residual_norms,
        "line_search_steps": line_search_steps,
        "used_fallback": fallback_used,
        "linear_iterations": linear_iterations,
        "first_step_residual_norm": residual_norms[0] if first_step_residual_norm is None else first_step_residual_norm,
        "converged": False,
    }, training_samples


def _pack_rollout_histories(stats_list: list[dict[str, Any]], max_iterations: int) -> tuple[np.ndarray, np.ndarray]:
    residual_history = np.full((len(stats_list), max_iterations), np.nan, dtype=float)
    line_search_history = np.full((len(stats_list), max_iterations), np.nan, dtype=float)
    for step_index, stats in enumerate(stats_list):
        residual_norms = np.asarray(stats["residual_norms"], dtype=float)
        line_search_steps = np.asarray(stats["line_search_steps"], dtype=float)
        residual_history[step_index, : min(len(residual_norms), max_iterations)] = residual_norms[:max_iterations]
        line_search_history[step_index, : min(len(line_search_steps), max_iterations)] = line_search_steps[:max_iterations]
    return residual_history, line_search_history


def simulate_implicit_euler(
    params: DoublePendulumParams | None = None,
    solver_config: ImplicitSolverConfig | None = None,
) -> SolverRollout:
    params = params or demo_params()
    solver_config = solver_config or quick_solver_config()
    time_grid = params.time_grid
    dt = float(time_grid[1] - time_grid[0])

    state = np.zeros((len(time_grid), 4), dtype=float)
    state[0] = params.initial_state.astype(float)
    step_stats: list[dict[str, Any]] = []
    explicit_guess_residual_norms: list[float] = []
    warm_start_residual_norms: list[float] = []
    for step in range(1, len(time_grid)):
        next_state, stats, _ = _solve_one_implicit_step(
            state[step - 1],
            dt,
            params,
            solver_config,
        )
        if not stats["converged"]:
            raise RuntimeError(f"Implicit Euler failed to converge at step {step}.")
        state[step] = next_state
        step_stats.append(stats)
        explicit_guess_residual_norms.append(float(stats["residual_norms"][0]))
        warm_start_residual_norms.append(float(stats["residual_norms"][0]))

    residual_history, line_search_history = _pack_rollout_histories(step_stats, solver_config.max_iterations)
    return SolverRollout(
        result=SimulationResult(time=time_grid, state=state, params=params),
        iteration_counts=np.asarray([len(stats["residual_norms"]) for stats in step_stats], dtype=int),
        linear_iteration_counts=np.zeros(len(step_stats), dtype=int),
        residual_history=residual_history,
        explicit_guess_residual_norms=np.asarray(explicit_guess_residual_norms, dtype=float),
        warm_start_residual_norms=np.asarray(warm_start_residual_norms, dtype=float),
        line_search_steps=line_search_history,
        fallback_steps=np.asarray([int(bool(stats["used_fallback"])) for stats in step_stats], dtype=int),
        converged=np.asarray([bool(stats["converged"]) for stats in step_stats], dtype=bool),
    )


def _predict_warmup_delta(model: nn.Module, iterate: np.ndarray, dt: float, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        features = torch.tensor(np.concatenate([iterate, [dt]]), dtype=torch.float32, device=device).unsqueeze(0)
        return model(features).squeeze(0).cpu().numpy()


def _model_preconditioner(model: nn.Module, iterate: np.ndarray, gradient: np.ndarray, dt: float, device: torch.device) -> np.ndarray:
    del gradient
    model.eval()
    with torch.no_grad():
        features = torch.tensor(np.concatenate([iterate, [dt]]), dtype=torch.float32, device=device).unsqueeze(0)
        matrix = model(features).squeeze(0).cpu().numpy()
    return matrix


def simulate_ai_aided_implicit_euler(
    model: nn.Module,
    params: DoublePendulumParams | None = None,
    solver_config: ImplicitSolverConfig | None = None,
) -> SolverRollout:
    params = params or demo_params()
    solver_config = solver_config or quick_solver_config()
    device = next(model.parameters()).device
    time_grid = params.time_grid
    dt = float(time_grid[1] - time_grid[0])

    state = np.zeros((len(time_grid), 4), dtype=float)
    state[0] = params.initial_state.astype(float)
    step_stats: list[dict[str, Any]] = []
    explicit_guess_residual_norms: list[float] = []
    warm_start_residual_norms: list[float] = []

    for step in range(1, len(time_grid)):
        current_state = state[step - 1]
        explicit_iterate = explicit_euler_guess(current_state, dt, params)
        explicit_residual_norm = float(
            np.linalg.norm(implicit_residual(explicit_iterate, current_state, dt, params))
        )
        warmup_delta = _predict_warmup_delta(model, explicit_iterate, dt, device)
        warm_iterate = explicit_iterate + warmup_delta
        warm_residual_norm = float(
            np.linalg.norm(implicit_residual(warm_iterate, current_state, dt, params))
        )

        next_state, stats, _ = _solve_one_implicit_step(
            current_state,
            dt,
            params,
            solver_config,
            preconditioner=None,
            initial_iterate=warm_iterate,
        )
        if not stats["converged"]:
            raise RuntimeError(f"AI-aided implicit Euler failed to converge at step {step}.")

        state[step] = next_state
        step_stats.append(stats)
        explicit_guess_residual_norms.append(explicit_residual_norm)
        warm_start_residual_norms.append(warm_residual_norm)

    residual_history, line_search_history = _pack_rollout_histories(step_stats, solver_config.max_iterations)
    return SolverRollout(
        result=SimulationResult(time=time_grid, state=state, params=params),
        iteration_counts=np.asarray([len(stats["residual_norms"]) for stats in step_stats], dtype=int),
        linear_iteration_counts=np.zeros(len(step_stats), dtype=int),
        residual_history=residual_history,
        explicit_guess_residual_norms=np.asarray(explicit_guess_residual_norms, dtype=float),
        warm_start_residual_norms=np.asarray(warm_start_residual_norms, dtype=float),
        line_search_steps=line_search_history,
        fallback_steps=np.zeros(len(step_stats), dtype=int),
        converged=np.asarray([bool(stats["converged"]) for stats in step_stats], dtype=bool),
    )


def simulate_gauss_newton_cg_implicit_euler(
    params: DoublePendulumParams | None = None,
    solver_config: ImplicitSolverConfig | None = None,
    preconditioner: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> SolverRollout:
    params = params or demo_params()
    solver_config = solver_config or quick_solver_config()
    time_grid = params.time_grid
    dt = float(time_grid[1] - time_grid[0])

    state = np.zeros((len(time_grid), 4), dtype=float)
    state[0] = params.initial_state.astype(float)
    step_stats: list[dict[str, Any]] = []
    explicit_guess_residual_norms: list[float] = []
    warm_start_residual_norms: list[float] = []
    for step in range(1, len(time_grid)):
        next_state, stats, _ = _solve_one_implicit_step(
            state[step - 1],
            dt,
            params,
            solver_config,
            preconditioner=preconditioner,
            projection_damping=solver_config.gauss_newton_damping,
            use_gauss_newton_cg=True,
        )
        if not stats["converged"]:
            raise RuntimeError(f"Gauss-Newton CG implicit Euler failed to converge at step {step}.")
        state[step] = next_state
        step_stats.append(stats)
        explicit_guess_residual_norms.append(float(stats["residual_norms"][0]))
        warm_start_residual_norms.append(float(stats["first_step_residual_norm"]))

    residual_history, line_search_history = _pack_rollout_histories(step_stats, solver_config.max_iterations)
    return SolverRollout(
        result=SimulationResult(time=time_grid, state=state, params=params),
        iteration_counts=np.asarray([len(stats["residual_norms"]) for stats in step_stats], dtype=int),
        linear_iteration_counts=np.asarray([int(stats["linear_iterations"]) for stats in step_stats], dtype=int),
        residual_history=residual_history,
        explicit_guess_residual_norms=np.asarray(explicit_guess_residual_norms, dtype=float),
        warm_start_residual_norms=np.asarray(warm_start_residual_norms, dtype=float),
        line_search_steps=line_search_history,
        fallback_steps=np.asarray([int(bool(stats["used_fallback"])) for stats in step_stats], dtype=int),
        converged=np.asarray([bool(stats["converged"]) for stats in step_stats], dtype=bool),
    )


def simulate_preconditioned_implicit_euler(
    model: nn.Module,
    params: DoublePendulumParams | None = None,
    solver_config: ImplicitSolverConfig | None = None,
) -> SolverRollout:
    params = params or demo_params()
    solver_config = solver_config or quick_solver_config()
    device = next(model.parameters()).device
    dt = float(params.time_grid[1] - params.time_grid[0])
    return simulate_gauss_newton_cg_implicit_euler(
        params=params,
        solver_config=solver_config,
        preconditioner=lambda iterate, gradient: _model_preconditioner(model, iterate, gradient, dt, device),
    )


def _build_warmup_training_dataset(
    params: DoublePendulumParams,
    split_config: TrajectorySplitConfig,
    solver_config: ImplicitSolverConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    trajectory_params = generate_trajectory_bank(params, split_config)
    dt = float(params.time_grid[1] - params.time_grid[0])

    train_samples: list[dict[str, np.ndarray]] = []
    validation_samples: list[dict[str, np.ndarray]] = []
    for trajectory_index, trajectory_param in enumerate(trajectory_params):
        current_state = trajectory_param.initial_state.astype(float)
        time_grid = trajectory_param.time_grid
        for _ in range(1, len(time_grid)):
            explicit_iterate = explicit_euler_guess(current_state, dt, trajectory_param)
            next_state, stats, _ = _solve_one_implicit_step(
                current_state,
                dt,
                trajectory_param,
                solver_config,
            )
            if not stats["converged"]:
                raise RuntimeError("Reference implicit solver failed while collecting training data.")
            sample = {
                "explicit_iterate": explicit_iterate.astype(np.float32),
                "target_delta": (next_state - explicit_iterate).astype(np.float32),
                "target_state": next_state.astype(np.float32),
            }
            current_state = next_state
            if trajectory_index < split_config.train_trajectories:
                train_samples.append(sample)
            else:
                validation_samples.append(sample)

    def stack(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        if not samples:
            raise ValueError("Need at least one implicit-step sample.")
        explicit_iterate = np.stack([sample["explicit_iterate"] for sample in samples]).astype(np.float32)
        target_delta = np.stack([sample["target_delta"] for sample in samples]).astype(np.float32)
        target_state = np.stack([sample["target_state"] for sample in samples]).astype(np.float32)
        features = np.concatenate(
            [explicit_iterate, np.full((len(samples), 1), dt, dtype=np.float32)],
            axis=1,
        )
        return {
            "features": features,
            "explicit_iterate": explicit_iterate,
            "target_delta": target_delta,
            "target_state": target_state,
        }

    return stack(train_samples), stack(validation_samples), dt


def _build_preconditioner_training_dataset(
    params: DoublePendulumParams,
    split_config: TrajectorySplitConfig,
    solver_config: ImplicitSolverConfig,
    projection_damping: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    trajectory_params = generate_trajectory_bank(params, split_config)
    dt = float(params.time_grid[1] - params.time_grid[0])
    reference_solver_config = replace(
        solver_config,
        max_iterations=max(solver_config.max_iterations, 32),
    )

    train_samples: list[dict[str, np.ndarray]] = []
    validation_samples: list[dict[str, np.ndarray]] = []
    for trajectory_index, trajectory_param in enumerate(trajectory_params):
        current_state = trajectory_param.initial_state.astype(float)
        time_grid = trajectory_param.time_grid
        for _ in range(1, len(time_grid)):
            next_state, stats, samples = _solve_one_implicit_step(
                current_state,
                dt,
                trajectory_param,
                reference_solver_config,
                collect_training_samples=True,
                projection_damping=projection_damping,
                use_gauss_newton_cg=True,
            )
            if not stats["converged"]:
                raise RuntimeError("Reference implicit solver failed while collecting training data.")
            current_state = next_state
            if trajectory_index < split_config.train_trajectories:
                train_samples.extend(samples)
            else:
                validation_samples.extend(samples)

    def stack(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        if not samples:
            raise ValueError("Need at least one nonlinear iteration sample.")
        iterate = np.stack([sample["iterate"] for sample in samples]).astype(np.float32)
        gradient = np.stack([sample["gradient"] for sample in samples]).astype(np.float32)
        target_matrix = np.stack([sample["target_matrix"] for sample in samples]).astype(np.float32)
        features = np.concatenate(
            [iterate, np.full((len(samples), 1), dt, dtype=np.float32)],
            axis=1,
        )
        target_action = np.einsum("bij,bj->bi", target_matrix, gradient)
        return {
            "features": features,
            "gradient": gradient,
            "target_matrix": target_matrix,
            "target_action": target_action.astype(np.float32),
        }

    return stack(train_samples), stack(validation_samples), dt


def _evaluate_warmup_dataset(
    model: nn.Module,
    dataset: dict[str, torch.Tensor],
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        predicted_delta = model(dataset["features"])
        delta_loss = F.mse_loss(predicted_delta, dataset["target_delta"])
        predicted_state = dataset["explicit_iterate"] + predicted_delta
        state_loss = F.mse_loss(predicted_state, dataset["target_state"])
    return float(delta_loss.detach().cpu()), float(state_loss.detach().cpu())


def _evaluate_preconditioner_dataset(
    model: nn.Module,
    dataset: dict[str, torch.Tensor],
    training_config: AidedTrainingConfig,
) -> tuple[float, float, float]:
    model.eval()
    with torch.no_grad():
        predicted_matrix = model(dataset["features"])
        predicted_action = torch.matmul(predicted_matrix, dataset["gradient"].unsqueeze(-1)).squeeze(-1)
        matrix_loss = F.mse_loss(predicted_matrix, dataset["target_matrix"])
        action_loss = F.mse_loss(predicted_action, dataset["target_action"])
        total_loss = (
            training_config.matrix_loss_weight * matrix_loss
            + training_config.action_loss_weight * action_loss
        )
    return (
        float(total_loss.detach().cpu()),
        float(matrix_loss.detach().cpu()),
        float(action_loss.detach().cpu()),
    )


def train_direct_warmup_solver(
    model_config: AidedModelConfig | None = None,
    training_config: AidedTrainingConfig | None = None,
    params: DoublePendulumParams | None = None,
    solver_config: ImplicitSolverConfig | None = None,
    split_config: TrajectorySplitConfig | None = None,
) -> tuple[nn.Module, TrainedAidedSolverResult]:
    model_config = model_config or quick_model_config()
    training_config = training_config or quick_training_config()
    params = params or demo_params()
    solver_config = solver_config or quick_solver_config()
    split_config = split_config or default_split_config()

    set_seed(training_config.seed)
    train_dataset_np, validation_dataset_np, dt = _build_warmup_training_dataset(
        params,
        split_config,
        solver_config,
    )

    device = get_device(training_config.device)
    model = build_warmup_model(model_config, device)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    train_dataset = {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in train_dataset_np.items()
    }
    validation_dataset = {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in validation_dataset_np.items()
    }

    batch_size = len(train_dataset["features"]) if training_config.batch_size is None else min(
        training_config.batch_size,
        len(train_dataset["features"]),
    )
    history = {
        "loss": [],
        "delta_loss": [],
        "state_loss": [],
        "val_loss": [],
        "val_delta_loss": [],
        "val_state_loss": [],
    }

    for epoch in range(training_config.epochs):
        model.train()
        permutation = torch.randperm(len(train_dataset["features"]), device=device)
        epoch_loss = 0.0
        epoch_state_loss = 0.0

        for start in range(0, len(train_dataset["features"]), batch_size):
            indices = permutation[start : start + batch_size]
            feature_batch = train_dataset["features"][indices]
            explicit_iterate_batch = train_dataset["explicit_iterate"][indices]
            target_delta_batch = train_dataset["target_delta"][indices]
            target_state_batch = train_dataset["target_state"][indices]

            optimizer.zero_grad()
            predicted_delta = model(feature_batch)
            delta_loss = F.mse_loss(predicted_delta, target_delta_batch)
            predicted_state = explicit_iterate_batch + predicted_delta
            state_loss = F.mse_loss(predicted_state, target_state_batch)
            delta_loss.backward()
            optimizer.step()

            batch_items = len(indices)
            epoch_loss += float(delta_loss.detach().cpu()) * batch_items
            epoch_state_loss += float(state_loss.detach().cpu()) * batch_items

        scheduler.step()

        history["loss"].append(epoch_loss / len(train_dataset["features"]))
        history["delta_loss"].append(history["loss"][-1])
        history["state_loss"].append(epoch_state_loss / len(train_dataset["features"]))
        val_delta_loss, val_state_loss = _evaluate_warmup_dataset(model, validation_dataset)
        history["val_loss"].append(val_delta_loss)
        history["val_delta_loss"].append(val_delta_loss)
        history["val_state_loss"].append(val_state_loss)
        if epoch % training_config.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}: loss={history['loss'][-1]:.6f}, "
                f"state_loss={history['state_loss'][-1]:.6f}, "
                f"val_loss={history['val_loss'][-1]:.6f}, "
                f"lr={current_lr:.6e}"
            )

    exact_rollout = simulate_implicit_euler(params=params, solver_config=solver_config)
    aided_rollout = simulate_ai_aided_implicit_euler(model, params=params, solver_config=solver_config)
    result = TrainedAidedSolverResult(
        variant=DIRECT_WARMUP_VARIANT,
        model_config=model_config,
        training_config=training_config,
        solver_config=solver_config,
        params=params,
        dt=dt,
        exact_rollout=exact_rollout,
        aided_rollout=aided_rollout,
        history=history,
        metadata={
            "train_trajectories": split_config.train_trajectories,
            "validation_trajectories": split_config.validation_trajectories,
            "total_trajectories": split_config.total_trajectories,
            "train_samples": int(len(train_dataset_np["features"])),
            "validation_samples": int(len(validation_dataset_np["features"])),
        },
    )
    return model, result


def train_projected_hessian_preconditioner(
    model_config: AidedModelConfig | None = None,
    training_config: AidedTrainingConfig | None = None,
    params: DoublePendulumParams | None = None,
    solver_config: ImplicitSolverConfig | None = None,
    split_config: TrajectorySplitConfig | None = None,
) -> tuple[nn.Module, TrainedAidedSolverResult]:
    model_config = model_config or quick_model_config()
    training_config = training_config or quick_training_config()
    params = params or demo_params()
    solver_config = solver_config or quick_solver_config()
    split_config = split_config or default_split_config()

    set_seed(training_config.seed)
    train_dataset_np, validation_dataset_np, dt = _build_preconditioner_training_dataset(
        params,
        split_config,
        solver_config,
        projection_damping=training_config.projection_damping,
    )

    device = get_device(training_config.device)
    model = build_preconditioner_model(model_config, device)
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    train_dataset = {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in train_dataset_np.items()
    }
    validation_dataset = {
        key: torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in validation_dataset_np.items()
    }

    batch_size = len(train_dataset["features"]) if training_config.batch_size is None else min(
        training_config.batch_size,
        len(train_dataset["features"]),
    )
    history = {
        "loss": [],
        "matrix_loss": [],
        "action_loss": [],
        "val_loss": [],
        "val_matrix_loss": [],
        "val_action_loss": [],
    }

    for epoch in range(training_config.epochs):
        model.train()
        permutation = torch.randperm(len(train_dataset["features"]), device=device)
        epoch_loss = 0.0
        epoch_matrix_loss = 0.0
        epoch_action_loss = 0.0

        for start in range(0, len(train_dataset["features"]), batch_size):
            indices = permutation[start : start + batch_size]
            feature_batch = train_dataset["features"][indices]
            gradient_batch = train_dataset["gradient"][indices]
            target_matrix_batch = train_dataset["target_matrix"][indices]
            target_action_batch = train_dataset["target_action"][indices]

            optimizer.zero_grad()
            predicted_matrix = model(feature_batch)
            predicted_action = torch.matmul(predicted_matrix, gradient_batch.unsqueeze(-1)).squeeze(-1)
            matrix_loss = F.mse_loss(predicted_matrix, target_matrix_batch)
            action_loss = F.mse_loss(predicted_action, target_action_batch)
            total_loss = (
                training_config.matrix_loss_weight * matrix_loss
                + training_config.action_loss_weight * action_loss
            )
            total_loss.backward()
            optimizer.step()

            batch_items = len(indices)
            epoch_loss += float(total_loss.detach().cpu()) * batch_items
            epoch_matrix_loss += float(matrix_loss.detach().cpu()) * batch_items
            epoch_action_loss += float(action_loss.detach().cpu()) * batch_items

        scheduler.step()

        history["loss"].append(epoch_loss / len(train_dataset["features"]))
        history["matrix_loss"].append(epoch_matrix_loss / len(train_dataset["features"]))
        history["action_loss"].append(epoch_action_loss / len(train_dataset["features"]))
        val_loss, val_matrix_loss, val_action_loss = _evaluate_preconditioner_dataset(
            model,
            validation_dataset,
            training_config,
        )
        history["val_loss"].append(val_loss)
        history["val_matrix_loss"].append(val_matrix_loss)
        history["val_action_loss"].append(val_action_loss)
        if epoch % training_config.log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}: total_loss={history['loss'][-1]:.6f}, "
                f"matrix_loss={history['matrix_loss'][-1]:.6f}, "
                f"action_loss={history['action_loss'][-1]:.6f}, "
                f"val_loss={history['val_loss'][-1]:.6f}, "
                f"lr={current_lr:.6e}"
            )

    exact_rollout = simulate_gauss_newton_cg_implicit_euler(params=params, solver_config=solver_config)
    aided_rollout = simulate_preconditioned_implicit_euler(model, params=params, solver_config=solver_config)
    result = TrainedAidedSolverResult(
        variant=PROJECTED_HESSIAN_VARIANT,
        model_config=model_config,
        training_config=training_config,
        solver_config=solver_config,
        params=params,
        dt=dt,
        exact_rollout=exact_rollout,
        aided_rollout=aided_rollout,
        history=history,
        metadata={
            "train_trajectories": split_config.train_trajectories,
            "validation_trajectories": split_config.validation_trajectories,
            "total_trajectories": split_config.total_trajectories,
            "train_samples": int(len(train_dataset_np["features"])),
            "validation_samples": int(len(validation_dataset_np["features"])),
        },
    )
    return model, result


def _serialize_rollout(rollout: SolverRollout) -> dict[str, Any]:
    return {
        "time": rollout.result.time,
        "state": rollout.result.state,
        "iteration_counts": rollout.iteration_counts,
        "linear_iteration_counts": rollout.linear_iteration_counts,
        "residual_history": rollout.residual_history,
        "explicit_guess_residual_norms": rollout.explicit_guess_residual_norms,
        "warm_start_residual_norms": rollout.warm_start_residual_norms,
        "line_search_steps": rollout.line_search_steps,
        "fallback_steps": rollout.fallback_steps,
        "converged": rollout.converged,
    }


def _deserialize_rollout(payload: dict[str, Any], params: DoublePendulumParams) -> SolverRollout:
    residual_history = np.asarray(payload["residual_history"])
    fallback_steps = np.asarray(payload.get("fallback_steps", np.zeros(len(residual_history), dtype=int)))
    linear_iteration_counts = np.asarray(
        payload.get("linear_iteration_counts", np.zeros(len(residual_history), dtype=int))
    )
    explicit_guess_residual_norms = np.asarray(
        payload.get("explicit_guess_residual_norms", residual_history[:, 0])
    )
    warm_start_residual_norms = np.asarray(
        payload.get("warm_start_residual_norms", residual_history[:, 0])
    )
    line_search_steps = np.asarray(
        payload.get("line_search_steps", np.full_like(residual_history, np.nan, dtype=float))
    )
    return SolverRollout(
        result=SimulationResult(
            time=np.asarray(payload["time"]),
            state=np.asarray(payload["state"]),
            params=params,
        ),
        iteration_counts=np.asarray(payload["iteration_counts"]),
        linear_iteration_counts=linear_iteration_counts,
        residual_history=residual_history,
        explicit_guess_residual_norms=explicit_guess_residual_norms,
        warm_start_residual_norms=warm_start_residual_norms,
        line_search_steps=line_search_steps,
        fallback_steps=fallback_steps,
        converged=np.asarray(payload["converged"]),
    )


def save_checkpoint_payload(*, model: nn.Module, result: TrainedAidedSolverResult, checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": result.variant,
        "model_config": asdict(result.model_config),
        "training_config": asdict(result.training_config),
        "solver_config": asdict(result.solver_config),
        "params": asdict(result.params),
        "dt": result.dt,
        "history": result.history,
        "exact_rollout": _serialize_rollout(result.exact_rollout),
        "aided_rollout": _serialize_rollout(result.aided_rollout),
        "metadata": result.metadata,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def _build_model_for_variant(
    variant: str,
    model_config: AidedModelConfig,
    device_name: str,
) -> nn.Module:
    device = get_device(device_name)
    if variant in {PROJECTED_HESSIAN_VARIANT, "gauss_newton_preconditioner"}:
        return build_preconditioner_model(model_config, device)
    return build_warmup_model(model_config, device)


def load_checkpoint(checkpoint_path: Path, device_name: str = "cpu") -> tuple[nn.Module, TrainedAidedSolverResult]:
    payload = torch.load(checkpoint_path, map_location=get_device(device_name), weights_only=False)
    model_config = AidedModelConfig(**payload["model_config"])
    training_config = AidedTrainingConfig(**payload["training_config"])
    solver_config = ImplicitSolverConfig(**payload["solver_config"])
    params = DoublePendulumParams(**payload["params"])
    variant = payload["variant"]

    model = _build_model_for_variant(variant, model_config, device_name)
    model.load_state_dict(payload["model_state_dict"])

    result = TrainedAidedSolverResult(
        variant=variant,
        model_config=model_config,
        training_config=training_config,
        solver_config=solver_config,
        params=params,
        dt=float(payload["dt"]),
        exact_rollout=_deserialize_rollout(payload["exact_rollout"], params),
        aided_rollout=_deserialize_rollout(payload["aided_rollout"], params),
        history={key: list(value) for key, value in payload["history"].items()},
        metadata=dict(payload.get("metadata", {})),
    )
    return model, result


def save_prediction_arrays(result: TrainedAidedSolverResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_error_norm = np.linalg.norm(result.aided_rollout.result.state - result.exact_rollout.result.state, axis=1)
    iteration_delta = result.exact_rollout.iteration_counts - result.aided_rollout.iteration_counts
    linear_iteration_delta = result.exact_rollout.linear_iteration_counts - result.aided_rollout.linear_iteration_counts
    warm_start_reduction = (
        result.aided_rollout.warm_start_residual_norms
        / np.maximum(result.aided_rollout.explicit_guess_residual_norms, 1e-16)
    )
    np.savez(
        output_path,
        time=result.time,
        dt=result.dt,
        exact_state=result.exact_rollout.result.state,
        aided_state=result.aided_rollout.result.state,
        exact_iteration_counts=result.exact_rollout.iteration_counts,
        aided_iteration_counts=result.aided_rollout.iteration_counts,
        exact_linear_iteration_counts=result.exact_rollout.linear_iteration_counts,
        aided_linear_iteration_counts=result.aided_rollout.linear_iteration_counts,
        exact_residual_history=result.exact_rollout.residual_history,
        aided_residual_history=result.aided_rollout.residual_history,
        exact_explicit_guess_residual_norms=result.exact_rollout.explicit_guess_residual_norms,
        aided_explicit_guess_residual_norms=result.aided_rollout.explicit_guess_residual_norms,
        aided_warm_start_residual_norms=result.aided_rollout.warm_start_residual_norms,
        warm_start_reduction=warm_start_reduction,
        exact_fallback_steps=result.exact_rollout.fallback_steps,
        aided_fallback_steps=result.aided_rollout.fallback_steps,
        state_error_norm=state_error_norm,
        iteration_delta=iteration_delta,
        linear_iteration_delta=linear_iteration_delta,
    )
    return output_path


def plot_prediction_summary(result: TrainedAidedSolverResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    explicit_residual = result.aided_rollout.explicit_guess_residual_norms
    warm_residual = result.aided_rollout.warm_start_residual_norms
    warm_start_ratio = warm_residual / np.maximum(explicit_residual, 1e-16)
    residual_removed = explicit_residual - warm_residual
    iteration_delta = result.exact_rollout.iteration_counts - result.aided_rollout.iteration_counts
    accepted_mask = result.aided_rollout.fallback_steps == 0
    accepted_label = "warmup" if result.variant == DIRECT_WARMUP_VARIANT else "accepted"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    min_residual = float(min(np.min(explicit_residual), np.min(warm_residual)))
    max_residual = float(max(np.max(explicit_residual), np.max(warm_residual)))
    diagonal = np.geomspace(max(min_residual, 1e-16), max_residual, 200)
    axes[0].scatter(
        explicit_residual[accepted_mask],
        warm_residual[accepted_mask],
        color="#2563eb",
        s=16,
        alpha=0.8,
        label=accepted_label,
    )
    if np.any(~accepted_mask):
        axes[0].scatter(
            explicit_residual[~accepted_mask],
            warm_residual[~accepted_mask],
            color="#f59e0b",
            s=20,
            alpha=0.9,
            label="fallback",
        )
    axes[0].plot(diagonal, diagonal, "--", color="#111827", linewidth=1.0, label="same residual")
    axes[0].set_title("Did the Learned Warm Start Beat Explicit Euler?")
    axes[0].set_xlabel(r"$\|r_{\mathrm{explicit}}\|$")
    axes[0].set_ylabel(r"$\|r_{\mathrm{warm}}\|$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    axes[0].text(
        0.02,
        0.98,
        f"median ratio = {np.median(warm_start_ratio):.3f}\n"
        f"better warmup = {(warm_start_ratio < 1.0).mean():.1%}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    ratio_min = max(float(np.min(warm_start_ratio)), 1e-3)
    ratio_max = max(float(np.max(warm_start_ratio)), 1.0) * 1.05
    ratio_bins = np.geomspace(ratio_min, ratio_max, 18)
    axes[1].hist(warm_start_ratio[accepted_mask], bins=ratio_bins, color="#2563eb", alpha=0.75, label=accepted_label)
    if np.any(~accepted_mask):
        axes[1].hist(warm_start_ratio[~accepted_mask], bins=ratio_bins, color="#f59e0b", alpha=0.8, label="fallback")
    axes[1].axvline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[1].set_title("How Much Residual Was Removed Before Newton?")
    axes[1].set_xlabel(r"$\|r_{\mathrm{warm}}\| / \|r_{\mathrm{explicit}}\|$")
    axes[1].set_ylabel("Count")
    axes[1].set_xscale("log")
    axes[1].legend(frameon=False)
    axes[1].text(
        0.02,
        0.98,
        f"median ratio = {np.median(warm_start_ratio):.3f}\n"
        f"median removed = {np.median(residual_removed):.3e}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    correction_values = np.arange(
        int(min(np.min(result.exact_rollout.iteration_counts), np.min(result.aided_rollout.iteration_counts))),
        int(max(np.max(result.exact_rollout.iteration_counts), np.max(result.aided_rollout.iteration_counts))) + 1,
    )
    exact_counts = np.array([np.sum(result.exact_rollout.iteration_counts == value) for value in correction_values])
    aided_counts = np.array([np.sum(result.aided_rollout.iteration_counts == value) for value in correction_values])
    bar_width = 0.35
    axes[2].bar(correction_values - bar_width / 2, exact_counts, width=bar_width, color="#2563eb", label="exact Newton")
    axes[2].bar(correction_values + bar_width / 2, aided_counts, width=bar_width, color="#dc2626", label="warmup + Newton")
    axes[2].set_title("Did Newton Need Fewer Corrections Afterwards?")
    axes[2].set_xlabel("Corrections")
    axes[2].set_ylabel("Count")
    axes[2].set_xticks(correction_values)
    axes[2].legend(frameon=False)

    axes[3].scatter(warm_start_ratio, iteration_delta, color="#111827", s=16, alpha=0.75)
    axes[3].axvline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[3].axhline(0.0, color="#111827", linestyle=":", linewidth=1.0)
    axes[3].set_title("Did a Better Warm Start Help Newton?")
    axes[3].set_xlabel(r"$\|r_{\mathrm{warm}}\| / \|r_{\mathrm{explicit}}\|$")
    axes[3].set_ylabel("exact - aided corrections")
    axes[3].set_xscale("log")
    corr = np.corrcoef(warm_start_ratio, iteration_delta)[0, 1] if len(warm_start_ratio) > 1 else np.nan
    axes[3].text(
        0.02,
        0.98,
        f"corr = {corr:.3f}",
        transform=axes[3].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    if result.variant == DIRECT_WARMUP_VARIANT:
        title = "AI-Aided Direct Warmup Solver"
    elif "preconditioner" in result.variant:
        title = "AI-Aided Gauss-Newton Preconditioner"
    else:
        title = "AI-Aided Implicit Solver"
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_loss_history(result: TrainedAidedSolverResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name, values in result.history.items():
        ax.plot(values, label=name.replace("_", " "))
    if result.variant == DIRECT_WARMUP_VARIANT:
        ax.set_title("Direct warmup training curves")
    else:
        ax.set_title("Projected-hessian training curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


class _TrainedAidedResultAdapter:
    def __init__(self, result: TrainedAidedSolverResult) -> None:
        self.variant = result.variant
        self.params = result.params
        self.reference = result.exact_reference
        self.prediction = result.aided_rollout.result.state

    @property
    def prediction_result(self) -> SimulationResult:
        return SimulationResult(time=self.reference.time, state=self.prediction, params=self.params)


def save_aided_animation(result: TrainedAidedSolverResult, output_path: Path) -> Path:
    from .inr import save_inr_animation

    return save_inr_animation(_TrainedAidedResultAdapter(result), output_path)


def save_comparison_animation(result: TrainedAidedSolverResult, output_path: Path) -> Path:
    from .inr import save_comparison_animation as save_inr_comparison_animation

    return save_inr_comparison_animation(_TrainedAidedResultAdapter(result), output_path)


def export_aided_solver_artifacts(result: TrainedAidedSolverResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "predictions": save_prediction_arrays(result, output_dir / f"{result.variant}_prediction.npz"),
        "summary": plot_prediction_summary(result, output_dir / f"{result.variant}_summary.png"),
        "loss": plot_loss_history(result, output_dir / f"{result.variant}_loss.png"),
        "animation": save_aided_animation(result, output_dir / f"{result.variant}_animation.gif"),
        "comparison_animation": save_comparison_animation(result, output_dir / f"{result.variant}_comparison.gif"),
    }
