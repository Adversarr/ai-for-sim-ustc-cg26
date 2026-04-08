from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class DoublePendulumParams:
    """Physical parameters and initial values for the double pendulum."""

    mass1: float = 1.0
    mass2: float = 1.0
    length1: float = 1.0
    length2: float = 1.0
    gravity: float = 9.81
    theta1_0: float = np.pi / 2
    theta2_0: float = np.pi / 2 + 0.2
    omega1_0: float = 0.0
    omega2_0: float = 0.0
    duration: float = 12.0
    fps: int = 25

    @property
    def initial_state(self) -> np.ndarray:
        """Return y(0) = [theta1, theta2, omega1, omega2]."""
        return np.array(
            [self.theta1_0, self.theta2_0, self.omega1_0, self.omega2_0],
            dtype=float,
        )

    @property
    def time_grid(self) -> np.ndarray:
        """Uniform time samples used for visualization and saved output."""
        frame_count = int(self.duration * self.fps) + 1
        return np.linspace(0.0, self.duration, frame_count)


@dataclass(frozen=True)
class SimulationResult:
    """Numerical solution returned by the ODE solver."""

    time: np.ndarray
    state: np.ndarray
    params: DoublePendulumParams


def state_derivative(_: float, state: np.ndarray, params: DoublePendulumParams) -> np.ndarray:
    """Compute dy/dt for y = [theta1, theta2, omega1, omega2]."""
    theta1, theta2, omega1, omega2 = state
    delta = theta1 - theta2

    m1 = params.mass1
    m2 = params.mass2
    l1 = params.length1
    l2 = params.length2
    g = params.gravity

    sin_delta = np.sin(delta)
    cos_delta = np.cos(delta)
    denominator = m1 + m2 * sin_delta**2

    theta1_dot = omega1
    theta2_dot = omega2

    omega1_dot = (
        m2 * g * np.sin(theta2) * cos_delta
        - m2 * sin_delta * (l1 * omega1**2 * cos_delta + l2 * omega2**2)
        - (m1 + m2) * g * np.sin(theta1)
    ) / (l1 * denominator)

    omega2_dot = (
        (m1 + m2)
        * (
            l1 * omega1**2 * sin_delta
            - g * np.sin(theta2)
            + g * np.sin(theta1) * cos_delta
        )
        + m2 * l2 * omega2**2 * sin_delta * cos_delta
    ) / (l2 * denominator)

    return np.array([theta1_dot, theta2_dot, omega1_dot, omega2_dot], dtype=float)


def simulate_double_pendulum(params: DoublePendulumParams | None = None) -> SimulationResult:
    """Integrate the double-pendulum ODE with SciPy."""
    params = params or DoublePendulumParams()

    solution = solve_ivp(
        fun=lambda t, y: state_derivative(t, y, params),
        t_span=(0.0, params.duration),
        y0=params.initial_state,
        t_eval=params.time_grid,
        method="DOP853",
        rtol=1e-9,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(f"solve_ivp failed: {solution.message}")

    return SimulationResult(
        time=solution.t,
        state=solution.y.T,
        params=params,
    )


def cartesian_positions(result: SimulationResult) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert angular coordinates into Cartesian positions."""
    theta1 = result.state[:, 0]
    theta2 = result.state[:, 1]
    l1 = result.params.length1
    l2 = result.params.length2

    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    return x1, y1, x2, y2


def total_energy(result: SimulationResult) -> np.ndarray:
    """Compute the total mechanical energy of the simulated system."""
    theta1 = result.state[:, 0]
    theta2 = result.state[:, 1]
    omega1 = result.state[:, 2]
    omega2 = result.state[:, 3]

    m1 = result.params.mass1
    m2 = result.params.mass2
    l1 = result.params.length1
    l2 = result.params.length2
    g = result.params.gravity

    speed1_sq = (l1 * omega1) ** 2
    speed2_sq = (
        speed1_sq
        + (l2 * omega2) ** 2
        + 2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)
    )

    kinetic = 0.5 * m1 * speed1_sq + 0.5 * m2 * speed2_sq
    potential = -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)
    return kinetic + potential
