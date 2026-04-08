from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .double_pendulum import SimulationResult, cartesian_positions, total_energy


def save_trajectory(result: SimulationResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, time=result.time, state=result.state)
    return output_path


def save_overview_figure(result: SimulationResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, _, x2, y2 = cartesian_positions(result)
    energy = total_energy(result)
    energy_shift = energy - energy[0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)

    axes[0].plot(result.time, result.state[:, 0], label=r"$\theta_1$")
    axes[0].plot(result.time, result.state[:, 1], label=r"$\theta_2$")
    axes[0].set_title("Angles over time")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Angle [rad]")
    axes[0].legend(frameon=False)

    axes[1].plot(x2, y2, color="#0f766e", linewidth=1.5)
    axes[1].scatter([x2[0], x2[-1]], [y2[0], y2[-1]], c=["#2563eb", "#dc2626"], s=35)
    axes[1].set_title("End-mass trajectory")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_aspect("equal", adjustable="box")

    axes[2].plot(result.time, energy_shift, color="#7c3aed")
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[2].set_title("Energy drift")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel(r"$E(t)-E(0)$")

    fig.suptitle("Double pendulum: state variables, trajectory, and integration output", fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_animation(result: SimulationResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x1, y1, x2, y2 = cartesian_positions(result)
    reach = result.params.length1 + result.params.length2

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_xlim(-reach - 0.2, reach + 0.2)
    ax.set_ylim(-reach - 0.2, reach + 0.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Double pendulum animation")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.25)

    rod, = ax.plot([], [], color="#111827", linewidth=2.0)
    trail, = ax.plot([], [], color="#0f766e", linewidth=1.2, alpha=0.8)
    masses = ax.scatter([0.0, 0.0], [0.0, 0.0], s=80, c=["#2563eb", "#dc2626"], zorder=3)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    def update(frame: int):
        rod.set_data([0.0, x1[frame], x2[frame]], [0.0, y1[frame], y2[frame]])
        trail_start = max(0, frame - 80)
        trail.set_data(x2[trail_start : frame + 1], y2[trail_start : frame + 1])
        masses.set_offsets(np.array([[x1[frame], y1[frame]], [x2[frame], y2[frame]]]))
        time_text.set_text(f"t = {result.time[frame]:.2f} s")
        return rod, trail, masses, time_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(result.time),
        interval=1000 / result.params.fps,
        blit=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=result.params.fps), dpi=150)
    plt.close(fig)
    return output_path
