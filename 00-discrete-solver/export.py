from __future__ import annotations

from pathlib import Path

from ai_for_sim.double_pendulum import simulate_double_pendulum
from ai_for_sim.visualize import save_animation, save_overview_figure, save_trajectory


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def export_double_pendulum_assets(output_dir: Path = ARTIFACTS_DIR) -> dict[str, Path]:
    """Run the solver once and write the teaching artifacts for this folder."""
    result = simulate_double_pendulum()
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "trajectory": save_trajectory(result, output_dir / "double_pendulum_trajectory.npz"),
        "figure": save_overview_figure(result, output_dir / "double_pendulum_overview.png"),
        "animation": save_animation(result, output_dir / "double_pendulum_animation.gif"),
    }


def main() -> None:
    outputs = export_double_pendulum_assets()
    print("Generated double pendulum artifacts:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
