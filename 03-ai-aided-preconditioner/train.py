from __future__ import annotations

from pathlib import Path

from ai_for_sim.aided_preconditioner import (
    quick_model_config,
    quick_solver_config,
    quick_training_config,
    save_checkpoint_payload,
    train_gauss_newton_preconditioner,
)
from ai_for_sim.paths import PRECONDITIONER_ARTIFACTS_DIR


CHECKPOINT_DIR = PRECONDITIONER_ARTIFACTS_DIR / "gauss_newton_preconditioner"
CHECKPOINT_PATH = CHECKPOINT_DIR / "gauss_newton_preconditioner_checkpoint.pt"


def train_and_save(checkpoint_path: Path = CHECKPOINT_PATH) -> tuple[Path, dict[str, float]]:
    model, result = train_gauss_newton_preconditioner(
        model_config=quick_model_config(),
        training_config=quick_training_config(),
        solver_config=quick_solver_config(),
    )
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    metrics = {
        "final_loss": result.history["loss"][-1],
        "final_val_loss": result.history["val_loss"][-1],
        "mean_exact_outer_iterations": float(result.exact_rollout.iteration_counts.mean()),
        "mean_preconditioned_outer_iterations": float(result.aided_rollout.iteration_counts.mean()),
        "mean_exact_cg_iterations": float(result.exact_rollout.linear_iteration_counts.mean()),
        "mean_preconditioned_cg_iterations": float(result.aided_rollout.linear_iteration_counts.mean()),
        "accepted_preconditioner_rate": float(result.metadata["accepted_preconditioner_rate"]),
        "trajectory_mse": float(result.metadata["trajectory_mse"]),
        "max_state_error": float(result.metadata["max_state_error"]),
    }
    return checkpoint_path, metrics


def main() -> None:
    checkpoint_path, metrics = train_and_save()
    print("Saved AI-aided preconditioner checkpoint (Gauss-Newton + CG, toy scale):")
    print(f"  checkpoint: {checkpoint_path}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
