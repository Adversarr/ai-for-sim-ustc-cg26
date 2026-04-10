from __future__ import annotations

from pathlib import Path

from ai_for_sim.aided_solver import (
    quick_model_config,
    quick_solver_config,
    quick_training_config,
    save_checkpoint_payload,
    train_direct_warmup_solver,
)
from ai_for_sim.paths import AIDED_ARTIFACTS_DIR


CHECKPOINT_PATH = AIDED_ARTIFACTS_DIR / "direct_warmup" / "direct_warmup_checkpoint.pt"


def train_and_save(checkpoint_path: Path = CHECKPOINT_PATH) -> tuple[Path, dict[str, float]]:
    model, result = train_direct_warmup_solver(
        model_config=quick_model_config(),
        training_config=quick_training_config(),
        solver_config=quick_solver_config(),
    )
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    metrics = {
        "final_loss": result.history["loss"][-1],
        "final_val_loss": result.history["val_loss"][-1],
        "mean_exact_iterations": float(result.exact_rollout.iteration_counts.mean()),
        "mean_aided_iterations": float(result.aided_rollout.iteration_counts.mean()),
        "accepted_warm_start_rate": float(1.0 - result.aided_rollout.fallback_steps.mean()),
    }
    return checkpoint_path, metrics


def main() -> None:
    checkpoint_path, metrics = train_and_save()
    print("Saved AI-aided implicit solver checkpoint:")
    print(f"  checkpoint: {checkpoint_path}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
