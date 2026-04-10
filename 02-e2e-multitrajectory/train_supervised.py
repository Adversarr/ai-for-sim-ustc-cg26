from __future__ import annotations

from pathlib import Path

from ai_for_sim.e2e_solver import (
    E2EModelConfig,
    TrajectorySplitConfig,
    default_trajectory_split_config,
    demo_params,
    quick_multitrajectory_supervised_config,
    save_checkpoint_payload,
    train_supervised_multitrajectory_e2e,
    tuned_supervised_model_config,
)


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "supervised" / "supervised_checkpoint.pt"


def train_and_save_supervised(
    checkpoint_path: Path = CHECKPOINT_PATH,
    model_config: E2EModelConfig | None = None,
    split_config: TrajectorySplitConfig | None = None,
) -> tuple[Path, dict[str, float]]:
    model, result = train_supervised_multitrajectory_e2e(
        model_config=model_config or tuned_supervised_model_config(),
        training_config=quick_multitrajectory_supervised_config(),
        params=demo_params(),
        split_config=split_config or default_trajectory_split_config(),
    )
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    metrics = {
        "final_loss": result.history["loss"][-1],
        "final_data_loss": result.history["data_loss"][-1],
        "final_val_loss": result.history["val_loss"][-1],
        "final_val_data_loss": result.history["val_data_loss"][-1],
    }
    return checkpoint_path, metrics


def main() -> None:
    checkpoint_path, metrics = train_and_save_supervised()
    print("Saved multi-trajectory supervised end-to-end solver checkpoint:")
    print(f"  checkpoint: {checkpoint_path}")
    print("  split: 400 train trajectories, 100 validation trajectories, default trajectory for test rollout")
    print("  batch_size: 64")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
