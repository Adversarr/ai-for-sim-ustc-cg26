from __future__ import annotations

from pathlib import Path

from ai_for_sim.e2e_solver import (
    E2EModelConfig,
    demo_params,
    quick_supervised_config,
    save_checkpoint_payload,
    tuned_supervised_model_config,
    train_supervised_e2e,
)
from ai_for_sim.paths import E2E_ARTIFACTS_DIR


CHECKPOINT_PATH = E2E_ARTIFACTS_DIR / "supervised" / "supervised_checkpoint.pt"


def train_and_save_supervised(
    checkpoint_path: Path = CHECKPOINT_PATH,
    model_config: E2EModelConfig | None = None,
) -> tuple[Path, dict[str, float]]:
    model, result = train_supervised_e2e(
        model_config=model_config or tuned_supervised_model_config(),
        training_config=quick_supervised_config(),
        params=demo_params(),
    )
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    metrics = {
        "final_loss": result.history["loss"][-1],
        "final_data_loss": result.history["data_loss"][-1],
    }
    return checkpoint_path, metrics


def main() -> None:
    checkpoint_path, metrics = train_and_save_supervised()
    print("Saved supervised end-to-end solver checkpoint:")
    print(f"  checkpoint: {checkpoint_path}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
