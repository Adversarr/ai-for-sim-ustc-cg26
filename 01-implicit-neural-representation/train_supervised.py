from __future__ import annotations

from pathlib import Path

from ai_for_sim.inr import (
    INRModelConfig,
    demo_params,
    quick_supervised_config,
    save_checkpoint_payload,
    train_supervised_inr,
)
from ai_for_sim.paths import INR_ARTIFACTS_DIR


CHECKPOINT_PATH = INR_ARTIFACTS_DIR / "supervised" / "supervised_checkpoint.pt"


def train_and_save_supervised(
    checkpoint_path: Path = CHECKPOINT_PATH,
    model_config: INRModelConfig | None = None,
) -> tuple[Path, dict[str, float]]:
    model, result = train_supervised_inr(
        model_config=model_config or INRModelConfig(),
        training_config=quick_supervised_config(),
        params=demo_params(),
    )
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    metrics = {
        "final_loss": result.history["loss"][-1],
    }
    return checkpoint_path, metrics


def main() -> None:
    checkpoint_path, metrics = train_and_save_supervised()
    print("Saved supervised INR checkpoint:")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  final_loss: {metrics['final_loss']:.6f}")


if __name__ == "__main__":
    main()
