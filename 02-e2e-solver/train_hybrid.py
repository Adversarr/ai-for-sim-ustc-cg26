from __future__ import annotations

from pathlib import Path

from ai_for_sim.e2e_solver import (
    E2EModelConfig,
    demo_params,
    quick_hybrid_config,
    save_checkpoint_payload,
    tuned_hybrid_model_config,
    train_hybrid_e2e,
)
from ai_for_sim.paths import E2E_ARTIFACTS_DIR


CHECKPOINT_PATH = E2E_ARTIFACTS_DIR / "hybrid" / "hybrid_checkpoint.pt"


def train_and_save_hybrid(
    checkpoint_path: Path = CHECKPOINT_PATH,
    model_config: E2EModelConfig | None = None,
) -> tuple[Path, dict[str, float]]:
    model, result = train_hybrid_e2e(
        model_config=model_config or tuned_hybrid_model_config(),
        training_config=quick_hybrid_config(),
        params=demo_params(),
    )
    save_checkpoint_payload(model=model, result=result, checkpoint_path=checkpoint_path)
    metrics = {
        "final_loss": result.history["loss"][-1],
        "final_data_loss": result.history["data_loss"][-1],
        "final_physics_loss": result.history["physics_loss"][-1],
    }
    return checkpoint_path, metrics


def main() -> None:
    checkpoint_path, metrics = train_and_save_hybrid()
    print("Saved hybrid end-to-end solver checkpoint:")
    print(f"  checkpoint: {checkpoint_path}")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    main()
