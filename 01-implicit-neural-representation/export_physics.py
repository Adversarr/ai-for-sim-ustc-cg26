from __future__ import annotations

from pathlib import Path

from ai_for_sim.inr import export_inr_artifacts, load_checkpoint
from ai_for_sim.paths import INR_ARTIFACTS_DIR


CHECKPOINT_PATH = INR_ARTIFACTS_DIR / "physics" / "physics_checkpoint.pt"
OUTPUT_DIR = INR_ARTIFACTS_DIR / "physics"


def export_physics_artifacts(
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    _, result = load_checkpoint(checkpoint_path)
    return export_inr_artifacts(result, output_dir)


def main() -> None:
    outputs = export_physics_artifacts()
    print("Exported physics-informed INR artifacts:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
