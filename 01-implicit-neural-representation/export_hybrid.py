from __future__ import annotations

from pathlib import Path

from ai_for_sim.inr import export_inr_artifacts, load_checkpoint
from ai_for_sim.paths import INR_ARTIFACTS_DIR


CHECKPOINT_PATH = INR_ARTIFACTS_DIR / "hybrid" / "hybrid_checkpoint.pt"
OUTPUT_DIR = INR_ARTIFACTS_DIR / "hybrid"


def export_hybrid_artifacts(
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    _, result = load_checkpoint(checkpoint_path)
    return export_inr_artifacts(result, output_dir)


def main() -> None:
    outputs = export_hybrid_artifacts()
    print("Exported hybrid INR artifacts:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
