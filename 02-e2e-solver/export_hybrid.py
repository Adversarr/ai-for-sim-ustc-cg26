from __future__ import annotations

from pathlib import Path

from ai_for_sim.e2e_solver import export_e2e_artifacts, load_checkpoint
from ai_for_sim.paths import E2E_ARTIFACTS_DIR


CHECKPOINT_PATH = E2E_ARTIFACTS_DIR / "hybrid" / "hybrid_checkpoint.pt"
OUTPUT_DIR = E2E_ARTIFACTS_DIR / "hybrid"


def export_hybrid_artifacts(
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    _, result = load_checkpoint(checkpoint_path)
    return export_e2e_artifacts(result, output_dir)


def main() -> None:
    outputs = export_hybrid_artifacts()
    print("Exported hybrid end-to-end solver artifacts:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
