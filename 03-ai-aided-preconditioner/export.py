from __future__ import annotations

from pathlib import Path

from ai_for_sim.aided_preconditioner import export_preconditioner_artifacts, load_checkpoint
from ai_for_sim.paths import PRECONDITIONER_ARTIFACTS_DIR


CHECKPOINT_PATH = (
    PRECONDITIONER_ARTIFACTS_DIR
    / "gauss_newton_preconditioner"
    / "gauss_newton_preconditioner_checkpoint.pt"
)
OUTPUT_DIR = PRECONDITIONER_ARTIFACTS_DIR / "gauss_newton_preconditioner"


def export_artifacts(
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    _, result = load_checkpoint(checkpoint_path)
    return export_preconditioner_artifacts(result, output_dir)


def main() -> None:
    outputs = export_artifacts()
    print("Exported AI-aided preconditioner artifacts (Gauss-Newton + CG, toy scale):")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
