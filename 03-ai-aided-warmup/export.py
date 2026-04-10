from __future__ import annotations

from pathlib import Path

from ai_for_sim.aided_solver import export_aided_solver_artifacts, load_checkpoint
from ai_for_sim.paths import AIDED_ARTIFACTS_DIR


CHECKPOINT_PATH = AIDED_ARTIFACTS_DIR / "direct_warmup" / "direct_warmup_checkpoint.pt"
OUTPUT_DIR = AIDED_ARTIFACTS_DIR / "direct_warmup"


def export_artifacts(
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    _, result = load_checkpoint(checkpoint_path)
    return export_aided_solver_artifacts(result, output_dir)


def main() -> None:
    outputs = export_artifacts()
    print("Exported AI-aided implicit solver artifacts:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
