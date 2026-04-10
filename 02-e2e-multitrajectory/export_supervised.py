from __future__ import annotations

from pathlib import Path

from ai_for_sim.e2e_solver import export_e2e_artifacts, load_checkpoint


ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
CHECKPOINT_PATH = ARTIFACTS_DIR / "supervised" / "supervised_checkpoint.pt"
OUTPUT_DIR = ARTIFACTS_DIR / "supervised"


def export_supervised_artifacts(
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    _, result = load_checkpoint(checkpoint_path)
    return export_e2e_artifacts(result, output_dir)


def main() -> None:
    outputs = export_supervised_artifacts()
    print("Exported multi-trajectory supervised end-to-end solver artifacts:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
