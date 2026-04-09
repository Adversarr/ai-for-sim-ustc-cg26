from __future__ import annotations

from pathlib import Path

from export_hybrid import export_hybrid_artifacts
from export_physics import export_physics_artifacts
from export_supervised import export_supervised_artifacts
from train_hybrid import train_and_save_hybrid
from train_physics import train_and_save_physics
from train_supervised import train_and_save_supervised


def run_all() -> dict[str, dict[str, Path]]:
    train_and_save_supervised()
    supervised_outputs = export_supervised_artifacts()

    train_and_save_physics()
    physics_outputs = export_physics_artifacts()

    train_and_save_hybrid()
    hybrid_outputs = export_hybrid_artifacts()

    return {
        "supervised": supervised_outputs,
        "physics": physics_outputs,
        "hybrid": hybrid_outputs,
    }


def main() -> None:
    outputs = run_all()
    print("Generated INR demo artifacts:")
    for variant, variant_outputs in outputs.items():
        print(f"  {variant}:")
        for name, path in variant_outputs.items():
            print(f"    {name}: {path}")


if __name__ == "__main__":
    main()
