from __future__ import annotations

from pathlib import Path

from export import export_artifacts
from train import train_and_save


def run_all() -> dict[str, Path]:
    train_and_save()
    return export_artifacts()


def main() -> None:
    outputs = run_all()
    print("Generated AI-aided preconditioner demo artifacts (Gauss-Newton + CG, toy scale):")
    for name, path in outputs.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
