from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DISCRETE_SOLVER_DIR = REPO_ROOT / "00-discrete-solver"
ARTIFACTS_DIR = DISCRETE_SOLVER_DIR / "artifacts"
INR_DIR = REPO_ROOT / "01-implicit-neural-representation"
INR_ARTIFACTS_DIR = INR_DIR / "artifacts"
