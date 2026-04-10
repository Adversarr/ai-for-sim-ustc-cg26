from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DISCRETE_SOLVER_DIR = REPO_ROOT / "00-discrete-solver"
ARTIFACTS_DIR = DISCRETE_SOLVER_DIR / "artifacts"
INR_DIR = REPO_ROOT / "01-implicit-neural-representation"
INR_ARTIFACTS_DIR = INR_DIR / "artifacts"
E2E_DIR = REPO_ROOT / "02-e2e-solver"
E2E_ARTIFACTS_DIR = E2E_DIR / "artifacts"
AIDED_SOLVER_DIR = REPO_ROOT / "03-ai-aided-warmup"
AIDED_ARTIFACTS_DIR = AIDED_SOLVER_DIR / "artifacts"
PRECONDITIONER_DIR = REPO_ROOT / "03-ai-aided-preconditioner"
PRECONDITIONER_ARTIFACTS_DIR = PRECONDITIONER_DIR / "artifacts"
