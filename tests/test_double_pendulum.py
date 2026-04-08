from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np

from ai_for_sim.double_pendulum import DoublePendulumParams, simulate_double_pendulum


def load_export_module():
    module_path = Path(__file__).resolve().parents[1] / "00-discrete-solver" / "export.py"
    spec = importlib.util.spec_from_file_location("discrete_solver_export", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_simulation_returns_finite_values():
    params = DoublePendulumParams(duration=2.0, fps=10)
    result = simulate_double_pendulum(params)

    assert result.state.shape == (21, 4)
    assert np.isfinite(result.state).all()
    assert np.allclose(result.time[[0, -1]], [0.0, 2.0])


def test_export_writes_expected_artifacts(tmp_path: Path):
    export_module = load_export_module()
    outputs = export_module.export_double_pendulum_assets(tmp_path)

    for key in ("trajectory", "figure", "animation"):
        assert outputs[key].exists()
        assert outputs[key].stat().st_size > 0

    data = np.load(outputs["trajectory"])
    assert data["state"].shape[1] == 4
    assert np.isfinite(data["state"]).all()
