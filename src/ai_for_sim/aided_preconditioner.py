from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from torch import nn

from .aided_solver import (
    AidedModelConfig,
    AidedTrainingConfig,
    ImplicitSolverConfig,
    PROJECTED_HESSIAN_VARIANT,
    SolverRollout,
    TrainedAidedSolverResult,
    default_split_config,
    demo_params,
    load_checkpoint as _load_aided_solver_checkpoint,
    plot_loss_history,
    save_aided_animation,
    save_comparison_animation,
    save_prediction_arrays,
    save_checkpoint_payload as _save_aided_solver_checkpoint,
    simulate_preconditioned_implicit_euler as simulate_gauss_newton_preconditioned_implicit_euler,
    train_projected_hessian_preconditioner,
)
from .e2e_solver import TrajectorySplitConfig


VARIANT_NAME = "gauss_newton_preconditioner"


def quick_model_config() -> AidedModelConfig:
    return AidedModelConfig(hidden_width=96, hidden_layers=3)


def quick_training_config() -> AidedTrainingConfig:
    return AidedTrainingConfig(
        epochs=500,
        learning_rate=1e-3,
        min_learning_rate=5e-5,
        batch_size=64,
        log_interval=10,
        matrix_loss_weight=0.15,
        action_loss_weight=1.0,
        projection_damping=1e-2,
        device="cpu",
    )


def quick_solver_config() -> ImplicitSolverConfig:
    return ImplicitSolverConfig(
        nonlinear_tol=1e-8,
        max_iterations=10,
        line_search_shrink=0.5,
        min_step_size=1e-4,
        jacobian_epsilon=1e-6,
        fallback_to_newton=True,
    )


def train_gauss_newton_preconditioner(
    model_config: AidedModelConfig | None = None,
    training_config: AidedTrainingConfig | None = None,
    params=None,
    solver_config: ImplicitSolverConfig | None = None,
    split_config: TrajectorySplitConfig | None = None,
) -> tuple[nn.Module, TrainedAidedSolverResult]:
    model, result = train_projected_hessian_preconditioner(
        model_config=model_config or quick_model_config(),
        training_config=training_config or quick_training_config(),
        params=params or demo_params(),
        solver_config=solver_config or quick_solver_config(),
        split_config=split_config or default_split_config(),
    )
    exact_state = result.exact_rollout.result.state
    aided_state = result.aided_rollout.result.state
    return model, replace(
        result,
        variant=VARIANT_NAME,
        metadata={
            **result.metadata,
            "trajectory_mse": float(np.mean((aided_state - exact_state) ** 2)),
            "max_state_error": float(np.max(np.linalg.norm(aided_state - exact_state, axis=1))),
            "accepted_preconditioner_rate": float(1.0 - np.mean(result.aided_rollout.fallback_steps)),
        },
    )


def save_checkpoint_payload(*, model: nn.Module, result: TrainedAidedSolverResult, checkpoint_path: Path) -> Path:
    serialized_result = result if result.variant == VARIANT_NAME else replace(result, variant=VARIANT_NAME)
    return _save_aided_solver_checkpoint(model=model, result=serialized_result, checkpoint_path=checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device_name: str = "cpu") -> tuple[nn.Module, TrainedAidedSolverResult]:
    model, result = _load_aided_solver_checkpoint(checkpoint_path, device_name=device_name)
    if result.variant not in {PROJECTED_HESSIAN_VARIANT, VARIANT_NAME}:
        raise ValueError("Checkpoint is not a Gauss-Newton preconditioner model.")
    result = replace(result, variant=VARIANT_NAME)
    return model, result


def plot_preconditioner_summary(result: TrainedAidedSolverResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    explicit_residual = result.aided_rollout.explicit_guess_residual_norms
    preconditioned_residual = result.aided_rollout.warm_start_residual_norms
    residual_ratio = preconditioned_residual / np.maximum(explicit_residual, 1e-16)
    residual_removed = explicit_residual - preconditioned_residual
    correction_delta = result.exact_rollout.linear_iteration_counts - result.aided_rollout.linear_iteration_counts
    accepted_mask = result.aided_rollout.fallback_steps == 0

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    min_residual = max(float(min(np.min(explicit_residual), np.min(preconditioned_residual))), 1e-16)
    max_residual = float(max(np.max(explicit_residual), np.max(preconditioned_residual)))
    diagonal = np.geomspace(min_residual, max_residual, 200)
    axes[0].scatter(
        explicit_residual[accepted_mask],
        preconditioned_residual[accepted_mask],
        color="#2563eb",
        s=16,
        alpha=0.8,
        label="accepted",
    )
    if np.any(~accepted_mask):
        axes[0].scatter(
            explicit_residual[~accepted_mask],
            preconditioned_residual[~accepted_mask],
            color="#f59e0b",
            s=20,
            alpha=0.9,
            label="fallback",
        )
    axes[0].plot(diagonal, diagonal, "--", color="#111827", linewidth=1.0, label="same residual")
    axes[0].set_title("Did the Learned CG Preconditioner Improve the First Step?")
    axes[0].set_xlabel(r"$\|r_{\mathrm{explicit}}\|$")
    axes[0].set_ylabel(r"$\|r_{\mathrm{preconditioned}}\|$")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].legend(frameon=False)
    axes[0].text(
        0.02,
        0.98,
        f"accepted = {accepted_mask.mean():.1%}\nmedian ratio = {np.median(residual_ratio):.3f}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    ratio_min = max(float(np.min(residual_ratio)), 1e-3)
    ratio_max = max(float(np.max(residual_ratio)), 1.0) * 1.05
    ratio_bins = np.geomspace(ratio_min, ratio_max, 18)
    axes[1].hist(residual_ratio[accepted_mask], bins=ratio_bins, color="#2563eb", alpha=0.75, label="accepted")
    if np.any(~accepted_mask):
        axes[1].hist(residual_ratio[~accepted_mask], bins=ratio_bins, color="#f59e0b", alpha=0.8, label="fallback")
    axes[1].axvline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[1].set_title("How Much Residual Was Removed Before Classical Corrections?")
    axes[1].set_xlabel(r"$\|r_{\mathrm{preconditioned}}\| / \|r_{\mathrm{explicit}}\|$")
    axes[1].set_ylabel("Count")
    axes[1].set_xscale("log")
    axes[1].legend(frameon=False)
    axes[1].text(
        0.02,
        0.98,
        f"median ratio = {np.median(residual_ratio):.3f}\nmedian removed = {np.median(residual_removed):.3e}",
        transform=axes[1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    correction_values = np.arange(
        int(min(np.min(result.exact_rollout.linear_iteration_counts), np.min(result.aided_rollout.linear_iteration_counts))),
        int(max(np.max(result.exact_rollout.linear_iteration_counts), np.max(result.aided_rollout.linear_iteration_counts))) + 1,
    )
    exact_counts = np.array([np.sum(result.exact_rollout.linear_iteration_counts == value) for value in correction_values])
    preconditioned_counts = np.array([np.sum(result.aided_rollout.linear_iteration_counts == value) for value in correction_values])
    bar_width = 0.35
    axes[2].bar(
        correction_values - bar_width / 2,
        exact_counts,
        width=bar_width,
        color="#2563eb",
        label="classical solver",
    )
    axes[2].bar(
        correction_values + bar_width / 2,
        preconditioned_counts,
        width=bar_width,
        color="#dc2626",
        label="with learned preconditioner",
    )
    axes[2].set_title("Did the Preconditioner Reduce CG Iterations?")
    axes[2].set_xlabel("Inner CG iterations per step")
    axes[2].set_ylabel("Count")
    axes[2].set_xticks(correction_values)
    axes[2].legend(frameon=False)
    axes[2].text(
        0.02,
        0.98,
        f"mean classical = {result.exact_rollout.linear_iteration_counts.mean():.3f}\n"
        f"mean aided = {result.aided_rollout.linear_iteration_counts.mean():.3f}",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    axes[3].scatter(residual_ratio, correction_delta, color="#111827", s=16, alpha=0.75)
    axes[3].axvline(1.0, color="#111827", linestyle="--", linewidth=1.0)
    axes[3].axhline(0.0, color="#111827", linestyle=":", linewidth=1.0)
    axes[3].set_title("Did Better Preconditioning Reduce CG Work?")
    axes[3].set_xlabel(r"$\|r_{\mathrm{preconditioned}}\| / \|r_{\mathrm{explicit}}\|$")
    axes[3].set_ylabel("classical - aided CG iterations")
    axes[3].set_xscale("log")
    if len(residual_ratio) > 1 and float(np.std(residual_ratio)) > 0.0 and float(np.std(correction_delta)) > 0.0:
        correlation = float(np.corrcoef(residual_ratio, correction_delta)[0, 1])
    else:
        correlation = float("nan")
    axes[3].text(
        0.02,
        0.98,
        f"corr = {correlation:.3f}",
        transform=axes[3].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.9},
    )

    fig.suptitle("AI-Aided Preconditioner in Gauss-Newton CG (Toy Scale)", fontsize=14)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_preconditioner_artifacts(result: TrainedAidedSolverResult, output_dir: Path) -> dict[str, Path]:
    export_result = result if result.variant == VARIANT_NAME else replace(result, variant=VARIANT_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "predictions": save_prediction_arrays(export_result, output_dir / f"{export_result.variant}_prediction.npz"),
        "summary": plot_preconditioner_summary(export_result, output_dir / f"{export_result.variant}_summary.png"),
        "loss": plot_loss_history(export_result, output_dir / f"{export_result.variant}_loss.png"),
        "animation": save_aided_animation(export_result, output_dir / f"{export_result.variant}_animation.gif"),
        "comparison_animation": save_comparison_animation(
            export_result,
            output_dir / f"{export_result.variant}_comparison.gif",
        ),
    }
