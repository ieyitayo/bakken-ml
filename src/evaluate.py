"""
evaluate.py
-----------
Model evaluation utilities for the Bakken Basin oil production model.

Provides:
  - compute_metrics()     : MAE, RMSE, R², MAPE on raw (un-logged) scale
  - feature_importance()  : ranked feature importances from tree models
  - plot_actual_vs_pred() : scatter plot for visual residual check
  - find_best_run()       : queries MLflow to identify the best experiment run

Usage (standalone):
    python src/evaluate.py --run-id <mlflow_run_id> \
                           --test-data data/processed/ml_ready.csv \
                           --config configs/config.yaml
"""

import argparse
import logging
import os
import pickle

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_transformed: bool = True,
) -> dict:
    """
    Compute MAE, RMSE, R², and MAPE on the **original BBL scale**.

    If log_transformed=True, both y_true and y_pred are assumed to be
    log(cum_oil_12mo) and are exponentiated before computing metrics.

    Returns:
        dict with keys: mae, rmse, r2, mape
    """
    if log_transformed:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    # Guard against division by zero in MAPE
    mask = y_true > 0
    y_true_safe = y_true[mask]
    y_pred_safe = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(
        np.mean(np.abs((y_true_safe - y_pred_safe) / y_true_safe)) * 100)

    metrics = {
        "mae":  round(mae,  2),
        "rmse": round(rmse, 2),
        "r2":   round(r2,   4),
        "mape": round(mape, 2),
    }

    logger.info(
        f"Metrics → MAE: {mae:,.0f} BBL | RMSE: {rmse:,.0f} BBL | "
        f"R²: {r2:.4f} | MAPE: {mape:.2f}%"
    )
    return metrics


# ─────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────

def feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 15,
    plot: bool = True,
    save_path: str | None = None,
) -> pd.DataFrame:
    """
    Extract and optionally plot feature importances from a fitted tree model.

    Works with sklearn RandomForest, GradientBoosting, and XGBoost.

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted descending.
    """
    # XGBoost exposes feature_importances_ via the sklearn API
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not expose feature_importances_; skipping.")
        return pd.DataFrame(columns=["feature", "importance"])

    importances = model.feature_importances_
    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if plot:
        top = fi_df.head(top_n)
        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color="#4e79a7")
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            logger.info(f"Feature importance plot saved → {save_path}")
        plt.show()

    return fi_df


# ─────────────────────────────────────────────
# Actual vs predicted plot
# ─────────────────────────────────────────────

def plot_actual_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_transformed: bool = True,
    title: str = "Actual vs Predicted — First-Year Oil (BBL)",
    save_path: str | None = None,
) -> None:
    """
    Scatter plot of actual vs predicted values with a perfect-fit reference line.
    Both axes are displayed on the raw BBL scale.
    """
    if log_transformed:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: actual vs predicted ──
    axes[0].scatter(y_true / 1e3, y_pred / 1e3,
                    alpha=0.3, s=10, color="#4e79a7")
    lim = max(y_true.max(), y_pred.max()) / 1e3 * 1.05
    axes[0].plot([0, lim], [0, lim], "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Cum Oil 12mo (K BBL)")
    axes[0].set_ylabel("Predicted Cum Oil 12mo (K BBL)")
    axes[0].set_title(title)
    axes[0].legend()

    # ── Right: residuals ──
    residuals = y_pred - y_true
    axes[1].scatter(y_true / 1e3, residuals / 1e3,
                    alpha=0.3, s=10, color="#f28e2b")
    axes[1].axhline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Actual Cum Oil 12mo (K BBL)")
    axes[1].set_ylabel("Residual (K BBL)")
    axes[1].set_title("Residuals")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info(f"Actual vs predicted plot saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# MLflow best-run selection
# ─────────────────────────────────────────────

def find_best_run(
    experiment_name: str,
    tracking_uri: str = "mlruns",
    metric: str = "test_r2",
    higher_is_better: bool = True,
) -> pd.Series:
    """
    Query MLflow to programmatically identify the best experiment run.

    Args:
        experiment_name:  Name of the MLflow experiment.
        tracking_uri:     Local or remote MLflow tracking URI.
        metric:           Metric to rank runs by (default: test_r2).
        higher_is_better: If True, select run with highest metric value.

    Returns:
        pandas Series with run_id, params, and metrics of the best run.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found in MLflow tracking at "
            f"'{tracking_uri}'. Run train.py first."
        )

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=[f"metrics.{metric} {'DESC' if higher_is_better else 'ASC'}"],
    )

    if runs_df.empty:
        raise ValueError("No finished runs found. Run train.py first.")

    best = runs_df.iloc[0]

    logger.info(f"Best run: {best['run_id']}")
    logger.info(f"  Model : {best.get('params.model_name', 'unknown')}")
    logger.info(f"  {metric}: {best[f'metrics.{metric}']:.4f}")

    # Print comparison table
    metric_cols = [c for c in runs_df.columns if c.startswith("metrics.")]
    param_cols = ["params.model_name", "params.algorithm"]
    display_cols = (
        ["run_id"] +
        [c for c in param_cols if c in runs_df.columns] +
        metric_cols
    )
    print("\n=== Experiment Run Comparison ===")
    print(runs_df[display_cols].to_string(index=False))

    return best


# ─────────────────────────────────────────────
# Load a saved model + transformers for inference
# ─────────────────────────────────────────────

def load_model_for_inference(
    run_id: str,
    encoders_path: str,
    scaler_path: str,
    tracking_uri: str = "mlruns",
):
    """
    Load the trained model artifact from MLflow plus the
    preprocessing transformers from disk.

    Returns:
        (model, encoders, scaler)
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Loaded model from MLflow run {run_id}")

    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, encoders, scaler


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Bakken model and compare MLflow runs"
    )
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument(
        "--run-id",    help="Specific MLflow run ID to evaluate")
    parser.add_argument("--test-data", default="data/processed/ml_ready.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Find best run ──
    best_run = find_best_run(
        experiment_name=cfg["mlflow"]["experiment_name"],
        tracking_uri=cfg["mlflow"]["tracking_uri"],
    )

    run_id = args.run_id or best_run["run_id"]

    # ── Load model + data ──
    model, encoders, scaler = load_model_for_inference(
        run_id=run_id,
        encoders_path=cfg["data"]["encoders_path"],
        scaler_path=cfg["data"]["scaler_path"],
        tracking_uri=cfg["mlflow"]["tracking_uri"],
    )

    df = pd.read_csv(args.test_data, index_col=0)
    feature_cols = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    log_transform = cfg["data"]["log_transform_target"]
    y = np.log1p(
        df["cum_oil_12mo"].values) if log_transform else df["cum_oil_12mo"].values

    y_pred = model.predict(X)
    metrics = compute_metrics(y, y_pred, log_transformed=log_transform)
    print("\nFinal evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k.upper():6s}: {v}")

    plot_actual_vs_pred(y, y_pred, log_transformed=log_transform)
    feature_importance(model, feature_cols)
