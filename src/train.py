"""
train.py
--------
Trains multiple ML model configurations on the Bakken Basin
first-year oil production dataset, logging every run to MLflow.

Each model config in configs/config.yaml becomes one MLflow run.
After all runs complete, the best model (by test R²) is identified
programmatically using mlflow.search_runs().

Usage:
    python src/train.py --config configs/config.yaml

    # Skip preprocessing if ml_ready.csv already exists:
    python src/train.py --config configs/config.yaml --skip-preprocess
"""

from preprocess import run_preprocessing
from evaluate import compute_metrics, feature_importance, find_best_run
import argparse
import logging
import os
import sys
import time

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Resolve imports whether run from repo root or src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Config loader
# ─────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {path}")
    return cfg


# ─────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────

def prepare_data(cfg: dict, skip_preprocess: bool = False):
    """
    Run preprocessing (or load cached result), apply outlier cap,
    optionally log-transform the target, and split train/test.

    Returns:
        X_train, X_test, y_train, y_test, feature_cols
    """
    processed_path = cfg["data"]["processed_path"]

    if skip_preprocess and os.path.exists(processed_path):
        logger.info(
            f"Skipping preprocessing — loading cached {processed_path}")
        df = pd.read_csv(processed_path, index_col=0)
    else:
        logger.info("Running full preprocessing pipeline …")
        df, _, _ = run_preprocessing(
            input_path=cfg["data"]["raw_path"],
            output_csv=processed_path,
            encoders_path=cfg["data"]["encoders_path"],
            scaler_path=cfg["data"]["scaler_path"],
        )

    # ── Feature columns ──
    num_feats = [c for c in cfg["features"]["numeric"] if c in df.columns]
    cat_feats = [c for c in cfg["features"]["categorical"] if c in df.columns]
    feature_cols = num_feats + cat_feats
    logger.info(f"Features used: {feature_cols}")

    # ── Outlier cap on target ──
    cap_pct = cfg["data"].get("outlier_cap_pct", 0.99)
    cap_val = df["cum_oil_12mo"].quantile(cap_pct)
    before = len(df)
    df = df[df["cum_oil_12mo"] <= cap_val].copy()
    logger.info(
        f"Outlier cap at {cap_pct*100:.0f}th pct ({cap_val:,.0f} BBL): "
        f"removed {before - len(df)} wells"
    )

    # ── Target ──
    if cfg["data"].get("log_transform_target", True):
        y = np.log1p(df["cum_oil_12mo"].values)
        logger.info("Target log-transformed: log(cum_oil_12mo + 1)")
    else:
        y = df["cum_oil_12mo"].values

    X = df[feature_cols].values

    # ── Train / test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )
    logger.info(
        f"Train: {X_train.shape[0]:,} wells | Test: {X_test.shape[0]:,} wells"
    )
    return X_train, X_test, y_train, y_test, feature_cols


# ─────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────

def build_model(algorithm: str, params: dict):
    """
    Instantiate the correct estimator from the algorithm name in config.

    Supported: 'random_forest', 'gradient_boosting', 'xgboost'
    """
    algo = algorithm.lower()
    if algo == "random_forest":
        return RandomForestRegressor(**params)
    elif algo == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    elif algo == "xgboost":
        return XGBRegressor(**params, verbosity=0)
    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            "Choose: random_forest | gradient_boosting | xgboost"
        )


# ─────────────────────────────────────────────
# Single training run
# ─────────────────────────────────────────────

def train_one_run(
    model_cfg: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    global_cfg: dict,
) -> str:
    """
    Train one model configuration and log everything to MLflow.

    Returns:
        MLflow run_id
    """
    name = model_cfg["name"]
    algorithm = model_cfg["algorithm"]
    params = model_cfg["params"]
    log_transform = global_cfg["data"].get("log_transform_target", True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting run: {name}  ({algorithm})")
    logger.info(f"Params: {params}")

    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id

        # ── Log metadata ──
        mlflow.set_tag("model_name",   name)
        mlflow.set_tag("algorithm",    algorithm)
        mlflow.set_tag("data_version", global_cfg["data"]["raw_path"])

        # ── Log all hyperparameters ──
        mlflow.log_param("model_name",        name)
        mlflow.log_param("algorithm",         algorithm)
        mlflow.log_param("log_transform",     log_transform)
        mlflow.log_param("test_size",         global_cfg["data"]["test_size"])
        mlflow.log_param("outlier_cap_pct",
                         global_cfg["data"]["outlier_cap_pct"])
        mlflow.log_param("n_train_wells",     X_train.shape[0])
        mlflow.log_param("n_features",        len(feature_cols))
        mlflow.log_param("feature_list",      str(feature_cols))
        for k, v in params.items():
            mlflow.log_param(k, v)

        # ── Train ──
        model = build_model(algorithm, params)
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time_sec", round(train_time, 2))
        logger.info(f"Training complete in {train_time:.1f}s")

        # ── Evaluate on both splits ──
        for split_name, X_split, y_split in [
            ("train", X_train, y_train),
            ("test",  X_test,  y_test),
        ]:
            y_pred = model.predict(X_split)
            metrics = compute_metrics(
                y_split, y_pred, log_transformed=log_transform)
            for metric_name, val in metrics.items():
                mlflow.log_metric(f"{split_name}_{metric_name}", val)

        # Report test metrics
        test_metrics = compute_metrics(
            y_test, model.predict(X_test), log_transformed=log_transform
        )
        logger.info(
            f"Test → R²: {test_metrics['r2']:.4f} | "
            f"MAE: {test_metrics['mae']:,.0f} BBL | "
            f"RMSE: {test_metrics['rmse']:,.0f} BBL | "
            f"MAPE: {test_metrics['mape']:.2f}%"
        )

        # ── Feature importances (log top features) ──
        if hasattr(model, "feature_importances_"):
            fi_df = feature_importance(model, feature_cols, plot=False)
            for _, row in fi_df.head(10).iterrows():
                safe_name = row["feature"].replace(" ", "_").replace("/", "_")
                mlflow.log_metric(f"fi_{safe_name}", round(
                    float(row["importance"]), 6))

            # Save importance CSV as artifact
            import tempfile
            fi_path = os.path.join(tempfile.gettempdir(), f"fi_{name}.csv")
            fi_df.to_csv(fi_path, index=False)
            mlflow.log_artifact(fi_path, artifact_path="feature_importance")

        # ── Log model artifact ──
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=global_cfg["mlflow"]["artifact_path"],
            registered_model_name=None,   # set a name here to use Model Registry
        )
        logger.info(f"Run {run_id} logged to MLflow ✓")

    return run_id


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def run_training(config_path: str, skip_preprocess: bool = False) -> str:
    """
    Train all model configurations defined in config.yaml.

    Returns:
        run_id of the best model (by test R²)
    """
    cfg = load_config(config_path)

    # ── MLflow setup ──
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    logger.info(
        f"MLflow experiment: '{cfg['mlflow']['experiment_name']}' "
        f"at '{cfg['mlflow']['tracking_uri']}'"
    )

    # ── Prepare data once (shared across all runs) ──
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(
        cfg, skip_preprocess=skip_preprocess
    )

    # ── Train each model config ──
    run_ids = []
    for model_cfg in cfg["models"]:
        run_id = train_one_run(
            model_cfg=model_cfg,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_cols=feature_cols,
            global_cfg=cfg,
        )
        run_ids.append(run_id)

    logger.info(f"\n{'='*60}")
    logger.info(f"All {len(run_ids)} runs complete.")

    # ── Programmatically find best run ──
    logger.info("\nQuerying MLflow to identify best run …")
    best = find_best_run(
        experiment_name=cfg["mlflow"]["experiment_name"],
        tracking_uri=cfg["mlflow"]["tracking_uri"],
        metric="test_r2",
        higher_is_better=True,
    )

    best_run_id = best["run_id"]
    best_r2 = best["metrics.test_r2"]
    best_name = best.get("params.model_name", "unknown")

    logger.info(f"\n🏆 Best model: {best_name}")
    logger.info(f"   Run ID : {best_run_id}")
    logger.info(f"   Test R²: {best_r2:.4f}")
    logger.info(
        f"\n   To load this model:\n"
        f"   mlflow.sklearn.load_model('runs:/{best_run_id}/model')\n"
        f"\n   Or run the app:\n"
        f"   python src/app.py --run-id {best_run_id}"
    )

    # ── Tag best run in MLflow ──
    with mlflow.start_run(run_id=best_run_id):
        mlflow.set_tag("best_model", "true")

    return best_run_id


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Bakken Basin oil production models with MLflow tracking"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocessing if ml_ready.csv already exists",
    )
    args = parser.parse_args()

    best_run_id = run_training(
        config_path=args.config,
        skip_preprocess=args.skip_preprocess,
    )
    print(f"\nBest run ID: {best_run_id}")
