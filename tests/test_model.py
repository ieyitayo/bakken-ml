"""
test_model.py
-------------
Model validation tests for the Bakken Basin oil production model.

Covers the 2 required categories:
  1. Prediction shape/type validation
  2. Minimum performance threshold on a known test sample

Run with:
    pytest tests/test_model.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluate import compute_metrics, feature_importance


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_dataset():
    """
    Generate a synthetic regression dataset that mimics the
    Bakken feature space. Used to train a fast test model.
    """
    rng = np.random.default_rng(42)
    n = 500

    proppant   = rng.uniform(3e6, 15e6, n)
    lateral    = rng.uniform(5000, 15000, n)
    perf_int   = rng.uniform(5000, 12000, n)
    oil_m1     = rng.uniform(2000, 12000, n)
    decline    = rng.uniform(0.1, 0.8, n)
    reservoir  = rng.integers(0, 5, n).astype(float)

    # Target: realistic function of features + modest noise
    # Noise is intentionally low here so the R² threshold test is
    # deterministic. Real-world thresholds live in configs/config.yaml.
    target = (
        0.3 * proppant / 1e6 * 5000
        + 0.4 * lateral / 1000 * 3000
        + 0.2 * oil_m1 * 10
        + rng.normal(0, 2000, n)          # reduced noise vs real data
    )
    target = np.clip(target, 10_000, 500_000)

    X = np.column_stack([proppant, lateral, perf_int, oil_m1, decline, reservoir])
    y = np.log1p(target)   # log-transformed target (as in real training)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="module")
def trained_model(synthetic_dataset):
    """Train a small RandomForest on the synthetic data."""
    X_train, _, y_train, _ = synthetic_dataset
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# Test 1: Prediction shape and type
# ─────────────────────────────────────────────

class TestPredictionShapeAndType:

    def test_predict_returns_numpy_array(self, trained_model, synthetic_dataset):
        """model.predict() must return a numpy ndarray."""
        _, X_test, _, _ = synthetic_dataset
        preds = trained_model.predict(X_test)
        assert isinstance(preds, np.ndarray), \
            f"Expected np.ndarray, got {type(preds)}"

    def test_predict_shape_matches_input(self, trained_model, synthetic_dataset):
        """Output shape must equal (n_samples,) — one prediction per row."""
        _, X_test, _, _ = synthetic_dataset
        preds = trained_model.predict(X_test)
        assert preds.shape == (X_test.shape[0],), \
            f"Expected shape ({X_test.shape[0]},), got {preds.shape}"

    def test_predict_single_row(self, trained_model, synthetic_dataset):
        """Model must handle a single-row input (inference use case)."""
        _, X_test, _, _ = synthetic_dataset
        single = X_test[[0]]
        pred = trained_model.predict(single)
        assert pred.shape == (1,), f"Single-row prediction shape wrong: {pred.shape}"

    def test_predictions_are_finite(self, trained_model, synthetic_dataset):
        """No prediction should be NaN or Inf."""
        _, X_test, _, _ = synthetic_dataset
        preds = trained_model.predict(X_test)
        assert np.all(np.isfinite(preds)), "Predictions contain NaN or Inf values"

    def test_predictions_are_positive_after_expm1(self, trained_model, synthetic_dataset):
        """
        After inverting the log transform (expm1), all predicted
        production values must be positive.
        """
        _, X_test, _, _ = synthetic_dataset
        preds_log = trained_model.predict(X_test)
        preds_bbl = np.expm1(preds_log)
        assert np.all(preds_bbl > 0), \
            "Some predictions are non-positive after expm1 inversion"


# ─────────────────────────────────────────────
# Test 2: Minimum performance threshold
# ─────────────────────────────────────────────

class TestMinimumPerformance:

    # Thresholds — deliberately lenient for synthetic data;
    # real data thresholds are set in configs/config.yaml
    MIN_R2   = 0.50
    MAX_MAPE = 40.0   # percent

    def test_r2_above_minimum_threshold(self, trained_model, synthetic_dataset):
        """
        R² on the held-out test set must exceed MIN_R2.
        A model that just predicts the mean has R²=0; random guessing is negative.
        """
        _, X_test, _, y_test = synthetic_dataset
        y_pred   = trained_model.predict(X_test)
        metrics  = compute_metrics(y_test, y_pred, log_transformed=True)
        assert metrics["r2"] >= self.MIN_R2, (
            f"R² = {metrics['r2']:.4f} is below the minimum threshold {self.MIN_R2}. "
            "Model needs retraining or feature engineering."
        )

    def test_mape_below_maximum_threshold(self, trained_model, synthetic_dataset):
        """
        Mean Absolute Percentage Error must be below MAX_MAPE percent.
        """
        _, X_test, _, y_test = synthetic_dataset
        y_pred  = trained_model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, log_transformed=True)
        assert metrics["mape"] <= self.MAX_MAPE, (
            f"MAPE = {metrics['mape']:.2f}% exceeds the maximum threshold {self.MAX_MAPE}%. "
            "Check for data leakage or preprocessing errors."
        )

    def test_metrics_dict_has_required_keys(self, trained_model, synthetic_dataset):
        """compute_metrics must always return all four required metric keys."""
        _, X_test, _, y_test = synthetic_dataset
        y_pred  = trained_model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, log_transformed=True)
        required = {"mae", "rmse", "r2", "mape"}
        assert required.issubset(metrics.keys()), \
            f"Missing metric keys: {required - metrics.keys()}"

    def test_mae_less_than_rmse(self, trained_model, synthetic_dataset):
        """
        MAE must always be <= RMSE (mathematical property).
        Violation indicates a bug in compute_metrics.
        """
        _, X_test, _, y_test = synthetic_dataset
        y_pred  = trained_model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, log_transformed=True)
        assert metrics["mae"] <= metrics["rmse"], (
            f"MAE ({metrics['mae']:.2f}) > RMSE ({metrics['rmse']:.2f}) — "
            "this is mathematically impossible and indicates a bug."
        )

    def test_feature_importance_covers_all_features(self, trained_model):
        """feature_importance must return one row per feature."""
        feature_names = ["proppant", "lateral", "perf_int",
                         "oil_m1", "decline", "reservoir"]
        fi_df = feature_importance(trained_model, feature_names, plot=False)
        assert len(fi_df) == len(feature_names), \
            f"Expected {len(feature_names)} rows, got {len(fi_df)}"

    def test_feature_importances_sum_to_one(self, trained_model):
        """Feature importances from a RandomForest must sum to 1.0."""
        feature_names = ["proppant", "lateral", "perf_int",
                         "oil_m1", "decline", "reservoir"]
        fi_df = feature_importance(trained_model, feature_names, plot=False)
        total = fi_df["importance"].sum()
        assert abs(total - 1.0) < 1e-6, \
            f"Feature importances sum to {total:.6f}, expected 1.0"
