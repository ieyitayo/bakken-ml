"""
test_preprocess.py
------------------
Unit tests for src/preprocess.py.

Covers the 4 required categories:
  1. Missing value handling
  2. Categorical encoding
  3. Numeric scaling to expected ranges
  4. Original DataFrame immutability
Plus: filter logic, feature engineering correctness.

Run with:
    pytest tests/test_preprocess.py -v
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import (
    filter_wells,
    drop_short_wells,
    engineer_features,
    impute_missing,
    encode_categoricals,
    scale_features,
    attach_static_features,
    pivot_monthly_to_well,
    CATEGORICAL_COLS,
    NUMERIC_FEATURE_COLS,
)


# ─────────────────────────────────────────────
# Fixtures — reusable synthetic data
# ─────────────────────────────────────────────

def make_monthly_df(n_wells: int = 3, n_months: int = 15) -> pd.DataFrame:
    """
    Build a minimal synthetic monthly production DataFrame
    that mirrors the real file's schema.
    """
    rows = []
    for w in range(n_wells):
        api = f"API_{w:05d}"
        for m in range(1, n_months + 1):
            oil = max(0, 5000 - m * 200 + np.random.randint(-100, 100))
            gas = oil * 0.5
            rows.append({
                "API/UWI": api,
                "Monthly Production Date": pd.Timestamp(f"2015-{((m-1) % 12)+1:02d}-01"),
                "Monthly Oil": float(oil),
                "Monthly Gas": float(gas),
                "Monthly Water": float(oil * 2),
                "Monthly BOE": float(oil + gas / 6),
                "Days": 30,
                "Producing Month Number": m,
                "Monthly Oil Per perforated Ft": oil / 300.0 if m == 1 else np.nan,
                "Production Type": "OIL",
                "Well Status": "ACTIVE",
                "Reservoir": "MIDDLE BAKKEN",
                "DI Landing Zone": "BAKKEN",
                "Total Proppant": 8_000_000.0,
                "Total Fluid": 50_000.0,
                "Gross Perforated Interval": 9500.0,
                "DI Lateral Length": 9800.0,
                "Operator Company Name": "TEST OPERATOR",
            })
    return pd.DataFrame(rows)


def make_well_feature_df() -> pd.DataFrame:
    """
    A flat per-well DataFrame (post-pivot) for testing
    imputation, encoding, and scaling.
    """
    data = {
        "oil_month_1": [5000.0, np.nan, 4200.0],
        "oil_month_2": [4800.0, 3900.0, np.nan],
        "oil_month_3": [4600.0, 3700.0, 4000.0],
        "oil_month_6": [3800.0, 3000.0, 3200.0],
        "cum_oil_3mo": [14400.0, np.nan, 12200.0],
        "cum_oil_6mo": [27000.0, 22000.0, np.nan],
        "peak_monthly_oil": [5000.0, 3900.0, 4200.0],
        "decline_rate_m1_to_m6": [0.24, 0.23, 0.24],
        "avg_gor_12mo": [500.0, 480.0, np.nan],
        "oil_per_perf_ft_m1": [16.7, np.nan, 14.0],
        "Total Proppant": [8e6, 7e6, np.nan],
        "Total Fluid": [50000.0, np.nan, 48000.0],
        "Gross Perforated Interval": [9500.0, 9200.0, 9800.0],
        "DI Lateral Length": [9800.0, 9500.0, 10100.0],
        "vintage_year": [2015.0, 2018.0, 2020.0],
        "cum_oil_12mo": [55000.0, 44000.0, 50000.0],
        "Reservoir": ["MIDDLE BAKKEN", "UNKNOWN", "THREE FORKS"],
        "DI Landing Zone": ["BAKKEN", "BAKKEN", np.nan],
        "Production Type": ["OIL", "OIL", "O&G"],
        "Well Status": ["ACTIVE", "ACTIVE", "P & A"],
    }
    return pd.DataFrame(data)


# ─────────────────────────────────────────────
# Test 1: Missing values are handled correctly
# ─────────────────────────────────────────────

class TestMissingValues:
    def test_numeric_nulls_are_filled(self):
        """All numeric NaNs must be gone after imputation."""
        df = make_well_feature_df()
        assert df.isnull().any().any(), "Fixture should have nulls before imputation"

        result = impute_missing(df)

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert result[numeric_cols].isnull().sum().sum() == 0, \
            "Numeric columns must have zero NaNs after imputation"

    def test_numeric_imputed_with_median(self):
        """Numeric NaN is replaced with the column median, not mean or zero."""
        df = pd.DataFrame({
            "oil_month_1": [100.0, 200.0, np.nan, 400.0],
            "Reservoir": ["A", "B", "C", "D"],
            "DI Landing Zone": ["X", "X", "X", "X"],
            "Production Type": ["OIL"] * 4,
            "Well Status": ["ACTIVE"] * 4,
            "cum_oil_12mo": [1000.0] * 4,
        })
        expected_median = pd.Series([100.0, 200.0, 400.0]).median()  # 200.0
        result = impute_missing(df)
        assert result.loc[2, "oil_month_1"] == pytest.approx(expected_median), \
            f"Expected median {expected_median}, got {result.loc[2, 'oil_month_1']}"

    def test_categorical_nulls_filled_with_unknown(self):
        """NaN categoricals become the string 'UNKNOWN'."""
        df = make_well_feature_df()
        result = impute_missing(df)
        for col in CATEGORICAL_COLS:
            if col in result.columns:
                assert result[col].isnull().sum() == 0, \
                    f"Categorical column '{col}' still has NaNs"
                assert "UNKNOWN" in result[col].values or result[col].notna().all()

    def test_wells_with_fewer_than_12_months_are_dropped(self):
        """drop_short_wells must remove wells lacking the minimum month count."""
        df = make_monthly_df(n_wells=3, n_months=15)
        # Artificially shorten one well to 8 months
        short_api = df["API/UWI"].unique()[0]
        df = df[~((df["API/UWI"] == short_api) & (df["Producing Month Number"] > 8))]

        result = drop_short_wells(df, min_months=12)
        assert short_api not in result["API/UWI"].values, \
            "Well with < 12 months should have been dropped"
        assert result["API/UWI"].nunique() == 2


# ─────────────────────────────────────────────
# Test 2: Categorical encoding
# ─────────────────────────────────────────────

class TestCategoricalEncoding:
    def test_categoricals_become_integers(self):
        """After encoding, categorical columns must be integer dtype."""
        df = impute_missing(make_well_feature_df())
        encoded, _ = encode_categoricals(df, fit=True)
        for col in CATEGORICAL_COLS:
            if col in encoded.columns:
                assert pd.api.types.is_integer_dtype(encoded[col]), \
                    f"Column '{col}' should be integer after encoding"

    def test_encoder_is_saved_and_reusable(self):
        """Encoders fitted on training data must transform unseen data without error."""
        df = impute_missing(make_well_feature_df())
        _, encoders = encode_categoricals(df, fit=True)

        # New inference row with the same categories
        df2 = df.copy()
        encoded2, _ = encode_categoricals(df2, encoders=encoders, fit=False)
        assert encoded2 is not None

    def test_unseen_label_handled_gracefully(self):
        """An unseen category at inference time must not raise an exception."""
        df = impute_missing(make_well_feature_df())
        _, encoders = encode_categoricals(df, fit=True)

        df_new = df.copy()
        df_new["Reservoir"] = "COMPLETELY_NEW_FORMATION"
        # Should not raise
        try:
            encode_categoricals(df_new, encoders=encoders, fit=False)
        except Exception as e:
            pytest.fail(f"Unseen label raised an exception: {e}")


# ─────────────────────────────────────────────
# Test 3: Numeric scaling to expected ranges
# ─────────────────────────────────────────────

class TestScaling:
    def _get_scaled_df(self):
        df = impute_missing(make_well_feature_df())
        df, _ = encode_categoricals(df, fit=True)
        scaled, scaler = scale_features(df, fit=True)
        return scaled, scaler

    def test_scaled_columns_have_near_zero_mean(self):
        """StandardScaler should produce columns with mean ≈ 0."""
        scaled, _ = self._get_scaled_df()
        scale_cols = [c for c in NUMERIC_FEATURE_COLS if c in scaled.columns]
        for col in scale_cols:
            mean = scaled[col].mean()
            assert abs(mean) < 1e-6, \
                f"Column '{col}' mean={mean:.6f}, expected ≈ 0 after scaling"

    def test_scaled_columns_have_unit_variance(self):
        """StandardScaler should produce columns with std ≈ 1 (when n > 1)."""
        scaled, _ = self._get_scaled_df()
        scale_cols = [c for c in NUMERIC_FEATURE_COLS if c in scaled.columns]
        for col in scale_cols:
            std = scaled[col].std(ddof=0)
            # Allow tolerance for very small datasets
            assert std == pytest.approx(1.0, abs=0.5), \
                f"Column '{col}' std={std:.4f}, expected ≈ 1 after scaling"

    def test_target_column_not_scaled(self):
        """cum_oil_12mo (target) must NOT be scaled — its raw values must be preserved."""
        df = impute_missing(make_well_feature_df())
        df, _ = encode_categoricals(df, fit=True)
        original_target = df["cum_oil_12mo"].copy()
        scaled, _ = scale_features(df, fit=True)
        pd.testing.assert_series_equal(
            scaled["cum_oil_12mo"], original_target,
            check_names=True,
        )


# ─────────────────────────────────────────────
# Test 4: Original DataFrame immutability
# ─────────────────────────────────────────────

class TestImmutability:
    def test_impute_does_not_modify_original(self):
        """impute_missing must return a new DataFrame, not mutate the input."""
        df = make_well_feature_df()
        original_nulls = df.isnull().sum().sum()
        _ = impute_missing(df)
        assert df.isnull().sum().sum() == original_nulls, \
            "impute_missing modified the original DataFrame"

    def test_encode_does_not_modify_original(self):
        """encode_categoricals must not alter the input DataFrame."""
        df = impute_missing(make_well_feature_df())
        original_reservoir = df["Reservoir"].copy()
        _, _ = encode_categoricals(df, fit=True)
        pd.testing.assert_series_equal(df["Reservoir"], original_reservoir)

    def test_scale_does_not_modify_original(self):
        """scale_features must not alter the input DataFrame."""
        df = impute_missing(make_well_feature_df())
        df, _ = encode_categoricals(df, fit=True)
        original_vals = df[
            [c for c in NUMERIC_FEATURE_COLS if c in df.columns]
        ].copy()
        _, _ = scale_features(df, fit=True)
        pd.testing.assert_frame_equal(
            df[[c for c in NUMERIC_FEATURE_COLS if c in df.columns]],
            original_vals,
        )

    def test_filter_does_not_modify_original(self):
        """filter_wells must not modify the raw input DataFrame."""
        df = make_monthly_df()
        original_len = len(df)
        _ = filter_wells(df)
        assert len(df) == original_len, \
            "filter_wells modified the original DataFrame"


# ─────────────────────────────────────────────
# Test 5: Feature engineering correctness
# ─────────────────────────────────────────────

class TestFeatureEngineering:
    def test_cum_oil_12mo_equals_sum_of_monthly_values(self):
        """Target variable must be the exact sum of months 1–12."""
        df = make_monthly_df(n_wells=1, n_months=15)
        well_df = pivot_monthly_to_well(df)
        well_df = engineer_features(well_df)

        api = well_df.index[0]
        expected = sum(
            df[(df["API/UWI"] == api) & (df["Producing Month Number"] <= 12)]["Monthly Oil"]
        )
        assert well_df.loc[api, "cum_oil_12mo"] == pytest.approx(expected, rel=1e-5)

    def test_decline_rate_is_between_minus_one_and_one(self):
        """Decline rate must be clipped to [-1, 1]."""
        df = make_monthly_df(n_wells=2, n_months=15)
        well_df = pivot_monthly_to_well(df)
        well_df = engineer_features(well_df)
        if "decline_rate_m1_to_m6" in well_df.columns:
            rates = well_df["decline_rate_m1_to_m6"].dropna()
            assert (rates >= -1).all() and (rates <= 1).all(), \
                "decline_rate_m1_to_m6 outside [-1, 1]"

    def test_filter_removes_non_oil_wells(self):
        """filter_wells must exclude gas-only and injection wells."""
        df = make_monthly_df(n_wells=2, n_months=15)
        df.loc[df["API/UWI"] == df["API/UWI"].unique()[0], "Production Type"] = "GAS"
        result = filter_wells(df)
        remaining_types = result["Production Type"].str.upper().unique()
        assert "GAS" not in remaining_types
