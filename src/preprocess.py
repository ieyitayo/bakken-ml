"""
preprocess.py
-------------
Transforms raw Bakken Basin monthly production CSV into a clean,
flat, ML-ready dataset (one row per well).

Pipeline:
  1. Load & parse dates
  2. Filter to oil-producing horizontal wells
  3. Drop wells with fewer than 12 months of production data
  4. Pivot monthly rows → per-well feature columns
  5. Engineer decline rate, GOR, normalized IP
  6. Attach static completion features from each well's Month-1 row
  7. Impute missing values
  8. Encode categoricals (LabelEncoder, saved for inference)
  9. Scale numerics (StandardScaler, saved for inference)
 10. Save processed dataset + transformers

Usage:
    python src/preprocess.py --input data/raw/monthly_production.csv \
                             --output data/processed/ml_ready.csv \
                             --encoders data/processed/encoders.pkl \
                             --scaler data/processed/scaler.pkl
"""

import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
VALID_PRODUCTION_TYPES = {"OIL", "O&G", "OIL & GAS"}
MIN_MONTHS_REQUIRED = 12        # wells must have at least 12 months to build target
TARGET_MONTHS = 12              # sum of months 1-N = target variable
STATIC_COLS = [                 # pulled from each well's month-1 row
    "Total Proppant",
    "Total Fluid",
    "Gross Perforated Interval",
    "DI Lateral Length",
    "Reservoir",
    "DI Landing Zone",
    "Production Type",
    "Well Status",
    "Operator Company Name",
]
CATEGORICAL_COLS = [
    "Reservoir",
    "DI Landing Zone",
    "Production Type",
    "Well Status",
]
NUMERIC_FEATURE_COLS = [
    "Total Proppant",
    "Total Fluid",
    "Gross Perforated Interval",
    "DI Lateral Length",
    "oil_month_1",
    "oil_month_2",
    "oil_month_3",
    "cum_oil_3mo",
    "cum_oil_6mo",
    "peak_monthly_oil",
    "decline_rate_m1_to_m6",
    "avg_gor_12mo",
    "oil_per_perf_ft_m1",
    "vintage_year",
]


# ─────────────────────────────────────────────
# Step 1 — Load
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load raw monthly production CSV and parse dates."""
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(
        filepath,
        parse_dates=["Monthly Production Date"],
        low_memory=False,
    )
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# Step 2 — Filter
# ─────────────────────────────────────────────
def filter_wells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only oil-producing wells with valid month numbers.
    Normalises Production Type strings before filtering.
    Also removes:
      - Negative production values (data errors — <0.01% of records per EDA)
      - Rows with Days <= 0 or Days is null (required for daily-rate features)
    """
    original_len = len(df)

    # Normalise production type
    df["Production Type"] = (
        df["Production Type"].astype(str).str.strip().str.upper()
    )
    df = df[df["Production Type"].isin(VALID_PRODUCTION_TYPES)].copy()
    logger.info(
        f"After production-type filter: {len(df):,} rows "
        f"(removed {original_len - len(df):,})"
    )

    # Remove negative production values (identified in EDA: ~0.005% of records)
    for col in ["Monthly Oil", "Monthly Gas", "Monthly Water", "Monthly BOE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                df = df[df[col] >= 0].copy()
                logger.info(f"Removed {neg_count:,} negative values in '{col}'")

    # Remove rows with Days <= 0 or null (required for daily-rate calculations)
    # EDA found 54,430 such rows (1.31% of dataset)
    if "Days" in df.columns:
        df["Days"] = pd.to_numeric(df["Days"], errors="coerce")
        before_days = len(df)
        df = df[df["Days"] > 0].copy()
        logger.info(
            f"Removed {before_days - len(df):,} rows with Days <= 0 or null"
        )

    # Keep only positive month numbers
    df = df[df["Producing Month Number"].notna()].copy()
    df["Producing Month Number"] = pd.to_numeric(
        df["Producing Month Number"], errors="coerce"
    )
    df = df[df["Producing Month Number"] >= 1].copy()

    # Coerce key numeric columns
    for col in ["Monthly Oil", "Monthly Gas", "Monthly Water",
                "Monthly BOE", "Monthly Oil Per perforated Ft"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"After month-number filter: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# Step 3 — Drop wells with < 12 months
# ─────────────────────────────────────────────
def drop_short_wells(df: pd.DataFrame, min_months: int = MIN_MONTHS_REQUIRED) -> pd.DataFrame:
    """Remove wells that don't have enough months to form the target."""
    month_counts = df.groupby("API/UWI")["Producing Month Number"].max()
    valid_apis = month_counts[month_counts >= min_months].index
    before = df["API/UWI"].nunique()
    df = df[df["API/UWI"].isin(valid_apis)].copy()
    after = df["API/UWI"].nunique()
    logger.info(
        f"Wells with >= {min_months} months: {after:,} "
        f"(dropped {before - after:,} short wells)"
    )
    return df


# ─────────────────────────────────────────────
# Step 4 — Pivot monthly rows into per-well columns
# ─────────────────────────────────────────────
def pivot_monthly_to_well(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each well, extract individual month values as columns.
    Only months 1–12 are used.
    Returns a DataFrame indexed by API/UWI.
    """
    logger.info("Pivoting monthly rows into per-well feature columns …")

    # Only work within the first 12 months for features + target
    df_12 = df[df["Producing Month Number"] <= TARGET_MONTHS].copy()

    # Pivot: one column per month for oil, gas, water
    oil_pivot = (
        df_12.pivot_table(
            index="API/UWI",
            columns="Producing Month Number",
            values="Monthly Oil",
            aggfunc="first",
        )
        .rename(columns=lambda m: f"oil_month_{int(m)}")
    )

    gas_pivot = (
        df_12.pivot_table(
            index="API/UWI",
            columns="Producing Month Number",
            values="Monthly Gas",
            aggfunc="first",
        )
        .rename(columns=lambda m: f"gas_month_{int(m)}")
    )

    # Normalised IP (oil per perforated ft, month 1 only)
    ip_norm = (
        df_12[df_12["Producing Month Number"] == 1]
        .set_index("API/UWI")["Monthly Oil Per perforated Ft"]
        .rename("oil_per_perf_ft_m1")
    )

    well_df = oil_pivot.join(gas_pivot, how="outer").join(ip_norm, how="left")
    logger.info(f"Pivoted dataframe: {well_df.shape}")
    return well_df


# ─────────────────────────────────────────────
# Step 5 — Engineer features
# ─────────────────────────────────────────────
def engineer_features(well_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute engineered features and the target variable.
    All operations are on the pivoted per-well DataFrame.
    """
    logger.info("Engineering features …")

    # ── Target: cumulative oil in first 12 months ──
    oil_cols = [f"oil_month_{m}" for m in range(1, TARGET_MONTHS + 1)]
    available_oil_cols = [c for c in oil_cols if c in well_df.columns]
    well_df["cum_oil_12mo"] = well_df[available_oil_cols].sum(axis=1, min_count=1)

    # ── Cumulative windows ──
    oil_3 = [f"oil_month_{m}" for m in range(1, 4) if f"oil_month_{m}" in well_df.columns]
    oil_6 = [f"oil_month_{m}" for m in range(1, 7) if f"oil_month_{m}" in well_df.columns]
    well_df["cum_oil_3mo"] = well_df[oil_3].sum(axis=1, min_count=1)
    well_df["cum_oil_6mo"] = well_df[oil_6].sum(axis=1, min_count=1)

    # ── Early month values ──
    for m in [1, 2, 3]:
        col = f"oil_month_{m}"
        if col not in well_df.columns:
            well_df[col] = np.nan

    # ── Peak monthly oil (months 1-12) ──
    well_df["peak_monthly_oil"] = well_df[available_oil_cols].max(axis=1)

    # ── Decline rate month 1 → month 6 ──
    # decline_rate = (m1 - m6) / m1  — positive means declining (normal)
    if "oil_month_1" in well_df.columns and "oil_month_6" in well_df.columns:
        m1 = well_df["oil_month_1"].replace(0, np.nan)
        m6 = well_df["oil_month_6"]
        well_df["decline_rate_m1_to_m6"] = (m1 - m6) / m1
        # Clip to [-1, 1] — anything outside is a data anomaly
        well_df["decline_rate_m1_to_m6"] = well_df["decline_rate_m1_to_m6"].clip(-1, 1)
    else:
        well_df["decline_rate_m1_to_m6"] = np.nan

    # ── Average GOR over months 1-12 ──
    gas_cols = [f"gas_month_{m}" for m in range(1, TARGET_MONTHS + 1)
                if f"gas_month_{m}" in well_df.columns]
    if gas_cols and available_oil_cols:
        total_gas = well_df[gas_cols].sum(axis=1, min_count=1)
        total_oil = well_df[available_oil_cols].sum(axis=1, min_count=1).replace(0, np.nan)
        well_df["avg_gor_12mo"] = total_gas / total_oil
        well_df["avg_gor_12mo"] = well_df["avg_gor_12mo"].clip(0, 50000)
    else:
        well_df["avg_gor_12mo"] = np.nan

    # Drop raw monthly columns — we only keep engineered features
    raw_monthly = [c for c in well_df.columns
                   if c.startswith("oil_month_") and
                   c not in ["oil_month_1", "oil_month_2", "oil_month_3"]]
    raw_monthly += [c for c in well_df.columns if c.startswith("gas_month_")]
    well_df.drop(columns=raw_monthly, inplace=True, errors="ignore")

    logger.info(f"After feature engineering: {well_df.shape[1]} columns")
    return well_df


# ─────────────────────────────────────────────
# Step 6 — Attach static completion features
# ─────────────────────────────────────────────
def attach_static_features(
    well_df: pd.DataFrame, df_raw: pd.DataFrame
) -> pd.DataFrame:
    """
    Pull static (time-invariant) features from each well's first recorded month.
    These are completion / header columns that don't change month-to-month.
    """
    logger.info("Attaching static completion features from Month 1 rows …")

    # Use the earliest available month per well to get static values
    static_src = (
        df_raw.sort_values("Producing Month Number")
        .groupby("API/UWI")
        .first()
        .reset_index()
    )
    available_static = [c for c in STATIC_COLS if c in static_src.columns]
    static_df = static_src.set_index("API/UWI")[available_static]

    # ── Vintage year: year of first production month ──
    # EDA shows steady improvement in well performance 2005→2020 (technology trend)
    if "Monthly Production Date" in df_raw.columns:
        vintage = (
            df_raw[df_raw["Producing Month Number"] == 1]
            .groupby("API/UWI")["Monthly Production Date"]
            .min()
            .dt.year
            .rename("vintage_year")
        )
        static_df = static_df.join(vintage, how="left")

    merged = well_df.join(static_df, how="left")
    logger.info(f"After attaching static features: {merged.shape}")
    return merged


# ─────────────────────────────────────────────
# Step 7 — Impute missing values
# ─────────────────────────────────────────────
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric columns  → median imputation.
    Categorical cols → fill with 'UNKNOWN'.
    Does NOT modify the original DataFrame.
    """
    logger.info("Imputing missing values …")
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()

    missing_after = df.isnull().sum().sum()
    logger.info(f"Missing values after imputation: {missing_after}")
    return df


# ─────────────────────────────────────────────
# Step 8 — Encode categoricals
# ─────────────────────────────────────────────
def encode_categoricals(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.

    Args:
        df:       Input DataFrame.
        encoders: Existing encoder dict (used during inference).
        fit:      If True, fit new encoders; if False, use provided ones.

    Returns:
        (encoded_df, encoders_dict)
    """
    logger.info(f"Encoding categoricals (fit={fit}) …")
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully during inference
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "UNKNOWN")
            df[col] = le.transform(df[col].astype(str))

    return df, encoders


# ─────────────────────────────────────────────
# Step 9 — Scale numerics
# ─────────────────────────────────────────────
def scale_features(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    StandardScale numeric feature columns.
    The target column (cum_oil_12mo) is intentionally excluded.

    Args:
        df:     Input DataFrame.
        scaler: Existing scaler (used during inference).
        fit:    If True, fit a new scaler; if False, use provided one.

    Returns:
        (scaled_df, scaler)
    """
    logger.info(f"Scaling numeric features (fit={fit}) …")
    df = df.copy()

    scale_cols = [c for c in NUMERIC_FEATURE_COLS if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    else:
        df[scale_cols] = scaler.transform(df[scale_cols])

    return df, scaler


# ─────────────────────────────────────────────
# Step 10 — Save artifacts
# ─────────────────────────────────────────────
def save_artifacts(
    df: pd.DataFrame,
    encoders: dict,
    scaler: StandardScaler,
    output_csv: str,
    encoders_path: str,
    scaler_path: str,
) -> None:
    """Persist the processed dataset and fitted transformers."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=True)
    logger.info(f"Saved processed dataset → {output_csv}  ({len(df):,} wells)")

    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)
    logger.info(f"Saved encoders → {encoders_path}")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler → {scaler_path}")


# ─────────────────────────────────────────────
# Public API — used by train.py and tests
# ─────────────────────────────────────────────
def run_preprocessing(
    input_path: str,
    output_csv: str = "data/processed/ml_ready.csv",
    encoders_path: str = "data/processed/encoders.pkl",
    scaler_path: str = "data/processed/scaler.pkl",
    fit: bool = True,
    encoders: dict | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, dict, StandardScaler]:
    """
    Full end-to-end preprocessing pipeline.

    Returns:
        (ml_ready_df, encoders, scaler)
    """
    df_raw = load_data(input_path)
    df_raw = filter_wells(df_raw)
    df_raw = drop_short_wells(df_raw)
    well_df = pivot_monthly_to_well(df_raw)
    well_df = engineer_features(well_df)
    well_df = attach_static_features(well_df, df_raw)
    well_df = impute_missing(well_df)
    well_df, encoders = encode_categoricals(well_df, encoders=encoders, fit=fit)
    well_df, scaler = scale_features(well_df, scaler=scaler, fit=fit)

    # Drop rows where target is null or zero
    before = len(well_df)
    well_df = well_df[well_df["cum_oil_12mo"].notna() & (well_df["cum_oil_12mo"] > 0)]
    logger.info(
        f"Dropped {before - len(well_df)} wells with null/zero target. "
        f"Final dataset: {len(well_df):,} wells"
    )

    if fit:
        save_artifacts(well_df, encoders, scaler, output_csv, encoders_path, scaler_path)

    return well_df, encoders, scaler


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bakken Basin — preprocess monthly production data")
    parser.add_argument("--input",    required=True,  help="Path to raw monthly production CSV")
    parser.add_argument("--output",   default="data/processed/ml_ready.csv")
    parser.add_argument("--encoders", default="data/processed/encoders.pkl")
    parser.add_argument("--scaler",   default="data/processed/scaler.pkl")
    args = parser.parse_args()

    run_preprocessing(
        input_path=args.input,
        output_csv=args.output,
        encoders_path=args.encoders,
        scaler_path=args.scaler,
    )
