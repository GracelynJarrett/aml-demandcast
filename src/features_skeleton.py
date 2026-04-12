"""
features.py — Feature engineering for DemandCast
=================================================
This module contains feature engineering logic for the NYC taxi demand
forecasting pipeline. It is imported by pipelines/build_features.py and
src/train.py.

Functions
---------
clean_data             Generic cleaning for raw trip-level DataFrame
create_temporal_features Add time-based features from the pickup datetime column
aggregate_to_hourly_demand Aggregate individual trips into hourly demand per zone
add_lag_features        Add lagged demand columns (1h, 24h, 168h) per zone

Constants
---------
FEATURE_COLS            Intentionally left empty for students to populate.
                        Keep this list in sync with train.py and dashboard.py
                        when you finalize feature choices.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Feature column contract (student exercise)
# ---------------------------------------------------------------------------
# IMPORTANT: Keep this list in sync with train.py and app/dashboard.py.
# Changing a name here without updating those files will break prediction.

FEATURE_COLS: list[str] = [
    "PULocationID",
    "hour",
    "day_of_week",
    "is_weekend",
    "is_rush_hour",
]


# ---------------------------------------------------------------------------
# 1. clean_data 
# ---------------------------------------------------------------------------

# AI prompt used: "Implement clean_data based on EDA with datetime parsing, month-boundary clipping, missing-value handling, and reusable outlier filters."
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw trip-level rows before feature engineering.

    Use thresholds determined during EDA (notebooks/02_eda_skeleton.ipynb). The defaults below are reasonable
    starting points — override them if your EDA revealed different
    breakpoints for your data sample.

    Cleaning strategy (student exercise)
    -----------------------------------
    Implement the data cleaning strategies you determined during exploratory
    data analysis (EDA). Do not hard-code specific thresholds in this
    template; instead document and apply the rules you identified (for
    example: outlier detection, sensible missing-value handling, sensor-error
    filters, or domain-specific rules). Justify your choices in the
    accompanying notebook and use the methods you found appropriate for the
    dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw trip-level DataFrame loaded from the parquet file.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame. Index is reset so it is contiguous after row drops.

    Examples
    --------
    >>> clean_df = clean_data(df)
    >>> print(f"Rows removed: {len(df) - len(clean_df)}")
    """
    cleaned = df.copy()

    # Ensure datetime columns are proper dtype if they exist.
    for dt_col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        if dt_col in cleaned.columns:
            cleaned[dt_col] = pd.to_datetime(cleaned[dt_col], errors="coerce")

    # Keep trips fully inside the dominant pickup month to avoid boundary leakage.
    if {"tpep_pickup_datetime", "tpep_dropoff_datetime"}.issubset(cleaned.columns):
        pickup_month = cleaned["tpep_pickup_datetime"].dt.to_period("M")
        dominant_month = pickup_month.mode(dropna=True)
        if len(dominant_month) > 0:
            target_month = dominant_month.iloc[0]
            in_month_mask = (
                cleaned["tpep_pickup_datetime"].dt.to_period("M").eq(target_month)
                & cleaned["tpep_dropoff_datetime"].dt.to_period("M").eq(target_month)
            )
            cleaned = cleaned.loc[in_month_mask]

    # Drop columns with high missingness, then apply column-aware imputations.
    missing_threshold = 30.0
    missing_pct = cleaned.isna().mean().mul(100)
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
    if cols_to_drop:
        cleaned = cleaned.drop(columns=cols_to_drop)

    fill_rules = {
        "passenger_count": cleaned["passenger_count"].median() if "passenger_count" in cleaned.columns else None,
        "RatecodeID": cleaned["RatecodeID"].mode(dropna=True).iloc[0] if "RatecodeID" in cleaned.columns and not cleaned["RatecodeID"].mode(dropna=True).empty else None,
        "store_and_fwd_flag": cleaned["store_and_fwd_flag"].mode(dropna=True).iloc[0] if "store_and_fwd_flag" in cleaned.columns and not cleaned["store_and_fwd_flag"].mode(dropna=True).empty else None,
        "congestion_surcharge": 0,
        "Airport_fee": 0,
    }
    for col, fill_value in fill_rules.items():
        if col in cleaned.columns and fill_value is not None:
            cleaned[col] = cleaned[col].fillna(fill_value)

    # Apply reusable domain filters only when relevant columns are present.
    if "trip_distance" in cleaned.columns:
        cleaned = cleaned.loc[(cleaned["trip_distance"] > 0) & (cleaned["trip_distance"] <= 50)]
    if "fare_amount" in cleaned.columns:
        cleaned = cleaned.loc[(cleaned["fare_amount"] > 0) & (cleaned["fare_amount"] <= 200)]
    if "passenger_count" in cleaned.columns:
        cleaned = cleaned.loc[(cleaned["passenger_count"] >= 1) & (cleaned["passenger_count"] <= 6)]

    return cleaned.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. create_temporal_features 
# ---------------------------------------------------------------------------

# AI prompt used: "Implement create_temporal_features with robust datetime handling and EDA-aligned temporal columns."
def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from the tpep_pickup_datetime column.

    All features are derived from a single source column so there is no risk
    of data leakage — we are only decomposing information already present at
    prediction time.

    New columns added
    -----------------
    pickup_hour : datetime64
        The pickup datetime floored to the nearest hour.
        Used as the groupby key in aggregate_to_hourly_demand().
    hour : int
        Hour of day (0–23).
    day_of_week : int
        Day of week (0 = Monday, 6 = Sunday). Use dt.dayofweek.
    is_weekend : int
        1 if day_of_week >= 5, else 0.
    month : int
        Month of year (1–12).
    is_rush_hour : int
        1 if (hour is 7, 8 OR hour is 17, 18) AND day_of_week < 5, else 0.
        Morning rush: 7–9am. Evening rush: 5–7pm. Weekdays only.

    Parameters
    ----------
    df : pd.DataFrame
        Trip-level DataFrame. Must contain column tpep_pickup_datetime.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with new feature columns appended.

    Examples
    --------
    >>> df = create_temporal_features(df)
    >>> df[['hour', 'day_of_week', 'is_weekend', 'is_rush_hour']].head()
    """
    if "tpep_pickup_datetime" not in df.columns:
        raise KeyError("Expected column 'tpep_pickup_datetime' in input DataFrame.")

    featured = df.copy()
    pickup_dt = pd.to_datetime(featured["tpep_pickup_datetime"], errors="coerce")

    featured["pickup_hour"] = pickup_dt.dt.floor("h")
    featured["hour"] = featured["pickup_hour"].dt.hour
    featured["day_of_week"] = featured["pickup_hour"].dt.dayofweek
    featured["is_weekend"] = (featured["day_of_week"] >= 5).astype(int)
    featured["month"] = featured["pickup_hour"].dt.month

    is_weekday = featured["day_of_week"] < 5
    is_rush = featured["hour"].isin([7, 8, 17, 18])
    featured["is_rush_hour"] = (is_weekday & is_rush).astype(int)

    return featured

# ---------------------------------------------------------------------------
# 3. aggregate_to_hourly_demand 
# ---------------------------------------------------------------------------

# AI prompt used: "Implement aggregate_to_hourly_demand with flexible input handling and output columns [PULocationID, hour, demand]."
def aggregate_to_hourly_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate individual trips into hourly demand counts per pickup zone.

    This function performs the core transformation that converts the raw
    trip-level data (one row per trip) into the modeling target (one row per
    zone per hour, where the value is the number of pickups).

    Input shape  : (n_trips, many columns)  — e.g. 2.5M rows for January 2024
    Output shape : (n_zones × n_hours, 3)   — e.g. ~260 zones × 744 hours

    Output columns
    --------------
    PULocationID : int
        Pickup zone ID (1–265 in NYC TLC data).
    hour : datetime64[ns]
        The hourly datetime bucket (floored pickup timestamp).
    demand : int
        Number of taxi pickups in this zone during this hour.
    
    Parameters
    ----------
    df : pd.DataFrame
        Trip-level DataFrame after create_temporal_features() has been called.
        Must contain columns: PULocationID, pickup_hour.

    Returns
    -------
    pd.DataFrame
        Aggregated demand DataFrame with columns [PULocationID, hour, demand].

    Examples
    --------
    >>> hourly = aggregate_to_hourly_demand(df)
    >>> print(hourly.shape)   # expect (n_zones * n_hours, 3)
    >>> hourly.head()
    """
    if "PULocationID" not in df.columns:
        raise KeyError("Expected column 'PULocationID' in input DataFrame.")

    # Output columns produced by this function:
    # 1) PULocationID -> pickup zone ID
    # 2) hour         -> hourly datetime bucket (floored pickup timestamp)
    # 3) demand       -> number of trips in each zone-hour group
    working = df.copy()

    # Prefer precomputed pickup_hour, otherwise derive it from pickup datetime.
    if "pickup_hour" not in working.columns:
        if "tpep_pickup_datetime" not in working.columns:
            raise KeyError("Expected 'pickup_hour' or 'tpep_pickup_datetime' in input DataFrame.")
        working["pickup_hour"] = pd.to_datetime(
            working["tpep_pickup_datetime"], errors="coerce"
        ).dt.floor("h")

    hourly = (
        working.groupby(["PULocationID", "pickup_hour"], dropna=False)
        .size()
        .reset_index(name="demand")
        .rename(columns={"pickup_hour": "hour"})
        .sort_values(["PULocationID", "hour"])
        .reset_index(drop=True)
    )

    return hourly


# ---------------------------------------------------------------------------
# 4. add_lag_features
# ---------------------------------------------------------------------------

# AI prompt used: "Implement add_lag_features using groupby(zone_col)[target_col].shift for 1h, 24h, and 168h lags."
def add_lag_features(
    df: pd.DataFrame,
    zone_col: str = "PULocationID",
    target_col: str = "demand",
) -> pd.DataFrame:
    """Add lagged demand features, computed separately for each zone.

    ⚠️  COMMON BUG WARNING ⚠️
    Lag features MUST be computed per zone using groupby. If you call
    df[target_col].shift(n) without groupby, you will bleed one zone's demand
    into the previous/next zone's lag column. This is a subtle data quality
    bug — the model will train without errors, but the features are wrong.

    Correct pattern:
        df[target_col].shift(n)                          ← WRONG
        df.groupby(zone_col)[target_col].shift(n)        ← CORRECT

    New columns added
    -----------------
    demand_lag_1h : float
        Demand for this zone 1 time-step ago (= 1 hour in the hourly table).
    demand_lag_24h : float
        Demand for this zone 24 time-steps ago (= same hour yesterday).
    demand_lag_168h : float
        Demand for this zone 168 time-steps ago (= same hour last week).

    Note: The first n rows for each zone will be NaN for a lag of n.
    Drop these rows after calling this function, or handle them in your
    training pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Hourly demand DataFrame returned by aggregate_to_hourly_demand().
        Must be sorted by zone and hour before calling this function.
        Must contain columns: zone_col, target_col.
    zone_col : str, optional
        Name of the zone identifier column. Default: 'PULocationID'.
    target_col : str, optional
        Name of the demand column to lag. Default: 'demand'.

    Returns
    -------
    pd.DataFrame
        DataFrame with three new lag columns appended.

    Examples
    --------
    >>> hourly = hourly.sort_values(['PULocationID', 'hour'])
    >>> hourly = add_lag_features(hourly, zone_col='PULocationID', target_col='demand')
    >>> hourly[['PULocationID', 'hour', 'demand', 'demand_lag_1h']].head(10)
    """
    missing = [col for col in [zone_col, target_col] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    lagged = df.copy()
    grouped_target = lagged.groupby(zone_col)[target_col]

    lag_map = {
        "demand_lag_1h": 1,
        "demand_lag_24h": 24,
        "demand_lag_168h": 168,
    }

    for lag_col, lag_steps in lag_map.items():
        lagged[lag_col] = grouped_target.shift(lag_steps)

    return lagged
