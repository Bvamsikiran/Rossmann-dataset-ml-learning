"""
Rossmann Sales ML Learning Lab - Relational Pipeline & Feature Engineering
Orchestrates multi-table joining, temporal decomposition, competitive dynamics,
and promotional modeling across all Rossmann relational datasets:
- train.csv + store.csv + store_states.csv + state_names.csv + weather.csv + googletrend.csv
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from backend.config import (
    GERMAN_STATE_FALLBACK_MAP,
    DEFAULT_SAMPLE_ROWS,
    DEFAULT_BASE_FEATURES,
    DEFAULT_STORE_FEATURES,
    DEFAULT_WEATHER_FEATURES,
    DEFAULT_TREND_FEATURES,
    DEFAULT_DATETIME_FEATURES,
)
from backend.data_loader import load_raw_table

logger = logging.getLogger("rossmann.pipeline")

# Cached merged DataFrame for high-performance interactive simulation
_MERGED_CACHE: Optional[pd.DataFrame] = None
_CACHE_METADATA: Dict[str, Any] = {}


def build_state_mapping(state_names_df: pd.DataFrame) -> Dict[str, str]:
    """
    Construct a clean mapping dictionary from state abbreviation (e.g. HE, NW)
    to the full federal state name (e.g. Hessen, NordrheinWestfalen).
    Handles composite abbreviations like 'HB,NI' gracefully.
    """
    mapping: Dict[str, str] = dict(GERMAN_STATE_FALLBACK_MAP)
    for _, row in state_names_df.iterrows():
        name = str(row.get("StateName", "")).strip()
        abbr_val = str(row.get("State", "")).strip()
        if not name or not abbr_val:
            continue
        # Split on commas for joint entries like 'HB,NI'
        for abbr in abbr_val.split(","):
            clean_abbr = abbr.strip()
            if clean_abbr:
                mapping[clean_abbr] = name
    return mapping


def parse_googletrend_weeks(gt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process googletrend.csv 'week' column ('YYYY-MM-DD - YYYY-MM-DD')
    into parsed start date and end date timestamps, and extract state codes
    from the 'file' column ('Rossmann_DE_SN' -> 'SN', 'Rossmann_DE' -> 'DE').
    """
    df = gt_df.copy()
    if "week" not in df.columns or "file" not in df.columns:
        return df

    # Extract state identifier
    def extract_state(file_str: str) -> str:
        s = str(file_str).strip()
        if s == "Rossmann_DE":
            return "DE"
        if s.startswith("Rossmann_DE_"):
            return s.replace("Rossmann_DE_", "").strip()
        return s

    df["TrendState"] = df["file"].apply(extract_state)

    # Parse week start date
    week_parts = df["week"].astype(str).str.split(" - ", expand=True)
    df["WeekStart"] = pd.to_datetime(week_parts[0], errors="coerce")
    df["WeekEnd"] = pd.to_datetime(week_parts[1], errors="coerce") if week_parts.shape[1] > 1 else df["WeekStart"]
    df["Year"] = df["WeekStart"].dt.year
    df["WeekOfYear"] = df["WeekStart"].dt.isocalendar().week.astype(int)

    return df


def calculate_competition_months(row: pd.Series) -> float:
    """
    Calculate the elapsed number of months since a competitor opened.
    Returns 0 if no competitor or opening date is not recorded.
    """
    year = row.get("Year")
    month = row.get("Month")
    comp_year = row.get("CompetitionOpenSinceYear")
    comp_month = row.get("CompetitionOpenSinceMonth")

    if pd.isna(comp_year) or pd.isna(comp_month) or pd.isna(year) or pd.isna(month):
        return 0.0

    comp_year = float(comp_year)
    comp_month = float(comp_month)
    if comp_year <= 0 or comp_month <= 0:
        return 0.0

    diff = (year - comp_year) * 12 + (month - comp_month)
    return max(0.0, diff)


def calculate_promo2_active(row: pd.Series) -> int:
    """
    Determine if Promo2 (continuous coupon campaign) is actively running
    for a store on the current date, accounting for interval months.
    """
    if row.get("Promo2") != 1:
        return 0

    p_year = row.get("Promo2SinceYear")
    p_week = row.get("Promo2SinceWeek")
    cur_year = row.get("Year")
    cur_week = row.get("WeekOfYear")
    cur_month = row.get("Month")
    interval = str(row.get("PromoInterval", "") or "")

    if pd.isna(p_year) or pd.isna(p_week) or pd.isna(cur_year) or pd.isna(cur_week):
        return 0

    # Must be after the campaign start week
    weeks_since = (cur_year - p_year) * 52 + (cur_week - p_week)
    if weeks_since < 0:
        return 0

    # Month matching (e.g. 'Jan,Apr,Jul,Oct')
    month_abbrs = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec"]
    cur_month_abbr = month_abbrs[int(cur_month)] if 1 <= int(cur_month) <= 12 else ""

    if cur_month_abbr and cur_month_abbr in interval:
        return 1
    elif not interval:
        return 1
    return 0


def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose datetime objects into calendar and cyclical features:
    Year, Month, Day, WeekOfYear, DayOfWeek, IsWeekend, Quarter, IsMonthStart, IsMonthEnd.
    """
    out = df.copy()
    if "Date" not in out.columns:
        return out

    date_series = pd.to_datetime(out["Date"])
    out["Year"] = date_series.dt.year.astype("int32")
    out["Month"] = date_series.dt.month.astype("int8")
    out["Day"] = date_series.dt.day.astype("int8")
    out["DayOfWeek"] = (date_series.dt.dayofweek + 1).astype("int8")  # 1=Mon, 7=Sun
    out["WeekOfYear"] = date_series.dt.isocalendar().week.astype("int8")
    out["Quarter"] = date_series.dt.quarter.astype("int8")
    out["IsWeekend"] = (out["DayOfWeek"] >= 6).astype("int8")
    out["IsMonthStart"] = date_series.dt.is_month_start.astype("int8")
    out["IsMonthEnd"] = date_series.dt.is_month_end.astype("int8")

    return out


def execute_multi_dataset_merge(
    sample_size: int = DEFAULT_SAMPLE_ROWS,
    include_store: bool = True,
    include_weather: bool = True,
    include_trends: bool = True,
    force_reload: bool = False,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Execute the full relational join pipeline across the 5 Rossmann tables:
    1. train (base sales transactions)
    2. store (characteristics) on Store
    3. store_states + state_names (geographic locations) on Store
    4. weather on [StateName, Date]
    5. googletrend on [State, WeekOfYear, Year]
    """
    global _MERGED_CACHE, _CACHE_METADATA

    # Check if cached merged dataset can be reused
    if (
        not force_reload
        and _MERGED_CACHE is not None
        and _CACHE_METADATA.get("sample_size") == sample_size
        and _CACHE_METADATA.get("include_weather") == include_weather
        and _CACHE_METADATA.get("include_trends") == include_trends
    ):
        logger.info(f"Using cached merged pipeline dataset ({len(_MERGED_CACHE)} rows)")
        return _MERGED_CACHE.copy(), dict(_CACHE_METADATA)

    start_time = time.time()
    tables_joined = ["train"]

    # 1. Load train dataset with sample limit
    logger.info(f"Step 1: Loading train dataset (sample_size={sample_size})...")
    train_df = load_raw_table("train", nrows=sample_size, force_reload=force_reload)

    # Date temporal feature decomposition
    merged = engineer_temporal_features(train_df)

    # 2. Join store metadata
    if include_store:
        logger.info("Step 2: Merging store characteristics on 'Store'...")
        store_df = load_raw_table("store", force_reload=force_reload)
        merged = merged.merge(store_df, on="Store", how="left")
        tables_joined.append("store")

        # Join store states & federal state names
        store_states_df = load_raw_table("store_states", force_reload=force_reload)
        state_names_df = load_raw_table("state_names", force_reload=force_reload)
        state_map = build_state_mapping(state_names_df)

        store_states_clean = store_states_df.copy()
        store_states_clean["StateName"] = store_states_clean["State"].map(state_map)
        merged = merged.merge(store_states_clean, on="Store", how="left")
        tables_joined.extend(["store_states", "state_names"])

        # Calculate competitive and promo dynamics
        merged["CompetitionDistance"] = pd.to_numeric(merged["CompetitionDistance"], errors="coerce")
        merged["LogCompetitionDistance"] = np.log1p(merged["CompetitionDistance"].fillna(0))
        merged["HasCompetition"] = (merged["CompetitionDistance"].notnull() & (merged["CompetitionDistance"] > 0)).astype("int8")
        merged["MonthsSinceCompetition"] = merged.apply(calculate_competition_months, axis=1).astype("float32")
        merged["Promo2Active"] = merged.apply(calculate_promo2_active, axis=1).astype("int8")

    # 3. Join weather daily observations
    if include_weather and "StateName" in merged.columns:
        logger.info("Step 3: Merging daily weather conditions on ['StateName', 'Date']...")
        try:
            weather_df = load_raw_table("weather", force_reload=force_reload)
            weather_clean = weather_df.rename(columns={"file": "StateName"}).copy()
            weather_clean["Date"] = pd.to_datetime(weather_clean["Date"])
            merged["Date"] = pd.to_datetime(merged["Date"])

            weather_cols = [
                "StateName",
                "Date",
                "Mean_TemperatureC",
                "Max_TemperatureC",
                "Min_TemperatureC",
                "Precipitationmm",
                "Mean_Humidity",
                "Max_Wind_SpeedKm_h",
                "CloudCover",
                "Events",
            ]
            avail_weather_cols = [c for c in weather_cols if c in weather_clean.columns]
            merged = merged.merge(weather_clean[avail_weather_cols], on=["StateName", "Date"], how="left")

            # Weather derived features
            if "Max_TemperatureC" in merged.columns and "Min_TemperatureC" in merged.columns:
                merged["TemperatureRange"] = (merged["Max_TemperatureC"] - merged["Min_TemperatureC"]).astype("float32")
            if "Precipitationmm" in merged.columns:
                merged["HasRain"] = (merged["Precipitationmm"].fillna(0) > 0).astype("int8")

            tables_joined.append("weather")
        except Exception as e:
            logger.warning(f"Weather join skipped due to error: {e}")

    # 4. Join Google search trends
    if include_trends and "State" in merged.columns:
        logger.info("Step 4: Merging weekly Google Trends search interest...")
        try:
            gt_df = load_raw_table("googletrend", force_reload=force_reload)
            gt_clean = parse_googletrend_weeks(gt_df)

            # National trend lookup for fallback
            de_trend = gt_clean[gt_clean["TrendState"] == "DE"][["Year", "WeekOfYear", "trend"]].rename(
                columns={"trend": "NationalTrend"}
            )
            merged = merged.merge(de_trend, on=["Year", "WeekOfYear"], how="left")

            # State specific trend lookup
            state_trend = gt_clean[gt_clean["TrendState"] != "DE"][["TrendState", "Year", "WeekOfYear", "trend"]].rename(
                columns={"TrendState": "State", "trend": "trend"}
            )
            merged = merged.merge(state_trend, on=["State", "Year", "WeekOfYear"], how="left")

            # Fill missing regional trends with national trend
            if "trend" in merged.columns and "NationalTrend" in merged.columns:
                merged["trend"] = merged["trend"].fillna(merged["NationalTrend"]).fillna(50)

            tables_joined.append("googletrend")
        except Exception as e:
            logger.warning(f"Google Trends join skipped due to error: {e}")

    # Remove duplicate columns if any resulted from joins
    merged = merged.loc[:, ~merged.columns.duplicated()]

    elapsed = time.time() - start_time
    logger.info(f"Pipeline merge completed in {elapsed:.2f}s (Final shape: {merged.shape})")

    # Generate summary metadata
    missing_summary = {col: int(merged[col].isnull().sum()) for col in merged.columns}
    
    # Store cache
    _MERGED_CACHE = merged.copy()
    _CACHE_METADATA = {
        "sample_size": sample_size,
        "include_store": include_store,
        "include_weather": include_weather,
        "include_trends": include_trends,
        "total_rows": len(merged),
        "total_columns": len(merged.columns),
        "tables_joined": tables_joined,
        "elapsed_seconds": round(elapsed, 2),
        "missing_summary": missing_summary,
    }

    return merged.copy(), dict(_CACHE_METADATA)


def get_feature_catalog(df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    """
    Inspect the merged dataframe and compile an annotated catalog of available features,
    specifying domain category, data type, numeric classification, and null metrics.
    """
    if df is None:
        if _MERGED_CACHE is not None:
            df = _MERGED_CACHE
        else:
            df, _ = execute_multi_dataset_merge(sample_size=1000)

    catalog: List[Dict[str, Any]] = []

    descriptions = {
        "Store": ("Store Identifier", "sales"),
        "DayOfWeek": ("Day of Week (1=Mon..7=Sun)", "temporal"),
        "Date": ("Date of Transaction", "temporal"),
        "Sales": ("Daily Turnover (Target Variable)", "sales"),
        "Customers": ("Customer Foot Traffic", "sales"),
        "Open": ("Store Open Indicator (0/1)", "sales"),
        "Promo": ("Store Promotion Running (0/1)", "sales"),
        "StateHoliday": ("State Holiday Marker (0, a, b, c)", "temporal"),
        "SchoolHoliday": ("Public School Holiday Indicator", "temporal"),
        "StoreType": ("Store Model Profile (a, b, c, d)", "store"),
        "Assortment": ("Assortment Level (a=basic, b=extra, c=extended)", "store"),
        "CompetitionDistance": ("Distance to Nearest Competitor in meters", "store"),
        "LogCompetitionDistance": ("Log-scaled Competition Distance", "store"),
        "HasCompetition": ("Competitor Proximity Indicator", "store"),
        "MonthsSinceCompetition": ("Months Competitor Has Been Operating", "store"),
        "Promo2": ("Continuing Coupon Promo Participation", "store"),
        "Promo2Active": ("Coupon Promo Currently Redeemable This Month", "store"),
        "State": ("Federal State Abbreviation (e.g. HE, NW)", "geographic"),
        "StateName": ("Full Federal State Name", "geographic"),
        "Mean_TemperatureC": ("Mean Daily Temperature (Celsius)", "weather"),
        "Max_TemperatureC": ("Maximum Daily Temperature", "weather"),
        "Min_TemperatureC": ("Minimum Daily Temperature", "weather"),
        "TemperatureRange": ("Daily Temperature Swing (Max - Min)", "weather"),
        "Precipitationmm": ("Daily Precipitation in millimeters", "weather"),
        "HasRain": ("Rain / Wet Weather Event Flag", "weather"),
        "Mean_Humidity": ("Mean Daily Humidity (%)", "weather"),
        "Max_Wind_SpeedKm_h": ("Max Wind Gust Velocity (km/h)", "weather"),
        "CloudCover": ("Cloud Cover Fraction (0-8)", "weather"),
        "trend": ("Regional Google Search Index for Rossmann", "trend"),
        "NationalTrend": ("Germany-wide Google Search Index", "trend"),
        "Year": ("Calendar Year", "temporal"),
        "Month": ("Calendar Month (1-12)", "temporal"),
        "Day": ("Day of Month (1-31)", "temporal"),
        "WeekOfYear": ("ISO Week of Year (1-52)", "temporal"),
        "Quarter": ("Calendar Quarter (1-4)", "temporal"),
        "IsWeekend": ("Weekend Flag (Saturday/Sunday)", "temporal"),
    }

    for col in df.columns:
        desc, cat = descriptions.get(col, (f"Feature column {col}", "other"))
        is_num = bool(pd.api.types.is_numeric_dtype(df[col]))
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / len(df)) * 100, 2) if len(df) > 0 else 0.0

        catalog.append({
            "name": col,
            "category": cat,
            "dtype": str(df[col].dtype),
            "description": desc,
            "is_numeric": is_num,
            "missing_count": null_count,
            "missing_percent": null_pct,
        })

    return catalog
