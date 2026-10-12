"""
Rossmann Sales ML Learning Lab - Data Loader Module
Handles discovery, validation, caching, and typed loading of the 6+ Rossmann dataset tables:
1. train.csv - Historical sales, customers, open status, promos
2. store.csv - Store types, assortments, competition distance & dates, Promo2
3. store_states.csv - Store to German state code mappings
4. state_names.csv - State abbreviations to full federal state names
5. weather.csv - Daily weather measurements by federal state
6. googletrend.csv - Weekly search trend interest by state and national level
7. test.csv - Test partition for competition evaluation
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

from backend.config import DATASET_DIR, REQUIRED_DATASET_FILES

logger = logging.getLogger("rossmann.data_loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Global in-memory cache for raw tables
_TABLE_CACHE: Dict[str, pd.DataFrame] = {}
_CACHE_TIMESTAMPS: Dict[str, float] = {}

# Explicit column dtypes for memory optimization and fast loading
TRAIN_DTYPES = {
    "Store": "int32",
    "DayOfWeek": "int8",
    "Sales": "float32",
    "Customers": "float32",
    "Open": "float32",
    "Promo": "int8",
    "StateHoliday": "string",
    "SchoolHoliday": "int8",
}

STORE_DTYPES = {
    "Store": "int32",
    "StoreType": "category",
    "Assortment": "category",
    "CompetitionDistance": "float32",
    "CompetitionOpenSinceMonth": "float32",
    "CompetitionOpenSinceYear": "float32",
    "Promo2": "int8",
    "Promo2SinceWeek": "float32",
    "Promo2SinceYear": "float32",
    "PromoInterval": "string",
}

STORE_STATES_DTYPES = {
    "Store": "int32",
    "State": "string",
}

STATE_NAMES_DTYPES = {
    "StateName": "string",
    "State": "string",
}

WEATHER_DTYPES = {
    "file": "string",
    "Max_TemperatureC": "float32",
    "Mean_TemperatureC": "float32",
    "Min_TemperatureC": "float32",
    "Dew_PointC": "float32",
    "MeanDew_PointC": "float32",
    "Min_DewpointC": "float32",
    "Max_Humidity": "float32",
    "Mean_Humidity": "float32",
    "Min_Humidity": "float32",
    "Max_Sea_Level_PressurehPa": "float32",
    "Mean_Sea_Level_PressurehPa": "float32",
    "Min_Sea_Level_PressurehPa": "float32",
    "Max_VisibilityKm": "float32",
    "Mean_VisibilityKm": "float32",
    "Min_VisibilitykM": "float32",
    "Max_Wind_SpeedKm_h": "float32",
    "Mean_Wind_SpeedKm_h": "float32",
    "Max_Gust_SpeedKm_h": "float32",
    "Precipitationmm": "float32",
    "CloudCover": "float32",
    "Events": "string",
    "WindDirDegrees": "float32",
}

GOOGLETREND_DTYPES = {
    "file": "string",
    "week": "string",
    "trend": "int32",
}


def get_file_path(table_key: str) -> Path:
    """Resolve full filesystem path for a given dataset key."""
    filename = REQUIRED_DATASET_FILES.get(table_key, f"{table_key}.csv")
    return DATASET_DIR / filename


def verify_dataset_directory() -> Tuple[bool, str]:
    """Verify that the target dataset directory exists and is readable."""
    if not DATASET_DIR.exists():
        msg = f"Target dataset directory does not exist: {DATASET_DIR}"
        logger.error(msg)
        return False, msg
    if not os.access(DATASET_DIR, os.R_OK):
        msg = f"Target dataset directory is not readable: {DATASET_DIR}"
        logger.error(msg)
        return False, msg
    return True, f"Dataset directory verified: {DATASET_DIR}"


def check_dataset_status() -> Dict[str, Any]:
    """
    Perform a complete health check on all required Rossmann CSV files.
    Returns file presence, byte sizes, MB, and modification timestamps.
    """
    dir_valid, msg = verify_dataset_directory()
    status: Dict[str, Any] = {
        "dataset_dir": str(DATASET_DIR),
        "directory_exists": dir_valid,
        "message": msg,
        "files": {},
        "all_required_present": True,
    }

    if not dir_valid:
        status["all_required_present"] = False
        return status

    for key, filename in REQUIRED_DATASET_FILES.items():
        file_path = DATASET_DIR / filename
        exists = file_path.exists()
        if not exists:
            status["all_required_present"] = False

        size_bytes = file_path.stat().st_size if exists else 0
        mod_time = time.ctime(file_path.stat().st_mtime) if exists else None

        status["files"][key] = {
            "name": filename,
            "exists": exists,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "modified_time": mod_time,
            "path": str(file_path),
        }

    return status


def load_raw_table(
    table_key: str,
    nrows: Optional[int] = None,
    force_reload: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load a specific Rossmann dataset table with typed memory optimization.
    Utilizes in-memory caching for auxiliary metadata tables (store, weather, states).
    """
    file_path = get_file_path(table_key)
    if not file_path.exists():
        raise FileNotFoundError(f"Requested dataset file does not exist: {file_path}")

    # Return cached dataframe if available and full load requested
    if use_cache and not force_reload and nrows is None and table_key in _TABLE_CACHE:
        logger.debug(f"Returning cached table for '{table_key}'")
        return _TABLE_CACHE[table_key].copy()

    logger.info(f"Loading '{table_key}' from {file_path.name} (nrows={nrows})...")
    start_time = time.time()

    if table_key == "train":
        df = pd.read_csv(
            file_path,
            nrows=nrows,
            parse_dates=["Date"],
            dtype=TRAIN_DTYPES,
            low_memory=False,
        )
    elif table_key == "store":
        df = pd.read_csv(
            file_path,
            nrows=nrows,
            dtype=STORE_DTYPES,
            low_memory=False,
        )
    elif table_key == "store_states":
        df = pd.read_csv(
            file_path,
            nrows=nrows,
            dtype=STORE_STATES_DTYPES,
            low_memory=False,
        )
    elif table_key == "state_names":
        df = pd.read_csv(
            file_path,
            nrows=nrows,
            dtype=STATE_NAMES_DTYPES,
            low_memory=False,
        )
    elif table_key == "weather":
        df = pd.read_csv(
            file_path,
            nrows=nrows,
            parse_dates=["Date"],
            dtype=WEATHER_DTYPES,
            low_memory=False,
        )
    elif table_key == "googletrend":
        df = pd.read_csv(
            file_path,
            nrows=nrows,
            dtype=GOOGLETREND_DTYPES,
            low_memory=False,
        )
    elif table_key == "test":
        df = pd.read_csv(
            file_path,
            nrows=nrows,
            parse_dates=["Date"],
            low_memory=False,
        )
    else:
        df = pd.read_csv(file_path, nrows=nrows, low_memory=False)

    elapsed = time.time() - start_time
    logger.info(f"Loaded '{table_key}' in {elapsed:.2f}s (shape: {df.shape})")

    # Cache tables if full table was loaded
    if use_cache and nrows is None and table_key in ["store", "store_states", "state_names", "weather", "googletrend"]:
        _TABLE_CACHE[table_key] = df
        _CACHE_TIMESTAMPS[table_key] = time.time()

    return df.copy()


def get_table_metadata(table_key: str, sample_rows: int = 5) -> Dict[str, Any]:
    """
    Inspect a table and return structural metadata, column data types,
    null value counts, distinct value counts, and an immediate JSON sample.
    """
    file_path = get_file_path(table_key)
    if not file_path.exists():
        return {
            "table_name": table_key,
            "exists": False,
            "error": f"File {file_path.name} not found in {DATASET_DIR}",
        }

    # Load small slice for preview
    df_preview = pd.read_csv(file_path, nrows=sample_rows, low_memory=False)
    
    # Check total rows without reading entire file into RAM if large
    total_rows = 0
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            total_rows = sum(1 for _ in f) - 1
    except Exception:
        total_rows = len(df_preview)

    null_counts: Dict[str, int] = {}
    dtypes: Dict[str, str] = {}
    for col in df_preview.columns:
        null_counts[col] = int(df_preview[col].isnull().sum())
        dtypes[col] = str(df_preview[col].dtype)

    # Sanitize NaN/Inf for JSON serialization
    clean_sample = df_preview.replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict(orient="records")

    return {
        "table_name": table_key,
        "file_name": file_path.name,
        "exists": True,
        "row_count": total_rows,
        "column_count": len(df_preview.columns),
        "columns": list(df_preview.columns),
        "dtypes": dtypes,
        "null_counts": null_counts,
        "sample": clean_sample,
    }


def clear_cache() -> Dict[str, Any]:
    """Evict all in-memory table caches to free memory."""
    evicted_keys = list(_TABLE_CACHE.keys())
    _TABLE_CACHE.clear()
    _CACHE_TIMESTAMPS.clear()
    logger.info(f"Evicted table cache for: {evicted_keys}")
    return {"status": "cache_cleared", "evicted_tables": evicted_keys}
