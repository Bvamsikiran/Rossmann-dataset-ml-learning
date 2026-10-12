"""
Rossmann Sales ML Learning Lab - Configuration & Constants
Defines dataset filesystem paths, server parameters, feature definitions,
and sampling limits for scalable local execution.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

# Primary dataset root path specified by the user
DEFAULT_DATASET_DIR = Path(r"G:\My Drive\Docs\CBP SEM -5\mds cbp\source materials\dataset\rossmann")

# Environment variable override if running in alternative environments
DATASET_DIR = Path(os.getenv("ROSSMANN_DATASET_DIR", str(DEFAULT_DATASET_DIR)))

# API Server Host and Port settings
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8000"))

# Sampling configuration to maintain real-time responsiveness
DEFAULT_SAMPLE_ROWS = 25000
MIN_SAMPLE_ROWS = 1000
MAX_SAMPLE_ROWS = 250000

# Required files dictionary for the Rossmann family
REQUIRED_DATASET_FILES: Dict[str, str] = {
    "train": "train.csv",
    "store": "store.csv",
    "store_states": "store_states.csv",
    "state_names": "state_names.csv",
    "weather": "weather.csv",
    "googletrend": "googletrend.csv",
    "test": "test.csv",
    "sample_submission": "sample_submission.csv",
}

# Core feature definitions
DEFAULT_BASE_FEATURES: List[str] = [
    "Customers",
    "Promo",
    "DayOfWeek",
    "SchoolHoliday",
    "Store",
    "Open",
]

DEFAULT_STORE_FEATURES: List[str] = [
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
]

DEFAULT_WEATHER_FEATURES: List[str] = [
    "Mean_TemperatureC",
    "Precipitationmm",
    "Mean_Humidity",
    "Max_Wind_SpeedKm_h",
    "CloudCover",
]

DEFAULT_TREND_FEATURES: List[str] = [
    "trend",
]

DEFAULT_DATETIME_FEATURES: List[str] = [
    "Year",
    "Month",
    "Day",
    "WeekOfYear",
    "IsWeekend",
]

# State mapping fallback in case of missing state rows
GERMAN_STATE_FALLBACK_MAP: Dict[str, str] = {
    "BW": "BadenWuerttemberg",
    "BY": "Bayern",
    "BE": "Berlin",
    "BB": "Brandenburg",
    "HB": "Bremen",
    "HH": "Hamburg",
    "HE": "Hessen",
    "MV": "MecklenburgVorpommern",
    "NI": "Niedersachsen",
    "NW": "NordrheinWestfalen",
    "RP": "RheinlandPfalz",
    "SL": "Saarland",
    "SN": "Sachsen",
    "ST": "SachsenAnhalt",
    "SH": "SchleswigHolstein",
    "TH": "Thueringen",
}
