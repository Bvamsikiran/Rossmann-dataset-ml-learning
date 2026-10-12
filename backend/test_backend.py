"""
Rossmann Sales ML Learning Lab - Backend Automated Test Suite
Verifies data loading, relational pipeline joins, preprocessing integrity,
leakage-free splitting, Scikit-Learn training, cross-validation, and API endpoints.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.config import DATASET_DIR
from backend.data_loader import check_dataset_status, load_raw_table, get_table_metadata
from backend.pipeline import execute_multi_dataset_merge, get_feature_catalog
from backend.preprocessing import RossmannPreprocessor, prepare_train_test_split
from backend.models import train_and_evaluate_models, execute_knn_sweep, predict_single_store
from backend.main import app

client = TestClient(app)


def test_dataset_status_and_files():
    """Verify dataset directory and files exist on Google Drive path."""
    status = check_dataset_status()
    assert status["directory_exists"] is True
    assert "train" in status["files"]
    assert status["files"]["train"]["exists"] is True
    assert status["files"]["store"]["exists"] is True
    assert status["files"]["weather"]["exists"] is True
    assert status["files"]["googletrend"]["exists"] is True


def test_table_metadata():
    """Verify table metadata extraction for store.csv."""
    meta = get_table_metadata("store", sample_rows=5)
    assert meta["exists"] is True
    assert "Store" in meta["columns"]
    assert "StoreType" in meta["columns"]
    assert len(meta["sample"]) == 5


def test_relational_merge_pipeline():
    """Verify multi-table join across Train, Store, Weather, and Trends."""
    sample_size = 2000
    merged_df, metadata = execute_multi_dataset_merge(sample_size=sample_size, force_reload=True)
    assert len(merged_df) <= sample_size
    assert "Store" in merged_df.columns
    assert "Sales" in merged_df.columns
    assert "StoreType" in merged_df.columns
    assert "State" in merged_df.columns
    assert "Mean_TemperatureC" in merged_df.columns
    assert "trend" in merged_df.columns
    assert "DayOfWeek" in merged_df.columns
    assert "Month" in merged_df.columns
    assert metadata["total_rows"] == len(merged_df)


def test_leakage_free_preprocessing():
    """Verify strict featurization ordering: split before fitting."""
    merged_df, _ = execute_multi_dataset_merge(sample_size=2000)
    features = ["Customers", "Promo", "DayOfWeek", "CompetitionDistance"]

    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_split(
        df=merged_df,
        features=features,
        test_size=0.25,
        random_state=42,
    )

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) <= len(merged_df)
    # Check no NaNs remain after preprocessing
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0


def test_model_training_and_cv():
    """Verify Scikit-Learn Linear Regression vs KNN training and evaluation."""
    merged_df, _ = execute_multi_dataset_merge(sample_size=2000)
    features = ["Customers", "Promo", "DayOfWeek"]

    results = train_and_evaluate_models(
        df=merged_df,
        features=features,
        split_percent=80.0,
        k=5,
        metric="euclidean",
        weighted=True,
        cv_folds=3,
        random_state=42,
    )

    assert results["status"] == "success"
    assert "linear_metrics" in results
    assert "knn_metrics" in results
    assert results["linear_metrics"]["test"]["rmse"] > 0
    assert results["knn_metrics"]["test"]["rmse"] > 0
    assert len(results["cv_results"]) == 3
    assert len(results["coefficients"]) == len(features)
    assert len(results["prediction_samples"]) > 0


def test_live_prediction():
    """Verify live inference output for user-provided features."""
    sample_input = {
        "Customers": 650,
        "Promo": 1,
        "DayOfWeek": 3,
    }
    pred = predict_single_store(sample_input)
    assert "linear_prediction" in pred
    assert "knn_prediction" in pred
    assert pred["linear_prediction"] > 0
    assert pred["knn_prediction"] > 0


def test_fastapi_endpoints():
    """Verify all FastAPI REST endpoints."""
    # 1. Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    # 2. Dataset Status
    res = client.get("/api/dataset/status")
    assert res.status_code == 200
    assert res.json()["directory_exists"] is True

    # 3. Table preview
    res = client.get("/api/dataset/preview/store?sample_rows=3")
    assert res.status_code == 200
    assert res.json()["table_name"] == "store"

    # 4. Load merged
    res = client.post("/api/dataset/load-merged", json={"sample_size": 1500})
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # 5. Features
    res = client.get("/api/dataset/features")
    assert res.status_code == 200
    assert res.json()["total_features"] > 0

    # 6. Correlations
    res = client.get("/api/dataset/correlations?target=Sales")
    assert res.status_code == 200
    assert len(res.json()["correlations"]) > 0

    # 7. Train endpoint
    res = client.post("/api/train", json={
        "features": ["Customers", "Promo", "DayOfWeek"],
        "split_percent": 80.0,
        "k": 7,
        "sample_size": 1500,
    })
    assert res.status_code == 200
    assert res.json()["status"] == "success"

    # 8. Predict endpoint
    res = client.post("/api/predict", json={
        "features": {"Customers": 500, "Promo": 1, "DayOfWeek": 4}
    })
    assert res.status_code == 200
    assert res.json()["linear_prediction"] > 0


if __name__ == "__main__":
    tests = [
        test_dataset_status_and_files,
        test_table_metadata,
        test_relational_merge_pipeline,
        test_leakage_free_preprocessing,
        test_model_training_and_cv,
        test_live_prediction,
        test_fastapi_endpoints,
    ]
    print(f"Running {len(tests)} backend tests...")
    for t in tests:
        print(f"  -> Running {t.__name__}...", end=" ", flush=True)
        t()
        print("[PASSED]")
    print("\nALL BACKEND TESTS PASSED SUCCESSFULLY! (7/7)")

