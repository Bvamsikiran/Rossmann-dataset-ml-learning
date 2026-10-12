"""
Rossmann Sales ML Learning Lab - FastAPI REST Application
Exposes production REST endpoints for:
- Multi-dataset health check & status
- Individual dataset table previews
- Relational pipeline joining (Train + Store + Weather + Google Trends)
- Dynamic feature catalog & correlation analysis
- Preprocessing & data audit
- Model training & cross-validation (Linear Regression vs KNN)
- KNN hyperparameter sweep
- Real-time retail store sales prediction
"""

import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import DATASET_DIR, HOST, PORT, DEFAULT_SAMPLE_ROWS
from backend.schemas import (
    HealthResponse,
    DatasetStatusResponse,
    TablePreviewResponse,
    DatasetFeaturesResponse,
    MergePipelineRequest,
    MergedDatasetResponse,
    PreprocessOptionsRequest,
    PreprocessAuditResponse,
    ModelTrainingRequest,
    ModelTrainingResponse,
    KNNSweepRequest,
    KNNSweepResponse,
    SinglePredictionRequest,
    SinglePredictionResponse,
)
from backend.data_loader import (
    check_dataset_status,
    get_table_metadata,
    clear_cache,
    load_raw_table,
)
from backend.pipeline import (
    execute_multi_dataset_merge,
    get_feature_catalog,
)
from backend.preprocessing import RossmannPreprocessor
from backend.models import (
    train_and_evaluate_models,
    execute_knn_sweep,
    predict_single_store,
)

logger = logging.getLogger("rossmann.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Initialize FastAPI application
app = FastAPI(
    title="Rossmann Sales ML Learning Lab API",
    description="High-performance machine learning backend for the Rossmann sales forecasting laboratory.",
    version="2.0.0",
)

# Enable Cross-Origin Resource Sharing (CORS) for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global active dataset reference
_CURRENT_DATASET: Optional[pd.DataFrame] = None


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """System health check and dataset readiness endpoint."""
    dir_exists = DATASET_DIR.exists()
    return {
        "status": "healthy",
        "version": "2.0.0",
        "dataset_dir": str(DATASET_DIR),
        "datasets_ready": dir_exists,
    }


@app.get("/api/dataset/status", response_model=DatasetStatusResponse)
def get_dataset_files_status():
    """Verify presence, sizes, and timestamps of all required Rossmann CSV files."""
    status_info = check_dataset_status()
    return status_info


@app.get("/api/dataset/preview/{table_name}", response_model=TablePreviewResponse)
def get_single_table_preview(table_name: str, sample_rows: int = Query(default=5, ge=1, le=50)):
    """Retrieve column schema, null counts, and a sample preview for any individual CSV table."""
    meta = get_table_metadata(table_name, sample_rows=sample_rows)
    if not meta.get("exists", False):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table '{table_name}' not found in dataset directory.",
        )
    return {
        "table_name": meta["table_name"],
        "row_count": meta["row_count"],
        "column_count": meta["column_count"],
        "columns": meta["columns"],
        "dtypes": meta["dtypes"],
        "null_counts": meta["null_counts"],
        "sample": meta["sample"],
    }


@app.post("/api/dataset/load-merged", response_model=MergedDatasetResponse)
def load_and_merge_datasets(req: MergePipelineRequest):
    """
    Execute the relational pipeline to merge train, store, store_states,
    state_names, weather, and googletrend on common keys.
    """
    global _CURRENT_DATASET
    try:
        merged_df, metadata = execute_multi_dataset_merge(
            sample_size=req.sample_size,
            include_store=req.include_store,
            include_weather=req.include_weather,
            include_trends=req.include_trends,
            seed=req.random_seed or 42,
        )
        _CURRENT_DATASET = merged_df

        # Calculate numerical summary statistics for key features
        summary_stats: Dict[str, Dict[str, float]] = {}
        numeric_cols = [c for c in merged_df.columns if pd.api.types.is_numeric_dtype(merged_df[c])]
        for col in numeric_cols[:15]:
            s = merged_df[col].dropna()
            if len(s) > 0:
                summary_stats[col] = {
                    "mean": round(float(s.mean()), 2),
                    "median": round(float(s.median()), 2),
                    "std": round(float(s.std()), 2),
                    "min": round(float(s.min()), 2),
                    "max": round(float(s.max()), 2),
                }

        # Safe sample preview
        preview_rows = merged_df.head(15).replace({np.nan: None, np.inf: None, -np.inf: None})
        sample_preview = preview_rows.to_dict(orient="records")

        return {
            "status": "success",
            "total_loaded_rows": metadata["total_rows"],
            "total_columns": metadata["total_columns"],
            "columns": list(merged_df.columns),
            "sample_preview": sample_preview,
            "summary_stats": summary_stats,
            "missing_summary": metadata["missing_summary"],
            "tables_joined": metadata["tables_joined"],
        }
    except Exception as e:
        logger.error(f"Merge pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Relational merge pipeline failed: {str(e)}",
        )


@app.get("/api/dataset/features", response_model=DatasetFeaturesResponse)
def get_features_list():
    """Retrieve the annotated catalog of features available for selection and training."""
    global _CURRENT_DATASET
    try:
        catalog = get_feature_catalog(_CURRENT_DATASET)
        default_selected = ["Customers", "Promo", "DayOfWeek", "SchoolHoliday", "Store", "CompetitionDistance"]
        avail_names = [f["name"] for f in catalog]
        final_selected = [f for f in default_selected if f in avail_names]

        return {
            "total_features": len(catalog),
            "features": catalog,
            "default_selected": final_selected,
        }
    except Exception as e:
        logger.error(f"Feature catalog error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/dataset/correlations")
def get_correlations(target: str = "Sales"):
    """Compute Pearson correlation matrix of numerical features against the target variable."""
    global _CURRENT_DATASET
    df = _CURRENT_DATASET
    if df is None:
        df, _ = execute_multi_dataset_merge(sample_size=10000)

    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not present in dataset.")

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    unique_num_cols = list(dict.fromkeys(num_cols))
    target_series = pd.to_numeric(df[target], errors="coerce")
    corr_matrix: List[Dict[str, Any]] = []

    for col in unique_num_cols:
        if col == target:
            continue
        try:
            col_series = pd.to_numeric(df[col], errors="coerce")
            valid_mask = col_series.notnull() & target_series.notnull()
            if valid_mask.sum() > 20:
                c_val = float(col_series[valid_mask].corr(target_series[valid_mask]))
                if not np.isnan(c_val):
                    corr_matrix.append({
                        "feature": col,
                        "correlation": round(c_val, 4),
                        "abs_correlation": round(abs(c_val), 4),
                    })
        except Exception:
            continue

    corr_matrix.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return {"target": target, "correlations": corr_matrix}


@app.post("/api/preprocess/audit", response_model=PreprocessAuditResponse)
def run_preprocess_audit(req: PreprocessOptionsRequest):
    """Run data cleaning and transformation audit, returning before/after distribution shifts."""
    global _CURRENT_DATASET
    df = _CURRENT_DATASET
    if df is None:
        df, _ = execute_multi_dataset_merge(sample_size=15000)

    preprocessor = RossmannPreprocessor(
        imputation_strategy=req.imputation_strategy,
        clip_outliers=req.clip_outliers,
        outlier_std_threshold=req.outlier_std_threshold,
        drop_closed=req.drop_closed,
        remove_duplicates=req.remove_duplicates,
    )

    clean_df = preprocessor.filter_raw_data(df)
    preprocessor.fit(clean_df)
    transformed_df = preprocessor.transform(clean_df)

    audit_dict = preprocessor.audit_.to_dict()
    return {
        "original_rows": audit_dict["original_rows"],
        "processed_rows": audit_dict["final_rows"],
        "rows_dropped": audit_dict["original_rows"] - audit_dict["final_rows"],
        "dropped_closed_count": audit_dict["closed_stores_dropped"],
        "duplicate_rows_count": audit_dict["duplicates_removed"],
        "imputed_columns": {k: 1 for k in audit_dict["imputed_values"].keys()},
        "encoded_features": audit_dict["encoded_columns"],
    }


@app.post("/api/train", response_model=ModelTrainingResponse)
def train_models(req: ModelTrainingRequest):
    """
    Train Linear Regression and K-Nearest Neighbors models using Scikit-Learn
    on the merged dataset with the specified features and parameters.
    """
    global _CURRENT_DATASET
    df = _CURRENT_DATASET
    sample_size = req.sample_size or DEFAULT_SAMPLE_ROWS

    if df is None or len(df) != sample_size:
        logger.info(f"Loading merged dataset for training (sample_size={sample_size})...")
        df, _ = execute_multi_dataset_merge(sample_size=sample_size, seed=req.seed or 42)
        _CURRENT_DATASET = df

    try:
        results = train_and_evaluate_models(
            df=df,
            features=req.features,
            split_percent=req.split_percent,
            k=req.k,
            metric=req.metric,
            weighted=req.weighted_knn,
            cv_folds=req.cv_folds,
            random_state=req.seed or 42,
        )
        return results
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(e)}",
        )


@app.post("/api/train/knn-sweep", response_model=KNNSweepResponse)
def run_knn_sweep(req: KNNSweepRequest):
    """Evaluate KNN performance across a sweep of K values to produce the bias-variance curve."""
    global _CURRENT_DATASET
    df = _CURRENT_DATASET
    if df is None:
        df, _ = execute_multi_dataset_merge(sample_size=req.sample_size or 10000)
        _CURRENT_DATASET = df

    try:
        sweep_data = execute_knn_sweep(
            df=df,
            features=req.features,
            k_values=req.k_values,
            metric=req.metric,
            weighted=req.weighted,
            sample_size=req.sample_size or 10000,
        )
        return sweep_data
    except Exception as e:
        logger.error(f"KNN sweep failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=SinglePredictionResponse)
def predict_store_sales(req: SinglePredictionRequest):
    """Real-time store turnover prediction based on user-supplied parameters."""
    try:
        prediction = predict_single_store(req.features)
        return prediction
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/cache/clear")
def clear_backend_cache():
    """Clear in-memory raw tables and merged cache."""
    global _CURRENT_DATASET
    _CURRENT_DATASET = None
    res = clear_cache()
    return res
